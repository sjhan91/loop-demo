import torch
import torch.nn as nn
import pytorch_lightning as pl

from utils.swiglu import SwiGLU
from utils.rmsnorm import RMSNorm

from torch.nn import functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding


class Attention(nn.Module):
    def __init__(self, dim_model=768, num_heads=12, dropout=0):
        super().__init__()

        self.num_heads = num_heads
        self.dropout = dropout

        self.to_qkv = nn.Linear(dim_model, dim_model * 3, bias=False)
        self.to_out = nn.Linear(dim_model, dim_model, bias=False)

        self.resid_drop = nn.Dropout(dropout, inplace=False)
        self.rotary_emb = RotaryEmbedding(dim=32)

    def forward(self, x):
        # input projection
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv)

        # rotary embedding
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        # attention - flash attention
        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            x = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout)

        # output projection
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.resid_drop(self.to_out(x))

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        dim_model=768,
        num_layers=12,
        num_heads=12,
        multiplier=4,
        dropout=0,
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim_model)
        self.norm2 = RMSNorm(dim_model)

        self.layers = nn.ModuleList([])
        self.dropout = nn.Dropout(dropout, inplace=False)

        ffn_hidden_layer = int(multiplier * dim_model)

        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim_model=dim_model, num_heads=num_heads, dropout=dropout),
                        SwiGLU(in_feat=dim_model, hidden_feat=ffn_hidden_layer),
                    ]
                )
            )

    def forward(self, x):
        for attention, ffn in self.layers:
            x = attention(self.norm1(x)) + x
            x = self.dropout(ffn(self.norm2(x))) + x

        return x


class GPT(pl.LightningModule):
    """
    The full GPT language model
    Refer to https://github.com/facebookresearch/xformers/blob/main/examples/microGPT.py
    """

    def __init__(
        self,
        vocab_size,
        pad_idx,
        dim_model=768,
        num_layers=12,
        num_heads=12,
        multiplier=4,
        lr=3e-4,
        dropout=0,
        weight_decay=0.1,
        max_length=2048,
        warm_up=2000,
        acc_batch=1,
    ):
        super().__init__()

        self.loss = 0
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.history = {}
        self.history["train_loss"] = []
        self.history["val_loss"] = []

        self.pad_idx = pad_idx
        self.vocab_size = vocab_size

        # embedding
        self.embedding = nn.Embedding(vocab_size, dim_model)

        # Transformers
        self.transformer = Transformer(
            dim_model=dim_model,
            num_layers=num_layers,
            num_heads=num_heads,
            multiplier=multiplier,
            dropout=dropout,
        )

        # decoder head
        self.ln_f = nn.LayerNorm(dim_model)
        self.head = nn.Linear(dim_model, vocab_size, bias=False)

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def configure_optimizers(self):
        # create the optimizer and the training schedule:
        # handle the per-param weight decay
        no_decay = ["bias", "norm", "embedding"]
        params_decay = [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)]
        params_nodecay = [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)]
        optim_groups = [
            {"params": params_decay, "weight_decay": self.hparams.weight_decay},
            {"params": params_nodecay, "weight_decay": 0},
        ]

        # start with a warm up, ramp up then cosine
        opt = AdamW(optim_groups, lr=self.hparams.lr, betas=(0.9, 0.95))

        def warm_decay(step):
            if step < self.hparams.warm_up:
                return step / self.hparams.warm_up
            return self.hparams.warm_up**0.5 * step**-0.5

        sch = {
            "scheduler": LambdaLR(opt, lr_lambda=[warm_decay, warm_decay]),
            "interval": "step",
            "frequency": 1,
            "name": "learning_rate",
        }

        return [opt], [sch]

    def forward(self, x):
        # predict the next tokens
        x = self.embedding(x)
        x = self.transformer(x)

        # translate the predictions into tokens
        x = self.ln_f(x)
        logits = self.head(x)

        return logits

    def pre_allocation(self, batch_size, data, margin=16):
        max_length = self.hparams.max_length + margin

        rand = torch.randint(self.vocab_size, (batch_size, max_length))
        rand = rand.type_as(data)

        logits = self(rand)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            rand.reshape(-1),
            ignore_index=self.pad_idx,
        )

        loss.backward()

    def training_step(self, batch, batch_idx):
        inputs, targets = batch["seq"][:, :-1], batch["seq"][:, 1:]
        batch_size = inputs.shape[0]

        opt = self.optimizers()
        sch = self.lr_schedulers()

        # pre-allocate GPU -> it is effective
        if batch_idx == 0:
            self.pre_allocation(batch_size, inputs, margin=24)
            opt.zero_grad()

        # forward pass
        logits = self(inputs)

        # compute the loss
        local_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.pad_idx,
        )

        # scale loss by 1 / N
        local_loss /= self.hparams.acc_batch
        self.loss += local_loss

        self.manual_backward(local_loss)

        # accumulate grads
        if (batch_idx + 1) % self.hparams.acc_batch == 0:
            opt.step()
            sch.step()
            opt.zero_grad()

            self.log("train_loss", self.loss, batch_size=batch_size, prog_bar=True)
            self.history["train_loss"].append(self.loss.item())

            self.loss = 0

        return local_loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch["seq"][:, :-1], batch["seq"][:, 1:]
        batch_size = inputs.shape[0]

        # forward pass
        logits = self(inputs)

        # compute the loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.pad_idx,
        )

        self.log("val_loss", loss, batch_size=batch_size, prog_bar=True, sync_dist=True)
        self.history["val_loss"].append(loss.item())

        return loss
