import torch
import pickle
import pytorch_lightning as pl

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from utils.utils import convert_cond
from loop_extraction.src.utils.constants import *


class SeqCollator:
    def __init__(self, pad_token):
        self.pad_token = pad_token

    def __call__(self, features):
        seq_list, cond_list = ([] for _ in range(2))

        batch = {}
        for feature in features:
            seq_list.append(feature[0])
            cond_list.append(feature[1])

        seq_list = pad_sequence(seq_list, batch_first=True, padding_value=self.pad_token)

        batch["seq"] = seq_list
        batch["cond"] = cond_list

        return batch


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        file_list,
        tokenizer,
        batch_size=32,
        num_workers=4,
        random_drop=0.5,
        pin_memory=True,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.pad_token = tokenizer.encode([PAD_TOKEN])[0]
        self.collator = SeqCollator(self.pad_token)

        self.train_data = DatasetSampler(file_list[0], tokenizer, random_drop)
        self.val_data = DatasetSampler(file_list[1], tokenizer, random_drop)
        self.test_data = DatasetSampler(file_list[2], tokenizer, random_drop)

    def return_dataloader(self):
        return (
            DataLoader(
                self.train_data,
                shuffle=True,
                collate_fn=self.collator,
                batch_size=self.hparams.batch_size,
                pin_memory=self.hparams.pin_memory,
                num_workers=self.hparams.num_workers,
            ),
            DataLoader(
                self.val_data,
                shuffle=False,
                collate_fn=self.collator,
                batch_size=self.hparams.batch_size,
                pin_memory=self.hparams.pin_memory,
                num_workers=self.hparams.num_workers,
            ),
            DataLoader(
                self.test_data,
                shuffle=True,
                collate_fn=self.collator,
                batch_size=self.hparams.batch_size,
                pin_memory=self.hparams.pin_memory,
                num_workers=self.hparams.num_workers,
            ),
        )


class DatasetSampler(Dataset):
    def __init__(self, file_list, tokenizer, random_drop=0.5):
        self.file_list = file_list
        self.tokenizer = tokenizer
        self.random_drop = random_drop

        # add start token
        self.tokenizer.add_tokens([START_TOKEN])
        self.start_token = self.tokenizer.encode_meta(START_TOKEN)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]

        # load data
        with open(file_path, "rb") as f:
            x = pickle.load(f)

        # generate conditions
        cond = convert_cond(x, self.tokenizer, random_drop=self.random_drop)

        cond = self.start_token + cond
        loop = self.tokenizer.encode(x["loop"])
        cond = cond + [loop[0]]

        # full sequence
        seq = torch.tensor(cond + loop[1:], dtype=torch.long)
        cond = torch.tensor(cond, dtype=torch.long)

        return seq, cond
