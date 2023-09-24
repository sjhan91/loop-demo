import torch
import random

from torch.nn import functional as F


def sizeof_number(number, currency=None):
    """
    Refer to https://gist.github.com/Abdelkrim/02e604fc38e7f35969e5552f13e4af0a

    format values per thousands : K-thousands, M-millions, B-billions.

    parameters:
    -----------
    number is the number you want to format
    currency is the prefix that is displayed if provided (€, $, £...)

    """

    currency = "" if currency is None else currency + " "

    for unit in ["", "K", "M"]:
        if abs(number) < 1000.0:
            return f"{currency}{number:0.2f} {unit}"
        number /= 1000.0

    return f"{currency}{number:6.2f} B"


def convert_cond(x, tokenizer, random_drop=0):
    """
    Encode music tokens with random drop
    """

    cond_keys_normal = [
        "inst",
        "mean_pitch",
        "mean_tempo",
        "mean_velocity",
        "mean_duration",
    ]

    cond_keys_special = [
        "density",
        "chord",
        "bar_length",
    ]

    cond = []
    # normal meta_tokens - sort for inst
    for key in cond_keys_normal:
        if random.random() >= random_drop:
            cond += sorted(tokenizer.encode(x[key]))

    # special meta_tokens
    for key in cond_keys_special:
        if random.random() >= random_drop:
            cond += tokenizer.encode_meta(x[key])

    return cond


@torch.no_grad()
def generate(
    x,
    model,
    device,
    tokens,
    temp=1,
    top_k=None,
    sample=False,
    max_length=1024,
):
    """
    Refer to https://colab.research.google.com/github/facebookresearch/xformers/blob/main/docs/source/xformers_mingpt.ipynb
    Refer to https://github.com/karpathy/minGPT/blob/master/mingpt/utils.py

    tokens[0]: start_token
    tokens[1]: end_token
    """

    model.eval()

    def top_k_logits(logits, k):
        v, _ = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[:, [-1]]] = -float("Inf")

        return out

    x = x.to(device)
    x = x.unsqueeze(0) if x.ndim == 1 else x

    count = 0
    cond_len = x.shape[1]

    while True:
        logits = model(x)

        # pluck the logits at the final step and scale by temp
        logits = logits[:, -1, :] / temp

        # optionally crop prob to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)

        # apply softmax to convert to prob
        probs = F.softmax(logits, dim=-1)

        # sample from the dist or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)

        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

        count += 1
        if ix == tokens[1] or count >= max_length:
            break

    # escape the batch dimension
    x = x[0, cond_len:].detach().cpu()

    # add bar_token
    x = torch.cat([torch.tensor([tokens[0]]), x])

    return x


def trim_tails(loop, end_time, fs):
    play_time = int(fs * end_time)

    loop_fs = loop.fluidsynth(fs=fs)
    diff = loop_fs.shape[0] - play_time

    rand = random.uniform(0, 0.2)
    margin = int(diff * rand) if diff > 0 else 0

    return loop_fs[: play_time + margin]
