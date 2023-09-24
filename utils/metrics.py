import torch
import numpy as np

from scipy import linalg
from torch.nn import functional as F


@torch.no_grad()
def compute_ppl(data_set, model, device, pad_idx=0):
    ppl = []

    model.eval()
    for batch_idx, batch in enumerate(data_set):
        inputs, targets = batch["seq"][:, :-1], batch["seq"][:, 1:]

        inputs = inputs.to(device)
        targets = targets.to(device)

        # forward pass
        logits = model(inputs)

        # compute the loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=pad_idx,
        )

        ppl.append(loss)

    ppl = torch.exp(torch.stack(ppl).mean())
    ppl = ppl.detach().cpu().numpy()

    return ppl


def compute_fid(feat1, feat2, eps=1e-6):
    """
    Refer to https://github.com/leognha/PyTorch-FID-score/blob/master/fid_score.py
    """

    mu1, sigma1 = np.mean(feat1, axis=0), np.cov(feat1.T)
    mu2, sigma2 = np.mean(feat2, axis=0), np.cov(feat2.T)

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # singular product
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # nmerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))

        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    mu_diff = diff.dot(diff)
    cov_diff = np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    return mu_diff + cov_diff
