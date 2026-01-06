# path: src/utils/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


def dice_coeff(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return ((2.0 * inter + eps) / (union + eps)).mean()


def miou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum(dim=(2, 3))
    union = ((pred + target) > 0.5).float().sum(dim=(2, 3))
    return ((inter + eps) / (union + eps)).mean()


def precision_recall(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    tp = (pred * target).sum(dim=(2, 3))
    fp = (pred * (1.0 - target)).sum(dim=(2, 3))
    fn = ((1.0 - pred) * target).sum(dim=(2, 3))
    prec = (tp + eps) / (tp + fp + eps)
    rec = (tp + eps) / (tp + fn + eps)
    return prec.mean(), rec.mean()


@dataclass
class ECEConfig:
    bins: int = 10


def expected_calibration_error(prob: torch.Tensor, target: torch.Tensor, cfg: ECEConfig) -> torch.Tensor:
    """
    Pixel-level ECE with cfg.bins bins (Sec.4.1).
    prob: (B,1,H,W) in [0,1]
    target: (B,1,H,W) in {0,1}
    """
    p = prob.flatten()
    y = target.flatten()
    correct = ((p > 0.5).float() == y).float()

    bins = cfg.bins
    ece = torch.zeros((), device=prob.device)

    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        mask = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        conf = p[mask].mean()
        acc = correct[mask].mean()
        w = mask.float().mean()
        ece = ece + w * torch.abs(acc - conf)

    return ece


def npv_image_level(prob: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """
    Image-level NPV (Sec.4.1):
    - confidence that a polyp exists in image = max p*(x,y)
    - predict polyp-negative if max < threshold
    NPV = TN / (TN + FN)
    """
    b = prob.shape[0]
    maxp = prob.view(b, -1).max(dim=1).values
    y_img = (target.view(b, -1).sum(dim=1) > 0).float()

    pred_neg = (maxp < threshold).float()
    tn = ((pred_neg == 1.0) & (y_img == 0.0)).float().sum()
    fn = ((pred_neg == 1.0) & (y_img == 1.0)).float().sum()
    return tn / (tn + fn + eps)
