# path: src/models/losses/dice.py
from __future__ import annotations

import torch


def dice_loss(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice loss.
    prob: (B,1,H,W) in [0,1]
    target: (B,1,H,W) in {0,1}
    """
    prob = prob.contiguous()
    target = target.contiguous()
    inter = (prob * target).sum(dim=(2, 3))
    union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()
