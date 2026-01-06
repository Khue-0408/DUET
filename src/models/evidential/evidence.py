# path: src/models/evidential/evidence.py
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def logits_to_dirichlet(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Eq. (4)(5): evidence e = softplus(logits), alpha = e + 1
    Returns:
      alpha: (B,K,H,W)
      e:     (B,K,H,W)
    """
    e = F.softplus(logits)
    alpha = e + 1.0
    return alpha, e


def dirichlet_mean(alpha: torch.Tensor) -> torch.Tensor:
    # mean = alpha / sum(alpha)
    s = alpha.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return alpha / s


def dirichlet_mean_binary(alpha: torch.Tensor) -> torch.Tensor:
    """
    Eq. (6): p_hat = alpha_polyp / sum_c alpha_c
    ASSUMPTION: channel 1 = polyp, channel 0 = background (consistent across repo).
    """
    s = alpha.sum(dim=1, keepdim=True).clamp_min(1e-12)
    p_polyp = alpha[:, 1:2, :, :] / s
    return p_polyp


def beta_variance_binary(alpha: torch.Tensor) -> torch.Tensor:
    """
    Expected variance of Bernoulli under Beta (binary Dirichlet) (end Sec.3.3):
    Var(p) = a*b / ((a+b)^2 * (a+b+1))
    """
    a = alpha[:, 1:2, :, :]
    b = alpha[:, 0:1, :, :]
    s = (a + b).clamp_min(1e-12)
    return (a * b) / (s * s * (s + 1.0))
