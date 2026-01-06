# path: src/models/fusion/evidence_fusion.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from src.models.evidential.evidence import dirichlet_mean_binary, beta_variance_binary


@dataclass
class EvidenceFusionConfig:
    eps: float = 1e-6
    detach_weights: bool = True
    type: str = "evidence_weighted"  # or "avg_prob" ablation


class EvidenceFusion(torch.nn.Module):
    """
    Evidence-level fusion:
    - Eq.(10): piH = SH / (SH+SL+eps)
    - Eq.(11): e* = piH eH + piL eL
    - Eq.(12): p* from alpha* = e* + 1
    """

    def __init__(self, cfg: EvidenceFusionConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        alpha_h: torch.Tensor, e_h: torch.Tensor,
        alpha_l: torch.Tensor, e_l: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cfg.type == "avg_prob":
            # Ablation: average probabilities (not paper fusion)
            p_h = dirichlet_mean_binary(alpha_h)
            p_l = dirichlet_mean_binary(alpha_l)
            p_f = 0.5 * (p_h + p_l)
            alpha_f = torch.cat([1.0 - p_f, p_f], dim=1) * 10.0  # pseudo alpha for uncertainty (NOT paper)
            u_f = beta_variance_binary(alpha_f)
            return alpha_f, p_f, u_f

        sh = alpha_h.sum(dim=1, keepdim=True)
        sl = alpha_l.sum(dim=1, keepdim=True)

        pi_h = sh / (sh + sl + self.cfg.eps)
        pi_l = 1.0 - pi_h

        if self.cfg.detach_weights:
            pi_h = pi_h.detach()
            pi_l = pi_l.detach()

        e_f = pi_h * e_h + pi_l * e_l  # Eq.(11)
        alpha_f = e_f + 1.0
        p_f = dirichlet_mean_binary(alpha_f)  # Eq.(12)

        u_f = beta_variance_binary(alpha_f)
        return alpha_f, p_f, u_f
