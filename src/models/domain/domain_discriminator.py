# path: src/models/domain/domain_discriminator.py
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DomainDiscriminatorConfig:
    in_dim: int
    hidden_dim: int = 256
    num_domains: int = 6  # set via config/meta
    dropout: float = 0.1


class DomainDiscriminator(nn.Module):
    """
    2-layer MLP domain classifier (Training Details p.17; Sec.3.1).
    Input: pooled feature vector from encoder.
    Output: logits over domains.
    """

    def __init__(self, cfg: DomainDiscriminatorConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
