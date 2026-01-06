# path: src/models/encoders/efficientnet_b4.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn
import timm


@dataclass
class EfficientNetB4Config:
    name: str = "tf_efficientnet_b4"
    pretrained: bool = True
    out_indices: Sequence[int] = (1, 2, 3, 4)


class EfficientNetB4Backbone(nn.Module):
    """
    EfficientNet-B4 backbone for EH (Sec.3.1, Sec.3.2).
    Uses timm features_only interface to output pyramid features.

    Paper trace:
    - EH is EfficientNet-B4 (Sec.3.1, p.10)
    - Multi-scale features feed FPN-style decoder (Sec.3.2-3.3)
    """

    def __init__(self, cfg: EfficientNetB4Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            cfg.name,
            pretrained=cfg.pretrained,
            features_only=True,
            out_indices=tuple(cfg.out_indices),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(x)

    def load_stage1_weights(self, ckpt_path: str, strict: bool = False) -> None:
        sd = torch.load(ckpt_path, map_location="cpu")
        if "model" in sd:
            sd = sd["model"]
        missing, unexpected = self.backbone.load_state_dict(sd, strict=strict)
        if not strict:
            _ = (missing, unexpected)
