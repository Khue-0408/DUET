# path: src/models/encoders/vit_mae.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import timm


@dataclass
class ViTMAEInit:
    mode: str = "timm_mae"          # {timm_mae, ckpt_path, none}
    timm_name: str = "vit_base_patch16_224.mae"
    ckpt_path: str = ""


@dataclass
class ViTMAEConfig:
    name: str = "vit_base_patch16_224"
    img_size: int = 512
    patch_size: int = 16
    feature_blocks: Sequence[int] = (2, 5, 8, 11)  # ASSUMPTION
    mae_init: ViTMAEInit = ViTMAEInit()
    embed_dim: int = 768


class ViTMAEBackbone(nn.Module):
    """
    ViT-Base/16 encoder for EL (Sec.3.2).
    Paper: init from MAE (Sec.3.2). Paper doesn't provide checkpoint path.

    Repo options:
    - mode=timm_mae: load timm MAE weights by model name (ASSUMPTION convenience)
    - mode=ckpt_path: user-provided checkpoint
    - mode=none: random init
    """

    def __init__(self, cfg: ViTMAEConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg.mae_init.mode == "timm_mae":
            self.vit = timm.create_model(cfg.mae_init.timm_name, pretrained=True)
        else:
            self.vit = timm.create_model(cfg.name, pretrained=False)

        if cfg.mae_init.mode == "ckpt_path" and cfg.mae_init.ckpt_path:
            sd = torch.load(cfg.mae_init.ckpt_path, map_location="cpu")
            if "model" in sd:
                sd = sd["model"]
            self.vit.load_state_dict(sd, strict=False)

        self._features: List[torch.Tensor] = []
        for i, blk_idx in enumerate(cfg.feature_blocks):
            self.vit.blocks[blk_idx].register_forward_hook(self._make_hook(i))

        self.proj = nn.ModuleList([
            nn.Conv2d(cfg.embed_dim, cfg.embed_dim, kernel_size=1) for _ in cfg.feature_blocks
        ])

    def _make_hook(self, idx: int):
        def hook(module, inputs, output):
            self._features.append(output)
        return hook

    def _tokens_to_map(self, tokens: torch.Tensor, hw: int) -> torch.Tensor:
        # tokens: (B, N+1, C) or (B,N,C)
        if tokens.dim() != 3:
            raise ValueError("Unexpected tokens shape")
        if tokens.size(1) == hw * hw + 1:
            tokens = tokens[:, 1:, :]  # drop cls
        b, n, c = tokens.shape
        fmap = tokens.transpose(1, 2).contiguous().view(b, c, hw, hw)
        return fmap

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        self._features = []

        # timm ViT expects fixed img_size? Many variants support arbitrary if pos_embed interpolates.
        _ = self.vit.forward_features(x)

        # ASSUMPTION: patch grid is img_size/patch_size (Sec.3.2)
        hw = x.shape[-1] // self.cfg.patch_size

        feats: List[torch.Tensor] = []
        for i, t in enumerate(self._features):
            fmap = self._tokens_to_map(t, hw)
            fmap = self.proj[i](fmap)
            feats.append(fmap)

        # Build a pyramid compatible with FPN: pick 4 levels at decreasing resolution.
        # ASSUMPTION: use the last tapped block as base and downsample for coarser levels.
        base = feats[-1]
        pyr = [
            torch.nn.functional.interpolate(base, scale_factor=2.0, mode="bilinear", align_corners=False),  # ~1/8
            base,                                                                                           # ~1/16
            torch.nn.functional.avg_pool2d(base, kernel_size=2, stride=2),                                  # ~1/32
            torch.nn.functional.avg_pool2d(base, kernel_size=4, stride=4),                                  # ~1/64
        ]
        return pyr
