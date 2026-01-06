# path: src/models/decoders/fpn_decoder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FPNDecoderConfig:
    fpn_dim: int = 256
    out_classes: int = 2
    dropout: float = 0.1


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FPNDecoder(nn.Module):
    """
    Lightweight FPN decoder (paper Sec.3.3: "lightweight FPN", p.12).
    ASSUMPTION: standard top-down FPN + smoothing convs.

    Input: list of features [C2,C3,C4,C5] at decreasing resolutions.
    Output: logits (B,2,H,W) at highest resolution among inputs, then upsample to input size in trainer.
    """

    def __init__(self, in_channels: List[int], cfg: FPNDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.lateral = nn.ModuleList([nn.Conv2d(ch, cfg.fpn_dim, kernel_size=1) for ch in in_channels])
        self.smooth = nn.ModuleList([ConvBNReLU(cfg.fpn_dim, cfg.fpn_dim) for _ in in_channels])

        self.dropout = nn.Dropout2d(cfg.dropout)
        self.head = nn.Conv2d(cfg.fpn_dim, cfg.out_classes, kernel_size=1)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        # feats: low->high? We assume feats is increasing level order (as returned by timm features_only)
        # We'll treat last as the lowest resolution.
        feats = list(feats)
        lat = [l(f) for l, f in zip(self.lateral, feats)]

        x = lat[-1]
        outs = [None] * len(lat)
        outs[-1] = self.smooth[-1](x)

        for i in range(len(lat) - 2, -1, -1):
            x = F.interpolate(x, size=lat[i].shape[-2:], mode="nearest") + lat[i]
            outs[i] = self.smooth[i](x)

        # merge all levels to the highest res (outs[0]) by upsampling and summing
        y = outs[0]
        for i in range(1, len(outs)):
            y = y + F.interpolate(outs[i], size=outs[0].shape[-2:], mode="bilinear", align_corners=False)

        y = self.dropout(y)
        return self.head(y)
