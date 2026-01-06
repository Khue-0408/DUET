# path: src/models/losses/region_weighting.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass
class RegionWeightConfig:
    wB: float = 3.0
    wC: float = 1.0
    wBG: float = 1.0
    re: int = 5   # ASSUMPTION: paper says 5-10 pixels
    rd: int = 10  # ASSUMPTION


def _morph_erode(mask: torch.Tensor, r: int) -> torch.Tensor:
    # mask: (B,1,H,W) in {0,1}
    if r <= 0:
        return mask
    k = 2 * r + 1
    return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=k, stride=1, padding=r)


def _morph_dilate(mask: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return mask
    k = 2 * r + 1
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=r)


def make_region_weights(gt_mask: torch.Tensor, cfg: RegionWeightConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Region masks for Eq.(9):
    - RC = eroded mask (core)
    - RB = dilated(mask, rd) - eroded(mask, re) (boundary band)
    - RBG = remaining pixels

    Returns:
    - w: (B,1,H,W)
    - region_id: (B,1,H,W) {0:bg,1:core,2:boundary}
    """
    m = (gt_mask > 0.5).float()
    core = _morph_erode(m, cfg.re)
    dil = _morph_dilate(m, cfg.rd)
    boundary = (dil - core).clamp(0.0, 1.0)

    bg = 1.0 - (core + boundary).clamp(0.0, 1.0)

    w = cfg.wC * core + cfg.wB * boundary + cfg.wBG * bg

    region_id = torch.zeros_like(m)
    region_id = torch.where(core > 0.5, torch.ones_like(region_id), region_id)
    region_id = torch.where(boundary > 0.5, torch.full_like(region_id, 2.0), region_id)

    return w, region_id
