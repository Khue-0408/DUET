# path: src/data/transforms.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2


@dataclass
class TransformConfig:
    image_size: int = 512
    augment: bool = True


def build_transforms(cfg: TransformConfig, for_stage1: bool = False) -> A.Compose:
    """
    Stage2/3: flips, rotations, color jitter (Training Details p.17).
    Stage1: can reuse same (paper doesn't forbid). Kept simple.
    """
    size = int(cfg.image_size)

    t = [A.Resize(size, size)]
    if cfg.augment:
        t += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=30, p=0.5),
            A.ColorJitter(p=0.5),
        ]
    t += [A.Normalize(), ToTensorV2()]

    return A.Compose(t, additional_targets={"mask": "mask", "pseudo_mask": "mask"})
