# path: src/data/datasets.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


@dataclass
class DatasetConfig:
    dataset_name: str
    root: str
    meta_csv: str = "meta.csv"
    split_dir: str = "splits"
    train_split: str = "train.txt"
    val_split: str = "val.txt"
    test_split: str = "test.txt"
    image_size: int = 512


class PolypMetaDataset(Dataset):
    """
    Generic dataset driven by meta.csv.
    Supports labeled and unlabeled usage.

    Returns dict with:
      - image: float tensor (3,H,W)
      - mask: float tensor (1,H,W) or None
      - pseudo_mask: float tensor (1,H,W) or None
      - domain_id: int
      - center_id: int
      - has_polyp: int
      - image_id: str (for saving)
    """

    def __init__(
        self,
        root: str,
        meta_csv: str,
        split_file: str,
        transform: Optional[Any] = None,
        pseudo_mask_key: str = "pseudo_mask",
    ) -> None:
        self.root = Path(root)
        self.meta = pd.read_csv(self.root / meta_csv)

        with open(self.root / split_file, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
        self.meta = self.meta[self.meta["image"].isin(ids)].reset_index(drop=True)

        self.transform = transform
        self.pseudo_mask_key = pseudo_mask_key

    def __len__(self) -> int:
        return len(self.meta)

    def _load_img(self, rel: str) -> np.ndarray:
        img = Image.open(self.root / rel).convert("RGB")
        return np.array(img)

    def _load_mask(self, rel: str) -> Optional[np.ndarray]:
        if not isinstance(rel, str) or rel.strip() == "" or rel.lower() == "nan":
            return None
        m = Image.open(self.root / rel).convert("L")
        arr = np.array(m)
        arr = (arr > 127).astype(np.uint8)
        return arr

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.meta.iloc[idx].to_dict()

        img = self._load_img(row["image"])
        mask = self._load_mask(row.get("mask", ""))
        pm = self._load_mask(row.get(self.pseudo_mask_key, ""))

        data = {"image": img}
        if mask is not None:
            data["mask"] = mask
        if pm is not None:
            data["pseudo_mask"] = pm

        if self.transform is not None:
            out = self.transform(**data)
            img_t = out["image"]
            mask_t = out.get("mask", None)
            pm_t = out.get("pseudo_mask", None)
        else:
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask)[None].float() if mask is not None else None
            pm_t = torch.from_numpy(pm)[None].float() if pm is not None else None

        sample: Dict[str, Any] = {
            "image": img_t,
            "mask": mask_t,
            "pseudo_mask": pm_t,
            "domain_id": int(row.get("domain_id", 0)),
            "center_id": int(row.get("center_id", -1)),
            "has_polyp": int(row.get("has_polyp", 1 if (mask_t is not None and mask_t.sum() > 0) else 0)),
            "image_id": Path(row["image"]).stem,
        }
        return sample
