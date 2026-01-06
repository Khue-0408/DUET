# path: src/data/samplers.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class LOCOConfig:
    enabled: bool = False
    dataset_name: str = "PolypGen"


def build_loco_splits_from_meta(root: str, meta_csv: str, out_dir: str) -> List[Tuple[int, str, str]]:
    """
    Build LOCO splits (Sec.4.1) given meta.csv with center_id.
    Writes:
      out_dir/fold_<k>/{train.txt,test.txt}
    Returns list of (heldout_center, train_split_path, test_split_path)
    """
    root_p = Path(root)
    meta = pd.read_csv(root_p / meta_csv)
    centers = sorted(meta["center_id"].unique().tolist())

    out = []
    for c in centers:
        fold_dir = root_p / out_dir / f"fold_{c}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_ids = meta[meta["center_id"] != c]["image"].tolist()
        test_ids = meta[meta["center_id"] == c]["image"].tolist()

        tr = fold_dir / "train.txt"
        te = fold_dir / "test.txt"
        tr.write_text("\n".join(train_ids), encoding="utf-8")
        te.write_text("\n".join(test_ids), encoding="utf-8")

        out.append((int(c), str(tr.relative_to(root_p)), str(te.relative_to(root_p))))
    return out
