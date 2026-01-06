# path: src/train/infer.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from src.models.duet import DUETModel
from src.train.train_duet import build_model
from src.utils.config import apply_overrides, load_yaml


def load_image(path: str, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return t


def save_gray(path: str, arr01: np.ndarray) -> None:
    arr = (np.clip(arr01, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--override", type=str, nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.override)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)

    ckpt = cfg.get("infer", {}).get("ckpt", "") or f'{cfg["logging"]["out_dir"]}/{cfg["exp"]["name"]}/checkpoints/duet_best.pt'
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd["model"], strict=False)
    model.eval()

    out_dir = Path(cfg["infer"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    size = int(cfg["data"]["image_size"])
    image_paths: List[str] = []
    if cfg["infer"]["image_path"]:
        image_paths = [cfg["infer"]["image_path"]]
    elif cfg["infer"]["image_dir"]:
        image_paths = [str(p) for p in Path(cfg["infer"]["image_dir"]).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    else:
        raise ValueError("Provide infer.image_path or infer.image_dir")

    with torch.no_grad():
        for ip in image_paths:
            x = load_image(ip, size)[None].to(device)
            out = model(x)

            p = out.p_f[0, 0].detach().cpu().numpy()
            u = out.u_f[0, 0].detach().cpu().numpy()
            m = (p >= 0.5).astype(np.float32)

            stem = Path(ip).stem
            save_gray(str(out_dir / f"{stem}_p.png"), p)
            save_gray(str(out_dir / f"{stem}_u.png"), u)
            save_gray(str(out_dir / f"{stem}_mask.png"), m)


if __name__ == "__main__":
    main()
