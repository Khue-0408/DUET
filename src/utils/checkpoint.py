# path: src/utils/checkpoint.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass
class CheckpointState:
    epoch: int
    step: int
    best_metric: float


def save_checkpoint(path: str, model: torch.nn.Module, optim: torch.optim.Optimizer, state: CheckpointState) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "epoch": state.epoch,
            "step": state.step,
            "best_metric": state.best_metric,
        },
        str(p),
    )


def load_checkpoint(path: str, model: torch.nn.Module, optim: Optional[torch.optim.Optimizer] = None) -> CheckpointState:
    sd = torch.load(path, map_location="cpu")
    model.load_state_dict(sd["model"], strict=False)
    if optim is not None and "optim" in sd:
        optim.load_state_dict(sd["optim"])
    return CheckpointState(epoch=int(sd.get("epoch", 0)), step=int(sd.get("step", 0)), best_metric=float(sd.get("best_metric", -1e9)))
