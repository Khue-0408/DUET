# path: src/utils/logger.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter


@dataclass
class Logger:
    out_dir: str
    exp_name: str
    enable_tb: bool = True

    def __post_init__(self) -> None:
        self.run_dir = Path(self.out_dir) / self.exp_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.tb = SummaryWriter(str(self.run_dir / "tb")) if self.enable_tb else None
        self.csv_path = self.run_dir / "metrics.csv"
        self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._csv_writer = None

    def log_scalars(self, step: int, scalars: Dict[str, float]) -> None:
        if self.tb is not None:
            for k, v in scalars.items():
                self.tb.add_scalar(k, v, step)

        if self._csv_writer is None:
            fieldnames = ["step"] + list(scalars.keys())
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
            if self._csv_file.tell() == 0:
                self._csv_writer.writeheader()

        row = {"step": step, **scalars}
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()
        self._csv_file.close()
