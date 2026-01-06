# path: src/utils/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_set(d: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def parse_value(v: str) -> Any:
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        if "." in v or "e" in v.lower():
            return float(v)
        return int(v)
    except Exception:
        return v


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    for ov in overrides:
        if "=" not in ov:
            continue
        k, v = ov.split("=", 1)
        deep_set(cfg, k.strip(), parse_value(v.strip()))
    return cfg
