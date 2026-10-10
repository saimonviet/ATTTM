"""Đọc cấu hình YAML và ghi đè từ CLI."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {path}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def override(cfg: dict[str, Any], dotted: dict[str, Any]) -> dict[str, Any]:
    """Trả về bản sao cfg với các khóa dạng 'train.epochs' bị ghi đè (bỏ qua giá trị None)."""
    out = copy.deepcopy(cfg)
    for key, val in dotted.items():
        if val is None:
            continue
        node = out
        parts = key.split(".")
        for k in parts[:-1]:
            node = node.setdefault(k, {})
        node[parts[-1]] = val
    return out


def resolve_path(p: str | Path) -> Path:
    """Đường dẫn tương đối được hiểu so với project root."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
