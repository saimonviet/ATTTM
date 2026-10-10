"""Lưu/đọc checkpoint kèm metadata (architecture, class mapping, normalization, cấu hình)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..data.cifar10 import CLASS_NAMES
from ..models.resnet import build_model


def save_checkpoint(path: str | Path, model: torch.nn.Module, cfg: dict, extra: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "arch": cfg["model"]["arch"],
        "num_classes": cfg["model"]["num_classes"],
        "class_names": CLASS_NAMES,
        "mean": list(cfg["data"]["mean"]),
        "std": list(cfg["data"]["std"]),
        "input_range": list(cfg["data"]["input_range"]),
        "config": cfg,
    }
    meta.update(extra or {})
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[torch.nn.Module, dict]:
    """Trả về (model ở eval mode, meta). Chỉ load file do chính project tạo ra."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}. Hãy train trước (xem README).")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    meta = ckpt["meta"]
    model = build_model(meta["arch"], meta["num_classes"], meta["mean"], meta["std"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), meta
