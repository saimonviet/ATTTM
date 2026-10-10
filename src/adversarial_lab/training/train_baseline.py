"""Huấn luyện baseline ResNet18 trên CIFAR-10 (clean training)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..data.cifar10 import get_loaders
from ..models.resnet import build_from_config
from ..seed import set_seed
from .engine import fit


def clean_step(model, x, y):
    out = model(x)
    loss = F.cross_entropy(out, y)
    return loss, {"acc": (out.argmax(1) == y).sum().item()}, y.size(0)


def clean_val_step(model, x, y):
    with torch.no_grad():
        return clean_step(model, x, y)


def train_baseline(cfg: dict, device: torch.device, ckpt_path, log_dir, max_samples=None, resume=False) -> dict:
    set_seed(cfg["seed"])
    tcfg = cfg["train"]
    train_loader, val_loader, _ = get_loaders(cfg, tcfg["batch_size"], max_samples)
    model = build_from_config(cfg)
    result = fit(
        model, train_loader, val_loader, clean_step, clean_val_step, cfg, tcfg,
        tcfg["epochs"], device, ckpt_path, log_dir, select_key="val_acc",
        extra_meta={"training": "baseline_clean_from_scratch", "n_train": len(train_loader.dataset), "n_val": len(val_loader.dataset)},
        resume=resume,
    )
    return result
