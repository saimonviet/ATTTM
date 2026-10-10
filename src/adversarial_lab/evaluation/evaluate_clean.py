"""Đánh giá model trên ảnh sạch."""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..utils.metrics import confusion_matrix, per_class_accuracy


@torch.no_grad()
def evaluate_clean(model: torch.nn.Module, loader: DataLoader, device: torch.device, class_names: list[str]) -> dict:
    """Clean accuracy, loss, per-class accuracy, confusion matrix, thời gian inference."""
    model.eval().to(device)
    n_cls = len(class_names)
    preds_all, labels_all, loss_sum, n = [], [], 0.0, 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        preds_all.append(out.argmax(1).cpu())
        labels_all.append(y.cpu())
        n += y.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inf_time = time.time() - t0
    preds, labels = torch.cat(preds_all), torch.cat(labels_all)
    cm = confusion_matrix(preds, labels, n_cls)
    pca = per_class_accuracy(cm)
    return {
        "n_samples": n,
        "accuracy": (preds == labels).float().mean().item(),
        "loss": loss_sum / n,
        "per_class_accuracy": {c: float(a) for c, a in zip(class_names, pca)},
        "confusion_matrix": cm.tolist(),
        "inference_time_sec": inf_time,
        "inference_ms_per_sample": 1000 * inf_time / n,
    }
