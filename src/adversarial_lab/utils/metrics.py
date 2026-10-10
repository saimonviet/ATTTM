"""Các metric: accuracy, ASR, chuẩn của perturbation, confusion matrix."""
from __future__ import annotations

import math

import torch


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Tỉ lệ dự đoán đúng. Tập rỗng -> NaN."""
    if labels.numel() == 0:
        return float("nan")
    return (preds == labels).float().mean().item()


def attack_success_rate(clean_correct: torch.Tensor, adv_correct: torch.Tensor) -> float:
    """ASR = #(đúng trước attack & sai sau attack) / #(đúng trước attack).

    Nếu không có mẫu nào đúng ban đầu thì trả về NaN (tránh chia cho 0);
    gọi hàm này cần kiểm tra bằng math.isnan và báo cáo riêng.
    ASR khác "adversarial accuracy": accuracy tính trên TOÀN BỘ mẫu sau attack.
    """
    clean_correct = clean_correct.bool()
    adv_correct = adv_correct.bool()
    n_base = int(clean_correct.sum().item())
    if n_base == 0:
        return float("nan")
    flipped = int((clean_correct & ~adv_correct).sum().item())
    return flipped / n_base


def is_nan(x: float) -> bool:
    return isinstance(x, float) and math.isnan(x)


def perturbation_norms(x: torch.Tensor, x_adv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(L_inf, L2) của (x_adv - x) cho từng mẫu, shape (N,), tính trong pixel space [0,1]."""
    d = (x_adv - x).flatten(1)
    return d.abs().amax(dim=1), d.norm(p=2, dim=1)


def confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """cm[i, j] = số mẫu nhãn thật i được dự đoán là j."""
    idx = labels.long() * num_classes + preds.long()
    return torch.bincount(idx, minlength=num_classes**2).view(num_classes, num_classes)


def per_class_accuracy(cm: torch.Tensor) -> torch.Tensor:
    totals = cm.sum(dim=1).float()
    return torch.where(totals > 0, cm.diag().float() / totals.clamp(min=1), torch.full_like(totals, float("nan")))
