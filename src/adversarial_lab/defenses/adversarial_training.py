"""Adversarial Training: bước huấn luyện sinh adversarial examples ngay trong vòng lặp.

Mục tiêu:  L = lambda * L_clean + (1 - lambda) * L_adv     (lambda = clean_weight, mặc định 0
= adversarial-only). Adversarial examples được sinh trên model hiện tại (white-box) với model ở
chế độ eval (BatchNorm dùng running stats khi tính attack), sau đó model chuyển sang train để cập
nhật trọng số bằng loss. Chỉ dùng tập train; validation dùng tập val (không dùng test).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..attacks import make_attack


def attack_factory_from_cfg(acfg: dict, amp: bool):
    """Tạo hàm model -> Attack từ cấu hình adv_train."""
    method = acfg["method"].lower()
    eps = acfg["epsilon"]

    def factory(model):
        if method == "fgsm":
            return make_attack("fgsm", model, eps, amp=amp)
        return make_attack("pgd", model, eps, steps=acfg["steps"], alpha=eps * acfg["alpha_ratio"],
                           random_start=acfg["random_start"], amp=amp)

    return factory


def make_adv_train_step(factory, adv_ratio: float = 1.0, clean_weight: float = 0.0):
    """Trả về train_step(model, x, y) -> (loss, stats, n) cho engine.fit."""
    if not 0.0 <= clean_weight <= 1.0 or not 0.0 <= adv_ratio <= 1.0:
        raise ValueError("clean_weight và adv_ratio phải nằm trong [0, 1]")

    def step(model, x, y):
        n = y.size(0)
        k = int(round(adv_ratio * n))
        model.eval()  # sinh attack với BN ở chế độ eval
        x_in = x.clone()
        if k > 0:
            x_in[:k] = factory(model)(x[:k], y[:k])
        model.train()
        out_adv = model(x_in)
        loss = F.cross_entropy(out_adv, y)
        if clean_weight > 0:
            loss = clean_weight * F.cross_entropy(model(x), y) + (1 - clean_weight) * loss
        return loss, {"acc": (out_adv.argmax(1) == y).sum().item()}, n

    return step


def make_adv_val_step(factory):
    """Validation: loss/accuracy trên ảnh đối kháng (attack sinh trên model đang train) + clean accuracy."""

    def step(model, x, y):
        x_adv = factory(model)(x, y)
        with torch.no_grad():
            out_adv, out_clean = model(x_adv), model(x)
            loss = F.cross_entropy(out_adv, y)
            stats = {"acc": (out_adv.argmax(1) == y).sum().item(), "clean_acc": (out_clean.argmax(1) == y).sum().item()}
        return loss, stats, y.size(0)

    return step
