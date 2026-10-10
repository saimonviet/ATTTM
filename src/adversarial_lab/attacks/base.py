"""Lớp cơ sở cho các attack L_inf trên không gian pixel [0, 1].

API thống nhất: attack = FGSM(model, epsilon, ...); x_adv = attack(x, y)
 - x: ảnh [0,1], shape (N,C,H,W);
 - y: nhãn thật (untargeted) hoặc nhãn mục tiêu (targeted=True).
Model nhận pixel [0,1] (Normalize nằm trong model) nên epsilon luôn là epsilon pixel.
Attack không thay đổi tham số model và không đổi chế độ train/eval của model.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


def project_linf(x_adv: torch.Tensor, x: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Chiếu x_adv vào quả cầu L_inf bán kính epsilon quanh x, rồi clamp về [0, 1]."""
    x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
    return x_adv.clamp(0.0, 1.0)


class Attack(ABC):
    name = "attack"

    def __init__(self, model: nn.Module, epsilon: float, targeted: bool = False, amp: bool = False, loss_scale: float = 1024.0):
        if epsilon < 0:
            raise ValueError(f"epsilon phải >= 0, nhận {epsilon}")
        self.model = model
        self.epsilon = float(epsilon)
        self.targeted = targeted
        self.amp = amp  # autocast khi sinh attack lúc train (đánh giá luôn dùng fp32, amp=False)
        self.loss_scale = loss_scale
        # bf16 có dải số như fp32 nên không underflow gradient; fp16 thì có (đã đo: ~11% gradient = 0)
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    def loss_gradient(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gradient của loss theo input x (chỉ theo x; không tính/đổi gradient của tham số).

        Untargeted: gradient của CE(model(x), y) (tăng loss).
        Targeted: gradient của -CE(model(x), y_target) (giảm loss về target).
        """
        with torch.enable_grad():
            x = x.detach().clone().requires_grad_(True)
            use_amp = self.amp and x.device.type == "cuda"
            # cache_enabled=False: attack có thể chạy lồng trong autocast fp16 của engine mà không lẫn dtype trọng số
            with torch.autocast(x.device.type, dtype=self.amp_dtype, enabled=use_amp, cache_enabled=False):
                logits = self.model(x)
            loss = F.cross_entropy(logits.float(), y, reduction="sum")
            if self.targeted:
                loss = -loss
            if use_amp and self.amp_dtype == torch.float16:
                loss = loss * self.loss_scale  # tránh underflow fp16; sign() không đổi
            (grad,) = torch.autograd.grad(loss, x)
        grad = grad.detach()
        if self.amp:  # fp16 có thể tràn số -> tránh NaN lan vào ảnh
            grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
        return grad

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.detach()
        y = y.detach()
        if self.epsilon == 0.0:
            return x.clone()
        return self.perturb(x, y).detach()

    @abstractmethod
    def perturb(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Sinh ảnh đối kháng (epsilon > 0)."""

    def config(self) -> dict:
        return {"attack": self.name, "epsilon": self.epsilon, "targeted": self.targeted}
