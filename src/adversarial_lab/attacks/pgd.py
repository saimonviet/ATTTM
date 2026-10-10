"""Projected Gradient Descent với ràng buộc L_inf (Madry et al., 2018)."""
from __future__ import annotations

import torch

from .base import Attack, project_linf


class PGD(Attack):
    """x_{t+1} = Proj_{B_inf(x, eps) ∩ [0,1]}(x_t + alpha * sign(grad_x loss(model(x_t), y))).

    alpha mặc định = epsilon / 4. random_start: khởi tạo x_0 = x + U(-eps, eps) (rồi clamp [0,1]).
    Trả về iterate cuối cùng (không chọn iterate tốt nhất).
    """

    name = "pgd"

    def __init__(self, model, epsilon: float, steps: int = 10, alpha: float | None = None, random_start: bool = True, **kw):
        super().__init__(model, epsilon, **kw)
        if steps < 0:
            raise ValueError("steps phải >= 0")
        self.steps = steps
        self.alpha = float(alpha) if alpha is not None else self.epsilon / 4
        self.random_start = random_start

    def perturb(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_adv = x.clone()
        if self.random_start:
            x_adv = project_linf(x + torch.empty_like(x).uniform_(-self.epsilon, self.epsilon), x, self.epsilon)
        for _ in range(self.steps):
            grad = self.loss_gradient(x_adv, y)
            x_adv = project_linf(x_adv + self.alpha * grad.sign(), x, self.epsilon)
        return x_adv

    def config(self) -> dict:
        return {**super().config(), "steps": self.steps, "alpha": self.alpha, "random_start": self.random_start}
