"""Fast Gradient Sign Method (Goodfellow et al., 2015)."""
from __future__ import annotations

import torch

from .base import Attack, project_linf


class FGSM(Attack):
    """x_adv = clip_[0,1](x + epsilon * sign(grad_x loss(model(x), y))).

    Một bước duy nhất; |x_adv - x|_inf <= epsilon theo pixel space [0,1].
    """

    name = "fgsm"

    def perturb(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        grad = self.loss_gradient(x, y)
        x_adv = x + self.epsilon * grad.sign()
        return project_linf(x_adv, x, self.epsilon)
