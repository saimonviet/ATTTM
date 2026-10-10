"""Gaussian Blur preprocessing defense (không train lại model).

Thứ tự xử lý đúng: ảnh pixel [0,1] -> Gaussian Blur (pixel space) -> Normalize -> backbone.
Vì Normalize nằm trong model (models/resnet.py), BlurDefendedModel chỉ cần blur ảnh [0,1] rồi gọi model.
Blur là phép tích chập vi phân được nên attack thích nghi (adaptive) có thể lan truyền gradient qua nó.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel_1d(kernel_size: int, sigma: float) -> torch.Tensor:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size phải là số lẻ dương, nhận {kernel_size}")
    if sigma <= 0:
        raise ValueError(f"sigma phải > 0, nhận {sigma}")
    ax = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    k = torch.exp(-(ax**2) / (2 * sigma**2))
    return k / k.sum()


class GaussianBlur(nn.Module):
    """Gaussian blur tách được (depthwise), padding reflect; đầu vào/ra trong [0,1]."""

    def __init__(self, kernel_size: int = 3, sigma: float = 1.0, channels: int = 3):
        super().__init__()
        self.kernel_size, self.sigma, self.channels = kernel_size, sigma, channels
        k = gaussian_kernel_1d(kernel_size, sigma)
        self.register_buffer("kh", k.view(1, 1, 1, -1).repeat(channels, 1, 1, 1))
        self.register_buffer("kv", k.view(1, 1, -1, 1).repeat(channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.kernel_size // 2
        if p == 0:
            return x
        x = F.pad(x, (p, p, p, p), mode="reflect")
        x = F.conv2d(x, self.kh, groups=self.channels)
        return F.conv2d(x, self.kv, groups=self.channels)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, sigma={self.sigma}"


class BlurDefendedModel(nn.Module):
    """Model + Gaussian Blur: logits = model(blur(x)). Trọng số model không đổi."""

    def __init__(self, model: nn.Module, kernel_size: int, sigma: float):
        super().__init__()
        self.blur = GaussianBlur(kernel_size, sigma)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.blur(x))


def blur_name(kernel_size: int, sigma: float) -> str:
    return f"gaussian_blur_k{kernel_size}_s{sigma:g}"
