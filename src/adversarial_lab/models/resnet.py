"""ResNet18 cho CIFAR-10 (ảnh 32x32), đầu vào pixel [0, 1]."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18


class NormalizedModel(nn.Module):
    """Gắn Normalize vào đầu model: input [0,1] -> (x-mean)/std -> backbone.

    Nhờ đó gradient/epsilon của attack luôn ở không gian pixel [0,1].
    """

    def __init__(self, backbone: nn.Module, mean, std):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone((x - self.mean) / self.std)


def build_model(arch: str = "resnet18", num_classes: int = 10, mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)) -> nn.Module:
    if arch != "resnet18":
        raise ValueError(f"Chưa hỗ trợ arch={arch!r}; hiện chỉ có 'resnet18'.")
    net = resnet18(weights=None, num_classes=num_classes)  # không tải trọng số ngoài
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    return NormalizedModel(net, mean, std)


def build_from_config(cfg: dict) -> nn.Module:
    return build_model(cfg["model"]["arch"], cfg["model"]["num_classes"], cfg["data"]["mean"], cfg["data"]["std"])
