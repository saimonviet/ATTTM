import pytest
import torch
import torch.nn as nn

from adversarial_lab.models.resnet import NormalizedModel


@pytest.fixture
def tiny_model():
    """CNN nhỏ (có Normalize bên trong như model thật) để test attack nhanh trên CPU."""
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(32, 10))
    return NormalizedModel(net, (0.49, 0.48, 0.45), (0.25, 0.24, 0.26)).eval()


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(1)
    x = torch.rand(8, 3, 32, 32, generator=g) * 0.6 + 0.2  # xa biên [0,1] để thấy rõ clamp khi eps lớn
    y = torch.randint(0, 10, (8,), generator=g)
    return x, y
