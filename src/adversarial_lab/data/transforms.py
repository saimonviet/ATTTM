"""Transform cho CIFAR-10.

Ảnh luôn ở dạng tensor [0, 1]; Normalize nằm trong model (xem models/resnet.py),
nhờ đó FGSM/PGD và epsilon luôn được định nghĩa trong không gian pixel [0, 1].
"""
from __future__ import annotations

from torchvision import transforms as T


def train_transform(augment: bool = True) -> T.Compose:
    ops = []
    if augment:
        ops += [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    ops.append(T.ToTensor())
    return T.Compose(ops)


def eval_transform() -> T.Compose:
    """Validation/test: không có augmentation ngẫu nhiên."""
    return T.Compose([T.ToTensor()])
