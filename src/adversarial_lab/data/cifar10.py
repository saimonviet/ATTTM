"""Dataset/DataLoader CIFAR-10 với split train/val/test cố định."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10

from .transforms import eval_transform, train_transform

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def _limit(ds: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None or max_samples >= len(ds):
        return ds
    return Subset(ds, list(range(max_samples)))


def get_datasets(
    root: str | Path = "data",
    val_size: int = 5000,
    seed: int = 42,
    augment: bool = True,
    max_samples: int | None = None,
    download: bool = True,
) -> tuple[Dataset, Dataset, Dataset]:
    """Trả về (train, val, test).

    Train/val được tách từ 50k ảnh train bằng permutation theo seed. Val dùng
    transform không augmentation. Test (10k) chỉ dùng để đánh giá cuối.
    max_samples cắt mỗi split (dùng cho smoke test).
    """
    root = str(root)
    full_train = CIFAR10(root, train=True, download=download, transform=train_transform(augment))
    full_val = CIFAR10(root, train=True, download=False, transform=eval_transform())
    test = CIFAR10(root, train=False, download=download, transform=eval_transform())

    n = len(full_train)
    if not 0 <= val_size < n:
        raise ValueError(f"val_size phải trong [0, {n}), nhận {val_size}")
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    train = Subset(full_train, train_idx)
    val = Subset(full_val, val_idx)
    return _limit(train, max_samples), _limit(val, max_samples), _limit(test, max_samples)


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int = 0, seed: int = 42) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=g if shuffle else None,
    )


def get_loaders(cfg: dict, batch_size: int | None = None, max_samples: int | None = None, augment: bool | None = None):
    """Tạo (train_loader, val_loader, test_loader) từ config."""
    d = cfg["data"]
    bs = batch_size or cfg["train"]["batch_size"]
    aug = cfg["train"].get("augment", True) if augment is None else augment
    train, val, test = get_datasets(d["root"], d["val_size"], cfg["seed"], aug, max_samples)
    nw = cfg.get("num_workers", 0)
    eval_bs = cfg.get("eval", {}).get("batch_size", 256)
    return (
        make_loader(train, bs, True, nw, cfg["seed"]),
        make_loader(val, eval_bs, False, nw),
        make_loader(test, eval_bs, False, nw),
    )


def get_test_loader(cfg: dict, max_samples: int | None = None, batch_size: int | None = None) -> DataLoader:
    """Chỉ test loader (dùng chung cho mọi model để so sánh công bằng)."""
    root = cfg["data"]["root"]
    test = _limit(CIFAR10(root, train=False, download=True, transform=eval_transform()), max_samples)
    return make_loader(test, batch_size or cfg.get("eval", {}).get("batch_size", 256), False, cfg.get("num_workers", 0))
