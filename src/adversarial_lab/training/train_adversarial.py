"""Huấn luyện robust model bằng FGSM/PGD adversarial training."""
from __future__ import annotations

import torch

from ..data.cifar10 import get_loaders
from ..defenses.adversarial_training import attack_factory_from_cfg, make_adv_train_step, make_adv_val_step
from ..models.resnet import build_from_config
from ..seed import set_seed
from ..utils.checkpoint import load_checkpoint
from .engine import fit


def train_adversarial(cfg: dict, device: torch.device, ckpt_path, log_dir, max_samples=None, resume=False,
                      init_checkpoint: str | None = None) -> dict:
    """mode='from_scratch' (mặc định) hoặc 'finetune' (khởi tạo từ init_checkpoint của baseline).

    Checkpoint tốt nhất chọn theo PGD accuracy trên tập validation (val_acc), không dùng test.
    """
    set_seed(cfg["seed"])
    a = cfg["adv_train"]
    amp = bool(a.get("amp", False)) and device.type == "cuda"
    train_loader, val_loader, _ = get_loaders(cfg, a["batch_size"], max_samples)
    model = build_from_config(cfg)
    mode = a["mode"]
    if mode == "finetune":
        if not init_checkpoint:
            raise ValueError("mode=finetune cần --init-checkpoint (checkpoint baseline)")
        base, _ = load_checkpoint(init_checkpoint, "cpu")
        model.load_state_dict(base.state_dict())
    elif mode != "from_scratch":
        raise ValueError(f"mode không hợp lệ: {mode!r} (from_scratch/finetune)")

    factory = attack_factory_from_cfg(a, amp)
    train_step = make_adv_train_step(factory, a["adv_ratio"], a["clean_weight"])
    val_step = make_adv_val_step(attack_factory_from_cfg(a, False))  # validation dùng fp32 cho attack
    meta = {
        "training": f"adversarial_{a['method']}_{mode}",
        "adv_config": {k: a[k] for k in ("method", "epsilon", "steps", "alpha_ratio", "random_start", "adv_ratio", "clean_weight")},
        "init_checkpoint": init_checkpoint if mode == "finetune" else None,
        "n_train": len(train_loader.dataset), "n_val": len(val_loader.dataset),
    }
    return fit(model, train_loader, val_loader, train_step, val_step, cfg, a, a["epochs"], device, ckpt_path, log_dir,
               select_key="val_acc", extra_meta=meta, resume=resume)
