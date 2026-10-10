"""experiment_manifest.json: ghi lại môi trường, cấu hình và checkpoint để tái lập thí nghiệm."""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import torchvision

from ..config import PROJECT_ROOT, resolve_path
from ..device import device_info, get_device


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _checkpoint_entry(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "status": "NOT_RUN"}
    e = {"path": str(path), "size_mb": round(path.stat().st_size / 2**20, 2), "sha256": _sha256(path)}
    meta = torch.load(path, map_location="cpu", weights_only=True)["meta"]
    for k in ("training", "best_epoch", "select_key", "train_time_sec_until_best", "train_config", "adv_config", "init_checkpoint"):
        if k in meta:
            e[k] = meta[k]
    e["seed"] = meta["config"].get("seed")
    e["arch"], e["normalization"] = meta["arch"], {"mean": meta["mean"], "std": meta["std"], "input_range": meta["input_range"]}
    return e


def build_manifest(cfg: dict, checkpoints: dict[str, str | Path], attack_csv: str | Path | None = None) -> dict:
    """Thu thập thông tin từ config hiện tại, các checkpoint thực sự tồn tại và bảng kết quả đã sinh."""
    device = get_device(cfg.get("device", "auto"))
    man = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": device_info(device),
        "seed": cfg["seed"],
        "dataset": {"name": "CIFAR-10", "train_val_split": f"{50000 - cfg['data']['val_size']}/{cfg['data']['val_size']} (seed {cfg['seed']})",
                    "test_size": 10000, "normalization": {k: cfg["data"][k] for k in ("mean", "std", "input_range")}},
        "model": cfg["model"],
        "hyperparameters": {"baseline_train": cfg["train"], "adversarial_train": cfg["adv_train"]},
        "checkpoints": {n: _checkpoint_entry(resolve_path(p)) for n, p in checkpoints.items()},
        "attack_configs": cfg["attacks"],
        "defense_configs": {"gaussian_blur": cfg["defense"]["gaussian_blur"],
                            "adversarial_training": {k: cfg["adv_train"][k] for k in ("method", "mode", "epsilon", "steps", "alpha_ratio", "random_start", "adv_ratio", "clean_weight")}},
        "evaluation": {"split": "CIFAR-10 test", "epsilon_space": "pixel [0, 1], L_inf", "batch_size": cfg["eval"]["batch_size"],
                       "max_samples": cfg["eval"]["max_samples"]},
    }
    if attack_csv and Path(attack_csv).exists():
        df = pd.read_csv(attack_csv)
        man["evaluation"]["n_samples_evaluated"] = sorted(int(v) for v in df["n_samples"].unique())
        man["evaluation"]["results_rows"] = len(df)
        man["evaluation"]["models_evaluated"] = sorted(df["eval_model"].unique().tolist())
    return man
