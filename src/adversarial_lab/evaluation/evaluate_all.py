"""Đánh giá tổng hợp: chạy phần còn thiếu (hoặc --force) rồi sinh bảng/hình/manifest.

Chỉ chạy lại những gì chưa có trong results/tables để không tốn thời gian khi đã có kết quả.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from ..config import resolve_path
from ..data.cifar10 import CLASS_NAMES, get_test_loader
from ..device import get_device
from ..seed import set_seed
from ..utils.checkpoint import load_checkpoint
from ..utils.cli import results_dir, save_json
from ..utils.logging import get_logger
from .build_reports import build_reports
from .evaluate_clean import evaluate_clean
from .run_attack import run_attack
from .run_blur import run_blur_defense


def _has(df: pd.DataFrame | None, source: str, eval_: str, attack: str) -> bool:
    return df is not None and bool(((df["source_model"] == source) & (df["eval_model"] == eval_) & (df["attack"] == attack)).any())


def evaluate_all(cfg: dict, models: dict[str, str], max_samples: int | None = None, force: bool = False,
                 reports_only: bool = False, blur: bool = True, adaptive: bool = True) -> dict:
    log = get_logger()
    out = results_dir(cfg)
    tables = out / "tables"
    csv = tables / "attack_results.csv"
    status = {}
    if not reports_only:
        set_seed(cfg["seed"])
        device = get_device(cfg["device"])
        for name, ck in models.items():
            path = resolve_path(ck)
            if not path.exists():
                log.warning(f"[{name}] chưa có checkpoint {path} -> NOT_RUN")
                status[name] = "NOT_RUN"
                continue
            if force or not (tables / f"{path.stem}_clean.json").exists():
                model, meta = load_checkpoint(path, device)
                res = evaluate_clean(model, get_test_loader(cfg, max_samples or cfg["eval"]["max_samples"]), device, CLASS_NAMES)
                res.update(train_time_sec=meta.get("train_time_sec_until_best"), best_epoch=meta.get("best_epoch"),
                           training=meta.get("training"), checkpoint=str(path),
                           note="train_time_sec lấy từ metadata checkpoint (thời gian đến epoch tốt nhất)")
                save_json(tables / f"{path.stem}_clean.json", res)
                log.info(f"[{name}] clean acc={res['accuracy']:.4f}")
            df = pd.read_csv(csv) if csv.exists() else None
            for atk, key in (("fgsm", "fgsm"), ("pgd", "pgd")):
                if force or not _has(df, name, name, atk):
                    run_attack(cfg, atk, cfg["attacks"][key]["epsilons"], str(path), max_samples=max_samples, model_name=name)
                    df = pd.read_csv(csv)
            status[name] = "done"
        base = next(iter(models))
        bpath = resolve_path(models[base])
        df = pd.read_csv(csv) if csv.exists() else None
        has_blur = df is not None and bool(df["defense"].astype(str).str.startswith("gaussian_blur").any())
        if blur and bpath.exists() and (force or not has_blur):
            run_blur_defense(cfg, str(bpath), max_samples=max_samples, model_name=base, adaptive=adaptive)
    ck_paths = {n: p for n, p in models.items()}
    outputs = build_reports(cfg, out, ck_paths)
    log.info("Đã tạo: " + ", ".join(f"{k}={v}" for k, v in outputs.items()))
    return {"status": status, "outputs": outputs}
