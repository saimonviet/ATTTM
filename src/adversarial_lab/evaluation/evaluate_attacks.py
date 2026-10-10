"""Đánh giá attack: accuracy, ASR, L_inf, L2, thời gian.

Mỗi dòng kết quả ghi rõ attack được TẠO TRÊN model nào (source_model) và
ĐÁNH GIÁ TRÊN model nào (eval_model). White-box: hai model trùng nhau.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..attacks import Attack, make_attack
from ..utils.metrics import attack_success_rate, perturbation_norms

AttackFactory = Callable[[torch.nn.Module], Attack]

# Các cột định danh một thí nghiệm (dùng để ghi đè dòng cũ khi chạy lại).
KEY_COLS = ["source_model", "eval_model", "attack", "epsilon", "steps", "alpha", "random_start", "defense", "n_samples"]
TABLE_COLS = KEY_COLS + [
    "clean_accuracy", "accuracy", "asr", "n_clean_correct", "mean_linf", "mean_l2",
    "attack_time_sec", "ms_per_sample", "note",
]


def _row_keys(df: pd.DataFrame) -> pd.Series:
    """Khóa chuỗi chuẩn hóa cho từng dòng (không phụ thuộc kiểu dữ liệu khi đọc lại từ CSV: 10 vs 10.0, True vs 'True')."""
    def norm(v):
        if isinstance(v, (float, int, np.floating, np.integer)) and not isinstance(v, (bool, np.bool_)):
            return "nan" if pd.isna(v) else f"{float(v):.8g}"
        return "nan" if pd.isna(v) else str(v)

    return df[KEY_COLS].apply(lambda r: "|".join(norm(v) for v in r), axis=1)


def _row_keys(df: pd.DataFrame) -> pd.Series:
    """Khóa chuỗi chuẩn hóa cho từng dòng (không phụ thuộc kiểu dữ liệu khi đọc lại từ CSV: 10 vs 10.0, True vs 'True')."""
    def norm(v):
        if isinstance(v, (float, int, np.floating, np.integer)) and not isinstance(v, (bool, np.bool_)):
            return "nan" if pd.isna(v) else f"{float(v):.8g}"
        return "nan" if pd.isna(v) else str(v)

    return df[KEY_COLS].apply(lambda r: "|".join(norm(v) for v in r), axis=1)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def _predict(model, x):
    probs = torch.softmax(model(x).float(), dim=1)
    conf, pred = probs.max(1)
    return pred, conf


def evaluate_attack(
    source_model: torch.nn.Module,
    eval_model: torch.nn.Module,
    loader: DataLoader,
    attack_factory: AttackFactory | None,
    device: torch.device,
    source_name: str,
    eval_name: str,
    attack_label: str = "clean",
    defense: str = "none",
    n_examples: int = 0,
    attack_cfg: dict | None = None,
) -> dict:
    """Sinh adversarial trên source_model, đánh giá trên eval_model.

    attack_factory=None -> chỉ đánh giá ảnh sạch (x_adv = x).
    Accuracy = trên TOÀN BỘ mẫu (ảnh đối kháng); ASR = mẫu đúng ban đầu bị lật / mẫu đúng ban đầu.
    Trả về dict (kèm khóa '_examples' nếu n_examples > 0).
    """
    source_model.eval().to(device)
    eval_model.eval().to(device)
    attack = attack_factory(source_model) if attack_factory else None
    clean_ok, adv_ok, linf, l2 = [], [], [], []
    attack_time, n, ex = 0.0, 0, None
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred_c, conf_c = _predict(eval_model, x)
        _sync(device)
        t0 = time.time()
        x_adv = attack(x, y) if attack else x.clone()
        _sync(device)
        attack_time += time.time() - t0
        pred_a, conf_a = _predict(eval_model, x_adv)
        li, l2n = perturbation_norms(x, x_adv)
        clean_ok.append((pred_c == y).cpu())
        adv_ok.append((pred_a == y).cpu())
        linf.append(li.cpu())
        l2.append(l2n.cpu())
        if n_examples and ex is None:
            k = min(n_examples, y.size(0))
            ex = {
                "x": x[:k].cpu(), "x_adv": x_adv[:k].cpu(), "y": y[:k].cpu(),
                "pred_clean": pred_c[:k].cpu(), "conf_clean": conf_c[:k].cpu(),
                "pred_adv": pred_a[:k].cpu(), "conf_adv": conf_a[:k].cpu(),
                "attack_cfg": attack_cfg or {},
            }
        n += y.size(0)
    clean_ok, adv_ok = torch.cat(clean_ok), torch.cat(adv_ok)
    asr = attack_success_rate(clean_ok, adv_ok)
    cfg = attack_cfg or {}
    row = {
        "source_model": source_name,
        "eval_model": eval_name,
        "attack": attack_label,
        "epsilon": cfg.get("epsilon", 0.0),
        "steps": cfg.get("steps", np.nan),
        "alpha": cfg.get("alpha", np.nan),
        "random_start": cfg.get("random_start", np.nan),
        "defense": defense,
        "n_samples": n,
        "clean_accuracy": clean_ok.float().mean().item(),
        "accuracy": adv_ok.float().mean().item(),
        "asr": asr,
        "n_clean_correct": int(clean_ok.sum()),
        "mean_linf": torch.cat(linf).mean().item(),
        "mean_l2": torch.cat(l2).mean().item(),
        "attack_time_sec": attack_time,
        "ms_per_sample": 1000 * attack_time / n,
        "note": "ASR=N/A (không có mẫu đúng ban đầu)" if math.isnan(asr) else "",
    }
    if ex is not None:
        row["_examples"] = ex
    return row


def sweep_attack(
    source_model, eval_model, loader, device, source_name, eval_name,
    attack: str, epsilons: list[float], steps: int = 10, alpha_ratio: float = 0.25,
    random_start: bool = True, defense: str = "none", n_examples: int = 0,
) -> list[dict]:
    """Chạy một attack qua nhiều epsilon; alpha = epsilon * alpha_ratio (PGD)."""
    rows = []
    for eps in epsilons:
        if attack == "pgd":
            kw = dict(steps=steps, alpha=eps * alpha_ratio, random_start=random_start)
            cfg = {"epsilon": eps, **kw}
        else:
            kw, cfg = {}, {"epsilon": eps}
        factory = lambda m, eps=eps, kw=kw: make_attack(attack, m, eps, **kw)
        rows.append(evaluate_attack(source_model, eval_model, loader, factory, device, source_name, eval_name,
                                    attack, defense, n_examples, cfg))
    return rows


def save_rows(rows: list[dict], csv_path: str | Path, examples_dir: str | Path | None = None) -> pd.DataFrame:
    """Gộp vào CSV: dòng cùng khóa thí nghiệm bị thay bằng dòng mới, các dòng khác giữ nguyên."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])[TABLE_COLS]
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        old = old[~_row_keys(old).isin(set(_row_keys(new)))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(csv_path, index=False)
    if examples_dir:
        ed = Path(examples_dir)
        ed.mkdir(parents=True, exist_ok=True)
        for r in rows:
            if "_examples" in r:
                e = r["_examples"]
                np.savez_compressed(
                    ed / f"{r['eval_model']}_{r['attack']}_eps{r['epsilon']}.npz",
                    **{k: v.numpy() for k, v in e.items() if torch.is_tensor(v)},
                )
    return new
