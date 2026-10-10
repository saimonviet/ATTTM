"""Đánh giá Gaussian Blur defense.

Hai kịch bản (luôn ghi rõ trong cột source_model / eval_model của bảng kết quả):
 1. Không thích nghi (transfer): adversarial tạo trên baseline (không biết blur), đánh giá trên baseline + blur.
 2. Thích nghi (adaptive): attack được tạo trực tiếp trên model đã gắn blur (gradient đi qua blur).
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..attacks import make_attack
from ..defenses.gaussian_blur import BlurDefendedModel, blur_name
from ..utils.metrics import attack_success_rate, perturbation_norms
from .evaluate_attacks import _predict, _sync


def attack_specs(cfg: dict) -> list[dict]:
    """Danh sách (attack, epsilon, ...) từ config: FGSM (gồm eps=0 = ảnh sạch) và PGD."""
    pc = cfg["attacks"]["pgd"]
    specs = [{"attack": "fgsm", "epsilon": float(e)} for e in cfg["attacks"]["fgsm"]["epsilons"]]
    specs += [{"attack": "pgd", "epsilon": float(e), "steps": pc["steps"], "alpha": float(e) * pc["alpha_ratio"],
               "random_start": pc["random_start"]} for e in pc["epsilons"]]
    return specs


def _make(spec: dict, model):
    kw = {k: spec[k] for k in ("steps", "alpha", "random_start") if k in spec}
    return make_attack(spec["attack"], model, spec["epsilon"], **kw)


@torch.no_grad()
def _eval_defended(defended, x, x_adv, y):
    pc, _ = _predict(defended, x)
    pa, _ = _predict(defended, x_adv)
    return pc == y, pa == y


def evaluate_transfer(source_model, source_name: str, variants: dict[str, BlurDefendedModel], loader: DataLoader,
                      device: torch.device, specs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Sinh adversarial MỘT lần trên source_model rồi đánh giá trên từng model có blur.

    Trả về (rows theo schema attack_results.csv, rows phân tích perturbation/mất thông tin do blur).
    """
    source_model.eval().to(device)
    for v in variants.values():
        v.eval().to(device)
    acc = {(i, v): {"c": [], "a": [], "l2_after": [], "info": []} for i in range(len(specs)) for v in variants}
    inp = {i: {"linf": [], "l2": [], "time": 0.0} for i in range(len(specs))}
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        for i, spec in enumerate(specs):
            _sync(device)
            t0 = time.time()
            x_adv = x.clone() if spec["epsilon"] == 0 else _make(spec, source_model)(x, y)
            _sync(device)
            inp[i]["time"] += time.time() - t0
            li, l2 = perturbation_norms(x, x_adv)
            inp[i]["linf"].append(li.cpu())
            inp[i]["l2"].append(l2.cpu())
            for vname, d in variants.items():
                c, a = _eval_defended(d, x, x_adv, y)
                with torch.no_grad():
                    bx, bxa = d.blur(x), d.blur(x_adv)
                acc[(i, vname)]["c"].append(c.cpu())
                acc[(i, vname)]["a"].append(a.cpu())
                acc[(i, vname)]["l2_after"].append((bxa - bx).flatten(1).norm(dim=1).cpu())
                acc[(i, vname)]["info"].append((bx - x).flatten(1).norm(dim=1).cpu())
        n += y.size(0)
    rows, analysis = [], []
    for i, spec in enumerate(specs):
        l2_in = torch.cat(inp[i]["l2"]).mean().item()
        for vname in variants:
            r = acc[(i, vname)]
            c, a = torch.cat(r["c"]), torch.cat(r["a"])
            asr = attack_success_rate(c, a)
            rows.append({
                "source_model": source_name, "eval_model": f"{source_name}+{vname}", "attack": spec["attack"],
                "epsilon": spec["epsilon"], "steps": spec.get("steps", np.nan), "alpha": spec.get("alpha", np.nan),
                "random_start": spec.get("random_start", np.nan), "defense": vname, "n_samples": n,
                "clean_accuracy": c.float().mean().item(), "accuracy": a.float().mean().item(), "asr": asr,
                "n_clean_correct": int(c.sum()), "mean_linf": torch.cat(inp[i]["linf"]).mean().item(), "mean_l2": l2_in,
                "attack_time_sec": inp[i]["time"], "ms_per_sample": 1000 * inp[i]["time"] / n,
                "note": "transfer: attack tạo trên model không có blur" + ("; ASR=N/A" if math.isnan(asr) else ""),
            })
            l2_after = torch.cat(r["l2_after"]).mean().item()
            analysis.append({
                "source_model": source_name, "defense": vname, "attack": spec["attack"], "epsilon": spec["epsilon"],
                "mean_l2_perturbation_before_blur": l2_in, "mean_l2_perturbation_after_blur": l2_after,
                "perturbation_ratio_after_over_before": l2_after / l2_in if l2_in > 0 else np.nan,
                "mean_l2_blur_info_loss_clean": torch.cat(r["info"]).mean().item(), "n_samples": n,
            })
    return rows, analysis


def save_analysis(rows: list[dict], path) -> pd.DataFrame:
    """Bảng phân tích blur; dòng cùng (source_model, defense, attack, epsilon) được thay thế."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        key = ["source_model", "defense", "attack", "epsilon"]
        m = old.merge(new[key], on=key, how="left", indicator=True)
        new = pd.concat([old[(m["_merge"] == "left_only").values], new], ignore_index=True)
    new.to_csv(path, index=False)
    return new


def build_variants(model, kernel_sizes: list[int], sigmas: list[float]) -> dict[str, BlurDefendedModel]:
    return {blur_name(k, s): BlurDefendedModel(model, k, s) for k in kernel_sizes for s in sigmas}
