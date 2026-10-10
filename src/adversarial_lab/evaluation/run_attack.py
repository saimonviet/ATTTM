"""Luồng chạy attack trên một checkpoint: đánh giá -> bảng CSV -> hình. Dùng bởi run_fgsm.py / run_pgd.py."""
from __future__ import annotations

from pathlib import Path

from ..config import resolve_path
from ..data.cifar10 import CLASS_NAMES, get_test_loader
from ..device import get_device
from ..seed import set_seed
from ..utils.checkpoint import load_checkpoint
from ..utils.cli import results_dir
from ..utils.logging import get_logger
from ..visualization.plot_results import plot_attack_curves
from ..visualization.visualize_attacks import plot_attack_examples
from .evaluate_attacks import save_rows, sweep_attack

import pandas as pd


def run_attack(cfg: dict, attack: str, epsilons: list[float], checkpoint: str, steps: int | None = None,
               alpha_ratio: float | None = None, random_start: bool | None = None,
               max_samples: int | None = None, model_name: str | None = None, n_examples: int = 6) -> pd.DataFrame:
    """White-box: attack tạo trên model và đánh giá trên chính model đó."""
    log = get_logger()
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    ckpt = resolve_path(checkpoint)
    model, _ = load_checkpoint(ckpt, device)
    name = model_name or ckpt.stem.removesuffix("_best")  # baseline_best.pth -> 'baseline' (khớp tên trong bảng tổng hợp)
    loader = get_test_loader(cfg, max_samples or cfg["eval"]["max_samples"])
    pc = cfg["attacks"]["pgd"]
    rows = sweep_attack(
        model, model, loader, device, name, name, attack, epsilons,
        steps=steps if steps is not None else pc["steps"],
        alpha_ratio=alpha_ratio if alpha_ratio is not None else pc["alpha_ratio"],
        random_start=pc["random_start"] if random_start is None else random_start,
        n_examples=n_examples,
    )
    out = results_dir(cfg)
    df = save_rows(rows, out / "tables" / "attack_results.csv", out / "examples")
    for r in rows:
        log.info(f"[{name}] {attack} eps={r['epsilon']}: acc={r['accuracy']:.4f} asr={r['asr']:.4f} "
                 f"linf={r['mean_linf']:.4f} l2={r['mean_l2']:.3f} t={r['attack_time_sec']:.1f}s (n={r['n_samples']})")
        if "_examples" in r and r["epsilon"] > 0:
            plot_attack_examples(r["_examples"], CLASS_NAMES, out / "figures" / f"examples_{name}_{attack}_eps{r['epsilon']}.png",
                                 f"{attack.upper()} on {name} (white-box), eps={r['epsilon']}")
    sub = df[(df["source_model"] == name) & (df["eval_model"] == name) & (df["defense"] == "none") & df["attack"].isin(["fgsm", "pgd"])]
    for metric in ("accuracy", "asr"):
        plot_attack_curves(sub, out / "figures" / f"{name}_{metric}_vs_epsilon.png",
                           f"{metric.upper() if metric == 'asr' else 'Accuracy'} vs epsilon - {name} (white-box)", metric)
    return df
