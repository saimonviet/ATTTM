"""Luồng chạy Gaussian Blur defense: transfer + adaptive -> bảng CSV -> hình. Dùng bởi scripts/run_blur.py."""
from __future__ import annotations

import pandas as pd
import torch

from ..attacks import make_attack
from ..config import resolve_path
from ..data.cifar10 import CLASS_NAMES, get_test_loader
from ..device import get_device
from ..seed import set_seed
from ..utils.checkpoint import load_checkpoint
from ..utils.cli import results_dir
from ..utils.logging import get_logger
from ..visualization.plot_results import plot_defense_curves
from ..visualization.visualize_attacks import plot_blur_examples
from .evaluate_attacks import _predict, save_rows, sweep_attack
from .evaluate_defenses import attack_specs, build_variants, evaluate_transfer, save_analysis


def run_blur_defense(cfg: dict, checkpoint: str, kernel_sizes: list[int] | None = None, sigmas: list[float] | None = None,
                     max_samples: int | None = None, model_name: str | None = None, adaptive: bool = True) -> pd.DataFrame:
    log = get_logger()
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    ckpt = resolve_path(checkpoint)
    model, _ = load_checkpoint(ckpt, device)
    name = model_name or ckpt.stem.removesuffix("_best")
    dc = cfg["defense"]["gaussian_blur"]
    variants = build_variants(model, kernel_sizes or dc["kernel_sizes"], sigmas or dc["sigmas"])
    loader = get_test_loader(cfg, max_samples or cfg["eval"]["max_samples"])
    out = results_dir(cfg)
    specs = attack_specs(cfg)

    log.info(f"[{name}] Kịch bản 1 (transfer): adversarial tạo trên {name}, đánh giá trên {len(variants)} cấu hình blur")
    rows, analysis = evaluate_transfer(model, name, variants, loader, device, specs)
    for r in rows:
        if r["epsilon"] in (0.0, 0.03):
            log.info(f"  {r['defense']:>24s} {r['attack']:>4s} eps={r['epsilon']}: acc={r['accuracy']:.4f} asr={r['asr']:.4f}")
    adaptive_rows = []
    if adaptive:
        pc, fc = cfg["attacks"]["pgd"], cfg["attacks"]["fgsm"]
        for vname, d in variants.items():
            log.info(f"[{name}] Kịch bản 2 (adaptive): attack tạo trực tiếp trên {name}+{vname}")
            dn = f"{name}+{vname}"
            for atk, eps in (("fgsm", [e for e in fc["epsilons"] if e > 0]), ("pgd", pc["epsilons"])):
                rs = sweep_attack(d, d, loader, device, dn, dn, atk, eps, steps=pc["steps"], alpha_ratio=pc["alpha_ratio"],
                                  random_start=pc["random_start"], defense=vname)
                for r in rs:
                    r["note"] = "adaptive: attack tạo trên model đã gắn blur"
                adaptive_rows += rs
    df = save_rows(rows + adaptive_rows, out / "tables" / "attack_results.csv")
    save_analysis(analysis, out / "tables" / "blur_analysis.csv")

    # Hình so sánh theo epsilon (none = baseline không blur, attack white-box trên baseline)
    ref = df[(df["source_model"] == name) & (df["eval_model"] == name) & (df["defense"] == "none") & df["attack"].isin(["fgsm", "pgd"])]
    order = ["none"] + list(variants)
    tr = pd.concat([ref, df[(df["source_model"] == name) & df["defense"].isin(variants) & (df["eval_model"] != name)]])
    ad = pd.concat([ref, df[df["source_model"].isin([f"{name}+{v}" for v in variants]) & (df["source_model"] == df["eval_model"])]])
    for metric in ("accuracy", "asr"):
        plot_defense_curves(tr, out / "figures" / f"{name}_blur_transfer_{metric}.png",
                            f"Gaussian Blur, non-adaptive (adversarial crafted on {name}, evaluated with blur) - {metric}", metric, order)
        if adaptive:
            plot_defense_curves(ad, out / "figures" / f"{name}_blur_adaptive_{metric}.png",
                                f"Gaussian Blur, adaptive attack (crafted through blur) - {metric}", metric, order)

    # Ví dụ trực quan: 4 ảnh test đầu tiên, PGD eps=0.03 tạo trên model không blur, blur cấu hình đầu tiên
    x, y = next(iter(loader))
    x, y = x[:4].to(device), y[:4].to(device)
    first = next(iter(variants.values()))
    xa = make_attack("pgd", model, 0.03, steps=cfg["attacks"]["pgd"]["steps"], alpha=0.03 * cfg["attacks"]["pgd"]["alpha_ratio"])(x, y)
    imgs = {"x": x, "x_blur": first.blur(x), "x_adv": xa, "x_adv_blur": first.blur(xa)}
    ex = {"y": y.cpu(), **{k: v.detach().cpu() for k, v in imgs.items()}}
    for key, img in zip(("clean", "clean_blur", "adv", "adv_blur"), imgs.values()):
        p, c = _predict(model, img.detach())
        ex["pred_" + key], ex["conf_" + key] = p.cpu(), c.cpu()
    plot_blur_examples(ex, CLASS_NAMES, out / "figures" / f"{name}_blur_examples.png",
                       f"{name}: effect of Gaussian Blur ({first.blur.extra_repr()}); PGD eps=0.03 crafted on {name} without blur")
    return df
