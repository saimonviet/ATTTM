"""Hình minh họa ảnh gốc / perturbation / ảnh đối kháng với dự đoán và confidence."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _img(t):
    return t.permute(1, 2, 0).clamp(0, 1).numpy()


def plot_blur_examples(ex: dict, class_names: list[str], out: str | Path, title: str = "") -> Path:
    """Mỗi hàng: ảnh sạch | sạch+blur | đối kháng | đối kháng+blur, kèm dự đoán và confidence của cùng một model.

    ex: x, x_blur, x_adv, x_adv_blur (N,3,32,32), y, và pred_*/conf_* cho từng cột (khóa 'clean','clean_blur','adv','adv_blur').
    """
    cols = [("x", "clean", "Clean"), ("x_blur", "clean_blur", "Clean + blur"), ("x_adv", "adv", "Adversarial"),
            ("x_adv_blur", "adv_blur", "Adversarial + blur")]
    n = ex["x"].shape[0]
    fig, axes = plt.subplots(n, 4, figsize=(10, 2.5 * n), squeeze=False)
    for i in range(n):
        for j, (img, key, name) in enumerate(cols):
            ax = axes[i, j]
            ax.imshow(_img(ex[img][i]), interpolation="nearest")
            ax.set_title(f"{name}\ntrue: {class_names[int(ex['y'][i])]}\npred: {class_names[int(ex['pred_' + key][i])]} "
                         f"({100 * float(ex['conf_' + key][i]):.1f}%)", fontsize=8)
            ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_attack_examples(ex: dict, class_names: list[str], out: str | Path, title: str = "") -> Path:
    """Mỗi hàng: ảnh gốc | perturbation (phóng đại) | ảnh đối kháng, kèm nhãn dự đoán và confidence.

    Perturbation được hiển thị bằng 0.5 + delta / (2*epsilon) (xám = không đổi) để nhìn thấy được;
    đây chỉ là scale để quan sát, KHÔNG phải độ lớn thực của perturbation.
    """
    n = ex["x"].shape[0]
    eps = max(float(ex["attack_cfg"].get("epsilon", 0.0)), 1e-8)
    fig, axes = plt.subplots(n, 3, figsize=(8.5, 2.6 * n), squeeze=False)
    for i in range(n):
        delta = ex["x_adv"][i] - ex["x"][i]
        vis = (0.5 + delta / (2 * eps)).clamp(0, 1)
        y, pc, pa = int(ex["y"][i]), int(ex["pred_clean"][i]), int(ex["pred_adv"][i])
        panels = [
            (_img(ex["x"][i]), f"Original\ntrue: {class_names[y]}\npred: {class_names[pc]} ({100*ex['conf_clean'][i]:.1f}%)"),
            (_img(vis), f"Perturbation (amplified)\nmax |δ| = {delta.abs().max():.3f}"),
            (_img(ex["x_adv"][i]), f"Adversarial\npred: {class_names[pa]} ({100*ex['conf_adv'][i]:.1f}%)"),
        ]
        for j, (im, t) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(im, interpolation="nearest")
            ax.set_title(t, fontsize=8)
            ax.axis("off")
    fig.suptitle(
        f"{title}\nPerturbation shown as 0.5 + δ/(2ε), ε={eps:g} (gray = no change; scaled for visibility only)",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97 if n > 3 else 0.9))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
