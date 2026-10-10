"""Biểu đồ kết quả (matplotlib, lưu PNG). Màu cố định theo thực thể, an toàn cho mù màu."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Palette theo thứ tự cố định (Okabe-Ito), gán theo thực thể, không xoay vòng.
COLORS = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7", "gray": "#6B7280"}
SERIES = list(COLORS.values())


def _style(ax) -> None:
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig, out) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_training_curves(history_csv: str | Path, out: str | Path, title: str = "Training history", acc_label: str = "accuracy") -> Path:
    """Loss và accuracy theo epoch (train vs validation). Nếu có val_clean_acc (adversarial training) thì vẽ thêm."""
    df = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df["epoch"], df["train_loss"], color=COLORS["blue"], lw=2, label="Train loss")
    axes[0].plot(df["epoch"], df["val_loss"], color=COLORS["orange"], lw=2, label="Validation loss")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title=f"{title} - loss")
    axes[1].plot(df["epoch"], 100 * df["train_acc"], color=COLORS["blue"], lw=2, label=f"Train {acc_label}")
    axes[1].plot(df["epoch"], 100 * df["val_acc"], color=COLORS["orange"], lw=2, label=f"Validation {acc_label}")
    if "val_clean_acc" in df:
        axes[1].plot(df["epoch"], 100 * df["val_clean_acc"], color=COLORS["green"], lw=2, ls="--", label="Validation clean accuracy")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title=f"{title} - accuracy")
    for ax in axes:
        _style(ax)
        ax.legend(frameon=False)
    return _save(fig, out)


ATTACK_COLOR = {"fgsm": COLORS["blue"], "pgd": COLORS["orange"]}  # màu theo attack, cố định
LINESTYLES = ["-", "--", ":", "-."]                                  # kiểu nét theo model/defense


def plot_attack_curves(df: pd.DataFrame, out: str | Path, title: str, metric: str = "accuracy", label_col: str = "eval_model") -> Path:
    """Accuracy (hoặc ASR) theo epsilon. Màu = attack, kiểu nét = model/defense (label_col).

    Chỉ gồm các dòng FGSM/PGD; epsilon trục x theo pixel space [0, 1].
    """
    ylabel = {"accuracy": "Accuracy on adversarial images (%)", "asr": "Attack success rate (%)"}[metric]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    labels = list(dict.fromkeys(df[label_col]))
    for li, lab in enumerate(labels):
        for atk in ("fgsm", "pgd"):
            d = df[(df[label_col] == lab) & (df["attack"] == atk)].sort_values("epsilon")
            if d.empty:
                continue
            ax.plot(d["epsilon"], 100 * d[metric], color=ATTACK_COLOR[atk], ls=LINESTYLES[li % 4], lw=2, marker="o", ms=5,
                    label=f"{lab} / {atk.upper()}")
    ax.set(xlabel="Epsilon (L∞, pixel scale [0, 1])", ylabel=ylabel, title=title, ylim=(-2, 102))
    _style(ax)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, out)


def plot_defense_curves(df: pd.DataFrame, out: str | Path, title: str, metric: str = "accuracy", order: list[str] | None = None) -> Path:
    """So sánh các defense (cột `defense`, 'none' = không phòng thủ) theo epsilon, mỗi attack một panel.

    Màu cố định theo defense (theo `order`, 'none' luôn xám).
    """
    ylabel = {"accuracy": "Accuracy on adversarial images (%)", "asr": "Attack success rate (%)"}[metric]
    names = order or list(dict.fromkeys(df["defense"]))
    palette = {n: COLORS["gray"] if n == "none" else SERIES[i % 5] for i, n in enumerate([m for m in names if m != "none"])}
    palette["none"] = COLORS["gray"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, atk in zip(axes, ("fgsm", "pgd")):
        for n in ["none"] + [m for m in names if m != "none"]:
            d = df[(df["defense"] == n) & (df["attack"] == atk)].sort_values("epsilon")
            if not d.empty:
                ax.plot(d["epsilon"], 100 * d[metric], color=palette[n], lw=2, marker="o", ms=5, label=n)
        ax.set(xlabel="Epsilon (L∞, pixel scale [0, 1])", title=f"{atk.upper()}", ylim=(-2, 102))
        _style(ax)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    return _save(fig, out)


def plot_confusion_matrix(cm, class_names: list[str], out: str | Path, title: str = "Confusion matrix") -> Path:
    """Confusion matrix (hàng = nhãn thật, cột = dự đoán), số mẫu."""
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set(xlabel="Predicted label", ylabel="True label", title=title)
    thr = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8, color="white" if cm[i, j] > thr else "#111827")
    fig.colorbar(im, ax=ax, label="Number of samples")
    return _save(fig, out)


def plot_final_bars(table: pd.DataFrame, out: str | Path, main_eps: float) -> Path:
    """So sánh cuối: Clean / FGSM / PGD accuracy (ở epsilon chính) cho mọi cấu hình trong bảng defense."""
    t = table[table["Clean Accuracy"].apply(lambda v: isinstance(v, float))].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(t) + 1.8))
    h = 0.26
    for j, (col, name, color) in enumerate([("Clean Accuracy", "Clean", COLORS["green"]), ("FGSM Accuracy", "FGSM", ATTACK_COLOR["fgsm"]),
                                             ("PGD Accuracy", "PGD", ATTACK_COLOR["pgd"])]):
        vals = pd.to_numeric(t[col], errors="coerce") * 100
        ys = np.arange(len(t)) + (j - 1) * h
        ax.barh(ys, vals.fillna(0), height=h * 0.9, color=color, label=name if name == "Clean" else f"{name} (eps={main_eps:g})")
        for y, v in zip(ys, vals):
            if not np.isnan(v):
                ax.text(v + 0.8, y, f"{v:.1f}", va="center", fontsize=7, color="#374151")
    ax.set_yticks(np.arange(len(t)), t["Label"], fontsize=8)
    ax.invert_yaxis()
    ax.set(xlabel="Accuracy (%)", title=f"Final comparison on CIFAR-10 test set (adversarial accuracy at eps={main_eps:g}, L∞ pixel scale)", xlim=(0, 108))
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    return _save(fig, out)


def plot_tradeoff(table: pd.DataFrame, out: str | Path, main_eps: float) -> Path:
    """Đánh đổi clean accuracy (x) và PGD accuracy (y); kích thước điểm tỉ lệ thời gian train (nhỏ = không train)."""
    t = table[table["Clean Accuracy"].apply(lambda v: isinstance(v, float))].reset_index(drop=True)
    groups = list(dict.fromkeys(t["Group"]))
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for gi, g in enumerate(groups):
        d = t[t["Group"] == g]
        sizes = 40 + 0.08 * pd.to_numeric(d["Training Time (s)"], errors="coerce").fillna(0)
        ax.scatter(100 * d["Clean Accuracy"], 100 * d["PGD Accuracy"], s=sizes, color=SERIES[gi % 5], label=g, edgecolor="white", linewidth=1.2, zorder=3)
    # Gộp nhãn các điểm trùng vị trí (làm tròn 1%) để không chồng chữ
    spots: dict[tuple, list[str]] = {}
    for _, r in t.iterrows():
        key = (round(100 * r["Clean Accuracy"]), round(100 * r["PGD Accuracy"]))
        spots.setdefault(key, []).append(r["Short"])
    offsets = [(7, 7), (7, -13), (-70, 7), (-70, -13)]
    for i, ((x, y), labs) in enumerate(spots.items()):
        ax.annotate(", ".join(dict.fromkeys(labs)), (x, y), textcoords="offset points", xytext=offsets[i % 4] if y < 5 else (7, 7), fontsize=7)
    ax.set(xlabel="Clean accuracy (%)", ylabel=f"PGD accuracy at eps={main_eps:g} (%)", xlim=(0, 100), ylim=(-3, 100),
           title="Trade-off: clean vs adversarial accuracy (point size ~ training time; top-right is best)")
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return _save(fig, out)
