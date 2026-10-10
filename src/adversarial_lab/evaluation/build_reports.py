"""Tổng hợp kết quả thực nghiệm thành bảng và biểu đồ cuối. Mọi số liệu lấy từ file kết quả đã chạy thật;
mục chưa chạy được ghi 'NOT_RUN', không tự tạo giá trị."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import resolve_path
from ..utils.manifest import build_manifest
from ..visualization.plot_results import plot_attack_curves, plot_final_bars, plot_tradeoff

NOT_RUN = "NOT_RUN"


def attack_table(df: pd.DataFrame) -> pd.DataFrame:
    """Bảng attack theo định dạng yêu cầu (kèm model tạo attack và defense để không nhầm lẫn)."""
    t = pd.DataFrame({
        "Model": df["eval_model"], "Attack": df["attack"], "Epsilon": df["epsilon"], "Accuracy": df["accuracy"], "ASR": df["asr"],
        "Mean L_inf": df["mean_linf"], "Mean L2": df["mean_l2"], "Time (s)": df["attack_time_sec"],
        "Attack source model": df["source_model"], "Defense": df["defense"], "N samples": df["n_samples"],
    })
    return t.sort_values(["Model", "Attack", "Epsilon"]).reset_index(drop=True)


def _row(df, source, eval_, attack, eps):
    d = df[(df["source_model"] == source) & (df["eval_model"] == eval_) & (df["attack"] == attack) & np.isclose(df["epsilon"], eps)]
    return d.iloc[0] if len(d) else None


def _train_label(training: str | None) -> str:
    if not training or training.startswith("baseline"):
        return "None (baseline)"
    m = re.match(r"adversarial_(\w+?)_(from_scratch|finetune)", training)
    return f"Adversarial Training ({m.group(1).upper()}, {'fine-tune' if m.group(2) == 'finetune' else 'from scratch'})" if m else training


def load_clean_json(tables_dir: Path, ckpt_path: Path) -> dict | None:
    p = tables_dir / f"{ckpt_path.stem}_clean.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def defense_table(df: pd.DataFrame, models: dict[str, Path], tables_dir: Path, main_eps: float) -> pd.DataFrame:
    """Mỗi dòng một cấu hình phòng thủ: white-box cho từng model; blur ở hai kịch bản (transfer, adaptive)."""
    cols = ["Model", "Defense", "Scenario", "Attack source model", "Epsilon", "Clean Accuracy", "FGSM Accuracy", "PGD Accuracy",
            "ASR (FGSM)", "ASR (PGD)", "Training Time (s)", "PGD attack time (s)", "Notes", "Label", "Group", "Short"]
    out = []

    def make(model, defense, scenario, source, eval_, train_time, notes, label, group):
        f, p = _row(df, source, eval_, "fgsm", main_eps), _row(df, source, eval_, "pgd", main_eps)
        ref = f if f is not None else p
        out.append({
            "Model": model, "Defense": defense, "Scenario": scenario, "Attack source model": source, "Epsilon": main_eps,
            "Clean Accuracy": ref["clean_accuracy"] if ref is not None else NOT_RUN,
            "FGSM Accuracy": f["accuracy"] if f is not None else NOT_RUN, "PGD Accuracy": p["accuracy"] if p is not None else NOT_RUN,
            "ASR (FGSM)": f["asr"] if f is not None else NOT_RUN, "ASR (PGD)": p["asr"] if p is not None else NOT_RUN,
            "Training Time (s)": train_time, "PGD attack time (s)": p["attack_time_sec"] if p is not None else NOT_RUN,
            "Notes": notes, "Label": label, "Group": group,
            "Short": label.split(" + blur ")[-1].split(" [")[0] if " + blur " in label else label,
        })

    base_name = next(iter(models))
    for name, ck in models.items():
        js = load_clean_json(tables_dir, ck)
        tt = js["train_time_sec"] if js and "train_time_sec" in js else NOT_RUN
        dl = _train_label(js.get("training") if js else None)
        make(name, dl, "white-box", name, name, tt, "attack tạo và đánh giá trên cùng model", name, dl.split(" (")[0])
    variants = sorted({v for v in df["defense"].unique() if str(v).startswith("gaussian_blur")})
    for v in variants:
        short = v.replace("gaussian_blur_", "blur ")
        make(base_name, v, "transfer (non-adaptive)", base_name, f"{base_name}+{v}", 0.0,
             "không train lại; adversarial tạo trên model không có blur", f"{base_name} + {short} [transfer]", "Gaussian Blur (non-adaptive)")
        make(base_name, v, "adaptive", f"{base_name}+{v}", f"{base_name}+{v}", 0.0,
             "không train lại; attack tạo trực tiếp qua blur", f"{base_name} + {short} [adaptive]", "Gaussian Blur (adaptive)")
    return pd.DataFrame(out)[cols]


def seed_summary(df: pd.DataFrame, main_eps: float) -> pd.DataFrame | None:
    """mean ± std (ddof=1) qua các seed. Model theo seed đặt tên '<base>_seed<N>'; model không hậu tố tính là một lần chạy (seed gốc)."""
    wb = df[(df["source_model"] == df["eval_model"]) & (df["defense"] == "none")]
    groups: dict[str, list[str]] = {}
    for m in wb["eval_model"].unique():
        groups.setdefault(re.sub(r"_seed\d+$", "", m), []).append(m)
    rows = []
    for base, members in groups.items():
        if len(members) < 2:
            continue
        per = []
        for m in members:
            f, p = _row(df, m, m, "fgsm", main_eps), _row(df, m, m, "pgd", main_eps)
            if f is None or p is None:
                continue
            per.append({"clean": f["clean_accuracy"], "fgsm": f["accuracy"], "pgd": p["accuracy"], "asr_pgd": p["asr"]})
        if len(per) < 2:
            continue
        p_df = pd.DataFrame(per)
        r = {"Model": base, "N seeds": len(per), "Epsilon": main_eps}
        for k in p_df:
            r[f"{k} mean"], r[f"{k} std"] = p_df[k].mean(), p_df[k].std(ddof=1)
        rows.append(r)
    return pd.DataFrame(rows) if rows else None


def to_markdown(df: pd.DataFrame, pct_cols: tuple[str, ...] = (), drop: tuple[str, ...] = ()) -> str:
    d = df.drop(columns=[c for c in drop if c in df]).copy()
    def fmt(c, v):
        if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
            if pd.isna(v):
                return "N/A"
            return f"{100 * v:.2f}%" if c in pct_cols else f"{v:.4g}"
        return str(v)
    head = "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "\n".join("| " + " | ".join(fmt(c, v) for c, v in zip(d.columns, r)) + " |" for r in d.itertuples(index=False)) + "\n"


def build_reports(cfg: dict, results: str | Path | None = None, checkpoints: dict | None = None) -> dict:
    """Đọc attack_results.csv và JSON clean, ghi bảng/hình cuối và experiment_manifest.json."""
    res = Path(results) if results else resolve_path(cfg["paths"]["results"])
    tables, figs = res / "tables", res / "figures"
    csv = tables / "attack_results.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Chưa có {csv}; hãy chạy các thí nghiệm trước (scripts/evaluate_all.py).")
    df = pd.read_csv(csv)
    models = {n: resolve_path(p) for n, p in (checkpoints or cfg["models"]).items()}
    eps = cfg["report"]["main_epsilon"]
    outputs = {}

    at = attack_table(df)
    at.to_csv(tables / "attack_table.csv", index=False)
    (tables / "attack_table.md").write_text(to_markdown(at, pct_cols=("Accuracy", "ASR")), encoding="utf-8")
    dt = defense_table(df, models, tables, eps)
    dt.to_csv(tables / "defense_table.csv", index=False)
    (tables / "defense_table.md").write_text(
        to_markdown(dt, pct_cols=("Clean Accuracy", "FGSM Accuracy", "PGD Accuracy", "ASR (FGSM)", "ASR (PGD)"), drop=("Label", "Group", "Short")), encoding="utf-8")
    outputs.update(attack_table=tables / "attack_table.csv", defense_table=tables / "defense_table.csv")

    ss = seed_summary(df, eps)
    if ss is not None:
        ss.to_csv(tables / "seed_summary.csv", index=False)
        outputs["seed_summary"] = tables / "seed_summary.csv"

    wb = df[(df["source_model"] == df["eval_model"]) & (df["defense"] == "none") & df["attack"].isin(["fgsm", "pgd"])]
    wb = wb[wb["eval_model"].isin(models)]
    if wb["eval_model"].nunique() >= 2:
        for metric in ("accuracy", "asr"):
            plot_attack_curves(wb, figs / f"comparison_models_{metric}_vs_epsilon.png",
                               f"{'ASR' if metric == 'asr' else 'Accuracy'} vs epsilon: model comparison (white-box, color = attack, line style = model)", metric)
    plot_final_bars(dt, figs / "final_comparison_bars.png", eps)
    plot_tradeoff(dt, figs / "tradeoff_clean_vs_pgd.png", eps)

    man = build_manifest(cfg, {n: str(p) for n, p in models.items()}, csv)
    man["report"] = {"main_epsilon": eps, "tables": [str(p) for p in outputs.values()],
                     "not_run": [f"{r['Model']}/{r['Scenario']}" for _, r in dt.iterrows()
                                 if NOT_RUN in (r["Clean Accuracy"], r["FGSM Accuracy"], r["PGD Accuracy"])]}
    mdir = res / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "experiment_manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    outputs["manifest"] = mdir / "experiment_manifest.json"
    return outputs
