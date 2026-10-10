import json

import numpy as np
import pandas as pd
import pytest

from adversarial_lab.config import load_config
from adversarial_lab.evaluation.build_reports import NOT_RUN, attack_table, build_reports, defense_table, seed_summary, to_markdown
from adversarial_lab.evaluation.evaluate_attacks import TABLE_COLS


def _rows(model, accs, source=None, defense="none", clean=0.9):
    """Dòng kết quả giả CHỈ dùng trong test để kiểm tra logic bảng (không phải kết quả thực nghiệm)."""
    out = []
    for atk, a in accs.items():
        out.append({"source_model": source or model, "eval_model": model, "attack": atk, "epsilon": 0.03, "steps": np.nan, "alpha": np.nan,
                    "random_start": np.nan, "defense": defense, "n_samples": 100, "clean_accuracy": clean, "accuracy": a,
                    "asr": 1 - a / clean, "n_clean_correct": 90, "mean_linf": 0.03, "mean_l2": 1.0, "attack_time_sec": 1.0,
                    "ms_per_sample": 10.0, "note": ""})
    return out


def test_attack_table_columns():
    df = pd.DataFrame(_rows("m", {"fgsm": 0.1, "pgd": 0.0}))[TABLE_COLS]
    t = attack_table(df)
    assert list(t.columns[:8]) == ["Model", "Attack", "Epsilon", "Accuracy", "ASR", "Mean L_inf", "Mean L2", "Time (s)"]


def test_defense_table_marks_missing_as_not_run(tmp_path):
    df = pd.DataFrame(_rows("baseline", {"fgsm": 0.1, "pgd": 0.0}))
    t = defense_table(df, {"baseline": tmp_path / "b.pth", "adv": tmp_path / "a.pth"}, tmp_path, 0.03)
    adv = t[t["Model"] == "adv"].iloc[0]
    assert adv["Clean Accuracy"] == NOT_RUN and adv["PGD Accuracy"] == NOT_RUN
    assert t[t["Model"] == "baseline"].iloc[0]["Training Time (s)"] == NOT_RUN  # không có JSON huấn luyện -> không bịa


def test_defense_table_blur_scenarios_and_training_json(tmp_path):
    rows = _rows("baseline", {"fgsm": 0.1, "pgd": 0.0}) + _rows("baseline+gaussian_blur_k3_s1", {"fgsm": 0.3, "pgd": 0.25}, source="baseline", defense="gaussian_blur_k3_s1", clean=0.6)
    rows += _rows("baseline+gaussian_blur_k3_s1", {"fgsm": 0.0, "pgd": 0.0}, source="baseline+gaussian_blur_k3_s1", defense="gaussian_blur_k3_s1", clean=0.6)
    (tmp_path / "b_clean.json").write_text(json.dumps({"train_time_sec": 12.5, "training": "baseline_clean_from_scratch"}))
    t = defense_table(pd.DataFrame(rows), {"baseline": tmp_path / "b.pth"}, tmp_path, 0.03)
    assert t[t["Model"] == "baseline"].iloc[0]["Training Time (s)"] == 12.5
    tr = t[t["Scenario"].str.startswith("transfer")].iloc[0]
    ad = t[t["Scenario"] == "adaptive"].iloc[0]
    assert tr["PGD Accuracy"] == 0.25 and ad["PGD Accuracy"] == 0.0
    assert tr["Attack source model"] == "baseline" and ad["Attack source model"] == "baseline+gaussian_blur_k3_s1"


def test_seed_summary_mean_std():
    rows = []
    for name, f, p in [("baseline", 0.10, 0.00), ("baseline_seed43", 0.20, 0.02), ("baseline_seed44", 0.30, 0.04)]:
        rows += _rows(name, {"fgsm": f, "pgd": p})
    s = seed_summary(pd.DataFrame(rows), 0.03)
    r = s.iloc[0]
    assert r["N seeds"] == 3 and r["fgsm mean"] == pytest.approx(0.2) and r["fgsm std"] == pytest.approx(0.1)
    assert seed_summary(pd.DataFrame(_rows("baseline", {"fgsm": 0.1, "pgd": 0.0})), 0.03) is None  # một seed: không tính std


def test_markdown_percent_and_nan():
    md = to_markdown(pd.DataFrame({"A": ["x"], "Accuracy": [0.12345], "ASR": [float("nan")]}), pct_cols=("Accuracy", "ASR"))
    assert "12.35%" in md and "N/A" in md


def test_build_reports_end_to_end_and_manifest(tmp_path, tiny_model):
    from adversarial_lab.utils.checkpoint import save_checkpoint

    cfg = load_config()
    res = tmp_path / "res"
    (res / "tables").mkdir(parents=True)
    rows = _rows("baseline", {"fgsm": 0.1, "pgd": 0.0}) + _rows("adv", {"fgsm": 0.5, "pgd": 0.4}, clean=0.8)
    pd.DataFrame(rows)[TABLE_COLS].to_csv(res / "tables" / "attack_results.csv", index=False)
    save_checkpoint(tmp_path / "baseline.pth", tiny_model, cfg, {"training": "baseline_clean_from_scratch", "best_epoch": 1})
    out = build_reports(cfg, res, {"baseline": str(tmp_path / "baseline.pth"), "adv": str(tmp_path / "missing.pth")})
    man = json.loads(out["manifest"].read_text(encoding="utf-8"))
    assert man["seed"] == cfg["seed"] and man["checkpoints"]["adv"]["status"] == NOT_RUN
    assert "sha256" in man["checkpoints"]["baseline"] and man["evaluation"]["n_samples_evaluated"] == [100]
    assert (res / "figures" / "final_comparison_bars.png").exists() and (res / "figures" / "tradeoff_clean_vs_pgd.png").exists()
    assert (res / "figures" / "comparison_models_accuracy_vs_epsilon.png").exists()
