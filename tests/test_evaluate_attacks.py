import math

import pandas as pd
import torch

from adversarial_lab.evaluation.evaluate_attacks import save_rows, sweep_attack

CPU = torch.device("cpu")


def _loader(batch):
    x, y = batch
    return [(x[:4], y[:4]), (x[4:], y[4:])]


def test_whitebox_eps_zero_matches_clean(tiny_model, batch):
    (r,) = sweep_attack(tiny_model, tiny_model, _loader(batch), CPU, "m", "m", "fgsm", [0.0])
    assert r["accuracy"] == r["clean_accuracy"] and r["mean_linf"] == 0.0
    assert r["n_samples"] == 8 and (r["asr"] == 0.0 or math.isnan(r["asr"]))


def test_metrics_consistent_and_bounded(tiny_model, batch):
    (r,) = sweep_attack(tiny_model, tiny_model, _loader(batch), CPU, "m", "m", "pgd", [0.03], steps=5)
    assert r["mean_linf"] <= 0.03 + 1e-6 and r["mean_l2"] > 0
    assert 0.0 <= r["accuracy"] <= 1.0
    # accuracy sau attack không thể vượt (số mẫu đúng ban đầu bị giữ lại + mẫu sai ban đầu bị lật thành đúng)/N
    if r["n_clean_correct"]:
        kept = (1 - r["asr"]) * r["n_clean_correct"]
        assert r["accuracy"] * r["n_samples"] >= kept - 1e-6


def test_save_rows_mixed_fgsm_pgd_roundtrip_dedup(tiny_model, batch, tmp_path):
    """Hồi quy: steps/random_start đọc lại từ CSV có kiểu khác (10 vs 10.0, True vs 'True') vẫn khớp khóa."""
    csv = tmp_path / "t.csv"
    f = sweep_attack(tiny_model, tiny_model, _loader(batch), CPU, "m", "m", "fgsm", [0.03])
    p = sweep_attack(tiny_model, tiny_model, _loader(batch), CPU, "m", "m", "pgd", [0.03], steps=10)
    save_rows(f + p, csv)
    p2 = sweep_attack(tiny_model, tiny_model, _loader(batch), CPU, "m", "m", "pgd", [0.03], steps=10)
    df = save_rows(p2, csv)
    assert len(df) == 2 and sorted(df["attack"]) == ["fgsm", "pgd"]


def test_save_rows_replaces_same_key(tiny_model, batch, tmp_path):
    csv = tmp_path / "t.csv"
    rows = sweep_attack(tiny_model, tiny_model, _loader(batch), CPU, "m", "m", "fgsm", [0.01, 0.03])
    save_rows(rows, csv)
    rows2 = sweep_attack(tiny_model, tiny_model, _loader(batch), CPU, "m", "m", "fgsm", [0.03])
    df = save_rows(rows2, csv)
    assert len(df) == 2 and len(pd.read_csv(csv)) == 2
