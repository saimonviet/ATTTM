"""Smoke test end-to-end nhỏ trên CPU: dữ liệu thật -> model -> FGSM -> PGD -> metric -> hình -> checkpoint.

Không huấn luyện (model khởi tạo ngẫu nhiên): chỉ kiểm tra các thành phần ghép nối đúng, không đánh giá chất lượng.
"""
from pathlib import Path

import pandas as pd
import pytest
import torch

from adversarial_lab.attacks import FGSM, PGD
from adversarial_lab.config import load_config, override
from adversarial_lab.data.cifar10 import CLASS_NAMES, get_test_loader
from adversarial_lab.evaluation.evaluate_attacks import save_rows, sweep_attack
from adversarial_lab.evaluation.evaluate_clean import evaluate_clean
from adversarial_lab.models.resnet import build_from_config
from adversarial_lab.utils.checkpoint import load_checkpoint, save_checkpoint
from adversarial_lab.utils.metrics import attack_success_rate, perturbation_norms
from adversarial_lab.visualization.plot_results import plot_attack_curves
from adversarial_lab.visualization.visualize_attacks import plot_attack_examples

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "data" / "cifar-10-batches-py").exists(), reason="cần dữ liệu CIFAR-10 đã tải")
CPU = torch.device("cpu")


def test_end_to_end_smoke(tmp_path):
    torch.manual_seed(0)
    cfg = override(load_config(), {"num_workers": 0})
    loader = get_test_loader(cfg, max_samples=8, batch_size=8)
    x, y = next(iter(loader))
    assert x.shape == (8, 3, 32, 32) and 0 <= x.min() and x.max() <= 1

    # tạo model, lưu và đọc lại checkpoint
    model = build_from_config(cfg).eval()
    ckpt = tmp_path / "m.pth"
    save_checkpoint(ckpt, model, cfg, {"training": "smoke"})
    model2, meta = load_checkpoint(ckpt, CPU)
    assert meta["class_names"] == CLASS_NAMES
    with torch.no_grad():
        assert torch.allclose(model(x), model2(x), atol=1e-5)

    # FGSM + PGD thỏa ràng buộc L_inf trên model ResNet18 thật
    xf = FGSM(model2, 0.03)(x, y)
    xp = PGD(model2, 0.03, steps=2, alpha=0.015)(x, y)
    for xa in (xf, xp):
        assert xa.min() >= 0 and xa.max() <= 1 and (xa - x).abs().max() <= 0.03 + 1e-6
    linf, l2 = perturbation_norms(x, xp)
    assert linf.max() <= 0.03 + 1e-6 and (l2 > 0).all()
    with torch.no_grad():
        c, a = model2(x).argmax(1) == y, model2(xp).argmax(1) == y
    asr = attack_success_rate(c, a)
    assert asr != asr or 0.0 <= asr <= 1.0  # NaN nếu không có mẫu đúng ban đầu

    # đánh giá clean + attack, lưu bảng và hình
    clean = evaluate_clean(model2, loader, CPU, CLASS_NAMES)
    assert clean["n_samples"] == 8 and len(clean["confusion_matrix"]) == 10
    rows = sweep_attack(model2, model2, loader, CPU, "smoke", "smoke", "fgsm", [0.0, 0.03], n_examples=4)
    rows += sweep_attack(model2, model2, loader, CPU, "smoke", "smoke", "pgd", [0.03], steps=2, n_examples=4)
    df = save_rows(rows, tmp_path / "attack_results.csv", tmp_path / "examples")
    assert len(df) == 3 and pd.read_csv(tmp_path / "attack_results.csv")["n_samples"].eq(8).all()
    ex = next(r["_examples"] for r in rows if r["attack"] == "pgd")
    fig1 = plot_attack_examples(ex, CLASS_NAMES, tmp_path / "figures" / "ex.png", "smoke")
    fig2 = plot_attack_curves(df, tmp_path / "figures" / "curves.png", "smoke", "accuracy", "eval_model")
    assert fig1.stat().st_size > 1000 and fig2.stat().st_size > 1000
    assert list((tmp_path / "examples").glob("*.npz"))
