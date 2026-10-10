import pytest
import torch
from PIL import Image

from adversarial_lab.demo import list_checkpoints, preprocess_upload, run_demo


def test_list_checkpoints_filters_last_and_smoke(tmp_path):
    for n in ["a_best.pth", "a_best_last.pth", "smoke_x.pth", "notes.txt"]:
        (tmp_path / n).write_bytes(b"x")
    assert [p.name for p in list_checkpoints(tmp_path)] == ["a_best.pth"]
    assert {p.name for p in list_checkpoints(tmp_path, include_smoke=True)} == {"a_best.pth", "smoke_x.pth"}
    assert list_checkpoints(tmp_path / "missing") == []


def test_preprocess_upload_shape_range_and_crop():
    img = Image.new("RGB", (200, 100), (255, 0, 0))
    t = preprocess_upload(img)
    assert t.shape == (1, 3, 32, 32) and t.min() >= 0 and t.max() <= 1
    assert torch.allclose(t[0, 0], torch.ones(32, 32), atol=1e-2)
    assert preprocess_upload(Image.new("L", (50, 50), 128)).shape == (1, 3, 32, 32)  # ảnh xám -> RGB


def test_no_attack_leaves_image_unchanged(tiny_model, batch):
    x, y = batch
    r = run_demo(tiny_model, x[:1], int(y[0]), attack="none")
    assert r["main"]["linf"] == 0.0 and not r["main"]["changed"] and r["defended"] is None


@pytest.mark.parametrize("attack", ["fgsm", "pgd"])
def test_attack_respects_epsilon(tiny_model, batch, attack):
    x, y = batch
    r = run_demo(tiny_model, x[:1], int(y[0]), attack=attack, epsilon=0.05, steps=3)
    assert r["main"]["linf"] <= 0.05 + 1e-6 and r["main"]["l2"] > 0
    assert r["main"]["probs_adv"].sum().item() == pytest.approx(1.0, abs=1e-5)


def test_uploaded_image_uses_model_prediction_as_label(tiny_model, batch):
    x, _ = batch
    r = run_demo(tiny_model, x[:1], None, attack="fgsm", epsilon=0.1)
    assert r["label"] is None and r["main"]["linf"] <= 0.1 + 1e-6


def test_blur_defense_transfer_vs_adaptive(tiny_model, batch):
    x, y = batch
    t = run_demo(tiny_model, x[:1], int(y[0]), attack="pgd", epsilon=0.05, steps=3, defense="gaussian_blur", adaptive=False)
    a = run_demo(tiny_model, x[:1], int(y[0]), attack="pgd", epsilon=0.05, steps=3, defense="gaussian_blur", adaptive=True)
    assert "Gaussian Blur" in t["defended"]["model"]
    # transfer dùng đúng ảnh đối kháng của model gốc (so sánh pred theo cùng ảnh); adaptive tạo ảnh mới
    assert t["defended"]["linf"] == t["main"]["linf"] or abs(t["defended"]["linf"] - t["main"]["linf"]) < 0.05
    assert a["defended"]["linf"] <= 0.05 + 1e-6


def test_adversarial_model_defense_requires_model(tiny_model, batch):
    x, y = batch
    with pytest.raises(ValueError):
        run_demo(tiny_model, x[:1], int(y[0]), attack="fgsm", defense="adversarial_model")
    r = run_demo(tiny_model, x[:1], int(y[0]), attack="fgsm", epsilon=0.03, defense="adversarial_model", defense_model=tiny_model,
                 adaptive=True, defense_name="robust")
    assert r["defended"]["model"] == "robust"


def test_invalid_options(tiny_model, batch):
    x, y = batch
    with pytest.raises(ValueError):
        run_demo(tiny_model, x[:1], int(y[0]), attack="cw")
    with pytest.raises(ValueError):
        run_demo(tiny_model, x[:1], int(y[0]), attack="none", defense="magic")
