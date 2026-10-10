"""Kiểm thử headless giao diện Streamlit (AppTest): không lỗi, không giả lập kết quả khi thiếu checkpoint."""
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")


def test_app_without_checkpoint_shows_instructions_not_results(tmp_path, monkeypatch):
    monkeypatch.setenv("ADV_LAB_CKPT_DIR", str(tmp_path))
    at = st_testing.AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Chưa có checkpoint" in e.value for e in at.error)
    assert len(at.metric) == 0  # không hiển thị số liệu bịa


@pytest.mark.skipif(not (ROOT / "data" / "cifar-10-batches-py").exists(), reason="cần dữ liệu CIFAR-10 đã tải")
def test_app_with_untrained_checkpoint_all_modes(tmp_path, monkeypatch):
    from adversarial_lab.config import load_config
    from adversarial_lab.models.resnet import build_from_config
    from adversarial_lab.utils.checkpoint import save_checkpoint

    cfg = load_config()
    save_checkpoint(tmp_path / "baseline_best.pth", build_from_config(cfg), cfg, {"training": "baseline_clean_from_scratch"})
    save_checkpoint(tmp_path / "advtrain_best.pth", build_from_config(cfg), cfg, {"training": "adversarial_pgd_from_scratch"})
    monkeypatch.setenv("ADV_LAB_CKPT_DIR", str(tmp_path))
    at = st_testing.AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception and len(at.metric) == 4
    for option in ("Gaussian Blur", "Model adversarially trained"):
        next(s for s in at.sidebar.selectbox if s.label == "Defense").select(option)
        at.run()
        assert not at.exception and len(at.metric) == 8  # hai khối kết quả
    next(s for s in at.sidebar.selectbox if s.label == "Attack").select("None")
    at.run()
    assert not at.exception and at.metric[0].value == "0.0000"
