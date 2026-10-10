import torch

from adversarial_lab.config import load_config
from adversarial_lab.data.cifar10 import CLASS_NAMES, get_datasets
from adversarial_lab.models.resnet import build_from_config
from adversarial_lab.utils.checkpoint import load_checkpoint, save_checkpoint


def test_split_disjoint_and_ranges():
    train, val, test = get_datasets("data", val_size=5000, seed=42, max_samples=None)
    assert len(train) == 45000 and len(val) == 5000 and len(test) == 10000
    assert set(train.indices).isdisjoint(val.indices)
    x, y = test[0]
    assert x.shape == (3, 32, 32) and 0.0 <= x.min() and x.max() <= 1.0
    assert 0 <= y < len(CLASS_NAMES)


def test_split_reproducible():
    a = get_datasets("data", 5000, 42)[1].indices
    b = get_datasets("data", 5000, 42)[1].indices
    assert a == b


def test_model_forward_and_checkpoint_roundtrip(tmp_path):
    cfg = load_config()
    model = build_from_config(cfg).eval()
    x = torch.rand(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)
    p = tmp_path / "m.pth"
    save_checkpoint(p, model, cfg, {"note": "test"})
    m2, meta = load_checkpoint(p)
    assert meta["class_names"] == CLASS_NAMES and meta["mean"] == cfg["data"]["mean"]
    assert torch.allclose(m2(x), out, atol=1e-6)
