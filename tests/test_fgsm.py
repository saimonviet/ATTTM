import copy

import pytest
import torch
import torch.nn.functional as F

from adversarial_lab.attacks import FGSM


def _loss(model, x, y):
    with torch.no_grad():
        return F.cross_entropy(model(x), y).item()


def test_eps_zero_identity(tiny_model, batch):
    x, y = batch
    assert torch.equal(FGSM(tiny_model, 0.0)(x, y), x)


@pytest.mark.parametrize("eps", [0.01, 0.03, 0.1, 0.5])
def test_range_and_linf_bound(tiny_model, batch, eps):
    x, y = batch
    xa = FGSM(tiny_model, eps)(x, y)
    assert xa.min() >= 0.0 and xa.max() <= 1.0
    assert (xa - x).abs().max() <= eps + 1e-6


def test_update_equals_eps_times_gradient_sign(tiny_model, batch):
    """Không bị clamp (x trong [0.2,0.8], eps nhỏ): x_adv - x == eps * sign(grad)."""
    x, y = batch
    eps = 0.05
    atk = FGSM(tiny_model, eps)
    g = atk.loss_gradient(x, y)
    xr = x.clone().requires_grad_(True)
    F.cross_entropy(tiny_model(xr), y, reduction="sum").backward()
    assert torch.allclose(g, xr.grad)
    delta = FGSM(tiny_model, eps)(x, y) - x
    assert torch.allclose(delta, eps * g.sign(), atol=1e-6)


def test_loss_increases_untargeted_and_decreases_targeted(tiny_model, batch):
    x, y = batch
    assert _loss(tiny_model, FGSM(tiny_model, 0.03)(x, y), y) > _loss(tiny_model, x, y)
    tgt = (y + 1) % 10
    xt = FGSM(tiny_model, 0.03, targeted=True)(x, tgt)
    assert _loss(tiny_model, xt, tgt) < _loss(tiny_model, x, tgt)


def test_model_parameters_unchanged(tiny_model, batch):
    x, y = batch
    before = copy.deepcopy(tiny_model.state_dict())
    FGSM(tiny_model, 0.03)(x, y)
    after = tiny_model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert all(p.grad is None for p in tiny_model.parameters())
    assert not tiny_model.training  # không đổi chế độ model


def test_epsilon_is_in_pixel_space_with_normalization(tiny_model, batch):
    """Model có Normalize bên trong: perturbation đo trên pixel [0,1] vẫn đúng bằng eps."""
    x, y = batch
    eps = 0.03
    d = (FGSM(tiny_model, eps)(x, y) - x).abs()
    assert d.max() <= eps + 1e-6 and d.max() > 0.9 * eps


def test_negative_epsilon_rejected(tiny_model):
    with pytest.raises(ValueError):
        FGSM(tiny_model, -0.1)
