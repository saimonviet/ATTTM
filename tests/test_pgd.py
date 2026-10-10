import copy

import pytest
import torch
import torch.nn.functional as F

from adversarial_lab.attacks import PGD, make_attack, project_linf


def _loss(model, x, y):
    with torch.no_grad():
        return F.cross_entropy(model(x), y).item()


@pytest.mark.parametrize("eps,steps", [(0.01, 10), (0.03, 10), (0.1, 5), (0.5, 3)])
def test_range_and_linf_bound(tiny_model, batch, eps, steps):
    x, y = batch
    xa = PGD(tiny_model, eps, steps=steps, random_start=True)(x, y)
    assert xa.min() >= 0.0 and xa.max() <= 1.0
    assert (xa - x).abs().max() <= eps + 1e-6


def test_random_start_within_ball_and_nonzero(tiny_model, batch):
    x, y = batch
    eps = 0.05
    xa = PGD(tiny_model, eps, steps=0, random_start=True)(x, y)
    d = (xa - x).abs()
    assert d.max() <= eps + 1e-6 and d.max() > 0


def test_no_random_start_no_steps_is_identity(tiny_model, batch):
    x, y = batch
    assert torch.equal(PGD(tiny_model, 0.05, steps=0, random_start=False)(x, y), x)


def test_eps_zero_identity(tiny_model, batch):
    x, y = batch
    assert torch.equal(PGD(tiny_model, 0.0, steps=10)(x, y), x)


def test_projection():
    x = torch.full((1, 3, 4, 4), 0.5)
    far = torch.cat([torch.full((1, 3, 4, 2), 5.0), torch.full((1, 3, 4, 2), -5.0)], dim=3)
    p = project_linf(far, x, 0.1)
    assert torch.allclose(p[..., :2], torch.full_like(p[..., :2], 0.6))
    assert torch.allclose(p[..., 2:], torch.full_like(p[..., 2:], 0.4))
    # clamp [0,1] có tác dụng khi x gần biên
    x0 = torch.full((1, 1, 2, 2), 0.02)
    assert project_linf(torch.full_like(x0, -1.0), x0, 0.1).min() == 0.0


def test_loss_increases_and_pgd_not_weaker_than_clean(tiny_model, batch):
    x, y = batch
    xa = PGD(tiny_model, 0.03, steps=10)(x, y)
    assert _loss(tiny_model, xa, y) > _loss(tiny_model, x, y)


def test_targeted_decreases_target_loss(tiny_model, batch):
    x, y = batch
    tgt = (y + 3) % 10
    xa = PGD(tiny_model, 0.05, steps=10, targeted=True)(x, tgt)
    assert _loss(tiny_model, xa, tgt) < _loss(tiny_model, x, tgt)


def test_model_parameters_unchanged(tiny_model, batch):
    x, y = batch
    before = copy.deepcopy(tiny_model.state_dict())
    PGD(tiny_model, 0.03, steps=5)(x, y)
    after = tiny_model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert all(p.grad is None for p in tiny_model.parameters())


def test_default_alpha_and_factory(tiny_model, batch):
    atk = make_attack("pgd", tiny_model, 0.04, steps=3)
    assert atk.alpha == pytest.approx(0.01) and atk.config()["steps"] == 3
    with pytest.raises(ValueError):
        make_attack("cw", tiny_model, 0.1)
