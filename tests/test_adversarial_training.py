import pytest
import torch
import torch.nn.functional as F

from adversarial_lab.attacks import PGD
from adversarial_lab.defenses.adversarial_training import make_adv_train_step, make_adv_val_step


def _factory(eps=0.03):
    return lambda m: PGD(m, eps, steps=3, alpha=eps / 2)


def test_step_returns_valid_loss_and_updates_weights(tiny_model, batch):
    x, y = batch
    tiny_model.train()
    opt = torch.optim.SGD(tiny_model.parameters(), lr=0.1)
    before = [p.detach().clone() for p in tiny_model.parameters()]
    step = make_adv_train_step(_factory(), 1.0, 0.0)
    loss, stats, n = step(tiny_model, x, y)
    assert torch.isfinite(loss) and n == 8 and 0 <= stats["acc"] <= 8
    assert tiny_model.training  # trở lại train mode sau khi sinh attack
    loss.backward()
    opt.step()
    assert any(not torch.equal(b, p) for b, p in zip(before, tiny_model.parameters()))


def test_adv_ratio_zero_equals_clean_loss(tiny_model, batch):
    x, y = batch
    tiny_model.eval()  # loại bỏ nhiễu BN để so sánh
    loss, _, _ = make_adv_train_step(_factory(), 0.0, 0.0)(tiny_model, x, y)
    # step đặt model.train(); tiny_model không có BN nên so sánh trực tiếp
    assert loss.item() == pytest.approx(F.cross_entropy(tiny_model(x), y).item(), rel=1e-5)


def test_clean_weight_mixes_losses(tiny_model, batch):
    x, y = batch
    torch.manual_seed(0)
    adv_only = make_adv_train_step(lambda m: PGD(m, 0.03, steps=3, random_start=False), 1.0, 0.0)(tiny_model, x, y)[0]
    clean = F.cross_entropy(tiny_model(x), y)
    mixed = make_adv_train_step(lambda m: PGD(m, 0.03, steps=3, random_start=False), 1.0, 0.5)(tiny_model, x, y)[0]
    assert mixed.item() == pytest.approx(0.5 * clean.item() + 0.5 * adv_only.item(), rel=1e-4)


def test_invalid_hyperparameters():
    with pytest.raises(ValueError):
        make_adv_train_step(_factory(), 1.5, 0.0)
    with pytest.raises(ValueError):
        make_adv_train_step(_factory(), 1.0, -0.1)


def test_val_step_reports_adv_and_clean(tiny_model, batch):
    x, y = batch
    loss, stats, n = make_adv_val_step(_factory())(tiny_model, x, y)
    assert n == 8 and set(stats) == {"acc", "clean_acc"} and torch.isfinite(loss)
