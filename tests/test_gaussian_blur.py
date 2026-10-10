import pytest
import torch

from adversarial_lab.attacks import FGSM
from adversarial_lab.defenses.gaussian_blur import BlurDefendedModel, GaussianBlur, gaussian_kernel_1d


def test_kernel_normalized_and_symmetric():
    k = gaussian_kernel_1d(5, 1.0)
    assert k.sum().item() == pytest.approx(1.0) and torch.allclose(k, k.flip(0)) and k.argmax() == 2


@pytest.mark.parametrize("ks,sigma", [(0, 1.0), (4, 1.0), (3, 0.0)])
def test_invalid_args(ks, sigma):
    with pytest.raises(ValueError):
        gaussian_kernel_1d(ks, sigma)


def test_constant_image_unchanged_and_range(batch):
    x, _ = batch
    blur = GaussianBlur(5, 1.0)
    c = torch.full((2, 3, 32, 32), 0.37)
    assert torch.allclose(blur(c), c, atol=1e-6)
    y = blur(x)
    assert y.shape == x.shape and y.min() >= 0.0 and y.max() <= 1.0


def test_kernel_size_one_is_identity(batch):
    x, _ = batch
    assert torch.equal(GaussianBlur(1, 1.0)(x), x)


def test_blur_reduces_noise_energy():
    torch.manual_seed(0)
    noise = torch.randn(4, 3, 32, 32) * 0.05 + 0.5
    assert GaussianBlur(3, 1.0)(noise).std() < noise.std()


def test_matches_2d_gaussian_reference():
    """Blur tách được == tích chập 2D với kernel ngoại tích (bên trong ảnh, không tính biên)."""
    torch.manual_seed(1)
    x = torch.rand(1, 3, 12, 12)
    k = gaussian_kernel_1d(3, 0.8)
    k2 = torch.outer(k, k).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    ref = torch.nn.functional.conv2d(x, k2, groups=3)
    assert torch.allclose(GaussianBlur(3, 0.8)(x)[..., 1:-1, 1:-1], ref, atol=1e-6)


def test_wrapper_order_blur_then_model(tiny_model, batch):
    x, _ = batch
    d = BlurDefendedModel(tiny_model, 3, 1.0).eval()
    assert torch.allclose(d(x), tiny_model(GaussianBlur(3, 1.0)(x)), atol=1e-6)


def test_gradient_flows_through_blur_for_adaptive_attack(tiny_model, batch):
    x, y = batch
    d = BlurDefendedModel(tiny_model, 3, 1.0).eval()
    g = FGSM(d, 0.03).loss_gradient(x, y)
    assert g.abs().sum() > 0
    xa = FGSM(d, 0.03)(x, y)
    assert (xa - x).abs().max() <= 0.03 + 1e-6


def test_model_parameters_not_modified_by_wrapper(tiny_model):
    before = {k: v.clone() for k, v in tiny_model.state_dict().items()}
    BlurDefendedModel(tiny_model, 5, 1.0)
    assert all(torch.equal(before[k], v) for k, v in tiny_model.state_dict().items())
