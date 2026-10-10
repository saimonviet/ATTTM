import math

import torch

from adversarial_lab.utils.metrics import accuracy, attack_success_rate, confusion_matrix, perturbation_norms, per_class_accuracy


def test_accuracy():
    assert accuracy(torch.tensor([0, 1, 2, 2]), torch.tensor([0, 1, 1, 2])) == 0.75


def test_asr_definition():
    clean = torch.tensor([True, True, True, False])
    adv = torch.tensor([False, True, False, False])
    # 3 mẫu đúng ban đầu, 2 bị lật -> 2/3 ; mẫu sai ban đầu không tính
    assert math.isclose(attack_success_rate(clean, adv), 2 / 3)


def test_asr_differs_from_accuracy():
    clean = torch.tensor([True, True, False, False])
    adv = torch.tensor([False, True, False, True])
    assert attack_success_rate(clean, adv) == 0.5
    assert adv.float().mean().item() == 0.5  # accuracy trùng giá trị ở ví dụ này nhưng khác định nghĩa
    adv2 = torch.tensor([False, False, True, True])
    assert attack_success_rate(clean, adv2) == 1.0 and adv2.float().mean().item() == 0.5


def test_asr_no_clean_correct_is_nan():
    z = torch.zeros(4, dtype=torch.bool)
    assert math.isnan(attack_success_rate(z, z))


def test_perturbation_norms():
    x = torch.zeros(2, 3, 2, 2)
    xa = x.clone()
    xa[0, 0, 0, 0] = 0.3
    xa[0, 1, 0, 0] = 0.4
    xa[1] += 0.1
    linf, l2 = perturbation_norms(x, xa)
    assert torch.allclose(linf, torch.tensor([0.4, 0.1]))
    assert torch.allclose(l2, torch.tensor([0.5, (12 * 0.01) ** 0.5]), atol=1e-6)


def test_confusion_and_per_class():
    preds = torch.tensor([0, 1, 1, 2])
    labels = torch.tensor([0, 1, 2, 2])
    cm = confusion_matrix(preds, labels, 3)
    assert cm.tolist() == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
    assert per_class_accuracy(cm).tolist() == [1.0, 1.0, 0.5]
