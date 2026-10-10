from .base import Attack, project_linf
from .fgsm import FGSM
from .pgd import PGD

__all__ = ["Attack", "FGSM", "PGD", "project_linf", "make_attack"]


def make_attack(name: str, model, epsilon: float, **kw) -> Attack:
    """Tạo attack theo tên ('fgsm' | 'pgd'). Tham số PGD: steps, alpha, random_start."""
    name = name.lower()
    if name == "fgsm":
        kw = {k: v for k, v in kw.items() if k in ("targeted", "amp", "loss_scale")}
        return FGSM(model, epsilon, **kw)
    if name == "pgd":
        return PGD(model, epsilon, **kw)
    raise ValueError(f"Attack không hỗ trợ: {name!r} (fgsm/pgd)")
