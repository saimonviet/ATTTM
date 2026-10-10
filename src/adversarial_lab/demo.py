"""Logic cho giao diện demo (không phụ thuộc Streamlit, có thể kiểm thử độc lập).

Giao diện chỉ gọi các hàm ở đây; không có thuật toán nào nằm trong app/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .attacks import make_attack
from .defenses.gaussian_blur import BlurDefendedModel
from .utils.metrics import perturbation_norms


def list_checkpoints(ckpt_dir: str | Path, include_smoke: bool = False) -> list[Path]:
    """Các checkpoint model (*.pth), bỏ file '_last' (trạng thái resume) và, mặc định, các file smoke test."""
    d = Path(ckpt_dir)
    if not d.exists():
        return []
    out = [p for p in sorted(d.glob("*.pth")) if not p.stem.endswith("_last")]
    return out if include_smoke else [p for p in out if not p.stem.startswith("smoke_")]


def preprocess_upload(img: Image.Image) -> torch.Tensor:
    """Ảnh tải lên -> tensor (1,3,32,32) trong [0,1]: RGB, cắt vuông ở giữa, resize 32x32.

    Ảnh ngoài CIFAR-10 nằm ngoài phân phối huấn luyện: kết quả không đại diện cho benchmark.
    """
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((32, 32), Image.LANCZOS)
    t = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)


@torch.no_grad()
def predict(model: torch.nn.Module, x: torch.Tensor) -> tuple[int, float, torch.Tensor]:
    """(lớp dự đoán, confidence, vector xác suất) cho ảnh đơn (1,3,32,32)."""
    probs = torch.softmax(model.eval()(x).float(), dim=1)[0]
    conf, pred = probs.max(0)
    return int(pred), float(conf), probs.cpu()


def craft(model, x, y, attack: str, epsilon: float, steps: int = 10, random_start: bool = True) -> torch.Tensor:
    """Sinh ảnh đối kháng trên model (white-box). attack: 'none' | 'fgsm' | 'pgd'."""
    if attack == "none" or epsilon == 0:
        return x.clone()
    if attack == "fgsm":
        return make_attack("fgsm", model, epsilon)(x, y)
    if attack == "pgd":
        return make_attack("pgd", model, epsilon, steps=steps, alpha=epsilon / 4, random_start=random_start)(x, y)
    raise ValueError(f"attack không hợp lệ: {attack!r}")


def summarize(model, x: torch.Tensor, x_adv: torch.Tensor, name: str) -> dict:
    pc, cc, prc = predict(model, x)
    pa, ca, pra = predict(model, x_adv)
    linf, l2 = perturbation_norms(x, x_adv)
    return {"model": name, "x_adv": x_adv.cpu(), "pred_clean": pc, "conf_clean": cc, "probs_clean": prc,
            "pred_adv": pa, "conf_adv": ca, "probs_adv": pra, "linf": float(linf), "l2": float(l2), "changed": pa != pc}


def run_demo(model, x: torch.Tensor, label: int | None, attack: str = "none", epsilon: float = 0.03, steps: int = 10,
             random_start: bool = True, defense: str = "none", defense_model=None, blur_kernel: int = 3, blur_sigma: float = 1.0,
             adaptive: bool = False, model_name: str = "model", defense_name: str = "defense") -> dict:
    """Chạy một lượt demo.

    - Nhãn dùng để tấn công: nhãn thật nếu là ảnh CIFAR-10; với ảnh tải lên (label=None) dùng dự đoán sạch của model.
    - defense: 'none' | 'gaussian_blur' | 'adversarial_model'.
        * gaussian_blur: mặc định đánh giá chính ảnh đối kháng đã tạo trên model gốc qua blur (không thích nghi);
          adaptive=True tạo attack trực tiếp qua blur.
        * adversarial_model: attack tạo trực tiếp trên defense_model (white-box) khi adaptive=True, ngược lại dùng ảnh
          đối kháng từ model gốc (transfer).
    """
    device = next(model.parameters()).device
    x = x.to(device)
    p0, _, _ = predict(model, x)
    target = torch.tensor([label if label is not None else p0], device=device)
    x_adv = craft(model, x, target, attack, epsilon, steps, random_start)
    res = {"label": label, "attack": attack, "epsilon": epsilon, "main": summarize(model, x, x_adv, model_name), "defended": None}
    if defense == "gaussian_blur":
        dm = BlurDefendedModel(model, blur_kernel, blur_sigma).to(device).eval()
        xd = craft(dm, x, target, attack, epsilon, steps, random_start) if adaptive else x_adv
        res["defended"] = summarize(dm, x, xd, f"{model_name} + Gaussian Blur (k={blur_kernel}, sigma={blur_sigma:g})")
    elif defense == "adversarial_model":
        if defense_model is None:
            raise ValueError("defense='adversarial_model' cần defense_model")
        defense_model = defense_model.to(device).eval()
        xd = craft(defense_model, x, target, attack, epsilon, steps, random_start) if adaptive else x_adv
        res["defended"] = summarize(defense_model, x, xd, defense_name)
    elif defense != "none":
        raise ValueError(f"defense không hợp lệ: {defense!r}")
    return res
