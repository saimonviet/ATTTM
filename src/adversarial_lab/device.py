"""Chọn thiết bị tính toán."""
from __future__ import annotations

import torch


def get_device(choice: str = "auto") -> torch.device:
    """auto -> cuda nếu có, ngược lại cpu. 'cuda' mà không có GPU sẽ báo lỗi rõ ràng."""
    choice = (choice or "auto").lower()
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Yêu cầu --device cuda nhưng PyTorch không thấy GPU. "
            "Cài bản torch CUDA hoặc dùng --device cpu/auto."
        )
    if choice not in ("cpu", "cuda"):
        raise ValueError(f"device không hợp lệ: {choice!r} (auto/cpu/cuda)")
    return torch.device(choice)


def device_info(device: torch.device) -> dict:
    info = {"device": str(device), "cuda_available": torch.cuda.is_available()}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        info.update(gpu_name=p.name, vram_gb=round(p.total_memory / 1024**3, 2))
    return info
