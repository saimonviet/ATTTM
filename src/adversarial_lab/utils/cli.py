"""Tham số CLI dùng chung cho các script."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

from ..config import load_config, override, resolve_path

for _s in (sys.stdout, sys.stderr):  # tiếng Việt khi chuyển hướng output ra file trên Windows (cp1252)
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None, help="giới hạn số mẫu mỗi split (smoke test)")
    p.add_argument("--output-dir", default=None, help="thư mục kết quả (mặc định: results)")
    p.add_argument("--checkpoint", default=None)
    return p


def common_overrides(args) -> dict:
    return {
        "device": args.device,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "paths.results": args.output_dir,
        "eval.batch_size": args.batch_size,
    }


def load_cfg(args, extra: dict | None = None) -> dict:
    return override(load_config(args.config), {**common_overrides(args), **(extra or {})})


def results_dir(cfg: dict, sub: str = "") -> Path:
    d = resolve_path(cfg["paths"]["results"]) / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path: str | Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path
