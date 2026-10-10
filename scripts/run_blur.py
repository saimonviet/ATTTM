"""Đánh giá Gaussian Blur defense (không train lại model): kịch bản transfer và adaptive.

Ví dụ:
    python scripts/run_blur.py --checkpoint checkpoints/baseline_best.pth --model-name baseline
    python scripts/run_blur.py --checkpoint checkpoints/baseline_best.pth --kernel-sizes 3 5 --sigmas 0.5 1.0 --no-adaptive
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial_lab.evaluation.run_blur import run_blur_defense
from adversarial_lab.utils.cli import add_common_args, load_cfg


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--kernel-sizes", type=int, nargs="+", default=None)
    p.add_argument("--sigmas", type=float, nargs="+", default=None)
    p.add_argument("--no-adaptive", action="store_true", help="bỏ qua attack thích nghi (nhanh hơn)")
    p.add_argument("--model-name", default=None)
    a = p.parse_args()
    cfg = load_cfg(a)
    run_blur_defense(cfg, a.checkpoint or cfg["train"]["checkpoint"], a.kernel_sizes, a.sigmas, a.max_samples,
                     a.model_name, adaptive=not a.no_adaptive)


if __name__ == "__main__":
    main()
