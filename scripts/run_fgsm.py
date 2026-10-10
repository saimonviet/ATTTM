"""Đánh giá FGSM (white-box) trên một checkpoint.

Ví dụ:
    python scripts/run_fgsm.py --checkpoint checkpoints/baseline_best.pth --epsilon 0.03
    python scripts/run_fgsm.py --checkpoint checkpoints/baseline_best.pth          # epsilon lấy từ config
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial_lab.evaluation.run_attack import run_attack
from adversarial_lab.utils.cli import add_common_args, load_cfg


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--epsilon", type=float, nargs="+", default=None, help="một hoặc nhiều epsilon (pixel [0,1])")
    p.add_argument("--model-name", default=None)
    a = p.parse_args()
    cfg = load_cfg(a)
    run_attack(cfg, "fgsm", a.epsilon or cfg["attacks"]["fgsm"]["epsilons"],
               a.checkpoint or cfg["train"]["checkpoint"], max_samples=a.max_samples, model_name=a.model_name)


if __name__ == "__main__":
    main()
