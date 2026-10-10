"""Đánh giá PGD L_inf (white-box) trên một checkpoint.

Ví dụ:
    python scripts/run_pgd.py --checkpoint checkpoints/baseline_best.pth --epsilon 0.03 --steps 10
    python scripts/run_pgd.py --checkpoint checkpoints/baseline_best.pth --alpha-ratio 0.25 --no-random-start
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial_lab.evaluation.run_attack import run_attack
from adversarial_lab.utils.cli import add_common_args, load_cfg


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--epsilon", type=float, nargs="+", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--alpha-ratio", type=float, default=None, help="alpha = epsilon * alpha_ratio")
    p.add_argument("--no-random-start", action="store_true")
    p.add_argument("--model-name", default=None)
    a = p.parse_args()
    cfg = load_cfg(a)
    run_attack(cfg, "pgd", a.epsilon or cfg["attacks"]["pgd"]["epsilons"], a.checkpoint or cfg["train"]["checkpoint"],
               steps=a.steps, alpha_ratio=a.alpha_ratio, random_start=False if a.no_random_start else None,
               max_samples=a.max_samples, model_name=a.model_name)


if __name__ == "__main__":
    main()
