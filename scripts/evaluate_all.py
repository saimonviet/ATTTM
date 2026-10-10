"""Đánh giá tổng hợp baseline / adversarial training / Gaussian Blur trên cùng test set, sinh bảng, hình, manifest.

Ví dụ:
    python scripts/evaluate_all.py --config configs/config.yaml
    python scripts/evaluate_all.py --reports-only          # chỉ dựng lại bảng/hình từ kết quả đã có
    python scripts/evaluate_all.py --force                 # chạy lại mọi đánh giá
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial_lab.evaluation.evaluate_all import evaluate_all
from adversarial_lab.utils.cli import add_common_args, load_cfg


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--baseline-checkpoint", default=None)
    p.add_argument("--adv-checkpoint", default=None)
    p.add_argument("--force", action="store_true", help="chạy lại cả những kết quả đã có")
    p.add_argument("--reports-only", action="store_true")
    p.add_argument("--no-blur", action="store_true")
    p.add_argument("--no-adaptive", action="store_true")
    a = p.parse_args()
    cfg = load_cfg(a)
    models = dict(cfg["models"])
    names = list(models)
    if a.baseline_checkpoint:
        models[names[0]] = a.baseline_checkpoint
    if a.adv_checkpoint:
        models[names[1]] = a.adv_checkpoint
    evaluate_all(cfg, models, a.max_samples, a.force, a.reports_only, not a.no_blur, not a.no_adaptive)


if __name__ == "__main__":
    main()
