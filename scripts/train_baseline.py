"""Huấn luyện baseline ResNet18 + đánh giá sạch trên test set.

Ví dụ:
    python scripts/train_baseline.py --config configs/config.yaml
    python scripts/train_baseline.py --max-samples 512 --epochs 2 --checkpoint checkpoints/smoke.pth --output-dir results_smoke
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial_lab.config import resolve_path
from adversarial_lab.data.cifar10 import CLASS_NAMES, get_test_loader
from adversarial_lab.device import device_info, get_device
from adversarial_lab.evaluation.evaluate_clean import evaluate_clean
from adversarial_lab.training.train_baseline import train_baseline
from adversarial_lab.utils.checkpoint import load_checkpoint
from adversarial_lab.utils.cli import add_common_args, load_cfg, results_dir, save_json
from adversarial_lab.utils.logging import get_logger
from adversarial_lab.visualization.plot_results import plot_confusion_matrix, plot_training_curves


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--resume", action="store_true", help="tiếp tục từ <checkpoint>_last.pth")
    p.add_argument("--overwrite", action="store_true", help="cho phép ghi đè checkpoint đã có")
    a = p.parse_args()
    cfg = load_cfg(a, {"train.epochs": a.epochs, "train.lr": a.lr, "train.batch_size": a.batch_size})
    log = get_logger()
    device = get_device(cfg["device"])
    ckpt = resolve_path(a.checkpoint or cfg["train"]["checkpoint"])
    if ckpt.exists() and not (a.resume or a.overwrite):
        sys.exit(f"Checkpoint đã tồn tại: {ckpt}. Dùng --resume, --overwrite hoặc --checkpoint khác.")
    log.info(f"Device: {device_info(device)} | epochs={cfg['train']['epochs']} bs={cfg['train']['batch_size']}")

    log_dir = resolve_path("logs") / ckpt.stem
    res = train_baseline(cfg, device, ckpt, log_dir, a.max_samples, a.resume)

    model, meta = load_checkpoint(ckpt, device)
    test = evaluate_clean(model, get_test_loader(cfg, a.max_samples), device, CLASS_NAMES)
    test.update(train_time_sec=res["train_time_sec"], best_epoch=meta["best_epoch"], best_val_acc=meta["best_val_acc"], checkpoint=str(ckpt))
    out = results_dir(cfg)
    test["training"] = meta["training"]
    save_json(out / "tables" / f"{ckpt.stem}_clean.json", test)
    plot_training_curves(log_dir / "history.csv", out / "figures" / f"{ckpt.stem}_training_curves.png", "Baseline ResNet18")
    plot_confusion_matrix(test["confusion_matrix"], CLASS_NAMES, out / "figures" / f"{ckpt.stem}_confusion_matrix.png", f"{ckpt.stem} - clean test confusion matrix")
    log.info(f"Clean test acc={test['accuracy']:.4f} loss={test['loss']:.4f} train_time={res['train_time_sec']:.0f}s -> {out}")


if __name__ == "__main__":
    main()
