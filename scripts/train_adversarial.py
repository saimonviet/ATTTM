"""Adversarial Training (FGSM/PGD) + đánh giá white-box FGSM/PGD trên model robust.

Ví dụ:
    python scripts/train_adversarial.py --config configs/config.yaml
    python scripts/train_adversarial.py --method fgsm --epsilon 0.03
    python scripts/train_adversarial.py --mode finetune --init-checkpoint checkpoints/baseline_best.pth --epochs 5
    python scripts/train_adversarial.py --epochs 1 --max-samples 512 --checkpoint checkpoints/smoke_at.pth --output-dir results_smoke
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial_lab.config import resolve_path
from adversarial_lab.data.cifar10 import CLASS_NAMES, get_test_loader
from adversarial_lab.device import device_info, get_device
from adversarial_lab.evaluation.evaluate_clean import evaluate_clean
from adversarial_lab.evaluation.run_attack import run_attack
from adversarial_lab.training.train_adversarial import train_adversarial
from adversarial_lab.utils.checkpoint import load_checkpoint
from adversarial_lab.utils.cli import add_common_args, load_cfg, results_dir, save_json
from adversarial_lab.utils.logging import get_logger
from adversarial_lab.visualization.plot_results import plot_confusion_matrix, plot_training_curves


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--method", choices=["pgd", "fgsm"], default=None)
    p.add_argument("--mode", choices=["from_scratch", "finetune"], default=None)
    p.add_argument("--init-checkpoint", default=None, help="checkpoint baseline (chỉ cho --mode finetune)")
    p.add_argument("--epsilon", type=float, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lambda-clean", type=float, default=None, help="lambda: trọng số clean loss (0 = adversarial-only)")
    p.add_argument("--adv-ratio", type=float, default=None, help="tỉ lệ mẫu trong batch bị tấn công")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-eval", action="store_true", help="bỏ qua đánh giá FGSM/PGD sau khi train")
    a = p.parse_args()
    cfg = load_cfg(a, {
        "adv_train.method": a.method, "adv_train.mode": a.mode, "adv_train.epsilon": a.epsilon, "adv_train.steps": a.steps,
        "adv_train.epochs": a.epochs, "adv_train.lr": a.lr, "adv_train.clean_weight": a.lambda_clean,
        "adv_train.adv_ratio": a.adv_ratio, "adv_train.batch_size": a.batch_size,
    })
    log, at = get_logger(), cfg["adv_train"]
    device = get_device(cfg["device"])
    default_ckpt = at["checkpoint"] if at["method"] == "pgd" else f"checkpoints/advtrain_{at['method']}_best.pth"
    ckpt = resolve_path(a.checkpoint or default_ckpt)
    if ckpt.exists() and not (a.resume or a.overwrite):
        sys.exit(f"Checkpoint đã tồn tại: {ckpt}. Dùng --resume, --overwrite hoặc --checkpoint khác.")
    log.info(f"Device: {device_info(device)} | {at['method'].upper()}-AT mode={at['mode']} eps={at['epsilon']} steps={at['steps']} "
             f"lambda={at['clean_weight']} adv_ratio={at['adv_ratio']} epochs={at['epochs']}")
    log_dir = resolve_path("logs") / ckpt.stem
    res = train_adversarial(cfg, device, ckpt, log_dir, a.max_samples, a.resume, a.init_checkpoint)

    model, meta = load_checkpoint(ckpt, device)
    test = evaluate_clean(model, get_test_loader(cfg, a.max_samples), device, CLASS_NAMES)
    test.update(train_time_sec=res["train_time_sec"], best_epoch=meta["best_epoch"], best_val_adv_acc=meta["best_val_acc"],
                training=meta["training"], adv_config=meta["adv_config"], checkpoint=str(ckpt))
    out = results_dir(cfg)
    save_json(out / "tables" / f"{ckpt.stem}_clean.json", test)
    plot_training_curves(log_dir / "history.csv", out / "figures" / f"{ckpt.stem}_training_curves.png",
                         f"{at['method'].upper()} adversarial training", acc_label="adversarial accuracy")
    plot_confusion_matrix(test["confusion_matrix"], CLASS_NAMES, out / "figures" / f"{ckpt.stem}_confusion_matrix.png",
                          f"{ckpt.stem} - clean test confusion matrix")
    log.info(f"Clean test acc={test['accuracy']:.4f} train_time={res['train_time_sec']:.0f}s")
    if not a.no_eval:
        name = ckpt.stem.replace("_best", "")
        run_attack(cfg, "fgsm", cfg["attacks"]["fgsm"]["epsilons"], str(ckpt), max_samples=a.max_samples, model_name=name)
        run_attack(cfg, "pgd", cfg["attacks"]["pgd"]["epsilons"], str(ckpt), max_samples=a.max_samples, model_name=name)


if __name__ == "__main__":
    main()
