"""Chạy toàn bộ pipeline: train baseline -> adversarial training -> đánh giá (attack, defense, bảng, hình, manifest).

Bước đã có checkpoint sẽ được bỏ qua (dùng --force để train lại và ghi đè).

Ví dụ:
    python scripts/run_pipeline.py --config configs/config.yaml
    python scripts/run_pipeline.py --smoke        # chạy thử rất nhỏ (512 mẫu, 1 epoch) vào results_smoke_pipeline/
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"Bước thất bại (mã {r.returncode}): {' '.join(cmd)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--baseline-checkpoint", default=None)
    p.add_argument("--adv-checkpoint", default=None)
    p.add_argument("--epochs", type=int, default=None, help="ghi đè số epoch cho cả hai lần train")
    p.add_argument("--force", action="store_true", help="train lại và ghi đè checkpoint đã có")
    p.add_argument("--smoke", action="store_true", help="chạy thử nhỏ, checkpoint/kết quả riêng, không đụng kết quả thật")
    a = p.parse_args()

    if a.smoke:
        a.max_samples = a.max_samples or 512
        a.epochs = a.epochs or 1
        a.output_dir = a.output_dir or "results_smoke_pipeline"
        a.baseline_checkpoint = a.baseline_checkpoint or "checkpoints/smoke_pipeline_baseline.pth"
        a.adv_checkpoint = a.adv_checkpoint or "checkpoints/smoke_pipeline_adv.pth"
        a.force = True

    py = sys.executable
    common = ["--config", a.config]
    for flag, val in (("--device", a.device), ("--seed", a.seed), ("--num-workers", a.num_workers),
                      ("--max-samples", a.max_samples), ("--output-dir", a.output_dir)):
        if val is not None:
            common += [flag, str(val)]
    sys.path.insert(0, str(ROOT / "src"))
    from adversarial_lab.config import load_config, resolve_path

    cfg = load_config(a.config)
    base_ck = a.baseline_checkpoint or cfg["models"]["baseline"]
    adv_ck = a.adv_checkpoint or cfg["models"]["advtrain_pgd"]
    t0 = time.time()

    if a.force or not resolve_path(base_ck).exists():
        cmd = [py, "scripts/train_baseline.py", *common, "--checkpoint", base_ck, "--overwrite"]
        run(cmd + (["--epochs", str(a.epochs)] if a.epochs else []))
    else:
        print(f"[bỏ qua] đã có {base_ck}")
    if a.force or not resolve_path(adv_ck).exists():
        cmd = [py, "scripts/train_adversarial.py", *common, "--checkpoint", adv_ck, "--overwrite", "--no-eval"]
        run(cmd + (["--epochs", str(a.epochs)] if a.epochs else []))
    else:
        print(f"[bỏ qua] đã có {adv_ck}")
    run([py, "scripts/evaluate_all.py", *common, "--baseline-checkpoint", base_ck, "--adv-checkpoint", adv_ck])
    print(f"\nHoàn tất pipeline sau {time.time() - t0:.0f}s. Kết quả: {a.output_dir or cfg['paths']['results']}/")


if __name__ == "__main__":
    main()
