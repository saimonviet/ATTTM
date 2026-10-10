"""Vòng lặp huấn luyện dùng chung cho baseline và adversarial training.

Hai loại training chỉ khác nhau ở `train_step`/`val_step`:
    step(model, x, y) -> (loss: Tensor, stats: dict[str, float], n: int)
stats là các *tổng* (vd {"acc": số mẫu đúng}); engine chia cho số mẫu để ra trung bình.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.checkpoint import save_checkpoint
from ..utils.logging import HistoryLogger, get_logger

Step = Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], tuple[torch.Tensor, dict, int]]


def build_optimizer(model: torch.nn.Module, tcfg: dict) -> torch.optim.Optimizer:
    name = tcfg["optimizer"].lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=tcfg["lr"], momentum=0.9, weight_decay=tcfg["weight_decay"], nesterov=True)
    raise ValueError(f"optimizer không hợp lệ: {name!r} (adamw/sgd)")


def build_scheduler(opt, tcfg: dict, epochs: int):
    if tcfg.get("scheduler", "none") == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    return None


def _run_epoch(model, loader, step: Step, device, train: bool, opt=None, scaler=None, amp=False, desc=""):
    model.train(train)
    loss_sum, n_total, sums = 0.0, 0, {}
    bar = tqdm(loader, desc=desc, leave=False)
    for x, y in bar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if train:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss, stats, n = step(model, x, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss, stats, n = step(model, x, y)
        loss_sum += loss.item() * n
        n_total += n
        for k, v in stats.items():
            sums[k] = sums.get(k, 0.0) + v
        bar.set_postfix(loss=loss_sum / n_total)
    out = {"loss": loss_sum / max(n_total, 1)}
    out.update({k: v / max(n_total, 1) for k, v in sums.items()})
    return out


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_step: Step,
    val_step: Step,
    cfg: dict,
    tcfg: dict,
    epochs: int,
    device: torch.device,
    ckpt_path: str | Path,
    log_dir: str | Path,
    select_key: str = "val_acc",
    extra_meta: dict | None = None,
    resume: bool = False,
) -> dict:
    """Huấn luyện, lưu best checkpoint (theo select_key, càng lớn càng tốt) và last checkpoint để resume."""
    log = get_logger()
    ckpt_path = Path(ckpt_path)
    last_path = ckpt_path.with_name(ckpt_path.stem + "_last.pth")
    model.to(device)
    opt = build_optimizer(model, tcfg)
    sched = build_scheduler(opt, tcfg, epochs)
    amp = bool(tcfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch, best, history, elapsed_before, bad = 0, -float("inf"), [], 0.0, 0
    if resume and last_path.exists():
        st = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        if sched and st.get("sched"):
            sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        start_epoch, best, elapsed_before = st["epoch"], st["best"], st["elapsed"]
        history, bad = st["history"], st["bad"]
        log.info(f"Resume từ epoch {start_epoch} ({last_path})")
    elif resume:
        log.warning(f"--resume nhưng không có {last_path}; train từ đầu.")

    hist_logger = HistoryLogger(log_dir, append=resume and start_epoch > 0)
    t0 = time.time()
    for epoch in range(start_epoch + 1, epochs + 1):
        te = time.time()
        tr = _run_epoch(model, train_loader, train_step, device, True, opt, scaler, amp, f"train {epoch}/{epochs}")
        va = _run_epoch(model, val_loader, val_step, device, False, amp=False, desc="val")
        if sched:
            sched.step()
        row = {f"train_{k}": v for k, v in tr.items()}
        row.update({f"val_{k}": v for k, v in va.items()})
        row["lr"] = opt.param_groups[0]["lr"]
        row["epoch_time_s"] = time.time() - te
        history.append({"epoch": epoch, **row})
        hist_logger.log(epoch, row)
        log.info(" | ".join(f"{k}={v:.4f}" for k, v in row.items()))

        score = row[select_key]
        if score > best:
            best, bad = score, 0
            elapsed = elapsed_before + time.time() - t0
            save_checkpoint(ckpt_path, model, cfg, {
                "best_epoch": epoch, "best_" + select_key: best, "select_key": select_key,
                "train_time_sec_until_best": elapsed, "train_config": tcfg, **(extra_meta or {}),
            })
        else:
            bad += 1
        torch.save({
            "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict() if sched else None,
            "scaler": scaler.state_dict(), "epoch": epoch, "best": best,
            "elapsed": elapsed_before + time.time() - t0, "history": history, "bad": bad,
        }, last_path)
        pat = tcfg.get("early_stop_patience", 0)
        if pat and bad >= pat:
            log.info(f"Early stopping sau {bad} epoch không cải thiện.")
            break
    hist_logger.close()
    total = elapsed_before + time.time() - t0
    return {"history": history, "best": best, "train_time_sec": total, "epochs_run": len(history), "ckpt_path": str(ckpt_path)}
