"""Logging: console logger và CSV history (có thể thêm TensorBoard)."""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path


def get_logger(name: str = "adversarial_lab") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        try:  # console Windows (cp1252) làm hỏng tiếng Việt
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


class HistoryLogger:
    """Ghi mỗi epoch một dòng vào CSV và (tùy chọn) TensorBoard."""

    def __init__(self, log_dir: str | Path, tensorboard: bool = True, append: bool = False):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.dir / "history.csv"
        self._header: list[str] | None = None
        if append and self.csv_path.exists():
            with open(self.csv_path, newline="", encoding="utf-8") as f:
                self._header = next(csv.reader(f), None)
        elif self.csv_path.exists():
            self.csv_path.unlink()
        self.tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(str(self.dir))
            except Exception:  # tensorboard không bắt buộc
                self.tb = None

    def log(self, epoch: int, row: dict) -> None:
        row = {"epoch": epoch, **row}
        if self._header is None:
            self._header = list(row)
            write_header = True
        else:
            write_header = False
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._header, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow(row)
        if self.tb:
            for k, v in row.items():
                if k != "epoch" and isinstance(v, (int, float)):
                    self.tb.add_scalar(k, v, epoch)

    def close(self) -> None:
        if self.tb:
            self.tb.close()
