# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import os
from typing import List, Dict, Any

import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_losses_csv(out_csv: str, train_losses: List[float], test_losses: List[float]) -> None:
    ensure_dir(os.path.dirname(out_csv) or ".")
    with open(out_csv, mode="w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "test_loss"])
        for i, (tr, te) in enumerate(zip(train_losses, test_losses), start=1):
            w.writerow([i, tr, te])


def save_metrics_excel(out_xlsx: str, metrics_per_epoch: List[Dict[str, Any]]) -> None:
    """
    metrics_per_epoch: list of dict, each dict contains keys like BACC/ACC/AUC/...
    """
    ensure_dir(os.path.dirname(out_xlsx) or ".")
    df = pd.DataFrame(metrics_per_epoch)
    df.index = df.index + 1
    df.index.name = "epoch"
    df.to_excel(out_xlsx, index=True)
