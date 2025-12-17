# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import compute_binary_metrics


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> Tuple[Dict[str, float], float]:
    model.eval()
    all_true, all_prob = [], []
    losses = []

    for padded, lengths, y, _ids in loader:
        padded = padded.to(device)
        lengths = lengths.to(device)
        labels = y.float().unsqueeze(1).to(device)  # [B,1]

        logits = model(padded, lengths)             # [B,1]
        loss = criterion(logits, labels)
        losses.append(float(loss.item()))

        prob = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()  # [B]
        yt = y.detach().cpu().numpy()                                   # [B]

        all_prob.append(prob)
        all_true.append(yt)

    y_true = np.concatenate(all_true, axis=0)
    y_prob = np.concatenate(all_prob, axis=0)

    pack = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=0.5)
    avg_loss = float(np.mean(losses)) if len(losses) else 0.0
    return pack.as_dict, avg_loss


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    losses = []

    for padded, lengths, y, _ids in loader:
        padded = padded.to(device)
        lengths = lengths.to(device)
        labels = y.float().unsqueeze(1).to(device)

        logits = model(padded, lengths)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if len(losses) else 0.0


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epochs: int,
    ckpt_dir: str,
    save_metric: str = "BACC",
) -> Tuple[List[float], List[float], List[Dict[str, float]], str, int, float]:
    os.makedirs(ckpt_dir, exist_ok=True)

    train_losses: List[float] = []
    test_losses: List[float] = []
    metrics_hist: List[Dict[str, float]] = []

    best_value = -1e18
    best_epoch = -1
    best_path = os.path.join(ckpt_dir, "best.pt")

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        met, te_loss = evaluate(model, test_loader, criterion, device)

        train_losses.append(round(tr_loss, 6))
        test_losses.append(round(te_loss, 6))

        met = dict(met)
        met["epoch"] = epoch
        metrics_hist.append(met)

        print(
            f"[Epoch {epoch:03d}] train_loss={tr_loss:.4f} | test_loss={te_loss:.4f} | "
            f"BACC={met['BACC']:.3f} ACC={met['ACC']:.3f} AUC={met['AUC']:.4f} MCC={met['MCC']:.4f}"
        )

        monitor_val = float(met.get(save_metric, -1e18))


        if monitor_val > best_value:
            best_value = monitor_val
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            print(f"  -> Saved BEST checkpoint: {best_path} ({save_metric}={monitor_val:.6f})")


    model.load_state_dict(torch.load(best_path, map_location=device))

    return train_losses, test_losses, metrics_hist, best_path, best_epoch, best_value

