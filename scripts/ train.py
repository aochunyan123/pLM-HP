# -*- coding: utf-8 -*-
from __future__ import annotations




import os
import torch
import torch.nn as nn

from src.config import get_train_args
from src.data import build_loaders
from src.model import BiLSTMClassifier
from src.record import ensure_dir, save_losses_csv, save_metrics_excel
from src.trainer import set_seed, fit


def main():
    args = get_train_args()
    set_seed(args.seed)

    ensure_dir(args.result_dir)
    ensure_dir(args.ckpt_dir)

    bundle = build_loaders(
        train_pkl=args.train_pkl,
        test_pkl=args.test_pkl,
        batch_size=args.batch_size,
        use_minmax=args.use_minmax,
        save_scaler=args.save_scaler if args.use_minmax else None,
    )

    print(f"Train set: pos={bundle.pos_count}, neg={bundle.neg_count}, input_dim={bundle.input_dim}")
    if args.pos_weight is None:
        # auto pos_weight = neg/pos
        pos_weight = float(bundle.neg_count / max(bundle.pos_count, 1))
    else:
        pos_weight = float(args.pos_weight)
    print(f"Using pos_weight={pos_weight:.4f}")

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    model = BiLSTMClassifier(
        input_dim=bundle.input_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_losses, test_losses, metrics_hist, best_path, best_epoch, best_value = fit(
        model=model,
        train_loader=bundle.train_loader,
        test_loader=bundle.test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        ckpt_dir=args.ckpt_dir,
        save_metric=args.save_metric,
    )

    print(f"Best checkpoint: {best_path} | epoch={best_epoch} | {args.save_metric}={best_value:.6f}")

    save_losses_csv(os.path.join(args.result_dir, "Loss.csv"), train_losses, test_losses)
    save_metrics_excel(os.path.join(args.result_dir, "Metrics.xlsx"), metrics_hist)
    print("Done. Saved Loss.csv and Metrics.xlsx")


if __name__ == "__main__":
    main()
