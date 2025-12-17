# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse


def get_train_args() -> argparse.Namespace:
    """Parse CLI args for training."""
    p = argparse.ArgumentParser(description="Train peptide sequence classifier (ESM residue embeddings as input).")

    # data
    p.add_argument("--train_pkl", type=str, default="./data/train_embeddings.pkl", help="Train embeddings .pkl")
    p.add_argument("--test_pkl", type=str, default="./data/test_embeddings.pkl", help="Test embeddings .pkl")
    p.add_argument("--save_scaler", type=str, default="./result/minmax_scaler.npz", help="Save min-max scaler")
    p.add_argument("--use_minmax", action="store_true", help="Apply per-dimension min-max using TRAIN tokens only")

    # runtime
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # optimization
    p.add_argument("--epochs", type=int, default=200, help="Training epochs")
    p.add_argument("--batch_size", type=int, default=256, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (AdamW)")
    p.add_argument("--pos_weight", type=float, default=None, help="BCE pos_weight. If None, auto = neg/pos")

    # model
    p.add_argument("--hidden_size", type=int, default=256, help="BiLSTM hidden size")
    p.add_argument("--num_layers", type=int, default=1, help="BiLSTM layers")
    p.add_argument("--dropout", type=float, default=0.2, help="Dropout")

    # checkpoint / logging
    p.add_argument("--result_dir", type=str, default="./result", help="Where to save metrics/loss")
    p.add_argument("--ckpt_dir", type=str, default="./Model_saved", help="Where to save checkpoints")
    p.add_argument("--save_metric", type=str, default="BACC", choices=["BACC", "ACC", "AUC", "MCC", "F1"],
                   help="Which metric to monitor for saving best model")

    return p.parse_args()
