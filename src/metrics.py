# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    matthews_corrcoef,
    precision_score,
)


@dataclass
class MetricPack:
    """
    Keep a consistent order (compatible with你原来的 metric 输出含义):
    [BACC(%), ACC(%), AUC, Sn(%), Sp(%), MCC, F1, Precision]
    """
    values: np.ndarray
    as_dict: Dict[str, float]


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> MetricPack:
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    y_pred = (y_prob > threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0

    # confusion -> Sn/Sp
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sn = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    sp = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # AUC/AP require both classes present
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
    else:
        auc = 0.0
        ap = 0.0

    values = np.array([bacc * 100, acc * 100, auc, sn * 100, sp * 100, mcc, f1, prec], dtype=float)
    as_dict = {
        "BACC": float(values[0]),
        "ACC": float(values[1]),
        "AUC": float(values[2]),
        "Sn": float(values[3]),
        "Sp": float(values[4]),
        "MCC": float(values[5]),
        "F1": float(values[6]),
        "Precision": float(values[7]),
        "AP": float(ap),
    }
    return MetricPack(values=values, as_dict=as_dict)
