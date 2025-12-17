# -*- coding: utf-8 -*-
from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def load_embeddings(pkl_path: str) -> Dict[str, np.ndarray]:
    """
    Load embeddings: dict {seq_id: np.ndarray(L, D)}.
    L = sequence length (residues), D = embedding dim.
    """
    with open(pkl_path, "rb") as f:
        embs = pickle.load(f)
    if not isinstance(embs, dict) or len(embs) == 0:
        raise ValueError(f"Invalid embeddings pkl: {pkl_path}")
    return embs


def infer_label_from_id(seq_id: str) -> int:
    """
    Infer label from sequence ID.
    Convention (you can customize):
      - Positive: id startswith 'p_' OR contains 'positive'
      - Negative: id startswith 'n_' OR contains 'negative'
    """
    k = seq_id.lower()
    if k.startswith("p_") or ("positive" in k):
        return 1
    if k.startswith("n_") or ("negative" in k):
        return 0
    raise ValueError(f"Cannot infer label from id='{seq_id}'. Please rename ids or implement your rule.")


def dict_to_lists(emb_dict: Dict[str, np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    """
    Input:  {id: np.ndarray(L, D)}
    Output: seqs(list[np.ndarray(L, D)]), labels(np.ndarray(N,)), ids(list[str])
    """
    seqs, labels, ids = [], [], []
    for k, arr in emb_dict.items():
        if not (isinstance(arr, np.ndarray) and arr.ndim == 2):
            raise ValueError(f"{k} expects np.ndarray(L, D), got {type(arr)} shape={getattr(arr,'shape',None)}")
        if arr.shape[0] <= 0:
            raise ValueError(f"{k} has empty sequence length.")
        seqs.append(arr.astype(np.float32))
        labels.append(infer_label_from_id(k))
        ids.append(k)
    return seqs, np.asarray(labels, dtype=np.int64), ids


def streaming_min_max(seqs: List[np.ndarray], eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Compute global per-dimension min/max over ALL tokens in TRAIN sequences."""
    min_vec, max_vec = None, None
    for x in seqs:  # (L, D)
        cur_min = x.min(axis=0)
        cur_max = x.max(axis=0)
        if min_vec is None:
            min_vec, max_vec = cur_min, cur_max
        else:
            min_vec = np.minimum(min_vec, cur_min)
            max_vec = np.maximum(max_vec, cur_max)

    if min_vec is None:
        raise ValueError("Empty seq list.")
    # avoid degenerate span
    span = np.maximum(max_vec - min_vec, eps)
    return min_vec.astype(np.float32), span.astype(np.float32)


def apply_minmax(seqs: List[np.ndarray], min_vec: np.ndarray, span: np.ndarray, clip: bool = True) -> List[np.ndarray]:
    """Apply min-max scaling: (x - min) / span."""
    out = []
    for x in seqs:
        y = (x - min_vec) / span
        if clip:
            y = np.clip(y, 0.0, 1.0)
        out.append(y.astype(np.float32))
    return out


class SeqDataset(Dataset):
    """Sequence-level classification dataset with variable length [L, D] inputs."""
    def __init__(self, seqs: List[np.ndarray], labels: np.ndarray, ids: List[str] | None = None):
        self.seqs = [torch.from_numpy(s) for s in seqs]          # each: [L, D]
        self.labels = torch.from_numpy(labels)                   # [N]
        self.ids = ids

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int):
        sid = self.ids[i] if self.ids is not None else ""
        return self.seqs[i], self.labels[i], sid


def collate_fn(batch):
    """
    batch: list of (seq[L,D], y, id)
    returns:
      padded:  [B, Lmax, D]
      lengths: [B]
      y:       [B]
      ids:     list[str]
    """
    seqs, ys, ids = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True)  # [B, Lmax, D]
    ys = torch.stack(ys)                           # [B]
    return padded, lengths, ys, list(ids)


@dataclass
class DataBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    input_dim: int
    pos_count: int
    neg_count: int


def build_loaders(
    train_pkl: str,
    test_pkl: str,
    batch_size: int,
    use_minmax: bool = False,
    save_scaler: str | None = None,
) -> DataBundle:
    train_embs = load_embeddings(train_pkl)
    test_embs = load_embeddings(test_pkl)

    train_seqs, y_train, ids_train = dict_to_lists(train_embs)
    test_seqs, y_test, ids_test = dict_to_lists(test_embs)

    if use_minmax:
        min_vec, span = streaming_min_max(train_seqs, eps=1e-6)
        train_seqs = apply_minmax(train_seqs, min_vec, span, clip=True)
        test_seqs = apply_minmax(test_seqs, min_vec, span, clip=True)
        if save_scaler is not None:
            np.savez(save_scaler, min_vec=min_vec, span=span)

    input_dim = int(train_seqs[0].shape[1])
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())

    train_loader = DataLoader(
        SeqDataset(train_seqs, y_train, ids_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        SeqDataset(test_seqs, y_test, ids_test),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return DataBundle(
        train_loader=train_loader,
        test_loader=test_loader,
        input_dim=input_dim,
        pos_count=pos,
        neg_count=neg,
    )
