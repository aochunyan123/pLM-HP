# -*- coding: utf-8 -*-
import argparse
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
import esm
from Bio import SeqIO
from tqdm import tqdm


def read_fasta(file_path: str) -> Dict[str, str]:
    """Return {seq_id: sequence}."""
    return {rec.id: str(rec.seq) for rec in SeqIO.parse(file_path, "fasta")}


def _make_batches(
    items: List[Tuple[str, str]],
    batch_converter,
    max_tokens_per_batch: int,
) -> List[List[Tuple[str, str]]]:
    """
    Simple length-aware batching by max tokens.
    items: [(name, seq), ...]
    """
    # sort by length to reduce padding
    items = sorted(items, key=lambda x: len(x[1]))
    batches: List[List[Tuple[str, str]]] = []
    cur: List[Tuple[str, str]] = []
    cur_tokens = 0

    for name, seq in items:
        # +2 for BOS/EOS
        tok = len(seq) + 2
        if cur and (cur_tokens + tok > max_tokens_per_batch):
            batches.append(cur)
            cur = []
            cur_tokens = 0
        cur.append((name, seq))
        cur_tokens += tok

    if cur:
        batches.append(cur)
    return batches


@torch.no_grad()
def get_per_residue_embeddings(
    seq_dict: Dict[str, str],
    model,
    alphabet,
    batch_converter,
    layer: int,
    device: str,
    max_tokens_per_batch: int = 2000,
    fp16: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Return dict: {seq_id: np.ndarray shape (L, D)}, where L excludes BOS/EOS.
    """
    model.eval().to(device)
    embeddings: Dict[str, np.ndarray] = {}

    items = list(seq_dict.items())
    batches = _make_batches(items, batch_converter, max_tokens_per_batch)

    for batch in tqdm(batches, desc="Extract ESM embeddings", unit="batch"):
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.to(device)

        if fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(batch_tokens, repr_layers=[layer], return_contacts=False)
        else:
            out = model(batch_tokens, repr_layers=[layer], return_contacts=False)

        reps = out["representations"][layer]  # [B, T, D]

        # for each sequence in the batch, slice out residues (remove BOS/EOS)
        pad = alphabet.padding_idx
        for i, (name, _seq) in enumerate(batch):
            # true token length including BOS/EOS (exclude padding)
            L = (batch_tokens[i] != pad).sum().item()
            # remove BOS (0) and EOS (L-1)
            per_res = reps[i, 1 : L - 1].detach().cpu().numpy()
            embeddings[name] = per_res

    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=str, required=True, help="Input fasta file")
    parser.add_argument("--out", type=str, required=True, help="Output pkl path")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--layer", type=int, default=12, help="Representation layer")
    parser.add_argument("--max_tokens", type=int, default=2000, help="Max tokens per batch")
    parser.add_argument("--fp16", action="store_true", help="Use autocast fp16 (cuda only)")
    args = parser.parse_args()

    seqs = read_fasta(args.fasta)

    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    embs = get_per_residue_embeddings(
        seq_dict=seqs,
        model=model,
        alphabet=alphabet,
        batch_converter=batch_converter,
        layer=args.layer,
        device=args.device,
        max_tokens_per_batch=args.max_tokens,
        fp16=args.fp16,
    )

    with open(args.out, "wb") as f:
        pickle.dump(embs, f)

    print(f"Saved: {args.out} | sequences: {len(embs)}")


if __name__ == "__main__":
    main()
