# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTMClassifier(nn.Module):
    """
    Sequence classifier for variable-length residue embeddings.

    Input:
      x: [B, L, D]
      lengths: [B] (true lengths, descending NOT required due to enforce_sorted=False)
    Output:
      logits: [B, 1]
    """
    def __init__(self, input_dim: int, hidden_size: int = 256, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _packed_out, (h_n, _c_n) = self.lstm(packed)

        # h_n: [num_layers*2, B, hidden]
        # take last layer's forward/backward hidden
        h_fwd = h_n[-2]   # [B, hidden]
        h_bwd = h_n[-1]   # [B, hidden]
        h = torch.cat([h_fwd, h_bwd], dim=1)  # [B, 2*hidden]

        h = self.dropout(h)
        logits = self.fc(h)  # [B,1]
        return logits
