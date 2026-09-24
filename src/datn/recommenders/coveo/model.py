from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn


class ActionAwareTwoTower(nn.Module):
    """Session tower + multimodal item tower with an auxiliary purchase task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        *,
        d_model: int = 96,
        n_actions: int = 6,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
        text_matrix: np.ndarray | None = None,
        image_matrix: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.item_id = nn.Embedding(vocab_size, d_model, padding_idx=0)
        nn.init.zeros_(self.item_id.weight)
        self.action = nn.Embedding(n_actions, d_model, padding_idx=0)
        self.position = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.text_proj = None
        self.image_proj = None
        if text_matrix is not None:
            self.register_buffer("text_matrix", torch.as_tensor(text_matrix, dtype=torch.float32))
            self.text_proj = nn.Linear(text_matrix.shape[1], d_model, bias=False)
        if image_matrix is not None:
            self.register_buffer("image_matrix", torch.as_tensor(image_matrix, dtype=torch.float32))
            self.image_proj = nn.Linear(image_matrix.shape[1], d_model, bias=False)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.purchase_head = nn.Linear(d_model, 1)

    def item_table(self) -> Tensor:
        table = self.item_id.weight
        if self.text_proj is not None:
            table = table + self.text_proj(self.text_matrix)
        if self.image_proj is not None:
            table = table + self.image_proj(self.image_matrix)
        table = torch.nn.functional.normalize(table, dim=-1)
        return torch.cat((torch.zeros_like(table[:1]), table[1:]), dim=0)

    def forward(self, items: Tensor, actions: Tensor, table: Tensor | None = None) -> Tensor:
        table = self.item_table() if table is None else table
        positions = torch.arange(items.shape[1], device=items.device).unsqueeze(0)
        hidden = table[items] * math.sqrt(self.d_model) + self.action(actions) + self.position(positions)
        hidden = self.dropout(hidden)
        causal = torch.triu(torch.ones(items.shape[1], items.shape[1], dtype=torch.bool, device=items.device), 1)
        hidden = self.encoder(hidden, mask=causal, src_key_padding_mask=items.eq(0))
        return self.norm(hidden)

    def encode(self, items: Tensor, actions: Tensor, table: Tensor | None = None) -> Tensor:
        hidden = self.forward(items, actions, table)
        last = items.ne(0).sum(1).clamp_min(1) - 1
        query = hidden[torch.arange(items.shape[0], device=items.device), last]
        return torch.nn.functional.normalize(query, dim=-1)


class ResidualListwiseReranker(nn.Module):
    """Small tabular ranker; output is a bounded residual over retrieval score."""

    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.15, max_delta: float = 2.0) -> None:
        super().__init__()
        self.max_delta = max_delta
        self.net = nn.Sequential(
            nn.LayerNorm(n_features), nn.Linear(n_features, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1),
        )

    def forward(self, features: Tensor, retrieval_score: Tensor) -> Tensor:
        delta = torch.tanh(self.net(features).squeeze(-1)) * self.max_delta
        return retrieval_score + delta
