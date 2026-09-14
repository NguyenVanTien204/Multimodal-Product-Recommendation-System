from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn

from .dataset import PAD_IDX


class UserTower(nn.Module):
    """SASRec-style self-attentive sequential encoder with content-aware item vectors.

    Two-tower retrieval, correctly: this module defines a *single* item space
        e_i = id_residual(i) + content_proj(clip(i))          -- Item Tower
    where `id_residual` is a trainable, zero-initialized embedding and `clip(i)` is
    the (frozen) concatenated image/text CLIP vector for item i. Zero-init matters:
    a catalog item that has never been interacted with keeps id_residual(i) == 0
    forever (no gradient reaches an index that's never sampled), so its item vector
    *degrades gracefully to pure content* instead of to a random, untrained vector --
    this is what actually fixes cold-start, not the late-fusion step downstream.

    The sequence encoder consumes e_i (not a plain ID embedding) at every position,
    so self-attention itself can exploit visual/textual similarity between items in
    a user's history, not just co-occurrence statistics.

    The Transformer's output hidden state at a user's most recent position, h_user,
    is the User Tower vector (a query). Retrieval/training score is ALWAYS the dot
    product score(u, i) = h_user(u) . e_i(i) -- never h_user + e_i. A user vector and
    an item vector live in the same space so they can be compared (dot product /
    cosine), but they are not interchangeable and must not be added together; doing
    so is what made the E_final = lambda*E_cf + (1-lambda)*E_content formula in the
    original docs/02-overview.md draft incoherent (E_cf there was a user vector,
    E_image/E_text were item vectors). See docs/user_tower_design.md for the
    corrected, score-level formulation used for the CF-vs-content ablation.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.2,
        content_matrix: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.id_residual = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        nn.init.zeros_(self.id_residual.weight)

        content_dim = 0 if content_matrix is None else int(content_matrix.shape[1])
        self.has_content = content_dim > 0
        if self.has_content:
            self.register_buffer("content_matrix", torch.from_numpy(content_matrix).float())
            self.content_proj = nn.Linear(content_dim, d_model, bias=False)
        else:
            self.content_matrix = None
            self.content_proj = None

        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN, matches SASRec's residual block ordering
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.out_norm = nn.LayerNorm(d_model)

    def embed_items(self, item_ids: Tensor) -> Tensor:
        """Content-aware item vectors e_i for an arbitrary-shaped index tensor.

        Single source of truth for the Item Tower: used for the sequence's input
        embeddings, for the positive/negative target vectors in the training loss,
        and (via `item_vectors`) for full-catalog retrieval scoring -- so the item
        actually scored against is always exactly the item fed into the encoder.
        """
        vecs = self.id_residual(item_ids)
        if self.has_content:
            vecs = vecs + self.content_proj(self.content_matrix[item_ids])
        return vecs

    def forward(self, item_seq: Tensor) -> Tensor:
        """item_seq: (B, L) right-padded item indices -> hidden states (B, L, d_model).

        Right-padding (see dataset.pad_right) plus causal masking guarantees every
        query position always has at least key 0 unmasked (position 0 is only padding
        for an empty sequence, which never occurs), so no attention row is ever fully
        masked and no NaN can leak through the value-weighted sum -- unlike
        left-padding, where the leading pad positions would have no valid key at all.
        """
        batch_size, seq_len = item_seq.shape
        pad_mask = item_seq == PAD_IDX  # (B, L), True where padded

        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.embed_items(item_seq) * math.sqrt(self.d_model)
        hidden = hidden + self.position_embedding(positions)
        hidden = self.embed_dropout(hidden)

        # Bool mask (True = disallowed), matching src_key_padding_mask's convention --
        # mixing a float additive mask with a bool padding mask is deprecated in
        # newer PyTorch and emits a warning on every forward call.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=item_seq.device), diagonal=1
        )
        hidden = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=pad_mask)
        return self.out_norm(hidden)

    def encode_user(self, item_seq: Tensor) -> Tensor:
        """Return the User Tower query vector h_user: hidden state at each sequence's
        last non-pad position, shape (B, d_model)."""
        hidden = self.forward(item_seq)
        lengths = (item_seq != PAD_IDX).sum(dim=1).clamp(min=1)
        last_pos = lengths - 1  # sequences are right-padded, so the last valid index is len-1
        return hidden[torch.arange(hidden.size(0), device=hidden.device), last_pos]

    def item_vectors(self) -> Tensor:
        """Full Item Tower matrix e_i, one row per catalog index (row 0 is the PAD sentinel)."""
        all_ids = torch.arange(self.id_residual.num_embeddings, device=self.id_residual.weight.device)
        return self.embed_items(all_ids)
