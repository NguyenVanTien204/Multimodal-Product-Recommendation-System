from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from datn.recommenders.user_tower.config import UserTowerConfig
from datn.recommenders.user_tower.dataset import EvalExample, ItemVocab, PAD_IDX
from datn.recommenders.user_tower.evaluate import full_ranking_evaluate
from datn.recommenders.user_tower.model import UserTower
from datn.recommenders.user_tower.train import _step_loss


def test_user_tower_forward_and_embedding():
    vocab_size = 50
    d_model = 128
    seq_len = 10
    content_dim = 64

    content_matrix = np.random.randn(vocab_size, content_dim).astype(np.float32)
    model = UserTower(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        d_model=d_model,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        dropout=0.1,
        content_matrix=content_matrix,
    )

    batch_size = 4
    input_seq = torch.randint(1, vocab_size, (batch_size, seq_len))
    # inject right padding
    input_seq[0, 5:] = PAD_IDX

    out = model(input_seq)
    assert out.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(out).any()

    user_vecs = model.encode_user(input_seq)
    assert user_vecs.shape == (batch_size, d_model)
    assert not torch.isnan(user_vecs).any()

    item_vecs = model.item_vectors()
    assert item_vecs.shape == (vocab_size, d_model)


def test_step_loss_with_multi_negatives():
    vocab_size = 30
    d_model = 64
    seq_len = 5
    model = UserTower(vocab_size=vocab_size, max_seq_len=seq_len, d_model=d_model, n_heads=2)

    input_seq = torch.randint(1, vocab_size, (2, seq_len))
    target_seq = torch.randint(1, vocab_size, (2, seq_len))
    cum_probs = torch.linspace(0, 1, vocab_size)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    loss = _step_loss(model, input_seq, target_seq, cum_probs, num_negatives=4, bce=bce)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert loss.item() > 0


def test_full_ranking_evaluate_multi_k():
    vocab_size = 20
    d_model = 32
    seq_len = 5
    model = UserTower(vocab_size=vocab_size, max_seq_len=seq_len, d_model=d_model, n_heads=2)

    vocab = ItemVocab(
        item2idx={f"item_{i}": i for i in range(1, vocab_size - 1)},
        num_items=vocab_size - 2,
    )

    examples = [
        EvalExample(
            user_id="u1",
            context=np.array([1, 2, 0, 0, 0], dtype=np.int64),
            target=3,
            seen={1, 2},
        ),
        EvalExample(
            user_id="u2",
            context=np.array([4, 5, 6, 0, 0], dtype=np.int64),
            target=7,
            seen={4, 5, 6},
        ),
    ]

    metrics = full_ranking_evaluate(
        model=model,
        examples=examples,
        vocab=vocab,
        ks=(5, 10),
        batch_size=2,
        exclude_seen=True,
        device=torch.device("cpu"),
    )

    assert metrics["n_users"] == 2
    assert "HitRate@5" in metrics
    assert "NDCG@5" in metrics
    assert "HitRate@10" in metrics
    assert "NDCG@10" in metrics
