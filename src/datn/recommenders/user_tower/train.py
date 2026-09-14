from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import UserTowerConfig
from .content import ContentSource, load_content_matrix
from .dataset import (
    PAD_IDX,
    SASRecTrainDataset,
    build_eval_examples,
    build_item_vocab,
    load_user_sequences,
)
from .evaluate import full_ranking_evaluate
from .model import UserTower

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_negative_sampler(
    sequences_train: dict[str, list[int]], num_items: int, device: torch.device, power: float = 0.75
) -> torch.Tensor:
    """Cumulative distribution over item popularity^power (word2vec/BPR-style).

    Uniform sampling over 152k items makes almost every negative an "easy" one --
    a random item is trivially distinguishable from what a user actually likes, so
    the model never has to sharpen its ranking among plausible/popular items. Biasing
    negatives towards popular items (still capped by **power < 1 so it doesn't only
    ever sample the single most popular item) yields harder, more informative
    negatives without the cost of true similarity-based hard-negative mining.
    """
    freq = np.ones(num_items + 1, dtype=np.float64)  # 1-smoothing; index 0 (PAD) excluded below
    freq[0] = 0.0
    for seq in sequences_train.values():
        for item in seq:
            freq[item] += 1.0
    weights = np.power(freq, power)
    weights[0] = 0.0
    probs = weights / weights.sum()
    cum_probs = np.cumsum(probs)
    cum_probs[-1] = 1.0  # guard against float round-off so searchsorted never overflows
    return torch.tensor(cum_probs, dtype=torch.float32, device=device)


def _sample_negatives(cum_probs: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    draws = torch.rand(shape, device=cum_probs.device)
    idx = torch.searchsorted(cum_probs, draws)
    return idx.clamp(min=1, max=cum_probs.numel() - 1)


def _step_loss(
    model: UserTower,
    input_seq: torch.Tensor,
    target_seq: torch.Tensor,
    cum_probs: torch.Tensor,
    num_negatives: int,
    bce: nn.BCEWithLogitsLoss,
) -> torch.Tensor:
    hidden = model(input_seq)  # (B, L, d)
    loss_mask = target_seq != PAD_IDX  # (B, L)

    pos_emb = model.embed_items(target_seq)  # (B, L, d)
    pos_logits = (hidden * pos_emb).sum(-1)  # (B, L)
    pos_loss = bce(pos_logits, torch.ones_like(pos_logits)) * loss_mask

    neg_ids = _sample_negatives(cum_probs, (*target_seq.shape, num_negatives))  # (B, L, K)
    neg_emb = model.embed_items(neg_ids)  # (B, L, K, d)
    neg_logits = (hidden.unsqueeze(2) * neg_emb).sum(-1)  # (B, L, K)
    neg_loss = bce(neg_logits, torch.zeros_like(neg_logits)) * loss_mask.unsqueeze(-1)

    denom = loss_mask.sum().clamp(min=1)
    return pos_loss.sum() / denom + neg_loss.sum() / (denom * num_negatives)


def train(config: UserTowerConfig) -> dict[str, float]:
    set_seed(config.train.seed)
    device = resolve_device(config.train.device)
    logger.info("Using device=%s", device)

    vocab = build_item_vocab(config.data.items_path)
    sequences = load_user_sequences(
        config.data.train_path, config.data.valid_path, config.data.test_path, vocab
    )
    logger.info("Catalog size=%d users=%d", vocab.num_items, len(sequences.train))

    train_dataset = SASRecTrainDataset(sequences.train, config.model.max_seq_len)
    logger.info("Trainable users (>=2 positive train interactions)=%d", len(train_dataset))
    train_loader = DataLoader(
        train_dataset, batch_size=config.train.batch_size, shuffle=True, drop_last=False
    )
    valid_examples = build_eval_examples(sequences, config.model.max_seq_len, "valid")
    logger.info("Valid eval users=%d", len(valid_examples))

    sources = [
        ContentSource(s.name, s.embeddings_path, s.metadata_path)
        for s in config.content.sources
        if s.name in config.content.enabled
    ]
    content_matrix = load_content_matrix(vocab, sources)
    logger.info(
        "Content sources enabled=%s content_dim=%d", [s.name for s in sources], content_matrix.shape[1]
    )

    model = UserTower(
        vocab_size=vocab.vocab_size,
        max_seq_len=config.model.max_seq_len,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        dropout=config.model.dropout,
        content_matrix=content_matrix if content_matrix.shape[1] > 0 else None,
    ).to(device)

    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.train.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=config.train.lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train.epochs, eta_min=1e-5
    )
    bce = nn.BCEWithLogitsLoss(reduction="none")
    cum_probs = build_negative_sampler(sequences.train, vocab.num_items, device)

    artifacts_dir = Path(config.data.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    best_ndcg = -1.0
    best_metrics: dict[str, float] = {}
    epochs_without_improvement = 0
    primary_k = config.eval.ks[0]

    for epoch in range(1, config.train.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for input_seq, target_seq in train_loader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            loss = _step_loss(
                model, input_seq, target_seq, cum_probs, config.train.num_negatives, bce
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
            optimizer.step()
            total_loss += float(loss.item()) * input_seq.size(0)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        avg_loss = total_loss / max(len(train_dataset), 1)
        logger.info(
            "epoch=%d loss=%.4f lr=%.6f time=%.1fs", epoch, avg_loss, current_lr, time.time() - t0
        )

        if epoch % config.train.eval_every != 0:
            continue

        metrics = full_ranking_evaluate(
            model,
            valid_examples,
            vocab,
            config.eval.ks,
            config.eval.batch_size,
            config.eval.exclude_seen,
            device,
        )
        logger.info("epoch=%d valid=%s", epoch, metrics)

        ndcg = metrics[f"NDCG@{primary_k}"]
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_metrics = metrics
            epochs_without_improvement = 0
            torch.save(model.state_dict(), artifacts_dir / "user_tower.pt")
            (artifacts_dir / "item2idx.json").write_text(
                json.dumps(vocab.item2idx), encoding="utf-8"
            )
            (artifacts_dir / "best_metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )
        else:
            epochs_without_improvement += config.train.eval_every

        if epochs_without_improvement >= config.train.patience:
            logger.info("Early stopping at epoch=%d (best NDCG@%d=%.4f)", epoch, primary_k, best_ndcg)
            break

    return best_metrics
