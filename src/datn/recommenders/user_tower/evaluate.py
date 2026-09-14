from __future__ import annotations

import numpy as np
import torch

from .dataset import PAD_IDX, EvalExample, ItemVocab
from .model import UserTower


@torch.no_grad()
def full_ranking_evaluate(
    model: UserTower,
    examples: list[EvalExample],
    vocab: ItemVocab,
    ks: tuple[int, ...],
    batch_size: int,
    exclude_seen: bool,
    device: torch.device,
) -> dict[str, float]:
    """Rank the full catalog per user and report HitRate/Recall/NDCG@k.

    Under this leave-last-out protocol each user has exactly one relevant target
    item, so Recall@k and HitRate@k are mathematically identical here; both are
    reported because docs/02-overview.md's result table names them separately.
    """
    model.eval()
    item_emb = model.item_vectors()  # (vocab_size, d_model)
    n = len(examples)
    if n == 0:
        return {f"{name}@{k}": float("nan") for name in ("HitRate", "Recall", "NDCG") for k in ks} | {
            "n_users": 0
        }

    hits = {k: 0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}

    for start in range(0, n, batch_size):
        batch = examples[start : start + batch_size]
        context = torch.from_numpy(np.stack([e.context for e in batch])).to(device)
        user_vecs = model.encode_user(context)  # (B, d_model)
        scores = user_vecs @ item_emb.T  # (B, vocab_size)
        scores[:, PAD_IDX] = float("-inf")
        scores[:, vocab.oov_idx] = float("-inf")
        if exclude_seen:
            for i, ex in enumerate(batch):
                if ex.seen:
                    idx = torch.tensor(list(ex.seen), device=device, dtype=torch.long)
                    scores[i, idx] = float("-inf")

        targets = torch.tensor([e.target for e in batch], device=device, dtype=torch.long)
        target_scores = scores.gather(1, targets.unsqueeze(1)).squeeze(1)
        rank = (scores > target_scores.unsqueeze(1)).sum(dim=1)  # 0-indexed rank of the target

        for k in ks:
            hit = rank < k
            hits[k] += int(hit.sum().item())
            ndcg = torch.where(hit, 1.0 / torch.log2(rank.float() + 2.0), torch.zeros_like(rank, dtype=torch.float))
            ndcgs[k] += float(ndcg.sum().item())

    metrics: dict[str, float] = {"n_users": n}
    for k in ks:
        metrics[f"HitRate@{k}"] = hits[k] / n
        metrics[f"Recall@{k}"] = hits[k] / n
        metrics[f"NDCG@{k}"] = ndcgs[k] / n
    return metrics
