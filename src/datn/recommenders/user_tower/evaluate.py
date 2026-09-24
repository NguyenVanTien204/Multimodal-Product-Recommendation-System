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
        user_vecs = model.encode_user(context, item_emb)  # (B, d_model)
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


@torch.no_grad()
def sampled_ranking_evaluate(
    model: UserTower,
    examples: list[EvalExample],
    vocab: ItemVocab,
    ks: tuple[int, ...] = (10, 20, 50),
    num_negatives: int = 99,
    batch_size: int = 512,
    device: torch.device = torch.device("cpu"),
    strategy: str = "random",
    popularity_weights: np.ndarray | None = None,
    seed: int = 20260813,
) -> dict[str, float]:
    """Rank 1 ground-truth positive item against N sampled negatives (Sampled Ranking Protocol).

    Commonly used in literature (NCF, SASRec, GRU4Rec) to evaluate top-K recommendation
    on a candidate pool of (1 + num_negatives) items per user (e.g. 1 + 99 = 100 items).

    Parameters:
    - num_negatives: number of negative items per user (default 99 or 999).
    - strategy: 'random' (uniform) or 'popularity' (popularity-weighted sampling).
    """
    model.eval()
    item_emb = model.item_vectors()
    n = len(examples)
    if n == 0:
        return {f"{name}@{k}": float("nan") for name in ("HitRate", "Recall", "NDCG") for k in ks} | {
            "MRR": float("nan"),
            "n_users": 0,
            "num_negatives": num_negatives,
            "strategy": strategy,
        }

    rng = np.random.default_rng(seed)
    num_items = vocab.num_items

    # Pre-calculate cumulative distribution if popularity strategy is chosen
    cum_probs = None
    if strategy == "popularity":
        if popularity_weights is not None:
            w = np.maximum(popularity_weights[1 : num_items + 1].astype(np.float64), 1e-6)
            probs = w / w.sum()
            cum_probs = np.cumsum(probs)
            cum_probs[-1] = 1.0
        else:
            raise ValueError("popularity_weights must be provided when strategy='popularity'")

    hits = {k: 0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}
    mrr_sum = 0.0

    for start in range(0, n, batch_size):
        batch = examples[start : start + batch_size]
        b_size = len(batch)

        # 1. Sample negatives per user excluding seen and target items
        cand_ids = np.zeros((b_size, 1 + num_negatives), dtype=np.int64)
        for i, ex in enumerate(batch):
            cand_ids[i, 0] = ex.target
            forbidden = ex.seen | {ex.target, PAD_IDX, vocab.oov_idx}

            sampled: list[int] = []
            while len(sampled) < num_negatives:
                needed = num_negatives - len(sampled)
                if strategy == "popularity":
                    rand_vals = rng.random(size=needed * 2)
                    draws = np.searchsorted(cum_probs, rand_vals) + 1
                    draws = np.clip(draws, 1, num_items)
                else:
                    draws = rng.integers(1, num_items + 1, size=needed * 2)

                for d in draws:
                    item_d = int(d)
                    if item_d not in forbidden and item_d not in sampled:
                        sampled.append(item_d)
                        if len(sampled) == num_negatives:
                            break
            cand_ids[i, 1:] = sampled

        # 2. Encode users and candidates
        context = torch.from_numpy(np.stack([e.context for e in batch])).to(device)
        user_vecs = model.encode_user(context, item_emb)  # (B, d_model)

        cand_tensor = torch.from_numpy(cand_ids).to(device)  # (B, 1 + num_negatives)
        cand_embs = model.embed_items(cand_tensor, item_emb)  # (B, 1 + num_negatives, d_model)

        # 3. Compute logits: dot product between user vector and candidates
        scores = (user_vecs.unsqueeze(1) * cand_embs).sum(dim=-1)  # (B, 1 + num_negatives)
        pos_score = scores[:, 0:1]  # (B, 1)
        neg_scores = scores[:, 1:]  # (B, num_negatives)

        # 0-indexed rank of positive item among candidate set
        rank = (neg_scores >= pos_score).sum(dim=1)  # (B,)

        for k in ks:
            hit = rank < k
            hits[k] += int(hit.sum().item())
            ndcg = torch.where(hit, 1.0 / torch.log2(rank.float() + 2.0), torch.zeros_like(rank, dtype=torch.float))
            ndcgs[k] += float(ndcg.sum().item())

        mrr_sum += float((1.0 / (rank.float() + 1.0)).sum().item())

    metrics: dict[str, float] = {
        "n_users": n,
        "num_negatives": num_negatives,
        "candidate_pool_size": 1 + num_negatives,
        "strategy": strategy,
        "MRR": mrr_sum / n,
    }
    for k in ks:
        metrics[f"HitRate@{k}"] = hits[k] / n
        metrics[f"Recall@{k}"] = hits[k] / n
        metrics[f"NDCG@{k}"] = ndcgs[k] / n
    return metrics
