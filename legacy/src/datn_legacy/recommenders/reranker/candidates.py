from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..user_tower.dataset import PAD_IDX, ItemVocab
from ..user_tower.model import UserTower

from .model import FEATURE_NAMES


@dataclass(frozen=True)
class CandidateBudget:
    """Per-source candidate caps, before de-duplication.

    Locked by the validation-selected "practical four-source union" recipe (see
    legacy/docs/logs/2026-09-22_balanced_retrieval_reranker_results.md); must match
    whatever budget a loaded reranker checkpoint was trained against, since the
    feature values (esp. reciprocal ranks) depend on which candidates exist at all.
    """

    tower: int = 1000
    popularity: int = 300
    content_centroid: int = 400
    last_item: int = 300

    @property
    def max_candidates(self) -> int:
        return self.tower + self.popularity + self.content_centroid + self.last_item


@dataclass(frozen=True)
class CandidateExample:
    """One user's context for candidate generation. `target` is only used for
    offline training/evaluation (force_target); serving code leaves it None."""

    context: np.ndarray  # right-padded item indices, shape (max_seq_len,)
    seen: frozenset[int]
    target: int | None = None


@dataclass(frozen=True)
class PopularityStats:
    """Log-popularity z-score and rank, computed once from train-split item counts."""

    score: np.ndarray  # (vocab_size,) z-scored log(1 + count), 0 for PAD/OOV
    order: np.ndarray  # item indices sorted by popularity, descending (no PAD/OOV)
    rank: np.ndarray  # (vocab_size,) 0-indexed popularity rank; num_items for unseen

    @staticmethod
    def from_sequences(vocab: ItemVocab, sequences: dict[str, list[int]]) -> "PopularityStats":
        counts = np.zeros(vocab.vocab_size, dtype=np.float32)
        for seq in sequences.values():
            if seq:
                np.add.at(counts, np.asarray(seq, dtype=np.int64), 1)

        score = np.zeros(vocab.vocab_size, dtype=np.float32)
        log_pop = np.log1p(counts[1 : vocab.num_items + 1])
        score[1 : vocab.num_items + 1] = (log_pop - log_pop.mean()) / max(float(log_pop.std()), 1e-6)

        order = np.lexsort((np.arange(vocab.vocab_size), -counts))
        order = order[(order != PAD_IDX) & (order != vocab.oov_idx)]

        rank = np.full(vocab.vocab_size, vocab.num_items, dtype=np.float32)
        rank[order] = np.arange(len(order))

        return PopularityStats(score=score, order=order, rank=rank)


def _reciprocal_rank(source: list[int], candidates: list[int], device: torch.device) -> torch.Tensor:
    ranks = {item: rank for rank, item in enumerate(source)}
    return torch.tensor(
        [1.0 / (ranks[item] + 1) if item in ranks else 0.0 for item in candidates], device=device
    )


@torch.no_grad()
def candidate_features(
    tower: UserTower,
    examples: Sequence[CandidateExample],
    vocab: ItemVocab,
    popularity: PopularityStats,
    budget: CandidateBudget = CandidateBudget(),
    device: torch.device | None = None,
    batch_size: int = 128,
    force_target: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Four-source candidate union (User Tower + popularity + content centroid +
    last-item content) with the 15-dim feature vector each candidate needs for
    `ResidualListwiseRanker`. Must stay numerically identical to the recipe the
    checkpoint was trained with (legacy/notebooks/reranker_training.ipynb, cell 5) --
    this is a straight port, not a reimplementation.

    Returns one (candidate_ids, features) pair per example, features shaped
    (num_candidates, len(FEATURE_NAMES)) in `FEATURE_NAMES` order.
    """
    if not tower.has_content:
        raise ValueError("candidate_features requires a content-aware UserTower (image+text enabled)")

    device = device or next(tower.parameters()).device
    table = tower.item_vectors()
    norm_table = F.normalize(table, dim=-1)
    raw_content = F.normalize(tower.content_matrix, dim=-1)

    rows: list[tuple[np.ndarray, np.ndarray]] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        context = torch.from_numpy(np.stack([ex.context for ex in batch])).to(device)
        users = tower.encode_user(context, table)
        scores = users @ table.T

        mask = context.ne(PAD_IDX)
        lengths = mask.sum(1, keepdim=True)
        positions = torch.arange(context.shape[1], device=device).float()[None]
        recency = (0.8 ** (lengths - 1 - positions).clamp(min=0)) * mask
        content_hist = raw_content[context]
        content_query = F.normalize((content_hist * recency.unsqueeze(-1)).sum(1), dim=-1)
        last_ids = context.gather(1, (lengths.long() - 1).clamp(min=0)).squeeze(1)
        content_scores = content_query @ raw_content.T
        last_scores = raw_content[last_ids] @ raw_content.T

        for matrix in (scores, content_scores, last_scores):
            matrix[:, PAD_IDX] = matrix[:, vocab.oov_idx] = float("-inf")
        for i, ex in enumerate(batch):
            if ex.seen:
                for matrix in (scores, content_scores, last_scores):
                    matrix[i, list(ex.seen)] = float("-inf")

        top_ut = torch.topk(scores, min(budget.tower, scores.shape[1]), dim=1).indices
        top_content = torch.topk(content_scores, min(budget.content_centroid, content_scores.shape[1]), dim=1).indices
        top_last = torch.topk(last_scores, min(budget.last_item, last_scores.shape[1]), dim=1).indices

        for i, ex in enumerate(batch):
            ut = top_ut[i].cpu().tolist()
            content = top_content[i].cpu().tolist()
            last = top_last[i].cpu().tolist()
            pop: list[int] = []
            for item in popularity.order:
                item = int(item)
                if item not in ex.seen:
                    pop.append(item)
                    if len(pop) == budget.popularity:
                        break

            candidates = list(dict.fromkeys(ut + pop + content + last))
            if force_target and ex.target is not None and ex.target not in candidates:
                candidates.append(ex.target)
            if not candidates:
                rows.append((np.empty(0, dtype=np.int64), np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)))
                continue

            ids = torch.tensor(candidates, device=device)
            raw = scores[i, ids]
            hist = context[i][context[i].ne(PAD_IDX)]
            sims = norm_table[ids] @ norm_table[hist].T if hist.numel() else torch.zeros(len(ids), 1, device=device)
            candidate_np = np.asarray(candidates, dtype=np.int64)

            rr_ut = _reciprocal_rank(ut, candidates, device)
            rr_content = _reciprocal_rank(content, candidates, device)
            rr_last = _reciprocal_rank(last, candidates, device)
            rr_pop = torch.from_numpy(1 / (popularity.rank[candidate_np] + 1)).to(device)

            features = torch.stack(
                (
                    raw,
                    torch.from_numpy(popularity.score[candidate_np]).to(device),
                    rr_ut,
                    rr_pop,
                    content_scores[i, ids],
                    last_scores[i, ids],
                    rr_content,
                    rr_last,
                    sims[:, -1],
                    sims.mean(1),
                    sims.max(1).values,
                    torch.full((len(ids),), float(len(hist)), device=device),
                    rr_ut.gt(0).float(),
                    rr_content.gt(0).float(),
                    rr_last.gt(0).float(),
                ),
                dim=1,
            )
            rows.append((candidate_np, features.cpu().numpy().astype(np.float32)))

    return rows
