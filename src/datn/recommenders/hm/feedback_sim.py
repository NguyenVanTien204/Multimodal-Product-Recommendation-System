"""Offline simulation of conversational feedback on top of the frozen H&M recommender.

A simulated shopper is shown `k` items per round. Items from their real future purchases (the test
split) are "liked" with probability `p_like`; other shown items are "disliked" with probability
`p_dislike` (most people give no feedback on most items). Three policies consume that feedback:

    none          ignore it (history never changes)
    like          liked items are appended to the tower history
    like+dislike  as `like`, plus a soft penalty on the nearest neighbours of disliked items

Every policy hides items it already showed, so the arms differ only in how feedback is *used*.
The model is never retrained; only its input history and the final ordering change. Pure Python/NumPy:
the ranking and neighbour functions are injected (see scripts/hm/sim_preference_feedback.py).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from ...agent.preferences import NEGATIVE_NEIGHBOURS, NEGATIVE_WEIGHT, neighbour_penalty

RRF_K = 60
ARMS = ("none", "like", "like+dislike")

# (histories, per-user excluded items) -> per-user ranked item list, best first
RankFn = Callable[[list[list[int]], list[set[int]]], list[list[int]]]
# disliked item -> {neighbour item: rank (1 = most similar)}
NeighbourFn = Callable[[int], Mapping[int, int]]


@dataclass(frozen=True)
class SimUser:
    context: tuple[int, ...]  # catalogue indices known before the conversation, oldest first
    truth: frozenset[int]  # items the user really bought afterwards (held out)


@dataclass
class SimResult:
    recall: dict[str, np.ndarray]  # arm -> (n_users, rounds): share of the user's true items shown so far
    hit: dict[str, np.ndarray]  # arm -> (n_users, rounds): 1 if at least one true item was shown so far
    shown_likes: dict[str, np.ndarray]  # arm -> (n_users, rounds): feedback volume actually given (liked items)
    shown_dislikes: dict[str, np.ndarray]


def _u01(seed: int, user: int, item: int) -> float:
    """Deterministic per (user, item): every arm sees the same simulated reaction to the same item (paired design)."""
    return random.Random(f"{seed}:{user}:{item}").random()


def rerank_with_penalty(ranked: Sequence[int], neighbour_maps: Sequence[Mapping[int, int]], k: int, pool: int = 100) -> list[int]:
    """Top-k of the first `pool` ranked items after subtracting the neighbourhood penalty from their RRF score."""
    scored = [
        (1.0 / (RRF_K + pos) - neighbour_penalty(neighbour_maps, item, NEGATIVE_WEIGHT, RRF_K), pos, item)
        for pos, item in enumerate(ranked[:pool], start=1)
    ]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [item for _, _, item in scored[:k]]


def simulate(
    users: Sequence[SimUser],
    rank_fn: RankFn,
    neighbour_fn: NeighbourFn,
    *,
    rounds: int = 5,
    k: int = 12,
    p_like: float = 0.8,
    p_dislike: float = 0.3,
    seed: int = 0,
    arms: Sequence[str] = ARMS,
    pool: int = 100,
) -> SimResult:
    n = len(users)
    recall = {a: np.zeros((n, rounds)) for a in arms}
    hit = {a: np.zeros((n, rounds)) for a in arms}
    n_likes = {a: np.zeros((n, rounds)) for a in arms}
    n_dislikes = {a: np.zeros((n, rounds)) for a in arms}
    neighbour_cache: dict[int, Mapping[int, int]] = {}

    def neighbours(item: int) -> Mapping[int, int]:
        if item not in neighbour_cache:
            neighbour_cache[item] = neighbour_fn(item)
        return neighbour_cache[item]

    for arm in arms:
        history = [list(u.context) for u in users]
        shown: list[set[int]] = [set() for _ in users]
        disliked: list[list[int]] = [[] for _ in users]
        for r in range(rounds):
            ranked = rank_fn([list(h) for h in history], [set(s) for s in shown])
            for i, user in enumerate(users):
                items = [x for x in ranked[i] if x not in shown[i]]
                if arm == "like+dislike" and disliked[i]:
                    maps = [neighbours(d) for d in disliked[i][-NEGATIVE_NEIGHBOURS:]]
                    top = rerank_with_penalty(items, maps, k, pool)
                else:
                    top = items[:k]
                shown[i].update(top)
                for item in top:
                    if item in user.truth:
                        if _u01(seed, i, item) < p_like:
                            n_likes[arm][i, r:] += 1
                            if arm != "none":
                                history[i].append(item)
                    elif _u01(seed, i, item) < p_dislike:
                        n_dislikes[arm][i, r:] += 1
                        disliked[i].append(item)
                found = len(user.truth & shown[i])
                recall[arm][i, r] = found / len(user.truth) if user.truth else 0.0
                hit[arm][i, r] = float(found > 0)
    return SimResult(recall, hit, n_likes, n_dislikes)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (b - a) over users with a 95% bootstrap CI: (mean, lo, hi)."""
    diff = np.asarray(b, dtype="float64") - np.asarray(a, dtype="float64")
    if diff.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, diff.size, size=(n_boot, diff.size))].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
