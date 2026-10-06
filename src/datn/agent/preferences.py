"""Interaction memory of the shopping assistant: typed, weighted, time-decayed feedback events.

Pure functions only (no I/O) so the same logic runs in the chat orchestrator, in the offline
simulation (`datn.recommenders.hm.feedback_sim`) and in unit tests. Nothing here is trained:
the profile only decides *which items go into the frozen User Tower's history* (positives),
*which items are never shown again* (negatives) and *which neighbourhoods get a soft penalty*.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

# Signed strength of one interaction. Purchases/carts come from the shop; like/dislike are explicit
# feedback given in the chat; view/click are implicit and weak on purpose.
EVENT_WEIGHTS: dict[str, float] = {"view": 0.1, "click": 0.3, "like": 0.7, "cart": 0.8, "purchase": 1.0, "dislike": -1.0}
# A reset kind replaces everything older for the same item ("I changed my mind" must win over the past).
RESET_KINDS = frozenset({"like", "dislike", "purchase"})

HALF_LIFE_DAYS = 30.0
POSITIVE_THRESHOLD = 0.25  # net score at/above which an item counts as a positive (tower history)
NEGATIVE_THRESHOLD = -0.25  # net score at/below which an item counts as a negative (excluded + penalised)
NEGATIVE_WEIGHT = 0.6  # same scale as the personal-rank boost in the orchestrator (PERSONAL_WEIGHT)
NEGATIVE_NEIGHBOURS = 4  # at most this many (most recent) negatives get a neighbourhood penalty
NEIGHBOUR_POOL = 30  # how many nearest products of a disliked item receive the soft penalty


@dataclass(frozen=True)
class PreferenceEvent:
    sku: str
    kind: str
    ts: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)  # idempotency key across chat session and shop DB
    source: str = "chat"

    def to_dict(self) -> dict:
        return {"event_id": self.event_id, "sku": self.sku, "kind": self.kind, "ts": self.ts, "source": self.source}

    @classmethod
    def from_dict(cls, d: Mapping) -> "PreferenceEvent":
        kwargs = {k: d[k] for k in ("sku", "kind", "ts", "event_id", "source") if d.get(k) is not None}
        return cls(**kwargs)


@dataclass(frozen=True)
class PreferenceProfile:
    scores: dict[str, float]  # sku -> net decayed score
    last_ts: dict[str, float]  # sku -> timestamp of its latest event

    @classmethod
    def from_events(
        cls, events: Iterable[PreferenceEvent], now: float | None = None, half_life_days: float = HALF_LIFE_DAYS
    ) -> "PreferenceProfile":
        now = time.time() if now is None else now
        seen: set[str] = set()
        by_sku: dict[str, list[PreferenceEvent]] = {}
        for e in events:
            if not e.sku or e.kind not in EVENT_WEIGHTS:
                continue
            if e.event_id:
                if e.event_id in seen:  # the same event relayed by the shop DB and still held in the chat session
                    continue
                seen.add(e.event_id)
            by_sku.setdefault(e.sku, []).append(e)
        scores: dict[str, float] = {}
        last: dict[str, float] = {}
        for sku, evs in by_sku.items():
            evs.sort(key=lambda e: e.ts)
            start = max((i for i, e in enumerate(evs) if e.kind in RESET_KINDS), default=0)
            scores[sku] = sum(
                EVENT_WEIGHTS[e.kind] * 0.5 ** (max(0.0, now - e.ts) / 86400.0 / half_life_days) for e in evs[start:]
            )
            last[sku] = evs[-1].ts
        return cls(scores, last)

    def positives(self, limit: int | None = None) -> list[str]:
        """Liked/bought/engaged SKUs, oldest first (the tower reads the most recent last)."""
        out = sorted((s for s, v in self.scores.items() if v >= POSITIVE_THRESHOLD), key=lambda s: self.last_ts[s])
        return out[-limit:] if limit else out

    def negatives(self, limit: int | None = None) -> list[str]:
        """Disliked SKUs, most recent first."""
        out = sorted((s for s, v in self.scores.items() if v <= NEGATIVE_THRESHOLD), key=lambda s: -self.last_ts[s])
        return out[:limit] if limit else out

    def summary(self) -> dict[str, int]:
        return {"liked_or_engaged": len(self.positives()), "disliked": len(self.negatives())}


def merge_history(shop_history: Sequence[str], positives: Sequence[str], negatives: Iterable[str], limit: int) -> list[str]:
    """Tower history = shop cart/orders (older) then chat-derived positives, latest occurrence wins, negatives removed."""
    drop = set(negatives)
    merged: list[str] = []
    for sku in [*shop_history, *positives]:
        if sku in drop:
            continue
        if sku in merged:
            merged.remove(sku)
        merged.append(sku)
    return merged[-limit:]


def neighbour_penalty(
    neighbour_ranks: Sequence[Mapping[int, int]], key: int, weight: float = NEGATIVE_WEIGHT, rrf_k: int = 60
) -> float:
    """Soft penalty for `key` being near a disliked item: the strongest neighbourhood wins (rank 1 = most similar)."""
    ranks = [r[key] for r in neighbour_ranks if key in r]
    return weight / (rrf_k + min(ranks)) if ranks else 0.0
