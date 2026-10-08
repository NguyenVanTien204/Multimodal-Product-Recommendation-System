from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)


@dataclass
class PersonalRanking:
    skus: list[str]  # best first
    scores: list[float]
    source: str  # "reranker" | "popularity_fallback"
    model_version: str

    def rank_of(self) -> dict[str, int]:
        return {sku: i for i, sku in enumerate(self.skus, start=1)}


class RecommenderClient:
    """HTTP client for the checkpointed Amazon model service (hm/apps/recommender).

    That service wraps the frozen User Tower (retrieval) + Residual Listwise
    Reranker checkpoints (`user_tower_balanced_v1` + `reranker_v2`). The chat
    agent treats it as an optional signal: any failure returns None and the
    caller falls back to content-based similarity, so chat never depends on it.
    """

    def __init__(self, base_url: str = "", timeout_s: float = 8.0, max_k: int = 50) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_k = max_k

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    async def health(self) -> dict:
        if not self.enabled:
            return {"enabled": False}
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                data = (await client.get(f"{self.base_url}/health")).json()
            return {"enabled": True, "reachable": True, **data}
        except Exception as exc:  # noqa: BLE001
            return {"enabled": True, "reachable": False, "error": str(exc)[:120]}

    async def recommend(self, history_skus: list[str], k: int = 50) -> PersonalRanking | None:
        if not self.enabled:
            return None
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.post(
                    f"{self.base_url}/recommend", json={"history": history_skus, "k": min(k, self.max_k)}
                )
            resp.raise_for_status()
            data = resp.json()
            recs = data.get("recommendations", [])
            return PersonalRanking(
                skus=[r["item_id"] for r in recs],
                scores=[float(r["score"]) for r in recs],
                source=data.get("source", "reranker"),
                model_version=data.get("model_version", ""),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("recommender unavailable: %s", exc)
            return None
