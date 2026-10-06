from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from ..retrieval import schema as S


@dataclass
class Review:
    review_id: int
    product_id: int
    rating: float
    helpful_vote: int
    title: str | None
    text: str
    score: float = 0.0  # query relevance (cosine), 0 when retrieved without a query
    is_mock: bool = False  # borrowed from another catalog (H&M has no review text)

    def snippet(self, limit: int = 320) -> str:
        body = " ".join(self.text.split())
        return body if len(body) <= limit else body[: limit - 1].rstrip() + "…"


def _to_review(point, score: float = 0.0) -> Review:
    p = point.payload or {}
    return Review(
        review_id=int(point.id),
        product_id=int(p.get(S.R_PRODUCT_ID, 0)),
        rating=float(p.get(S.R_RATING, 0.0)),
        helpful_vote=int(p.get(S.R_HELPFUL, 0)),
        title=p.get(S.R_TITLE),
        text=str(p.get(S.R_TEXT, "")),
        score=score,
        is_mock=bool(p.get(S.R_IS_MOCK, False)),
    )


class ReviewRetriever:
    """Retrieves review evidence *scoped to specific products*.

    Grounding the LLM in reviews of the exact items being discussed (rather than
    the globally most similar reviews) is what prevents answers about product A
    from quoting reviews of product B. Ranking = semantic similarity to the user's
    question, with a small log-scaled boost for community-endorsed reviews.
    """

    HELPFUL_BOOST = 0.02

    def __init__(self, client: QdrantClient, collection: str = S.REVIEWS_COLLECTION) -> None:
        self.client = client
        self.collection = collection

    def _rank(self, reviews: list[Review]) -> list[Review]:
        return sorted(reviews, key=lambda r: r.score + self.HELPFUL_BOOST * math.log1p(r.helpful_vote), reverse=True)

    def for_products(
        self, product_ids: Sequence[int], query_vector: np.ndarray | None, per_product: int = 3
    ) -> dict[int, list[Review]]:
        """Top reviews per product; semantic if a query vector is given, else most helpful."""
        out: dict[int, list[Review]] = {}
        if not product_ids:
            return out
        if query_vector is None:
            for pid in product_ids:
                out[pid] = self.most_helpful(pid, per_product)
            return out

        requests = [
            qm.QueryRequest(
                query=query_vector.tolist(),
                using=S.VECTOR_TEXT,
                filter=qm.Filter(must=[qm.FieldCondition(key=S.R_PRODUCT_ID, match=qm.MatchValue(value=int(pid)))]),
                limit=max(per_product * 3, 6),
                with_payload=True,
            )
            for pid in product_ids
        ]
        responses = self.client.query_batch_points(self.collection, requests=requests)
        for pid, resp in zip(product_ids, responses):
            ranked = self._rank([_to_review(p, float(p.score)) for p in resp.points])
            out[pid] = ranked[:per_product]
        return out

    def most_helpful(self, product_id: int, limit: int = 3) -> list[Review]:
        points, _ = self.client.scroll(
            self.collection,
            scroll_filter=qm.Filter(must=[qm.FieldCondition(key=S.R_PRODUCT_ID, match=qm.MatchValue(value=int(product_id)))]),
            limit=64,
            with_payload=True,
        )
        return sorted((_to_review(p) for p in points), key=lambda r: (r.helpful_vote, len(r.text)), reverse=True)[:limit]

    def critical(self, product_id: int, limit: int = 2) -> list[Review]:
        """Low-rated reviews, so comparisons surface drawbacks and not only praise."""
        points, _ = self.client.scroll(
            self.collection,
            scroll_filter=qm.Filter(
                must=[
                    qm.FieldCondition(key=S.R_PRODUCT_ID, match=qm.MatchValue(value=int(product_id))),
                    qm.FieldCondition(key=S.R_RATING, range=qm.Range(lte=2.0)),
                ]
            ),
            limit=32,
            with_payload=True,
        )
        return sorted((_to_review(p) for p in points), key=lambda r: (r.helpful_vote, len(r.text)), reverse=True)[:limit]

    def rating_summary(self, product_id: int) -> dict[str, float]:
        """Exact count / mean over the indexed reviews of one product."""
        points, _ = self.client.scroll(
            self.collection,
            scroll_filter=qm.Filter(must=[qm.FieldCondition(key=S.R_PRODUCT_ID, match=qm.MatchValue(value=int(product_id)))]),
            limit=1000,
            with_payload=[S.R_RATING],
        )
        ratings = [float(p.payload[S.R_RATING]) for p in points]
        if not ratings:
            return {"count": 0, "mean": 0.0, "positive_share": 0.0}
        return {
            "count": len(ratings),
            "mean": sum(ratings) / len(ratings),
            "positive_share": sum(r >= 4 for r in ratings) / len(ratings),
        }


def flatten(reviews_by_product: dict[int, list[Review]]) -> Iterable[Review]:
    for reviews in reviews_by_product.values():
        yield from reviews
