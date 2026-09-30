from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from . import schema as S
from .filters import Hit, SearchFilters

log = logging.getLogger(__name__)

# Weighted reciprocal-rank fusion. Raw cosine scores are NOT comparable across
# named vectors (text->text is ~0.6, text->image is ~0.3 in CLIP space), so
# candidates are fused on rank and the raw scores are kept alongside for
# explanations. k=60 is the value from the original RRF paper.
RRF_K = 60
DEFAULT_WEIGHTS = {"text_query": {S.VECTOR_TEXT: 1.0, S.VECTOR_IMAGE: 0.8}, "image_query": {S.VECTOR_IMAGE: 1.0, S.VECTOR_TEXT: 0.6}}


def rrf_fuse(rankings: dict[str, list[Hit]], weights: dict[str, float], k: int = RRF_K) -> list[Hit]:
    """Fuse several ranked hit lists into one, keeping per-signal score/rank."""
    merged: dict[int, Hit] = {}
    for signal, hits in rankings.items():
        w = weights.get(signal, 1.0)
        for rank, hit in enumerate(hits, start=1):
            target = merged.setdefault(hit.product_id, Hit(hit.product_id, hit.payload))
            target.score += w / (k + rank)
            target.scores[signal] = hit.score
            target.ranks[signal] = rank
    return sorted(merged.values(), key=lambda h: h.score, reverse=True)


class HybridSearcher:
    """Multimodal hybrid retrieval over the `products` collection.

    Combines (a) dense vector search on the text and image named vectors with
    (b) hard payload filters (price/brand/category/rating/exclusions) evaluated
    inside Qdrant, so filters shrink the candidate set *before* ranking instead
    of leaving fewer than k results after a post-filter.
    """

    def __init__(
        self,
        client: QdrantClient,
        products: str = S.PRODUCTS_COLLECTION,
        reviews: str = S.REVIEWS_COLLECTION,
    ) -> None:
        self.client = client
        self.products = products
        self.reviews = reviews

    # ---- health ------------------------------------------------------------------
    def ready(self) -> dict[str, int]:
        out = {}
        for name in (self.products, self.reviews):
            try:
                out[name] = self.client.count(name, exact=False).count if self.client.collection_exists(name) else 0
            except Exception:  # noqa: BLE001 - health probes must never raise
                out[name] = -1
        return out

    # ---- vector search -----------------------------------------------------------
    def _search_vector(
        self, vector: np.ndarray, using: str, filters: SearchFilters | None, limit: int
    ) -> list[Hit]:
        flt = filters.to_qdrant() if filters else None
        res = self.client.query_points(
            self.products,
            query=vector.tolist(),
            using=using,
            query_filter=flt,
            limit=limit,
            with_payload=True,
            # The collection uses a smaller on-disk INT8 HNSW graph to keep the
            # full RAG stack within laptop RAM.  64 preserves a useful candidate
            # pool here because RRF combines two independent vector searches.
            search_params=qm.SearchParams(hnsw_ef=64),
        ).points
        return [Hit(int(p.id), dict(p.payload or {}), float(p.score)) for p in res]

    def search(
        self,
        text_vector: np.ndarray | None = None,
        image_vector: np.ndarray | None = None,
        filters: SearchFilters | None = None,
        k: int = 10,
        pool: int = 100,
    ) -> list[Hit]:
        """Top-k products for a text and/or image query vector (unit-norm, 1024-d)."""
        if text_vector is None and image_vector is None:
            raise ValueError("search() needs a text and/or image query vector")
        rankings: dict[str, list[Hit]] = {}
        weights: dict[str, float] = {}
        if text_vector is not None:
            for using, w in DEFAULT_WEIGHTS["text_query"].items():
                sig = f"text_query->{using}"
                rankings[sig] = self._search_vector(text_vector, using, filters, pool)
                weights[sig] = w
        if image_vector is not None:
            for using, w in DEFAULT_WEIGHTS["image_query"].items():
                sig = f"image_query->{using}"
                rankings[sig] = self._search_vector(image_vector, using, filters, pool)
                weights[sig] = w
        fused = rrf_fuse(rankings, weights)
        return fused[:k] if k else fused

    # ---- item-to-item ------------------------------------------------------------
    def vectors_for(self, product_ids: Sequence[int]) -> dict[int, dict[str, np.ndarray]]:
        recs = self.client.retrieve(self.products, ids=list(product_ids), with_vectors=True, with_payload=False)
        return {
            int(r.id): {k: np.asarray(v, dtype=np.float32) for k, v in (r.vector or {}).items()} for r in recs
        }

    def similar(
        self,
        product_ids: Sequence[int],
        filters: SearchFilters | None = None,
        k: int = 10,
        pool: int = 100,
    ) -> list[Hit]:
        """Products closest to the centroid of the given products (content-based)."""
        vecs = self.vectors_for(product_ids)
        if not vecs:
            return []
        exclude = tuple(set(product_ids) | set((filters.exclude_product_ids if filters else ())))
        flt = (filters or SearchFilters()).with_(exclude_product_ids=exclude)
        rankings: dict[str, list[Hit]] = {}
        for using in (S.VECTOR_IMAGE, S.VECTOR_TEXT):
            stack = [v[using] for v in vecs.values() if using in v]
            if not stack:
                continue
            centroid = np.mean(stack, axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
            rankings[f"similar->{using}"] = self._search_vector(centroid, using, flt, pool)
        fused = rrf_fuse(rankings, {name: 1.0 for name in rankings})
        return fused[:k]

    # ---- lookups -----------------------------------------------------------------
    def get_products(self, product_ids: Iterable[int]) -> dict[int, dict]:
        ids = [int(i) for i in product_ids]
        if not ids:
            return {}
        recs = self.client.retrieve(self.products, ids=ids, with_payload=True, with_vectors=False)
        return {int(r.id): dict(r.payload or {}) for r in recs}

    def products_by_sku(self, skus: Sequence[str]) -> dict[str, Hit]:
        if not skus:
            return {}
        flt = qm.Filter(must=[qm.FieldCondition(key=S.P_ITEM_ID, match=qm.MatchAny(any=list(skus)))])
        points, _ = self.client.scroll(self.products, scroll_filter=flt, limit=len(skus), with_payload=True)
        return {str(p.payload.get(S.P_ITEM_ID)): Hit(int(p.id), dict(p.payload)) for p in points}

    def top_brands(self, limit: int = 3000) -> dict[str, int]:
        """Brand -> product count, via Qdrant's facet API (>= v1.12)."""
        try:
            res = self.client.facet(self.products, key=S.P_BRAND, limit=limit)
            return {str(h.value): int(h.count) for h in res.hits}
        except Exception as exc:  # noqa: BLE001
            log.warning("brand facet unavailable: %s", exc)
            return {}

    def popular(self, filters: SearchFilters | None = None, k: int = 10) -> list[Hit]:
        """Most-reviewed products (popularity prior), used when there is no query and no history."""
        flt = filters.to_qdrant() if filters else None
        points, _ = self.client.scroll(
            self.products,
            scroll_filter=flt,
            limit=k,
            with_payload=True,
            order_by=qm.OrderBy(key=S.P_REVIEW_COUNT, direction=qm.Direction.DESC),
        )
        return [Hit(int(p.id), dict(p.payload), 1.0 / (RRF_K + i)) for i, p in enumerate(points, 1)]

    # ---- degraded mode -----------------------------------------------------------
    def lexical_search(self, text: str, filters: SearchFilters | None = None, k: int = 10) -> list[Hit]:
        """Keyword fallback over `title` when no encoder is available.

        Requires the full-text index created by the indexer. Words are AND-ed by
        Qdrant's `MatchText`; we try the full phrase and then progressively fewer
        tokens so a long natural-language query still returns something.
        """
        tokens = [t for t in text.lower().split() if len(t) > 2]
        base = filters.to_qdrant() if filters else None
        for cut in range(len(tokens), 0, -1):
            conds = [qm.FieldCondition(key=S.P_TITLE, match=qm.MatchText(text=" ".join(tokens[:cut])))]
            must = list(conds) + list(base.must or []) if base else conds
            flt = qm.Filter(must=must, must_not=base.must_not if base else None)
            points, _ = self.client.scroll(self.products, scroll_filter=flt, limit=k, with_payload=True)
            if points:
                return [Hit(int(p.id), dict(p.payload), 1.0 / (RRF_K + i)) for i, p in enumerate(points, 1)]
        return []
