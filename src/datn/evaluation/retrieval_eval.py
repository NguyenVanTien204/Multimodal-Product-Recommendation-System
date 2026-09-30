from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
from qdrant_client import QdrantClient

from ..retrieval import schema as S
from ..retrieval.encoder import JinaClipEncoder
from ..retrieval.filters import SearchFilters
from ..retrieval.search import HybridSearcher

log = logging.getLogger(__name__)


def _hit_metrics(ranks: list[int | None], ks=(1, 5, 10, 50)) -> dict[str, float]:
    n = len(ranks)
    out = {f"HR@{k}": sum(r is not None and r <= k for r in ranks) / n for k in ks}
    out["MRR@50"] = sum(1.0 / r for r in ranks if r is not None and r <= 50) / n
    return out


def review_to_product(searcher: HybridSearcher, encoder: JinaClipEncoder, client: QdrantClient, n: int, seed: int) -> dict:
    """Natural-language proxy: use a buyer's review text as the query and check that
    the reviewed product is retrieved. Compares text-only, image-only (cross-modal)
    and fused retrieval, which is the ablation for 'does multimodal fusion help?'."""
    rng = np.random.default_rng(seed)
    points, _ = client.scroll(S.REVIEWS_COLLECTION, limit=20000, with_payload=True)
    pool = [p for p in points if 60 <= len(p.payload[S.R_TEXT]) <= 400]
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    sample = [pool[i] for i in idx]
    texts = [p.payload[S.R_TEXT] for p in sample]
    targets = [int(p.payload[S.R_PRODUCT_ID]) for p in sample]
    qvecs = encoder.encode_query(texts)

    ranks: dict[str, list[int | None]] = {"text_only": [], "image_only": [], "fused": []}
    t0 = time.perf_counter()
    for vec, target in zip(qvecs, targets):
        for name, using in (("text_only", S.VECTOR_TEXT), ("image_only", S.VECTOR_IMAGE)):
            hits = searcher._search_vector(vec, using, None, 50)
            ranks[name].append(next((i for i, h in enumerate(hits, 1) if h.product_id == target), None))
        fused = searcher.search(text_vector=vec, k=50, pool=100)
        ranks["fused"].append(next((i for i, h in enumerate(fused, 1) if h.product_id == target), None))
    elapsed = time.perf_counter() - t0
    return {
        "n_queries": len(sample),
        "avg_search_ms_all_three": round(1000 * elapsed / len(sample), 1),
        **{name: _hit_metrics(r) for name, r in ranks.items()},
    }


def cross_modal_alignment(searcher: HybridSearcher, client: QdrantClient, n: int, seed: int) -> dict:
    """Item image vector as the query, its own text vector as the target (and vice-versa)."""
    rng = np.random.default_rng(seed)
    ids = rng.choice(np.arange(1, 152_087), size=n, replace=False).tolist()
    vecs = searcher.vectors_for(ids)
    res = {}
    for src, dst in ((S.VECTOR_IMAGE, S.VECTOR_TEXT), (S.VECTOR_TEXT, S.VECTOR_IMAGE)):
        ranks = []
        for pid, v in vecs.items():
            if src not in v:
                continue
            hits = searcher._search_vector(v[src], dst, None, 50)
            ranks.append(next((i for i, h in enumerate(hits, 1) if h.product_id == pid), None))
        res[f"{src}_to_{dst}"] = _hit_metrics(ranks)
    return {"n_items": len(vecs), **res}


def filter_compliance(searcher: HybridSearcher, encoder: JinaClipEncoder) -> dict:
    """Hard constraints must hold for 100% of returned products (they are enforced inside Qdrant)."""
    queries = ["black running shoes", "white dress shirt", "leather handbag", "gold necklace", "summer dress", "men's watch", "warm winter jacket"]
    qvecs = encoder.encode_query(queries)
    checks = {"max_price": 0, "price_band": 0, "brand": 0, "min_rating": 0, "total": 0}
    for vec in qvecs:
        for name, flt in (
            ("max_price", SearchFilters(max_price=400_000)),
            ("price_band", SearchFilters(min_price=300_000, max_price=700_000)),
            ("min_rating", SearchFilters(min_rating=4.5)),
        ):
            hits = searcher.search(text_vector=vec, filters=flt, k=20)
            ok = all(flt.matches(h.payload, h.product_id) for h in hits) and len(hits) > 0
            checks[name] += ok
        checks["total"] += 1
    return {**checks, "note": "counts of queries (out of total) where every returned product satisfied the constraint"}


def review_evidence_coverage(client: QdrantClient) -> dict:
    total = client.count(S.PRODUCTS_COLLECTION, exact=True).count
    with_reviews = client.count(
        S.PRODUCTS_COLLECTION,
        count_filter={"must": [{"key": S.P_REVIEW_COUNT, "range": {"gte": 1}}]},
        exact=True,
    ).count
    return {
        "products": total,
        "products_with_review_stats": with_reviews,
        "coverage": with_reviews / total,
        "review_points": client.count(S.REVIEWS_COLLECTION, exact=True).count,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="datn-eval-retrieval")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--output", type=Path, default=Path("data/artifacts/rag_eval/retrieval_eval.json"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    client = QdrantClient(url=args.qdrant_url, timeout=120)
    searcher = HybridSearcher(client)
    encoder = JinaClipEncoder(device=args.device).load()

    report = {
        "seed": args.seed,
        "encoder_device": encoder.device,
        "review_to_product": review_to_product(searcher, encoder, client, args.n, args.seed),
        "cross_modal_alignment": cross_modal_alignment(searcher, client, args.n, args.seed),
        "filter_compliance": filter_compliance(searcher, encoder),
        "coverage": review_evidence_coverage(client),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
