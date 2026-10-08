"""I/O around datn.catalog.mock_reviews: read the Amazon reviews/products from Qdrant, write the H&M mock review files.

Stage 1 `candidates`: for each Amazon product that has reviews, its nearest H&M products (+ a printed similarity
profile and sample pairs, to choose the similarity floor by looking at real matches).
Stage 2 `build`: assign every acceptable review to one H&M product and write
    <serving>/hm_reviews_meta.parquet, <serving>/hm_review_embeddings.npy (512-d, row-aligned), <serving>/hm_reviews_manifest.json
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
from qdrant_client import QdrantClient

from . import mock_reviews as M

log = logging.getLogger(__name__)
DIM = 512


def load_amazon_reviews(client: QdrantClient, collection: str = "reviews", cache: Path | None = None) -> tuple[list[dict], np.ndarray]:
    """All embedded Amazon reviews (payload + the text vector cut to 512-d) in a deterministic (point id) order.
    Reading 295k points out of Qdrant takes ~8 minutes, so the result is cached next to the outputs."""
    if cache is not None and (cache / "_cache_amazon_reviews.parquet").is_file() and (cache / "_cache_amazon_review_vecs.npy").is_file():
        log.info("amazon reviews: using cache in %s", cache)
        return pl.read_parquet(cache / "_cache_amazon_reviews.parquet").to_dicts(), np.load(cache / "_cache_amazon_review_vecs.npy")
    rows: list[dict] = []
    vecs: list[np.ndarray] = []
    offset, t0 = None, time.time()
    while True:
        points, offset = client.scroll(collection, limit=2000, offset=offset, with_payload=True, with_vectors=["text"])
        for p in points:
            pl_ = p.payload or {}
            rows.append(
                {
                    "review_id": int(p.id), "source_item_id": pl_.get("item_id"), "rating": float(pl_.get("rating", 0.0)),
                    "helpful_vote": int(pl_.get("helpful_vote") or 0), "verified_purchase": bool(pl_.get("verified_purchase")),
                    "review_title": pl_.get("title"), "text": str(pl_.get("text") or ""),
                }
            )
            vecs.append(np.asarray(p.vector["text"], dtype=np.float32)[:DIM])
        if len(rows) % 20000 < 2000:
            log.info("reviews read: %d (%.0fs)", len(rows), time.time() - t0)
        if offset is None:
            break
    order = np.argsort([r["review_id"] for r in rows], kind="stable")
    rows = [rows[i] for i in order]
    mat = M.truncate_renorm(np.stack(vecs)[order], DIM)
    if cache is not None:
        pl.DataFrame(rows).write_parquet(cache / "_cache_amazon_reviews.parquet")
        np.save(cache / "_cache_amazon_review_vecs.npy", mat)
    return rows, mat


def load_amazon_products(client: QdrantClient, collection: str = "products") -> dict[str, dict]:
    out: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = client.scroll(collection, limit=5000, offset=offset, with_payload=["item_id", "title", "brand", "category"], with_vectors=False)
        for p in points:
            pl_ = p.payload or {}
            if pl_.get("item_id"):
                out[pl_["item_id"]] = {"title": pl_.get("title"), "brand": (pl_.get("brand") or "").strip(), "category": pl_.get("category")}
        if offset is None:
            return out


def _hm_side(serving: Path) -> tuple[pl.DataFrame, np.ndarray]:
    shop = pl.read_parquet(serving / "shop_catalog.parquet").filter(pl.col("active"))
    text = np.load(serving / "text_embeddings.npy", mmap_mode="r")
    rows = np.asarray(shop["product_id"].to_list()) - 1
    return shop, M.truncate_renorm(np.asarray(text[rows], dtype=np.float32), DIM)


def _amazon_side(amazon_dir: Path, asins: list[str]) -> np.ndarray:
    meta = pl.read_parquet(amazon_dir / "text_embedding_metadata.parquet", columns=["item_id"])["item_id"].to_list()
    row_of = {a: i for i, a in enumerate(meta)}
    text = np.load(amazon_dir / "text_embeddings.npy", mmap_mode="r")
    keep = [a for a in asins if a in row_of]
    mat = M.truncate_renorm(np.asarray(text[[row_of[a] for a in keep]], dtype=np.float32), DIM)
    return keep, mat


WIDE_K = 20  # candidates fetched per Amazon product before the audience constraint trims them to `keep_k`


def _matches(client: QdrantClient, reviews: list[dict], serving: Path, amazon_dir: Path):
    """(asins, idx, sim, shop): top WIDE_K H&M neighbours per reviewed Amazon product; cached (3 min of matrix products)."""
    shop, h_mat = _hm_side(serving)
    cache = serving / "_cache_matches.npz"
    if cache.is_file():
        z = np.load(cache, allow_pickle=True)
        if int(z["n_hm"]) == shop.height:
            return list(z["asins"]), z["idx"], z["sim"], shop
    asins = sorted({r["source_item_id"] for r in reviews if r["source_item_id"]})
    keep, a_mat = _amazon_side(amazon_dir, asins)
    log.info("matching %d amazon products against %d h&m products", len(keep), shop.height)
    t0 = time.time()
    idx, sim = M.top_matches(a_mat, h_mat, k=WIDE_K, chunk=512)
    log.info("matching done in %.0fs", time.time() - t0)
    np.savez(cache, asins=np.array(keep, dtype=object), idx=idx, sim=sim, n_hm=shop.height)
    return keep, idx, sim, shop


def _constrained_candidates(keep, idx, sim, shop, titles: dict[str, str | None], keep_k: int = 8):
    """asin -> [(hm_product_id, sim)] best first, restricted to H&M audiences compatible with the Amazon title."""
    pids, audience = shop["product_id"].to_list(), shop["audience"].to_list()
    out: dict[str, list[tuple[int, float]]] = {}
    constrained = 0
    for a, row, srow in zip(keep, idx, sim):
        hosts = M.allowed_audiences(titles.get(a))
        constrained += hosts is not None
        pairs = [(int(pids[j]), float(s)) for j, s in zip(row, srow) if hosts is None or audience[j] in hosts]
        out[a] = pairs[:keep_k]
    log.info("audience constraint applied to %d / %d amazon products", constrained, len(keep))
    return out


def run_candidates(client: QdrantClient, *, serving: Path, amazon_dir: Path, k: int = 5, samples_per_bin: int = 6) -> None:
    reviews, _ = load_amazon_reviews(client, cache=serving)
    keep, idx, sim, shop = _matches(client, reviews, serving, amazon_dir)
    idx, sim = idx[:, :k], sim[:, :k]

    products = load_amazon_products(client)
    names, pids = shop["name"].to_list(), shop["product_id"].to_list()
    out = pl.DataFrame(
        {
            "asin": keep,
            "amazon_title": [products.get(a, {}).get("title") for a in keep],
            "amazon_brand": [products.get(a, {}).get("brand") for a in keep],
            "hm_product_ids": [[int(pids[j]) for j in row] for row in idx],
            "hm_names": [[names[j] for j in row] for row in idx],
            "sims": [row.tolist() for row in sim],
        }
    )
    out.write_parquet(serving / "mock_candidates.parquet")

    top1 = sim[:, 0]
    print("top-1 similarity percentiles:", {q: round(float(np.quantile(top1, q / 100)), 3) for q in (5, 25, 50, 75, 95)})
    rng = np.random.default_rng(0)
    for lo, hi in ((0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)):
        sel = np.where((top1 >= lo) & (top1 < hi))[0]
        print(f"\n=== top-1 sim in [{lo}, {hi}): {len(sel)} amazon products ({len(sel) / len(keep):.1%}) ===")
        for i in rng.choice(sel, size=min(samples_per_bin, len(sel)), replace=False) if len(sel) else []:
            print(f"  {top1[i]:.3f} | AMZ: {(out['amazon_title'][int(i)] or '')[:70]!r}  ->  H&M: {out['hm_names'][int(i)][0]!r}")


def run_build(client: QdrantClient, *, serving: Path, amazon_dir: Path, min_sim: float, cap_per_item: int, k: int = 8) -> dict:
    reviews, r_vec = load_amazon_reviews(client, cache=serving)
    for r in reviews:
        r["text"] = M.clean_review_text(r["text"])
        r["review_title"] = M.clean_review_text(r["review_title"]) or None
    log.info("amazon reviews: %d", len(reviews))
    toks = [M.tokens(r["text"]) for r in reviews]
    products = load_amazon_products(client)
    brand_counts = Counter(p["brand"].strip().lower() for p in products.values() if p["brand"])
    brand_set = M.make_brand_set(brand_counts, M.document_frequency(toks))
    log.info("brand screen: %d names (of %d distinct brand strings)", len(brand_set), len(brand_counts))

    keep, idx, sim, shop = _matches(client, reviews, serving, amazon_dir)
    pids = shop["product_id"].to_list()
    candidates = _constrained_candidates(keep, idx, sim, shop, {a: p.get("title") for a, p in products.items()}, keep_k=k)
    colour = {int(p): json.loads(a).get("colour") for p, a in zip(pids, shop["attributes"].to_list())}
    sku_of = dict(zip(pids, shop["sku"].to_list()))

    row_of = {r["review_id"]: i for i, r in enumerate(reviews)}
    assigned, rejected = M.assign_reviews(reviews, candidates, colour, min_sim=min_sim, cap_per_item=cap_per_item, brand_set=brand_set)
    emb = r_vec[[row_of[a["review_id"]] for a in assigned]]
    meta = pl.DataFrame(
        {
            "review_id": [a["review_id"] for a in assigned],
            "product_id": [a["product_id"] for a in assigned],
            "item_id": [sku_of[a["product_id"]] for a in assigned],
            "rating": [a["rating"] for a in assigned],
            "helpful_vote": [a["helpful_vote"] for a in assigned],
            "verified_purchase": [a["verified_purchase"] for a in assigned],
            "review_title": [a["review_title"] for a in assigned],
            "review_text": [a["text"] for a in assigned],
            "source_item_id": [a["source_item_id"] for a in assigned],
            "match_score": [a["match_score"] for a in assigned],
        }
    )
    meta.write_parquet(serving / "hm_reviews_meta.parquet")
    np.save(serving / "hm_review_embeddings.npy", emb.astype(np.float32))

    per_item = meta.group_by("product_id").agg(pl.len().alias("n"), pl.col("rating").mean().alias("avg"))
    manifest = {
        "min_sim": min_sim, "cap_per_item": cap_per_item, "amazon_reviews": len(reviews), "assigned": len(assigned), "rejected": rejected,
        "products_with_reviews": per_item.height, "active_products": shop.height,
        "coverage": per_item.height / shop.height, "reviews_per_covered_product_mean": float(per_item["n"].mean()),
        "rating_distribution": {str(k): int(v) for k, v in meta["rating"].value_counts().sort("rating").iter_rows()},
        "match_score_quantiles": {str(q): float(meta["match_score"].quantile(q / 100)) for q in (5, 25, 50, 75, 95)},
        "brand_screen_size": len(brand_set), "dim": DIM,
    }
    (serving / "hm_reviews_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest
