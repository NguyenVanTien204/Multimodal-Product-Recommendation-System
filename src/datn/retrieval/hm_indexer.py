"""Qdrant `products` collection for the H&M catalog (512-d image + text vectors, H&M facet payload).

Same shape as the Amazon indexer (named vectors `image`/`text`, point id = Postgres products.id) so search,
filters and the RAG gateway keep working; only the payload facets differ (no brand, real taxonomy).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from . import schema as S
from .indexer import _HNSW_EF_CONSTRUCT, _HNSW_M, _QUANTIZATION, _clip

log = logging.getLogger(__name__)
_DESC_CHARS = 700


def _vec_params() -> qm.VectorParams:
    return qm.VectorParams(
        size=S.VECTOR_SIZE,
        distance=qm.Distance.COSINE,
        on_disk=True,
        quantization_config=_QUANTIZATION,
        hnsw_config=qm.HnswConfigDiff(m=_HNSW_M, ef_construct=_HNSW_EF_CONSTRUCT),
    )


def ensure_hm_products_collection(client: QdrantClient, recreate: bool = False) -> None:
    name = S.PRODUCTS_COLLECTION
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(name, vectors_config={S.VECTOR_IMAGE: _vec_params(), S.VECTOR_TEXT: _vec_params()}, on_disk_payload=True)
    have = client.get_collection(name).payload_schema
    for f in (S.P_ITEM_ID, S.P_CATEGORY_SLUG, S.P_AUDIENCE, S.P_PRODUCT_TYPE, S.P_PRODUCT_GROUP, S.P_COLOUR, S.P_APPEARANCE, S.P_SECTION, S.P_DEPARTMENT):
        if f not in have:
            client.create_payload_index(name, f, qm.PayloadSchemaType.KEYWORD)
    for f in (S.P_PRICE, S.P_AVG_RATING):
        if f not in have:
            client.create_payload_index(name, f, qm.PayloadSchemaType.FLOAT)
    for f in (S.P_CATEGORY_ID, S.P_REVIEW_COUNT, S.P_SOLD_28D):
        if f not in have:
            client.create_payload_index(name, f, qm.PayloadSchemaType.INTEGER)
    for f in (S.P_HAS_IMAGE, S.P_HAS_TEXT, S.P_IMAGE_FALLBACK, S.P_PRICE_ESTIMATED, S.P_REVIEWS_MOCK):
        if f not in have:
            client.create_payload_index(name, f, qm.PayloadSchemaType.BOOL)
    if S.P_TITLE not in have:  # keyword fallback search when the encoder is unavailable
        client.create_payload_index(
            name, S.P_TITLE, qm.TextIndexParams(type=qm.TextIndexType.TEXT, tokenizer=qm.TokenizerType.WORD, lowercase=True, min_token_len=2)
        )


def hm_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """Qdrant payload of one shop product (`row`: a Postgres products row joined with its category slug; attributes = parsed JSON)."""
    attrs = row.get("attributes") or {}
    payload = {
        S.P_PRODUCT_ID: int(row["product_id"]),
        S.P_ITEM_ID: row["item_id"],
        S.P_TITLE: row["name"],
        S.P_CATEGORY_ID: row.get("category_id"),
        S.P_CATEGORY_SLUG: row.get("category_slug"),
        S.P_PRICE: float(row["price_vnd"]),
        S.P_PRICE_ESTIMATED: True,  # H&M prices are normalised, see datn.catalog.hm
        S.P_IMAGE_URL: row.get("image_url") or None,
        S.P_DESCRIPTION: _clip(row.get("description"), _DESC_CHARS),
        S.P_AUDIENCE: row.get("audience"),
        S.P_PRODUCT_TYPE: attrs.get("product_type") or None,
        S.P_PRODUCT_GROUP: attrs.get("product_group") or None,
        S.P_COLOUR: attrs.get("colour") or None,
        S.P_APPEARANCE: attrs.get("appearance") or None,
        S.P_SECTION: attrs.get("section") or None,
        S.P_DEPARTMENT: attrs.get("department") or None,
        S.P_SOLD_28D: int(attrs.get("sold_28d") or 0),
        S.P_REVIEW_COUNT: 0,  # filled by the (mock) review step
        S.P_REVIEWS_MOCK: True,
        S.P_HAS_IMAGE: True,
        S.P_HAS_TEXT: True,
        S.P_IMAGE_FALLBACK: False,
    }
    return {k: v for k, v in payload.items() if v is not None}


def load_shop_products(pg_dsn: str) -> pl.DataFrame:
    import psycopg

    query = (
        "SELECT p.id AS product_id, p.sku AS item_id, p.name, p.description, p.price::float8 AS price_vnd, p.image_url, "
        "p.category_id, c.slug AS category_slug, p.audience, p.attributes::text AS attributes "
        "FROM products p LEFT JOIN categories c ON c.id = p.category_id WHERE p.is_active ORDER BY p.id"
    )
    with psycopg.connect(pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(query)
        cols = [d.name for d in cur.description]
        rows = cur.fetchall()
    return pl.DataFrame(rows, schema=cols, orient="row")


def index_hm_products(client: QdrantClient, *, pg_dsn: str, serving_dir: Path, recreate: bool = False, batch_size: int = 256) -> int:
    ensure_hm_products_collection(client, recreate=recreate)
    shop = load_shop_products(pg_dsn)
    items = pl.read_parquet(serving_dir / "items.parquet", columns=["item_id"])["item_id"].to_list()
    text = np.load(serving_dir / "text_embeddings.npy", mmap_mode="r")
    image = np.load(serving_dir / "image_embeddings.npy", mmap_mode="r")
    if text.shape[1] != S.VECTOR_SIZE or image.shape[1] != S.VECTOR_SIZE:
        raise ValueError(f"embeddings are {text.shape[1]}-d, schema expects {S.VECTOR_SIZE}-d (set DATN_VECTOR_SIZE=512)")
    pids, skus = shop["product_id"].to_list(), shop["item_id"].to_list()
    if any(items[p - 1] != s for p, s in zip(pids, skus)):
        raise ValueError("Postgres products no longer line up with items.parquet (product_id != row + 1)")

    total, t0 = 0, time.time()
    for start in range(0, shop.height, batch_size):
        points = []
        for row in shop.slice(start, batch_size).iter_rows(named=True):
            row["attributes"] = json.loads(row["attributes"]) if row.get("attributes") else {}
            r = row["product_id"] - 1
            points.append(
                qm.PointStruct(
                    id=int(row["product_id"]),
                    vector={S.VECTOR_IMAGE: np.asarray(image[r], dtype=np.float32).tolist(), S.VECTOR_TEXT: np.asarray(text[r], dtype=np.float32).tolist()},
                    payload=hm_payload(row),
                )
            )
        client.upsert(S.PRODUCTS_COLLECTION, points=points, wait=True)
        total += len(points)
        if (start // batch_size) % 40 == 0:
            log.info("products %d/%d (%.0f/s)", total, shop.height, total / max(time.time() - t0, 1e-6))
    log.info("indexed %d H&M products in %.1fs", total, time.time() - t0)
    return total


# ---- mock reviews ----------------------------------------------------------------
def review_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """`reviews` payload of one borrowed review. `is_mock` is what lets the UI/answers say so honestly."""
    from .reviews import MAX_REVIEW_CHARS

    payload = {
        S.R_PRODUCT_ID: int(row["product_id"]),
        S.R_ITEM_ID: row["item_id"],
        S.R_RATING: float(row["rating"]),
        S.R_HELPFUL: int(row.get("helpful_vote") or 0),
        S.R_VERIFIED: bool(row.get("verified_purchase")),
        S.R_TITLE: row.get("review_title") or None,
        S.R_TEXT: str(row["review_text"])[:MAX_REVIEW_CHARS],
        S.R_IS_MOCK: True,
        "source_item_id": row.get("source_item_id"),
        "match_score": round(float(row.get("match_score") or 0.0), 4),
    }
    return {k: v for k, v in payload.items() if v is not None}


def product_review_stats(meta: pl.DataFrame) -> dict[int, tuple[int, float]]:
    g = meta.group_by("product_id").agg(pl.len().alias("n"), pl.col("rating").mean().alias("avg"))
    return {int(p): (int(n), round(float(a), 3)) for p, n, a in g.iter_rows()}


def import_hm_reviews(client: QdrantClient, *, serving_dir: Path, recreate: bool = True, batch_size: int = 256) -> int:
    """Load hm_reviews_meta.parquet/hm_review_embeddings.npy (see datn.catalog.mock_reviews_job) into `reviews`,
    then write review_count / avg_rating onto the matching `products` points."""
    from .indexer import ensure_reviews_collection

    ensure_reviews_collection(client, recreate=recreate)
    meta = pl.read_parquet(serving_dir / "hm_reviews_meta.parquet")
    vectors = np.load(serving_dir / "hm_review_embeddings.npy", mmap_mode="r")
    if vectors.shape[0] != meta.height or vectors.shape[1] != S.VECTOR_SIZE:
        raise ValueError(f"vectors {vectors.shape} do not fit {meta.height} reviews / {S.VECTOR_SIZE}-d schema")
    t0 = time.time()
    for start in range(0, meta.height, batch_size):
        chunk = meta.slice(start, batch_size)
        block = np.asarray(vectors[start : start + chunk.height], dtype=np.float32)
        client.upsert(
            S.REVIEWS_COLLECTION,
            points=[
                qm.PointStruct(id=int(row["review_id"]), vector={S.VECTOR_TEXT: vec.tolist()}, payload=review_payload(row))
                for row, vec in zip(chunk.iter_rows(named=True), block)
            ],
            wait=True,
        )
        if (start // batch_size) % 100 == 0:
            log.info("reviews %d/%d (%.0f/s)", start + chunk.height, meta.height, (start + chunk.height) / max(time.time() - t0, 1e-6))

    stats = product_review_stats(meta)
    ops = []
    for pid, (n, avg) in stats.items():
        ops.append(qm.SetPayloadOperation(set_payload=qm.SetPayload(payload={S.P_REVIEW_COUNT: n, S.P_AVG_RATING: avg, S.P_REVIEWS_MOCK: True}, points=[pid])))
        if len(ops) == 500:
            client.batch_update_points(S.PRODUCTS_COLLECTION, update_operations=ops, wait=True)
            ops = []
    if ops:
        client.batch_update_points(S.PRODUCTS_COLLECTION, update_operations=ops, wait=True)
    log.info("imported %d mock reviews; %d products now carry review_count/avg_rating", meta.height, len(stats))
    return meta.height
