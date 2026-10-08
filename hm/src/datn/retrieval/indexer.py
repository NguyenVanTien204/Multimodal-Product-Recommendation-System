from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from . import schema as S
from .encoder import JinaClipEncoder
from .reviews import (
    MAX_REVIEW_CHARS,
    MAX_REVIEWS_PER_PRODUCT,
    load_reviews,
    review_stats,
    review_text,
    select_reviews,
)

log = logging.getLogger(__name__)

_DESC_CHARS = 700
_FEAT_CHARS = 500

# The development machine runs all three 1024-d collections locally.  Keeping
# full-precision vectors and every payload resident makes Qdrant the second
# largest RAM user after Jina CLIP.  INT8 scalar quantization keeps retrieval
# accurate enough for the subsequent RRF/reranking stages while moving vectors
# and payloads out of the resident set.
_HNSW_M = 16
_HNSW_EF_CONSTRUCT = 96
_QUANTIZATION = qm.ScalarQuantization(
    scalar=qm.ScalarQuantizationConfig(
        type=qm.ScalarType.INT8,
        quantile=0.99,
        always_ram=False,
    )
)


# ---- collections -----------------------------------------------------------------
def _vec_params() -> qm.VectorParams:
    return qm.VectorParams(
        size=S.VECTOR_SIZE,
        distance=qm.Distance.COSINE,
        on_disk=True,
        quantization_config=_QUANTIZATION,
        hnsw_config=qm.HnswConfigDiff(m=_HNSW_M, ef_construct=_HNSW_EF_CONSTRUCT),
    )


def ensure_products_collection(client: QdrantClient, recreate: bool = False) -> None:
    name = S.PRODUCTS_COLLECTION
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            name,
            vectors_config={S.VECTOR_IMAGE: _vec_params(), S.VECTOR_TEXT: _vec_params()},
            on_disk_payload=True,
        )
    have = client.get_collection(name).payload_schema
    kw = (S.P_ITEM_ID, S.P_BRAND, S.P_CATEGORY_SLUG)
    for f in kw:
        if f not in have:
            client.create_payload_index(name, f, qm.PayloadSchemaType.KEYWORD)
    for f in (S.P_PRICE, S.P_AVG_RATING):
        if f not in have:
            client.create_payload_index(name, f, qm.PayloadSchemaType.FLOAT)
    for f in (S.P_CATEGORY_ID, S.P_REVIEW_COUNT):
        if f not in have:
            client.create_payload_index(name, f, qm.PayloadSchemaType.INTEGER)
    for f in (S.P_HAS_IMAGE, S.P_HAS_TEXT, S.P_IMAGE_FALLBACK, S.P_PRICE_ESTIMATED):
        if f not in have:
            client.create_payload_index(name, f, qm.PayloadSchemaType.BOOL)
    if S.P_TITLE not in have:  # keyword fallback search when the encoder is unavailable
        client.create_payload_index(
            name,
            S.P_TITLE,
            qm.TextIndexParams(type=qm.TextIndexType.TEXT, tokenizer=qm.TokenizerType.WORD, lowercase=True, min_token_len=2),
        )


def ensure_reviews_collection(client: QdrantClient, recreate: bool = False) -> None:
    name = S.REVIEWS_COLLECTION
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            name,
            vectors_config={S.VECTOR_TEXT: _vec_params()},
            on_disk_payload=True,
        )
    have = client.get_collection(name).payload_schema
    if S.R_PRODUCT_ID not in have:
        client.create_payload_index(name, S.R_PRODUCT_ID, qm.PayloadSchemaType.INTEGER)
    if S.R_RATING not in have:
        client.create_payload_index(name, S.R_RATING, qm.PayloadSchemaType.FLOAT)


# ---- source data -----------------------------------------------------------------
def load_marketplace_catalog(pg_dsn: str) -> pl.DataFrame:
    """Read the shop's own product rows (the source of truth for id/price/category)."""
    import psycopg

    query = (
        "SELECT p.id AS product_id, p.sku AS item_id, p.name, p.description AS shop_description, "
        "p.price::float8 AS price_vnd, p.image_url, p.category_id, c.slug AS category_slug, p.is_active "
        "FROM products p LEFT JOIN categories c ON c.id = p.category_id ORDER BY p.id"
    )
    with psycopg.connect(pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(query)
        cols = [d.name for d in cur.description]
        rows = cur.fetchall()
    return pl.DataFrame(rows, schema=cols, orient="row")


def _clip(value: object, limit: int) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _embedding_lookup(emb_path: Path, meta_path: Path) -> tuple[np.ndarray, dict[str, int]]:
    emb = np.load(emb_path, mmap_mode="r")
    meta = pl.read_parquet(meta_path, columns=["item_id"])
    if emb.shape[0] != meta.height:
        raise ValueError(f"{emb_path.name}: {emb.shape[0]} vectors vs {meta.height} metadata rows")
    return emb, {iid: i for i, iid in enumerate(meta["item_id"].to_list())}


def _dominant_duplicate_rows(emb: np.ndarray) -> set[int]:
    """Rows sharing the single largest byte-identical vector: the black placeholder
    used when an item photo failed to download (see multimodal_embeddings_report.md)."""
    contiguous = np.ascontiguousarray(emb)
    view = contiguous.view([("", contiguous.dtype)] * contiguous.shape[1]).reshape(-1)
    _, inverse, counts = np.unique(view, return_inverse=True, return_counts=True)
    if counts.max() <= 1:
        return set()
    return set(np.flatnonzero(counts[inverse] == counts.max()).tolist())


# ---- products --------------------------------------------------------------------
def index_products(
    client: QdrantClient,
    *,
    pg_dsn: str,
    data_dir: Path,
    recreate: bool = False,
    batch_size: int = 256,
) -> int:
    """Build the `products` collection from the shop DB + the offline Jina CLIP v2 vectors.

    Point id = Postgres `products.id`, so the gateway can hydrate hits with a plain
    primary-key lookup, and `item_id` = SKU (ASIN) is the key the recommender
    checkpoint understands.
    """
    ensure_products_collection(client, recreate=recreate)

    shop = load_marketplace_catalog(pg_dsn).filter(pl.col("is_active"))
    items = pl.read_parquet(data_dir / "items.parquet", columns=["item_id", "brand", "category", "features", "description", "price"]).rename(
        {"description": "raw_description", "price": "raw_price_usd"}
    )
    catalog = shop.join(items, on="item_id", how="left")
    log.info("catalog rows: %d (active shop products)", catalog.height)

    img, img_row = _embedding_lookup(data_dir / "embedding/image_embeddings.npy", data_dir / "embedding/image_embedding_metadata.parquet")
    txt, txt_row = _embedding_lookup(data_dir / "embedding/text_embeddings.npy", data_dir / "embedding/text_embedding_metadata.parquet")
    fallback_rows = _dominant_duplicate_rows(img)
    log.info("image-fallback (placeholder) vectors: %d", len(fallback_rows))

    stats = review_stats(load_reviews(data_dir, set(catalog["item_id"].to_list())))

    total = 0
    t0 = time.time()
    for start in range(0, catalog.height, batch_size):
        chunk = catalog.slice(start, batch_size)
        points: list[qm.PointStruct] = []
        for row in chunk.iter_rows(named=True):
            sku = row["item_id"]
            vec: dict[str, list[float]] = {}
            ir, tr = img_row.get(sku), txt_row.get(sku)
            if ir is not None:
                vec[S.VECTOR_IMAGE] = np.asarray(img[ir], dtype=np.float32).tolist()
            if tr is not None:
                vec[S.VECTOR_TEXT] = np.asarray(txt[tr], dtype=np.float32).tolist()
            if not vec:
                continue
            n_reviews, avg_rating = stats.get(sku, (0, None))
            raw_price = row.get("raw_price_usd")
            payload = {
                S.P_PRODUCT_ID: int(row["product_id"]),
                S.P_ITEM_ID: sku,
                S.P_TITLE: row["name"],
                S.P_BRAND: (row.get("brand") or None),
                S.P_CATEGORY: row.get("category"),
                S.P_CATEGORY_ID: row["category_id"],
                S.P_CATEGORY_SLUG: row["category_slug"],
                S.P_PRICE: float(row["price_vnd"]),
                S.P_PRICE_ESTIMATED: not (raw_price is not None and raw_price > 0),
                S.P_IMAGE_URL: row["image_url"] or None,
                S.P_DESCRIPTION: _clip(row.get("raw_description") or row.get("shop_description"), _DESC_CHARS),
                S.P_FEATURES: _clip(row.get("features"), _FEAT_CHARS),
                S.P_REVIEW_COUNT: n_reviews,
                S.P_AVG_RATING: round(avg_rating, 3) if avg_rating is not None else None,
                S.P_HAS_IMAGE: ir is not None,
                S.P_HAS_TEXT: tr is not None,
                S.P_IMAGE_FALLBACK: ir is not None and ir in fallback_rows,
            }
            points.append(qm.PointStruct(id=int(row["product_id"]), vector=vec, payload={k: v for k, v in payload.items() if v is not None}))
        if points:
            client.upsert(S.PRODUCTS_COLLECTION, points=points, wait=True)
            total += len(points)
        if (start // batch_size) % 40 == 0:
            log.info("products %d/%d (%.0f/s)", total, catalog.height, total / max(time.time() - t0, 1e-6))
    log.info("indexed %d products in %.1fs", total, time.time() - t0)
    return total


# ---- reviews ---------------------------------------------------------------------
def index_reviews(
    client: QdrantClient,
    encoder: JinaClipEncoder,
    *,
    pg_dsn: str,
    data_dir: Path,
    progress_file: Path,
    recreate: bool = False,
    chunk_size: int = 2048,
    limit: int | None = None,
    per_product: int = MAX_REVIEWS_PER_PRODUCT,
) -> int:
    """Embed every review with Jina CLIP v2 and upsert into `reviews` (resumable).

    Ordering is deterministic (see `load_reviews`), and `progress_file` records how
    many rows are already stored, so re-running after an interruption continues
    instead of re-encoding hours of work.
    """
    ensure_reviews_collection(client, recreate=recreate)
    shop = load_marketplace_catalog(pg_dsn).filter(pl.col("is_active")).select("item_id", "product_id")
    sku_to_pid = dict(zip(shop["item_id"].to_list(), shop["product_id"].to_list()))
    reviews = load_reviews(data_dir, set(sku_to_pid))
    reviews = select_reviews(reviews, per_product)
    if limit:
        reviews = reviews.head(limit)
    n = reviews.height

    done = 0
    if progress_file.exists() and not recreate:
        saved = json.loads(progress_file.read_text())
        if saved.get("total") == n:
            done = int(saved["done"])
    log.info("reviews to index: %d (resuming at %d)", n, done)

    resumed_from = done
    t0 = time.time()
    for start in range(done, n, chunk_size):
        chunk = reviews.slice(start, chunk_size)
        titles = chunk["review_title"].to_list()
        texts = chunk["review_text"].to_list()
        vectors = encoder.encode_passages([review_text(t, x) for t, x in zip(titles, texts)])
        points = []
        for row, vec in zip(chunk.iter_rows(named=True), vectors):
            points.append(
                qm.PointStruct(
                    id=int(row["review_id"]),
                    vector={S.VECTOR_TEXT: vec.tolist()},
                    payload={
                        S.R_PRODUCT_ID: int(sku_to_pid[row["item_id"]]),
                        S.R_ITEM_ID: row["item_id"],
                        S.R_RATING: float(row["rating"]),
                        S.R_HELPFUL: int(row["helpful_vote"] or 0),
                        S.R_VERIFIED: bool(row["verified_purchase"]),
                        S.R_TITLE: row["review_title"] or None,
                        S.R_TEXT: row["review_text"][:MAX_REVIEW_CHARS],
                    },
                )
            )
        for i in range(0, len(points), 256):  # Qdrant caps a JSON request at 32 MB
            client.upsert(S.REVIEWS_COLLECTION, points=points[i : i + 256], wait=True)
        done = start + chunk.height
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        progress_file.write_text(json.dumps({"done": done, "total": n}))
        log.info("reviews %d/%d (%.0f/s)", done, n, (done - resumed_from) / max(time.time() - t0, 1e-6))
    log.info("indexed reviews in %.1fs", time.time() - t0)
    return n


def import_reviews(
    client: QdrantClient,
    *,
    embeddings_path: Path,
    meta_path: Path,
    recreate: bool = True,
    batch_size: int = 256,
) -> int:
    """Load review vectors that were computed elsewhere (see the Kaggle notebook).

    `meta_path` is the parquet written by the notebook (review_id, product_id,
    item_id, rating, helpful_vote, verified_purchase, review_title, review_text);
    row i of it belongs to row i of the .npy. Recreates the collection by default
    so points from an earlier, partial local run cannot linger.
    """
    ensure_reviews_collection(client, recreate=recreate)
    vectors = np.load(embeddings_path, mmap_mode="r")
    meta = pl.read_parquet(meta_path)
    if vectors.shape[0] != meta.height:
        raise ValueError(f"{embeddings_path.name}: {vectors.shape[0]} vectors vs {meta.height} metadata rows")
    if vectors.shape[1] != S.VECTOR_SIZE:
        raise ValueError(f"expected {S.VECTOR_SIZE}-d vectors, got {vectors.shape[1]}")

    t0 = time.time()
    for start in range(0, meta.height, batch_size):
        chunk = meta.slice(start, batch_size)
        block = np.asarray(vectors[start : start + chunk.height], dtype=np.float32)
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        block = block / np.clip(norms, 1e-12, None)  # float16 storage -> renormalise
        points = [
            qm.PointStruct(
                id=int(row["review_id"]),
                vector={S.VECTOR_TEXT: vec.tolist()},
                payload={
                    S.R_PRODUCT_ID: int(row["product_id"]),
                    S.R_ITEM_ID: row["item_id"],
                    S.R_RATING: float(row["rating"]),
                    S.R_HELPFUL: int(row["helpful_vote"] or 0),
                    S.R_VERIFIED: bool(row["verified_purchase"]),
                    S.R_TITLE: row["review_title"] or None,
                    S.R_TEXT: row["review_text"][:MAX_REVIEW_CHARS],
                },
            )
            for row, vec in zip(chunk.iter_rows(named=True), block)
        ]
        client.upsert(S.REVIEWS_COLLECTION, points=points, wait=True)
        if (start // batch_size) % 100 == 0:
            log.info("imported %d/%d reviews (%.0f/s)", start + chunk.height, meta.height, (start + chunk.height) / max(time.time() - t0, 1e-6))
    log.info("imported %d reviews in %.1fs", meta.height, time.time() - t0)
    return meta.height
