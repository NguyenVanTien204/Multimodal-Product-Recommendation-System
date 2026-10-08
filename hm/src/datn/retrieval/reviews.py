"""Review selection shared by the local indexer and the Kaggle embedding notebook.

Kept free of qdrant/torch imports so it runs anywhere polars does. Both sides call
`select_reviews` on the same parquet files, so `review_id` (position in the sorted,
de-duplicated table) is identical everywhere and embeddings computed on Kaggle can
be imported locally without re-encoding.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

REVIEW_FILES = ("train.parquet", "valid.parquet", "test.parquet", "candidate_interactions.parquet")
MIN_REVIEW_CHARS = 20
MAX_REVIEW_CHARS = 600  # what we embed and store; ~150 tokens
MAX_REVIEWS_PER_PRODUCT = 4  # most-helpful first; enough evidence per item, bounds indexing time

_COLUMNS = ["user_id", "item_id", "rating", "timestamp", "verified_purchase", "review_title", "review_text", "helpful_vote"]


def catalog_product_ids(items_path: Path) -> dict[str, int]:
    """SKU (ASIN) -> shop product id. The marketplace importer assigned
    `products.id = row position in items.parquet + 1`, so the map can be rebuilt
    without a database (needed on Kaggle)."""
    items = pl.read_parquet(items_path, columns=["item_id"])
    return {sku: i + 1 for i, sku in enumerate(items["item_id"].to_list())}


def load_reviews(data_dir: Path, catalog_skus: set[str]) -> pl.DataFrame:
    """All usable reviews for catalog items, de-duplicated and deterministically ordered.

    The four interaction files overlap (a user-item pair can appear in several),
    so we keep the latest row per pair.
    """
    frames = [pl.read_parquet(data_dir / name, columns=_COLUMNS) for name in REVIEW_FILES if (data_dir / name).exists()]
    if not frames:
        raise FileNotFoundError(f"no review parquet files under {data_dir}")
    return (
        pl.concat(frames)
        .filter(pl.col("item_id").is_in(list(catalog_skus)))
        .with_columns(
            pl.col("review_title").fill_null("").str.strip_chars(),
            pl.col("review_text").fill_null("").str.strip_chars(),
        )
        .filter(pl.col("review_text").str.len_chars() >= MIN_REVIEW_CHARS)
        .sort("timestamp", descending=True)
        .unique(subset=["user_id", "item_id"], keep="first", maintain_order=True)
        .sort(["item_id", "user_id"])
        .with_row_index("review_id")
    )


def review_stats(reviews: pl.DataFrame) -> dict[str, tuple[int, float]]:
    agg = reviews.group_by("item_id").agg(pl.len().alias("n"), pl.col("rating").mean().alias("avg"))
    return {r["item_id"]: (int(r["n"]), float(r["avg"])) for r in agg.iter_rows(named=True)}


def select_reviews(reviews: pl.DataFrame, per_product: int = MAX_REVIEWS_PER_PRODUCT) -> pl.DataFrame:
    """Keep the most community-endorsed (then most informative) reviews of each product.

    `review_id` values from `load_reviews` are preserved (they are NOT renumbered),
    so a subset selected with a different `per_product` still lines up with ids
    already stored in Qdrant.
    """
    return (
        reviews.with_columns(pl.col("review_text").str.len_chars().alias("_len"))
        .sort(["item_id", "helpful_vote", "_len", "user_id"], descending=[False, True, True, False])
        .group_by("item_id", maintain_order=True)
        .head(per_product)
        .drop("_len")
        .sort(["item_id", "user_id"])
    )


def review_text(title: str | None, text: str) -> str:
    """The exact string that is embedded for a review."""
    body = f"{title}. {text}" if title else text
    return body[:MAX_REVIEW_CHARS]
