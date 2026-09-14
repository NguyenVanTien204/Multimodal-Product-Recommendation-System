from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .schema import (
    FIELD_BRAND,
    FIELD_CATEGORY,
    FIELD_HAS_IMAGE,
    FIELD_HAS_TEXT,
    FIELD_IMAGE_URL,
    FIELD_IS_IMAGE_FALLBACK,
    FIELD_ITEM_ID,
    FIELD_PRICE,
    FIELD_TITLE,
    VECTOR_IMAGE,
    VECTOR_TEXT,
)

log = logging.getLogger(__name__)


def _load_items_index(items_path: Path) -> pl.DataFrame:
    """Read the catalog and assign each item_id its canonical point id (row position).

    This row position is the single source of truth for point ids across both
    modalities, so an image import and a later text import land on the same
    point regardless of which metadata file's own ordering is trusted.
    """
    items = pl.read_parquet(
        items_path,
        columns=["item_id", "title", "brand", "category", "price", "image_url"],
    )
    return items.with_row_index("point_id")


def _duplicate_vector_mask(embeddings: np.ndarray) -> np.ndarray:
    """Flag rows belonging to the single largest exact-duplicate vector cluster.

    A handful of catalog images fail to download and get replaced by one shared
    black placeholder before encoding, producing one dominant cluster of
    byte-identical vectors; genuinely distinct product photos essentially never
    collide. See docs/multimodal_embeddings_report.md section 2.3.D.
    """
    contiguous = np.ascontiguousarray(embeddings)
    view = contiguous.view([("", contiguous.dtype)] * contiguous.shape[1]).reshape(-1)
    _, inverse, counts = np.unique(view, return_inverse=True, return_counts=True)
    group_sizes = counts[inverse]
    dominant = counts.max()
    return (group_sizes == dominant) & (dominant > 1)


def _payload_row(row: dict) -> dict:
    payload = {FIELD_ITEM_ID: row["item_id"], FIELD_TITLE: row["title"]}
    if row["brand"]:
        payload[FIELD_BRAND] = row["brand"]
    if row["category"]:
        payload[FIELD_CATEGORY] = row["category"]
    if row["price"] is not None:
        payload[FIELD_PRICE] = row["price"]
    if row["image_url"]:
        payload[FIELD_IMAGE_URL] = row["image_url"]
    return payload


def _iter_points(
    vector_name: str,
    embeddings: np.ndarray,
    metadata: pl.DataFrame,
    items_index: pl.DataFrame,
    fallback_mask: np.ndarray | None,
) -> Iterator[qm.PointStruct]:
    item_to_row = {r["item_id"]: r for r in items_index.iter_rows(named=True)}
    skipped = 0
    for i, item_id in enumerate(metadata["item_id"]):
        row = item_to_row.get(item_id)
        if row is None:
            skipped += 1
            continue
        payload = _payload_row(row)
        if vector_name == VECTOR_IMAGE:
            payload[FIELD_HAS_IMAGE] = True
            if fallback_mask is not None:
                payload[FIELD_IS_IMAGE_FALLBACK] = bool(fallback_mask[i])
        elif vector_name == VECTOR_TEXT:
            payload[FIELD_HAS_TEXT] = True
        yield qm.PointStruct(
            id=int(row["point_id"]),
            vector={vector_name: embeddings[i].tolist()},
            payload=payload,
        )
    if skipped:
        log.warning("Skipped %d embedding rows whose item_id is not in items.parquet", skipped)


def import_embeddings(
    client: QdrantClient,
    config: dict,
    vector_name: str,
    embeddings_path: Path,
    metadata_path: Path,
    detect_fallback: bool = False,
) -> int:
    """Upsert one modality's vectors (image or text) into the shared `products` collection.

    Points are keyed by their row position in items.parquet (see _load_items_index),
    so running this once for images and later for text merges onto the same points
    via partial vector updates instead of creating duplicates.
    """
    collection = config["collection"]["name"]
    items_index = _load_items_index(Path(config["paths"]["items"]))
    metadata = pl.read_parquet(metadata_path)
    embeddings = np.load(embeddings_path, mmap_mode="r")
    if embeddings.shape[0] != metadata.shape[0]:
        raise ValueError(
            f"{embeddings_path} has {embeddings.shape[0]} rows but "
            f"{metadata_path} has {metadata.shape[0]} rows"
        )

    fallback_mask = None
    if detect_fallback:
        fallback_mask = _duplicate_vector_mask(np.asarray(embeddings))
        log.info(
            "Flagged %d/%d vectors as shared fallback duplicates",
            int(fallback_mask.sum()),
            len(fallback_mask),
        )

    points = _iter_points(vector_name, embeddings, metadata, items_index, fallback_mask)
    import_cfg = config.get("import", {})
    client.upload_points(
        collection_name=collection,
        points=points,
        batch_size=import_cfg.get("batch_size", 256),
        parallel=import_cfg.get("parallel", 2),
        wait=import_cfg.get("wait", False),
    )
    count = client.count(collection).count
    log.info("Collection %r now holds %d points", collection, count)
    return count
