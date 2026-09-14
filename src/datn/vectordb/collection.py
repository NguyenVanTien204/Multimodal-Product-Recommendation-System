from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .schema import (
    BOOL_INDEX_FIELDS,
    FLOAT_INDEX_FIELDS,
    KEYWORD_INDEX_FIELDS,
    VECTOR_IMAGE,
    VECTOR_TEXT,
)

log = logging.getLogger(__name__)


def _vector_params(config: dict) -> qm.VectorParams:
    coll = config["collection"]
    hnsw = coll.get("hnsw", {})
    return qm.VectorParams(
        size=coll["vector_size"],
        distance=qm.Distance(coll.get("distance", "Cosine")),
        on_disk=coll.get("on_disk_vectors", False),
        hnsw_config=qm.HnswConfigDiff(
            m=hnsw.get("m", 32),
            ef_construct=hnsw.get("ef_construct", 128),
        ),
    )


def ensure_collection(client: QdrantClient, config: dict, recreate: bool = False) -> str:
    """Create the `products` collection (image + text named vectors) if missing.

    Both named vectors are declared upfront even though text embeddings don't exist
    yet: Qdrant lets a point omit a named vector, so image-only points can be
    upserted now and gain a "text" vector later via a partial upsert, with no
    collection migration needed when the text pipeline finishes.
    """
    name = config["collection"]["name"]
    exists = client.collection_exists(name)
    if exists and recreate:
        log.info("Dropping existing collection %r (recreate=True)", name)
        client.delete_collection(name)
        exists = False

    if not exists:
        vectors = _vector_params(config)
        client.create_collection(
            collection_name=name,
            vectors_config={VECTOR_IMAGE: vectors, VECTOR_TEXT: vectors},
            shard_number=config["collection"].get("shard_number", 1),
            replication_factor=config["collection"].get("replication_factor", 1),
            on_disk_payload=config["collection"].get("on_disk_payload", False),
        )
        log.info("Created collection %r", name)
    else:
        log.info("Collection %r already exists, reusing it", name)

    _ensure_payload_indexes(client, name)
    return name


def _ensure_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    existing = client.get_collection(collection_name).payload_schema
    for field in KEYWORD_INDEX_FIELDS:
        if field not in existing:
            client.create_payload_index(collection_name, field, field_schema=qm.PayloadSchemaType.KEYWORD)
    for field in FLOAT_INDEX_FIELDS:
        if field not in existing:
            client.create_payload_index(collection_name, field, field_schema=qm.PayloadSchemaType.FLOAT)
    for field in BOOL_INDEX_FIELDS:
        if field not in existing:
            client.create_payload_index(collection_name, field, field_schema=qm.PayloadSchemaType.BOOL)
