from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .dataset import ItemVocab

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContentSource:
    name: str
    embeddings_path: Path
    metadata_path: Path


def load_content_matrix(vocab: ItemVocab, sources: list[ContentSource]) -> np.ndarray:
    """Build a (vocab.vocab_size, D) content matrix aligned to `vocab`'s index space.

    D is the concatenation of every enabled source's embedding dimension (e.g. 1024
    for image-only, 2048 once text embeddings are also enabled). Row 0 (PAD) and any
    catalog item missing from a source's metadata stay zero for that source's slice --
    this is precisely what lets a true cold-start item (no interaction history at all)
    still receive a meaningful item vector in UserTower: its trainable id-residual is
    zero-initialized (see model.py), so at init, and for as long as it stays unseen,
    its item vector *is* its CLIP content projection, not a random/untrained embedding.
    """
    if not sources:
        return np.zeros((vocab.vocab_size, 0), dtype=np.float32)

    blocks: list[np.ndarray] = []
    for source in sources:
        vectors = np.load(source.embeddings_path)
        item_ids = pl.read_parquet(source.metadata_path, columns=["item_id"])["item_id"].to_list()
        if len(item_ids) != vectors.shape[0]:
            raise ValueError(
                f"content source '{source.name}': metadata rows ({len(item_ids)}) != "
                f"embedding rows ({vectors.shape[0]})"
            )
        block = np.zeros((vocab.vocab_size, vectors.shape[1]), dtype=np.float32)
        matched = 0
        for row_idx, item_id in enumerate(item_ids):
            idx = vocab.item2idx.get(item_id)
            if idx is not None:
                block[idx] = vectors[row_idx]
                matched += 1
        if matched < vocab.num_items:
            missing = vocab.num_items - matched
            # Expected to be 0 in this project (image/text extraction covers the full
            # catalog), but never silently ship a content-aware model with holes.
            logger.warning(
                "content source '%s': %d/%d catalog items have no vector (fall back to zero)",
                source.name,
                missing,
                vocab.num_items,
            )
        blocks.append(block)

    return np.concatenate(blocks, axis=1)
