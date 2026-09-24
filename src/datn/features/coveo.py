from __future__ import annotations

import ast
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from datn.data.coveo import sha256


def _vector(value: str | None, dim: int) -> np.ndarray | None:
    if not value:
        return None
    try:
        data = np.asarray(ast.literal_eval(value), dtype=np.float32)
    except (SyntaxError, ValueError, TypeError):
        return None
    return data if data.shape == (dim,) and np.isfinite(data).all() else None


def _normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return np.divide(array, norms, out=np.zeros_like(array), where=norms > 0)


def build_coveo_embeddings(
    catalog_csv: Path,
    items_parquet: Path,
    output_dir: Path,
    *,
    vector_dim: int = 50,
) -> dict[str, object]:
    """Parse supplied Coveo vectors, align to item_idx, normalize and checkpoint."""
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Embedding version already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    try:
        item_ids = pl.read_parquet(items_parquet, columns=["item_id"])["item_id"].to_list()
        lookup = {item_id: index + 1 for index, item_id in enumerate(item_ids)}
        text = np.zeros((len(item_ids) + 1, vector_dim), dtype=np.float32)
        image = np.zeros_like(text)
        text_mask = np.zeros(len(item_ids) + 1, dtype=np.uint8)
        image_mask = np.zeros_like(text_mask)
        with catalog_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                index = lookup.get(row["product_sku_hash"])
                if index is None:
                    continue
                value = _vector(row.get("description_vector"), vector_dim)
                if value is not None:
                    text[index], text_mask[index] = value, 1
                value = _vector(row.get("image_vector"), vector_dim)
                if value is not None:
                    image[index], image_mask[index] = value, 1
        text, image = _normalize_rows(text), _normalize_rows(image)
        np.save(output_dir / "text_embeddings.npy", text)
        np.save(output_dir / "image_embeddings.npy", image)
        pl.DataFrame({
            "item_id": ["__PAD__", *item_ids],
            "item_idx": np.arange(len(item_ids) + 1),
            "has_text": text_mask,
            "has_image": image_mask,
        }).write_parquet(output_dir / "embedding_metadata.parquet")
        report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "vector_dim": vector_dim,
            "normalized": True,
            "rows": len(item_ids) + 1,
            "text_coverage": float(text_mask[1:].mean()) if item_ids else 0.0,
            "image_coverage": float(image_mask[1:].mean()) if item_ids else 0.0,
            "files": {},
        }
        for name in ("text_embeddings.npy", "image_embeddings.npy", "embedding_metadata.parquet"):
            path = output_dir / name
            report["files"][name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        (output_dir / "embedding_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    except Exception:
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
