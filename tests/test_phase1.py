from __future__ import annotations

import gzip
import json
from pathlib import Path

import polars as pl

from datn.data.pipeline import run


def _jsonl_gz(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")
        output.write("{broken\n")


def test_pipeline_is_streamed_and_reports_join_quality(tmp_path: Path) -> None:
    reviews, metadata = tmp_path / "reviews.jsonl.gz", tmp_path / "meta.jsonl.gz"
    _jsonl_gz(reviews, [
        {"user_id": "u1", "parent_asin": "p1", "rating": 5, "timestamp": 1, "title": "good"},
        {"user_id": "u2", "parent_asin": "missing", "rating": 4, "timestamp": 2},
    ])
    _jsonl_gz(metadata, [{
        "parent_asin": "p1", "title": "Shoe", "description": ["Light", "shoe"],
        "features": ["black"], "categories": ["Fashion", "Shoes"],
        "images": [{"large": "https://example.test/p1.jpg"}], "price": 12.5,
    }])
    config = {
        "paths": {
            "reviews": str(reviews), "metadata": str(metadata),
            "interactions_output": str(tmp_path / "interactions.parquet"),
            "items_output": str(tmp_path / "items.parquet"), "report": str(tmp_path / "report.md"),
        },
        "resources": {
            "max_memory_gb": 0.125, "reserve_gb": 0.01, "min_budget_mb": 16,
            "min_batch_rows": 1, "initial_batch_rows": 1,
        },
        "pipeline": {},
    }
    _, stats = run(config)
    assert stats["interactions"].malformed_rows == 1
    assert pl.read_parquet(tmp_path / "interactions.parquet").height == 2
    assert pl.read_parquet(tmp_path / "items.parquet")["category"].item() == "Shoes"
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Interactions without matching metadata | 1" in report
    assert "installed RAM is not treated as usable RAM" in report
