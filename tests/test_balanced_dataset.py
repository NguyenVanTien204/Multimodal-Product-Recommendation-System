from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from datn.data.balanced import prepare_balanced_dataset


def test_balanced_dataset_uses_positive_kcore_and_positive_targets(tmp_path: Path) -> None:
    interactions = pl.DataFrame(
        {
            "user_id": [
                "u1", "u1", "u1", "u1", "u1", "u1", "u1",
                "u2", "u2", "u2", "u2", "u2", "u2",
                "u3", "u3", "u3",
            ],
            "item_id": [
                "i1", "i2", "i3", "i4", "i5", "i6", "i7",
                "i1", "i2", "i3", "i4", "i5", "i6",
                "i1", "i8", "i9",
            ],
            "rating": [5.0, 5.0, 1.0, 4.0, 3.0, 5.0, 2.0,
                       5.0, 4.0, 2.0, 5.0, 5.0, 4.0,
                       5.0, 5.0, 5.0],
            "timestamp": [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 1, 2, 3],
            "verified_purchase": [True] * 16,
            "is_positive": [1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
        }
    )
    source = tmp_path / "events.parquet"
    interactions.write_parquet(source)
    items = pl.DataFrame(
        {
            "item_id": [f"i{i}" for i in range(1, 10)],
            "title": [f"Item {i}" for i in range(1, 10)],
        }
    )
    items_path = tmp_path / "items.parquet"
    items.write_parquet(items_path)

    output = tmp_path / "balanced_v1"
    result = prepare_balanced_dataset(
        {
            "paths": {
                "interactions": [str(source)],
                "items": str(items_path),
                "output_dir": str(output),
            },
            "dataset": {
                "name": "test-balanced",
                "min_user_positive_degree": 4,
                "min_item_positive_degree": 2,
                "positive_rating": 4.0,
                "strong_negative_max_rating": 2.0,
                "verified_only": True,
            },
            "resources": {"memory_limit": "256MB", "threads": 1},
        }
    )

    assert result.manifest["positive_core"] == {
        "rows": 8,
        "users": 2,
        "items": 4,
        "item_degree_min": 2,
        "item_degree_median": 2.0,
        "item_degree_mean": 2.0,
        "item_degree_max": 2,
    }
    assert set(pl.read_parquet(output / "items.parquet")["item_id"]) == {"i1", "i2", "i4", "i6"}

    train = pl.read_parquet(output / "train.parquet")
    valid = pl.read_parquet(output / "valid.parquet")
    test = pl.read_parquet(output / "test.parquet")
    assert train.filter(pl.col("rating") <= 2).height == 0
    assert valid.select("item_id").to_series().to_list() == ["i4", "i4"]
    assert test.select("item_id").to_series().to_list() == ["i6", "i6"]
    assert valid.filter(pl.col("rating") < 4).is_empty()
    assert test.filter(pl.col("rating") < 4).is_empty()
    assert all(value == 0 for value in result.manifest["checks"].values())


def test_balanced_dataset_refuses_to_overwrite_version(tmp_path: Path) -> None:
    source = tmp_path / "events.parquet"
    pl.DataFrame(
        {
            "user_id": ["u1"],
            "item_id": ["i1"],
            "rating": [5.0],
            "timestamp": [1],
            "verified_purchase": [True],
            "is_positive": [1],
        }
    ).write_parquet(source)
    items = tmp_path / "items.parquet"
    pl.DataFrame({"item_id": ["i1"]}).write_parquet(items)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        prepare_balanced_dataset(
            {
                "paths": {
                    "interactions": [str(source)],
                    "items": str(items),
                    "output_dir": str(output),
                }
            }
        )
