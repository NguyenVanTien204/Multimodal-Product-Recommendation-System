from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from datn.data.coveo import CoveoPrepareConfig, prepare_coveo_dataset
from datn.data.coveo_dense import DenseSliceConfig, build_dense_coveo_slice
from datn.features.coveo import build_coveo_embeddings
from datn.recommenders.coveo.model import ActionAwareTwoTower, ResidualListwiseReranker
from datn.recommenders.coveo.pipeline import (
    RerankerTrainConfig,
    RetrievalTrainConfig,
    generate_candidates,
    train_reranker,
    train_retrieval,
)


def _write_fixture(root: Path) -> None:
    root.mkdir()
    items = [f"sku-{i}" for i in range(8)]
    with (root / "browsing_train.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["session_id_hash", "event_type", "product_action", "product_sku_hash", "server_timestamp_epoch_ms", "hashed_url"])
        for session in range(30):
            writer.writerow([f"s-{session:02d}", "event_product", "detail", items[session % 8], session * 100 + 1, "url"])
            writer.writerow([f"s-{session:02d}", "event_product", "purchase", items[(session + 1) % 8], session * 100 + 2, "url"])
    with (root / "search_train.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["session_id_hash", "query_vector", "clicked_skus_hash", "product_skus_hash", "server_timestamp_epoch_ms"])
        writer.writerow(["s-00", str([0.0] * 50), "['sku-2']", "['sku-2']", 3])
    with (root / "sku_to_content.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["product_sku_hash", "description_vector", "category_hash", "image_vector", "price_bucket"])
        for index, item in enumerate(items):
            vector = [float(index + 1)] + [0.0] * 49
            writer.writerow([item, str(vector), f"root/cat-{index % 2}", str(vector), float(index % 10)])


def _write_dense_fixture(root: Path) -> dict[str, Path]:
    root.mkdir(); day_ms = 86_400_000
    active = [f"active-{i}" for i in range(10)]; sparse = [f"sparse-{i}" for i in range(5)]
    with (root / "browsing_train.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["session_id_hash", "event_type", "product_action", "product_sku_hash", "server_timestamp_epoch_ms", "hashed_url"])
        for day in range(30):
            for session in range(4):
                sid = f"active-{day:02d}-{session}"
                for offset in range(4):
                    action = "purchase" if offset == 3 else "detail"
                    writer.writerow([sid, "event_product", action, active[(day + session + offset) % 10], day * day_ms + session * 10_000 + offset, "url"])
            sid = f"sparse-{day:02d}"
            for offset in range(3):
                writer.writerow([sid, "event_product", "detail", sparse[(day + offset) % 5], day * day_ms + 50_000 + offset, "url"])
    with (root / "search_train.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["session_id_hash", "query_vector", "clicked_skus_hash", "product_skus_hash", "server_timestamp_epoch_ms"])
    with (root / "sku_to_content.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["product_sku_hash", "description_vector", "category_hash", "image_vector", "price_bucket"])
        for item in active:
            writer.writerow([item, str([1.0] + [0.0] * 49), "root-a/sub-a", str([1.0] + [0.0] * 49), 2.0])
        for item in sparse:
            writer.writerow([item, str([0.0, 1.0] + [0.0] * 48), "root-b/sub-b", str([0.0, 1.0] + [0.0] * 48), 6.0])
    return {name: root / name for name in ("browsing_train.csv", "search_train.csv", "sku_to_content.csv")}


def test_prepare_embeddings_and_models(tmp_path: Path) -> None:
    raw = tmp_path / "raw"; _write_fixture(raw)
    processed = tmp_path / "processed"
    manifest = prepare_coveo_dataset(raw, processed, CoveoPrepareConfig(max_sessions=None))
    assert manifest["counts"]["train"]["sessions"] > 0
    assert manifest["counts"]["valid"]["sessions"] > 0
    assert manifest["counts"]["test"]["sessions"] > 0
    ranges = manifest["split_time_ranges"]
    assert ranges[0]["max_ts"] <= ranges[1]["max_ts"] <= ranges[2]["max_ts"]
    for split in ("train", "valid", "test"):
        frame = pl.read_parquet(processed / f"{split}.parquet")
        assert {"user_id", "item_id", "action", "action_weight", "event_order"} <= set(frame.columns)

    embedding_dir = tmp_path / "embeddings"
    report = build_coveo_embeddings(raw / "sku_to_content.csv", processed / "items.parquet", embedding_dir)
    assert report["text_coverage"] == pytest.approx(1.0)
    text = np.load(embedding_dir / "text_embeddings.npy")
    assert np.linalg.norm(text[1:], axis=1).tolist() == pytest.approx([1.0] * 8)

    retriever = ActionAwareTwoTower(9, 4, d_model=8, n_heads=2, n_layers=1, text_matrix=text, image_matrix=text)
    query = retriever.encode(torch.tensor([[1, 2, 0, 0]]), torch.tensor([[1, 4, 0, 0]]))
    assert query.shape == (1, 8)
    assert torch.isfinite(query).all()
    ranker = ResidualListwiseReranker(8, hidden=16)
    scores = ranker(torch.randn(5, 8), torch.randn(5))
    assert scores.shape == (5,)


def test_processed_version_is_immutable(tmp_path: Path) -> None:
    raw = tmp_path / "raw"; _write_fixture(raw); processed = tmp_path / "processed"
    prepare_coveo_dataset(raw, processed, CoveoPrepareConfig())
    with pytest.raises(FileExistsError):
        prepare_coveo_dataset(raw, processed, CoveoPrepareConfig())


def test_tiny_two_stage_training(tmp_path: Path) -> None:
    raw = tmp_path / "raw"; _write_fixture(raw)
    processed = tmp_path / "processed"
    prepare_coveo_dataset(raw, processed, CoveoPrepareConfig())
    embeddings = tmp_path / "embeddings"
    build_coveo_embeddings(raw / "sku_to_content.csv", processed / "items.parquet", embeddings)
    retrieval_dir = tmp_path / "retrieval"
    retrieval_metrics = train_retrieval(processed, embeddings, retrieval_dir, RetrievalTrainConfig(
        max_seq_len=4, d_model=8, n_heads=2, n_layers=1, batch_size=8,
        epochs=1, patience=1, device="cpu",
    ))
    assert (retrieval_dir / "best_retrieval.pt").is_file()
    assert retrieval_metrics["test"]["n_sessions"] > 0

    reranker_dir = tmp_path / "reranker"; candidate_dir = reranker_dir / "candidates"
    rr_cfg = RerankerTrainConfig(candidate_k=8, retrieval_k=5, popularity_k=3,
        max_sessions_per_split=None, batch_size=4, epochs=1, patience=1, device="cpu")
    for split in ("train", "valid", "test"):
        report = generate_candidates(split, processed, embeddings, retrieval_dir / "best_retrieval.pt",
                                     candidate_dir / f"{split}.parquet", rr_cfg)
        assert report["rows"] > 0
    metrics = train_reranker(candidate_dir / "train.parquet", candidate_dir / "valid.parquet",
                             candidate_dir / "test.parquet", reranker_dir, rr_cfg)
    assert (reranker_dir / "best_reranker.pt").is_file()
    assert metrics["test"]["conditional_sessions"] > 0


def test_dense_temporal_slice_keeps_coherent_active_catalog(tmp_path: Path) -> None:
    source = _write_dense_fixture(tmp_path / "raw")
    output = tmp_path / "dense"
    manifest = build_dense_coveo_slice(source, output, DenseSliceConfig(
        window_days_min=21, window_days_max=28, target_events=400,
        target_events_min=300, target_events_max=500, min_session_events=3,
        max_session_events=50, min_item_events=5, include_search_clicks=False,
    ))
    assert manifest["random_or_hash_sampling"] is False
    assert manifest["selected_window"]["category_node"].startswith("root-a")
    assert manifest["graph"]["events"] >= 300
    items = pl.read_parquet(output / "items.parquet")
    assert items.height == 10
    assert items["category_path"].str.starts_with("root-a").all()
    session_sets = [set(pl.read_parquet(output / f"{split}.parquet")["user_id"].to_list()) for split in ("train", "valid", "test")]
    assert session_sets[0].isdisjoint(session_sets[1])
    assert session_sets[0].isdisjoint(session_sets[2])
    assert session_sets[1].isdisjoint(session_sets[2])
