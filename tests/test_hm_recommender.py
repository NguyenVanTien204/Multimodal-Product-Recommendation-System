"""HMRecommender on a tiny synthetic catalog (no real checkpoint, no GPU)."""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest
import torch

lgb = pytest.importorskip("lightgbm")

from datn.recommenders.hm import FEATURES_V2, SOURCES, HMConfig, HMRecommender, date_to_day, interleave_cold  # noqa: E402

N, D_MODEL, DIM = 300, 16, 8
AS_OF = dt.date(2020, 9, 16)
CFG = HMConfig(tower_k=40, recent_k=20, repeat_k=10, new_k=10, content_k=20, popular_k=10, cold_k=10, prior_days=56)


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    d = tmp_path_factory.mktemp("hm")
    rng = np.random.default_rng(0)
    ids = [f"{108000000 + i:010d}" for i in range(N)]
    types = rng.choice(["Trousers", "Dress", "Sweater"], N).tolist()
    pl.DataFrame({
        "item_id": ids, "product_type_name": types, "department_name": rng.choice(["A", "B"], N).tolist(),
        "colour_group_name": rng.choice(["Black", "White", "Red"], N).tolist(),
    }).write_parquet(d / "items.parquet")
    np.save(d / "text.npy", (t := rng.normal(size=(N, DIM))).astype("float32") / np.linalg.norm(t, axis=1, keepdims=True))
    np.save(d / "image.npy", (m := rng.normal(size=(N, DIM))).astype("float32") / np.linalg.norm(m, axis=1, keepdims=True))

    degree = np.zeros(N + 1)
    degree[1:241] = rng.integers(1, 50, 240)   # items 241..300 are tower-cold (degree 0)
    ckpt = {
        "state": {
            "image_gate": torch.tensor(0.0), "id_gate": torch.tensor(-2.0), "id_keep": torch.ones(N + 1, 1),
            "degree": torch.tensor(degree), "id.weight": torch.randn(N + 1, D_MODEL),
            "text_proj.weight": torch.randn(D_MODEL, DIM), "image_proj.weight": torch.randn(D_MODEL, DIM),
        },
        "item_ids": ids, "degree": degree, "modality": "multimodal",
        "serving": {"active_days": 0, "cold_bonus": 0.2, "pop_weight": 0.1},
    }
    torch.save(ckpt, d / "tower.pt")

    # daily counts: items 1..240 sold over time; 241..270 first sold after the clock; 271..300 never sold
    start = date_to_day(AS_OF) - 120
    rows = []
    for i in range(240):
        for day in rng.integers(start, date_to_day(AS_OF) + 6, 8):
            rows.append((ids[i], dt.date(1970, 1, 1) + dt.timedelta(days=int(day)), int(rng.integers(1, 5)), float(rng.random() * 0.1)))
    for i in range(240, 270):
        rows.append((ids[i], AS_OF + dt.timedelta(days=2), 3, 0.05))
    pl.DataFrame(rows, schema=["article_id", "t_dat", "n", "price_sum"], orient="row").write_parquet(d / "daily.parquet")

    x = rng.normal(size=(2000, len(FEATURES_V2))).astype("float32")
    y = (x[:, 0] + 0.5 * x[:, 3] + rng.normal(scale=0.5, size=2000) > 1).astype(int)
    booster = lgb.train({"objective": "lambdarank", "verbose": -1, "min_data_in_leaf": 5}, lgb.Dataset(x, label=y, group=[100] * 20), num_boost_round=10)
    booster.save_model(str(d / "ranker.txt"))
    return d, ids


@pytest.fixture(scope="module")
def rec(artifacts):
    d, _ = artifacts
    return HMRecommender.from_artifacts(
        tower_path=d / "tower.pt", ranker_path=d / "ranker.txt", items_path=d / "items.parquet", text_path=d / "text.npy",
        image_path=d / "image.npy", daily_counts_path=d / "daily.parquet", as_of=AS_OF, config=CFG,
    )


def test_stats_use_only_the_past(rec):
    st = rec.stats
    assert st["pop_all"][241:271].sum() == 0               # sold only after the clock -> unseen
    assert set(range(241, 301)) <= set(st["cold_pool_t"].tolist())
    assert st["seen"][1] == 1.0 and st["seen"][250] == 0.0
    assert st["age"][250] == -1


def test_features_have_24_columns_and_union_is_deduped(rec):
    g = rec.candidates_and_features([[5, 9, 12], []])
    for one in g:
        assert one["x"].shape == (len(one["items"]), len(FEATURES_V2))
        assert len(set(one["items"].tolist())) == len(one["items"]) and 0 not in one["items"]
        assert np.isfinite(one["x"]).all()
        assert set(one["sources"]) == set(SOURCES)
    cols = {n: i for i, n in enumerate(FEATURES_V2)}
    first = g[0]
    repeat_rows = np.isin(first["items"], [5, 9, 12])
    assert (first["x"][repeat_rows, cols["repeat_count"]] > 0).all()
    assert (first["x"][~repeat_rows, cols["repeat_count"]] == 0).all()
    cold_rows = first["items"] > 240
    assert (first["x"][cold_rows, cols["is_cold"]] == 1).all()


def test_recommend_excludes_history_and_maps_back_to_article_ids(rec, artifacts):
    _, ids = artifacts
    hist = [ids[4], ids[8], "not-an-item"]
    out = rec.recommend(hist, k=10)
    got = [r.item_id for r in out.recommendations]
    assert len(got) == 10 and ids[4] not in got and ids[8] not in got
    assert all(g in set(ids) for g in got)
    assert [r.score for r in out.recommendations] == sorted([r.score for r in out.recommendations], reverse=True)
    assert rec.recommend(hist, k=10, exclude_history=False).recommendations  # repeat purchase allowed when asked


def test_empty_history_still_ranks_by_recent_sales(rec):
    out = rec.recommend([], k=8)
    assert len(out.recommendations) == 8 and out.source == "hm_reranker_no_history"


def test_allowed_mask_filters_items_without_image(rec, artifacts):
    _, ids = artifacts
    allowed = np.ones(N + 1, dtype=bool)
    allowed[:150] = False
    out = rec.recommend([ids[3]], k=10, allowed=allowed)
    assert out.recommendations and all(int(r.item_id) - 108000000 + 1 > 150 for r in out.recommendations)


def test_cold_slots_every_n(rec):
    ranked, _ = rec.rank_many([[3, 4]], cold_every=5)[0]
    cold = set(rec.candidates_and_features([[3, 4]])[0]["cold_list"])
    positions = [i + 1 for i, item in enumerate(ranked.tolist()) if item in cold]
    assert positions[:3] == [5, 10, 15]


def test_interleave_cold_helper():
    assert interleave_cold([1, 2, 3, 4, 5, 6], [9, 8], every=3) == [1, 2, 9, 3, 4, 8, 5, 6]
