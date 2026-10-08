"""Thống kê mô tả và chẩn đoán bộ dữ liệu H&M `hm_v1` (mẫu 50k khách) — số liệu dùng cho hm/docs/01.

Chạy:  python hm/scripts/dataset_stats.py [--data data/hm] [--out data/artifacts/hm/dataset_stats.json]
Chỉ cần polars + numpy; không dùng GPU, ~1 phút.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/hm")
parser.add_argument("--out", default=None)
args = parser.parse_args()
D = Path(args.data)

items = pl.read_parquet(D / "items.parquet")
N = items.height
vocab = {a: i + 1 for i, a in enumerate(items["item_id"].to_list())}
splits = ("train", "rerank_train", "valid", "test")
ev = {s: pl.read_parquet(D / f"{s}.parquet").with_columns(pl.col("article_id").replace_strict(vocab, default=0).alias("idx")) for s in splits}
out: dict = {"catalog_items": N}

print("== Cửa sổ dữ liệu")
out["splits"] = {}
for s, d in ev.items():
    row = {"events": d.height, "users": d["customer_id"].n_unique(), "items": d["article_id"].n_unique(), "first_day": str(d["t_dat"].min()), "last_day": str(d["t_dat"].max())}
    out["splits"][s] = row
    print(f"  {s:<13}{row}")

day = {s: d["t_dat"].cast(pl.Int32).to_numpy() for s, d in ev.items()}
idx = {s: d["idx"].to_numpy() for s, d in ev.items()}
cutoff = {"rerank_train": int(day["rerank_train"].min()), "valid": int(day["valid"].min()), "test": int(day["test"].min())}
alld = np.concatenate([day[s] for s in splits]); alli = np.concatenate([idx[s] for s in splits])
first_seen = np.full(N + 1, 10 ** 9); np.minimum.at(first_seen, alli, alld)

print("\n== Cold-start theo cửa sổ (trên mẫu 50k khách)")
train_items = set(idx["train"].tolist())
out["cold"] = {}
for s in ("rerank_train", "valid", "test"):
    c = cutoff[s]
    vs_train = float(np.mean([i not in train_items for i in idx[s].tolist()]))
    never_sold = float(np.mean(first_seen[idx[s]] >= c))
    out["cold"][s] = {"events_unseen_by_tower_train": vs_train, "events_never_sold_before_window": never_sold}
    print(f"  {s:<13} sự kiện chưa có trong train: {vs_train:.3f} | chưa có giao dịch nào trước cửa sổ: {never_sold:.3f}")

print("\n== Tính thời vụ: item đích đã bán trong N ngày trước cửa sổ (mẫu 50k khách)")
out["recency"] = {}
for s in ("valid", "test"):
    c = cutoff[s]
    row = {}
    for n in (7, 14, 28, 56):
        keep = (alld < c) & (alld >= c - n)
        sold = np.zeros(N + 1, bool); sold[alli[keep]] = True
        row[f"sold_prev_{n}d"] = float(sold[idx[s]].mean())
    row["distinct_items_in_window"] = int(len(set(idx[s].tolist())))
    prev = (alld < c) & (alld >= c - 7)
    row["distinct_items_sold_prev_7d"] = int(len(set(alli[prev].tolist())))
    out["recency"][s] = row
    print(f"  {s:<6}{row}")

print("\n== Hành vi người dùng")
tr = ev["train"].group_by("customer_id").agg(pl.len().alias("n"), pl.col("t_dat").n_unique().alias("days"))
te = ev["test"].group_by("customer_id").agg(pl.len().alias("n"))
hist = {}
for u, a in ev["train"].select("customer_id", "article_id").iter_rows():
    hist.setdefault(u, set()).add(a)
repeat = float(np.mean([a in hist.get(u, ()) for u, a in ev["test"].select("customer_id", "article_id").iter_rows()]))
train_users = set(tr["customer_id"].to_list())
out["users"] = {"train_events_mean": float(tr["n"].mean()), "train_events_median": float(tr["n"].median()), "train_days_mean": float(tr["days"].mean()),
                "test_targets_mean": float(te["n"].mean()), "test_targets_median": float(te["n"].median()), "test_repeat_rate_vs_train": repeat,
                "test_users_without_train_history": float(1 - np.mean([u in train_users for u in te["customer_id"].to_list()]))}
print(" ", out["users"])

print("\n== Popularity toàn thời gian (train) trúng bao nhiêu sự kiện test")
cnt = np.bincount(idx["train"], minlength=N + 1)
out["popularity_hit_share"] = {}
for k in (12, 100, 1000):
    top = set(np.argsort(-cnt)[:k].tolist())
    out["popularity_hit_share"][k] = float(np.mean([i in top for i in idx["test"].tolist()]))
print(" ", out["popularity_hit_share"])

meta_cols = {c: float(items[c].is_null().mean()) for c in items.columns}
out["items_null_share"] = meta_cols
print("\n== Tỉ lệ null của metadata item:", {k: round(v, 4) for k, v in meta_cols.items() if v})

if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=float), encoding="utf8")
    print("\nđã ghi", args.out)
