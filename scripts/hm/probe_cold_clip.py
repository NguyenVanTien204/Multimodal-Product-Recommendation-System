"""Thăm dò cold-start THẬT SỰ trên H&M: xếp hạng riêng trong pool item chưa có giao dịch nào trước cutoff bằng CLIP.

Pool = item chưa bán trong mẫu trước ngày bắt đầu cửa sổ test. Đích = item thuộc pool được mua ở tuần test.
So sánh Hit@K (trong pool) của: độ giống CLIP với lịch sử user, prior "giống hàng mới ra gần đây", và tổng hai z-score.
Chạy:  python scripts/hm/probe_cold_clip.py [--data data/hm]   (chỉ cần polars + numpy, ~1 phút, CPU)
Kết quả đã dẫn trong docs/hm/03_experiments_and_results.md (mục E10).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/hm")
parser.add_argument("--profile-items", type=int, default=10, help="số item gần nhất dùng làm hồ sơ người dùng")
args = parser.parse_args()
D = Path(args.data)

items = pl.read_parquet(D / "items.parquet")
N = items.height
vocab = {a: i + 1 for i, a in enumerate(items["item_id"].to_list())}
nz = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)
C = nz(nz(np.load(D / "text_embeddings.npy").astype("float32")) + nz(np.load(D / "image_embeddings.npy").astype("float32")))   # hàng i-1 = item idx i

ev = {}
for name in ("train", "rerank_train", "valid", "test"):
    ev[name] = (pl.read_parquet(D / f"{name}.parquet").with_columns(pl.col("article_id").replace_strict(vocab, default=0).alias("idx"))
                .filter(pl.col("idx") > 0).sort(["customer_id", "t_dat"]))
cut = int(ev["test"]["t_dat"].cast(pl.Int32).min())
al = pl.concat([ev[n].select("idx", "t_dat") for n in ev])
idx, day = al["idx"].to_numpy(), al["t_dat"].cast(pl.Int32).to_numpy()
first = np.full(N + 1, 10 ** 9); np.minimum.at(first, idx, day)
pool = np.where(first[1:] >= cut)[0] + 1
print(f"pool cold: {len(pool):,}/{N:,} item")

hist = pl.concat([ev[n] for n in ("train", "rerank_train", "valid")]).group_by("customer_id", maintain_order=True).agg(pl.col("idx"))
hist = dict(zip(hist["customer_id"].to_list(), hist["idx"].to_list()))
poolset = set(pool.tolist())
targets: dict[str, set[int]] = {}
for u, i in zip(ev["test"]["customer_id"].to_list(), ev["test"]["idx"].to_list()):
    if i in poolset:
        targets.setdefault(u, set()).add(i)
users = [u for u in targets if hist.get(u)]
print(f"user có đích cold: {len(users)} | sự kiện: {sum(len(v) for v in targets.values())} | item cold distinct được mua: {len(set().union(*targets.values()))}")

Cp = C[pool - 1]
def launch_prior(days: int, k: int = 10) -> tuple[np.ndarray, int]:
    new = np.where((first < cut) & (first >= cut - days))[0]
    sims = Cp @ C[new - 1].T
    return np.sort(sims, axis=1)[:, -k:].mean(1), len(new)

U = np.stack([nz(C[np.asarray(hist[u][-args.profile_items:]) - 1].mean(0, keepdims=True))[0] for u in users])
user_sim = U @ Cp.T
z = lambda x: (x - x.mean(-1, keepdims=True)) / x.std(-1, keepdims=True).clip(1e-6)
pos = [set(np.where(np.isin(pool, list(targets[u])))[0].tolist()) for u in users]

def hit(score: np.ndarray, k: int) -> float:
    top = np.argpartition(-score, k, axis=1)[:, :k]
    return float(np.mean([len(set(top[i].tolist()) & pos[i]) > 0 for i in range(len(users))]))

print(f"random Hit@100 ≈ {100 / len(pool):.4f} × số đích/user")
for k in (100, 500):
    print(f"\n-- Hit@{k} trong pool cold ({len(users)} user)")
    print(f"  chỉ độ giống CLIP với lịch sử user        {hit(user_sim, k):.4f}")
    for days in (14, 28, 56):
        prior, n_new = launch_prior(days)
        prior_mat = np.tile(prior, (len(users), 1))
        print(f"  prior {days:>2}d ({n_new:>4} item mới)  chỉ prior {hit(prior_mat, k):.4f} | z(user)+z(prior) {hit(z(user_sim) + z(prior_mat), k):.4f} | 0.5·z(user)+z(prior) {hit(0.5 * z(user_sim) + z(prior_mat), k):.4f}")
