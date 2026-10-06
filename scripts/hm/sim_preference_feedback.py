"""Does remembering chat feedback improve recommendations? Offline simulation on the H&M test users.

Each simulated shopper starts from their real purchase history, is shown 12 items per round and reacts
using their held-out future purchases as ground truth (see datn/recommenders/hm/feedback_sim.py).
Arms: none (feedback ignored) | like (liked items join the tower history) | like+dislike (+ soft penalty).

    python scripts/hm/sim_preference_feedback.py --users 300                  # ~1 min on CPU after loading
    python scripts/hm/sim_preference_feedback.py --users 2799 --out reports/hm/feedback_sim.json

Needs <data>/{train,rerank_train,valid,test}.parquet and the serving bundle <data>/serving (the same one the
recommender service loads). Memory: a few GB for the embeddings; on the 8 GB VM stop the rag container first.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datn.agent.preferences import NEIGHBOUR_POOL  # noqa: E402
from datn.recommenders.hm import HMRecommender  # noqa: E402
from datn.recommenders.hm.feedback_sim import ARMS, SimUser, paired_bootstrap, simulate  # noqa: E402


def load_users(bundle: Path, data: Path, n_users: int, seed: int) -> list[SimUser]:
    items = pl.read_parquet(bundle / "items.parquet", columns=["item_id"])["item_id"].to_list()
    vocab = {x: i + 1 for i, x in enumerate(items)}

    def histories(name: str) -> dict[str, list[int]]:
        d = (pl.read_parquet(data / f"{name}.parquet", columns=["customer_id", "article_id", "t_dat"])
             .with_columns(pl.col("article_id").replace_strict(vocab, default=0).alias("idx")).filter(pl.col("idx") > 0)
             .sort(["customer_id", "t_dat"]))
        g = d.group_by("customer_id", maintain_order=True).agg(pl.col("idx"))
        return dict(zip(g["customer_id"].to_list(), g["idx"].to_list()))

    train, selection, valid, test = (histories(n) for n in ("train", "rerank_train", "valid", "test"))
    eligible = sorted(u for u in test if test[u])
    if n_users and n_users < len(eligible):
        rng = np.random.default_rng(seed)
        eligible = sorted(rng.choice(eligible, size=n_users, replace=False).tolist())
    return [SimUser(tuple(train.get(u, []) + selection.get(u, []) + valid.get(u, [])), frozenset(test[u])) for u in eligible]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=300, help="0 = all test users")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--k", type=int, default=12, help="items shown per round")
    ap.add_argument("--p-like", type=float, default=0.8, help="chance a shown true item is liked")
    ap.add_argument("--p-dislike", type=float, default=0.3, help="chance a shown non-purchased item is disliked")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", type=Path, default=ROOT / "data/hm", help="train/rerank_train/valid/test.parquet (user histories)")
    ap.add_argument("--serving", type=Path, default=None, help="model bundle (tower.pt, reranker_lgbm.txt, items, embeddings); default <data>/serving")
    ap.add_argument("--out", type=Path, default=None, help="optional JSON dump of the aggregated results")
    args = ap.parse_args()
    bundle = args.serving or args.data / "serving"

    users = load_users(bundle, args.data, args.users, args.seed)
    t0 = time.time()
    rec = HMRecommender.from_artifacts(
        tower_path=bundle / "tower.pt", ranker_path=bundle / "reranker_lgbm.txt", items_path=bundle / "items.parquet",
        text_path=bundle / "text_embeddings.npy", image_path=bundle / "image_embeddings.npy",
        daily_counts_path=bundle / "item_daily_counts.parquet", as_of=dt.date(2020, 9, 16), device="cpu",
    )
    print(f"loaded in {time.time() - t0:.0f}s | users: {len(users)} | rounds: {args.rounds} x {args.k} items")

    content = rec.content_t  # (n_items+1, d), unit rows: the same CLIP content space the shop's Qdrant neighbours come from

    def neighbours(item: int) -> dict[int, int]:
        sims = content @ content[item]
        sims[0] = sims[item] = -torch.inf
        return {int(i): r for r, i in enumerate(torch.topk(sims, NEIGHBOUR_POOL).indices.tolist(), start=1)}

    def rank(histories: list[list[int]], excludes: list[set[int]]) -> list[list[int]]:
        return [r.tolist() for r, _ in rec.rank_many(histories, exclude=excludes, cold_every=0)]

    t1 = time.time()
    res = simulate(users, rank, neighbours, rounds=args.rounds, k=args.k, p_like=args.p_like, p_dislike=args.p_dislike, seed=args.seed)
    print(f"simulated in {time.time() - t1:.0f}s\n")

    rounds = range(args.rounds)
    print(f"{'arm':<14}" + "".join(f"  recall@r{r + 1:<3}" for r in rounds) + "   hit@last   likes/user  dislikes/user")
    for arm in ARMS:
        print(f"{arm:<14}" + "".join(f"  {res.recall[arm][:, r].mean():<10.4f}" for r in rounds)
              + f"   {res.hit[arm][:, -1].mean():<8.4f}   {res.shown_likes[arm][:, -1].mean():<10.2f}  {res.shown_dislikes[arm][:, -1].mean():.2f}")

    print("\npaired differences in recall (95% bootstrap CI over users)")
    summary: dict[str, dict] = {}
    for a, b in (("none", "like"), ("like", "like+dislike"), ("none", "like+dislike")):
        row = []
        for r in rounds:
            mean, lo, hi = paired_bootstrap(res.recall[a][:, r], res.recall[b][:, r], seed=args.seed)
            row.append({"round": r + 1, "mean": mean, "lo": lo, "hi": hi})
        summary[f"{b} - {a}"] = {"recall": row}
        print(f"  {b} - {a:<13}" + "".join(f"  r{x['round']}: {x['mean']:+.4f} [{x['lo']:+.4f},{x['hi']:+.4f}]" for x in row[1:]))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "users": len(users),
            "recall": {a: res.recall[a].mean(axis=0).tolist() for a in ARMS},
            "hit": {a: res.hit[a].mean(axis=0).tolist() for a in ARMS},
            "paired": summary,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
