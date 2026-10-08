"""Full-ranking test metrics of the ported pipeline vs hm/checkpoints/reranker_metrics.json (no notebook code involved).

    python hm/scripts/eval_port.py --users 100        # timing / smoke
    python hm/scripts/eval_port.py                    # all 2,799 test users (CPU, a few minutes)
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hm" / "src"))

from datn.recommenders.hm import HMRecommender  # noqa: E402

KS = (12, 50, 100)


def metrics(rankings: list[np.ndarray], truths: list[set[int]]) -> dict[str, float]:
    out = {f"HitRate@{k}": [] for k in KS} | {f"Recall@{k}": [] for k in KS}
    for ranked, truth in zip(rankings, truths):
        if not truth:
            continue
        for k in KS:
            found = len(truth.intersection(ranked[:k].tolist()))
            out[f"HitRate@{k}"].append(float(found > 0))
            out[f"Recall@{k}"].append(found / len(truth))
    return {"evaluated_users": len(out["HitRate@12"]), **{k: float(np.mean(v)) for k, v in out.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=0, help="0 = all test users")
    ap.add_argument("--data", type=Path, default=ROOT / "data/hm")
    ap.add_argument("--ckpt-dir", type=Path, default=ROOT / "hm/checkpoints")
    ap.add_argument("--tower", type=Path, default=None, help="default: <ckpt-dir>/best_retrieval_refit.pt (data/hm/serving/tower.pt is the slim copy)")
    args = ap.parse_args()

    items = pl.read_parquet(args.data / "items.parquet", columns=["item_id"])["item_id"].to_list()
    vocab = {x: i + 1 for i, x in enumerate(items)}

    def histories(name: str) -> dict[str, list[int]]:
        d = (pl.read_parquet(args.data / f"{name}.parquet", columns=["customer_id", "article_id", "t_dat"])
             .with_columns(pl.col("article_id").replace_strict(vocab, default=0).alias("idx")).filter(pl.col("idx") > 0)
             .sort(["customer_id", "t_dat"]))
        g = d.group_by("customer_id", maintain_order=True).agg(pl.col("idx"))
        return dict(zip(g["customer_id"].to_list(), g["idx"].to_list()))

    train, selection, valid, test = (histories(n) for n in ("train", "rerank_train", "valid", "test"))
    users = sorted(test)[: args.users or None]
    context = [train.get(u, []) + selection.get(u, []) + valid.get(u, []) for u in users]
    truths = [set(test[u]) for u in users]

    t0 = time.time()
    rec = HMRecommender.from_artifacts(
        tower_path=args.tower or args.ckpt_dir / "best_retrieval_refit.pt", ranker_path=args.ckpt_dir / "reranker_lgbm.txt",
        items_path=args.data / "items.parquet", text_path=args.data / "text_embeddings.npy", image_path=args.data / "image_embeddings.npy",
        daily_counts_path=args.data / "item_daily_counts.parquet", as_of=dt.date(2020, 9, 16), device="cpu",
    )
    print(f"loaded in {time.time() - t0:.0f}s | users: {len(users)}")

    ref = json.loads((args.ckpt_dir / "reranker_metrics.json").read_text(encoding="utf-8"))["test"]["full_ranking"]
    for cold_every, ref_key in ((0, "reranker"), (10, "reranker+cold_slot/10"), (5, "reranker+cold_slot/5")):
        t1 = time.time()
        ranked = [r for r, _ in rec.rank_many(context, cold_every=cold_every)]
        got = metrics(ranked, truths)
        want = ref[ref_key]["overall"]
        print(f"\n[{ref_key}]  {len(users)/(time.time()-t1):.1f} users/s")
        for k in ("evaluated_users", *(f"HitRate@{k}" for k in KS), *(f"Recall@{k}" for k in KS)):
            print(f"  {k:<16} port={got[k]:<10.4f} notebook={want[k]:<10.4f} diff={got[k] - want[k]:+.4f}")
        if args.users:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
