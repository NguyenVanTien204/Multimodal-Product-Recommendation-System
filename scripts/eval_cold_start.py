"""Why is the User Tower / Reranker blind to cold items?  (cold = item with 0 positive train interactions)

Measures, on the balanced dataset:
  1. how much of the eval actually IS cold-start (targets, catalog);
  2. cold-target rank: over the whole catalog vs inside the cold pool only (does the content pathway rank cold items sensibly?);
  3. systematic score bias of the tower against cold items and how many cold items ever get recommended;
  4. cheap fixes: an additive cold bonus (tuned on VALID) and reserved cold slots in the top-10, with the warm/cold trade-off.
Usage: PYTHONPATH=src:scripts python scripts/eval_cold_start.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_baselines import D, P, mask_scores, ranks_from_scores  # noqa: E402

from datn.recommenders.reranker.inference import RerankerPipeline  # noqa: E402
from datn.recommenders.user_tower.content import ContentSource, load_content_matrix  # noqa: E402
from datn.recommenders.user_tower.dataset import PAD_IDX, build_eval_examples, build_item_vocab, load_user_sequences  # noqa: E402

torch.set_num_threads(4)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = build_item_vocab(P / "items.parquet")
seqs = load_user_sequences(P / "train.parquet", P / "valid.parquet", P / "test.parquet", vocab)
V = vocab.vocab_size
deg = torch.zeros(V, device=dev)
for s in seqs.train.values():
    deg[torch.tensor(s, device=dev)] += 1
real = torch.zeros(V, dtype=torch.bool, device=dev); real[1:vocab.num_items + 1] = True
cold = (deg == 0) & real
warm = (deg > 0) & real
print(f"catalog={vocab.num_items}  cold items (0 positive train)={int(cold.sum())} ({cold.sum().item()/vocab.num_items*100:.1f}%)", flush=True)

src = [ContentSource("image", D / "embedding/image_embeddings.npy", D / "embedding/image_embedding_metadata.parquet"),
       ContentSource("text", D / "embedding/text_embeddings.npy", D / "embedding/text_embedding_metadata.parquet")]
pipe = RerankerPipeline.from_artifacts(
    user_tower_dir=D / "artifacts/user_tower_balanced_v1", reranker_dir=D / "artifacts/reranker_v2", items_path=P / "items.parquet",
    train_path=P / "train.parquet", valid_path=P / "valid.parquet", test_path=P / "test.parquet", content_sources=src, device="auto")
item_emb = pipe.tower.item_vectors()
cm = load_content_matrix(vocab, src)
half = cm.shape[1] // 2
cm = np.concatenate([cm[:, :half] / (np.linalg.norm(cm[:, :half], axis=1, keepdims=True) + 1e-9),
                     cm[:, half:] / (np.linalg.norm(cm[:, half:], axis=1, keepdims=True) + 1e-9)], axis=1) / math.sqrt(2)
X = torch.tensor(cm, device=dev)


def prep(split):
    ex = build_eval_examples(seqs, 20, split)
    hist = [[int(i) for i in e.context if i != PAD_IDX] for e in ex]
    return ex, hist, [sorted(e.seen) for e in ex], torch.tensor([e.target for e in ex], device=dev)


@torch.no_grad()
def scores_for(ex, hist, s, e):
    ctx = torch.from_numpy(np.stack([x.context for x in ex[s:e]])).to(pipe.device)
    tw = (pipe.tower.encode_user(ctx, item_emb) @ item_emb.T).to(dev)
    C = torch.stack([(torch.tensor([0.7 ** (len(h) - 1 - j) for j in range(len(h))], device=dev)[:, None] * X[h]).sum(0) for h in hist[s:e]])
    return tw, C @ X.T


def rank_of(sc, seen, tgt, oov, pool=None):
    sc = sc.clone(); mask_scores(sc, seen, oov)
    if pool is not None:
        keep = pool.expand_as(sc).clone(); keep[torch.arange(len(tgt)), tgt] = True
        sc[~keep] = float("-inf")
    return ranks_from_scores(sc, tgt).cpu().numpy()


report: dict = {"cold_items": int(cold.sum()), "catalog": vocab.num_items}
BS = 512
out_split: dict = {}
for split in ("valid", "test"):
    ex, hist, seen, tgt = prep(split)
    n = len(ex)
    is_cold_t = cold[tgt].cpu().numpy()
    print(f"\n[{split}] users={n}  cold targets={is_cold_t.sum()} ({is_cold_t.mean()*100:.2f}%)", flush=True)
    R = {k: np.empty(n) for k in ("tower_all", "tower_pool", "content_all", "content_pool")}
    gap, in_top100, distinct = [], 0, set()
    bonus_grid = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
    Rb = {b: np.empty(n) for b in bonus_grid}
    slots = {r: np.empty(n) for r in (0, 1, 2, 3)}
    for s in range(0, n, BS):
        e = min(n, s + BS)
        tw, ct = scores_for(ex, hist, s, e)
        t = tgt[s:e]
        R["tower_all"][s:e] = rank_of(tw, seen[s:e], t, vocab.oov_idx)
        R["tower_pool"][s:e] = rank_of(tw, seen[s:e], t, vocab.oov_idx, cold[None, :])
        R["content_all"][s:e] = rank_of(ct, seen[s:e], t, vocab.oov_idx)
        R["content_pool"][s:e] = rank_of(ct, seen[s:e], t, vocab.oov_idx, cold[None, :])
        # bias: standardised gap between mean score of cold vs warm items, per user
        w_sc, c_sc = tw[:, warm], tw[:, cold]
        gap.append(((c_sc.mean(1) - w_sc.mean(1)) / w_sc.std(1)).cpu().numpy())
        tmask = tw.clone(); mask_scores(tmask, seen[s:e], vocab.oov_idx)
        top = tmask.topk(100, dim=1).indices
        in_top100 += int(cold[top].any(dim=1).sum()); distinct.update(top[cold[top]].cpu().tolist())
        # additive cold bonus in units of the user's score std over warm items
        std = w_sc.std(1, keepdim=True)
        for b in bonus_grid:
            Rb[b][s:e] = rank_of(tw + b * std * cold[None, :].float(), seen[s:e], t, vocab.oov_idx)
        # reserved cold slots: top (10-r) of tower + best r cold items by content (r=0 == tower)
        cmask = ct.clone(); mask_scores(cmask, seen[s:e], vocab.oov_idx); cmask[:, ~cold] = float("-inf")
        for r in slots:
            for i in range(e - s):
                tl = tmask[i].topk(10 - r).indices.tolist() if r < 10 else []
                cl = [x for x in cmask[i].topk(r + 3).indices.tolist() if x not in tl][:r] if r else []
                lst = tl + cl
                slots[r][s + i] = lst.index(int(t[i])) if int(t[i]) in lst else 1e9
    cw = ~is_cold_t
    def hr(rk, mask, K):
        return float(np.mean(rk[mask] < K)) if mask.sum() else float("nan")
    res = {
        "n_users": n, "cold_target_share": float(is_cold_t.mean()),
        "cold_target_median_pct_rank_tower": float(np.median(R["tower_all"][is_cold_t]) / vocab.num_items * 100),
        "cold_target_median_pct_rank_content": float(np.median(R["content_all"][is_cold_t]) / vocab.num_items * 100),
        "cold_pool_size": int(cold.sum()),
        "cold_pool_random_HR@10_pct": 1000.0 / int(cold.sum()),
        "in_cold_pool_only": {m: {f"HR@{K}": hr(R[f"{m}_pool"], is_cold_t, K) for K in (10, 50, 100)} for m in ("tower", "content")},
        "global_cold_targets": {m: {f"HR@{K}": hr(R[f"{m}_all"], is_cold_t, K) for K in (10, 50, 100)} for m in ("tower", "content")},
        "tower_std_score_gap_cold_minus_warm": float(np.mean(np.concatenate(gap))),
        "users_with_any_cold_item_in_top100": in_top100 / n,
        "distinct_cold_items_ever_in_top100": len(distinct),
        "bonus_sweep": {str(b): {"overall": {f"HR@{K}": hr(Rb[b], np.ones(n, bool), K) for K in (10, 50)},
                                 "cold_targets": {f"HR@{K}": hr(Rb[b], is_cold_t, K) for K in (10, 50)},
                                 "warm_targets": {f"HR@{K}": hr(Rb[b], cw, K) for K in (10, 50)}} for b in bonus_grid},
        "reserved_cold_slots_top10": {str(r): {"overall": hr(slots[r], np.ones(n, bool), 10), "cold": hr(slots[r], is_cold_t, 10),
                                               "warm": hr(slots[r], cw, 10)} for r in slots},
    }
    out_split[split] = res
    print(json.dumps({k: v for k, v in res.items() if k not in ("bonus_sweep", "reserved_cold_slots_top10")}, indent=1), flush=True)
    print("cold-bonus sweep (units of warm-score std):  overall HR@10/50 | cold HR@10/50 | warm HR@10/50")
    for b, v in res["bonus_sweep"].items():
        print(f"  b={b:4s} {v['overall']['HR@10']*100:5.2f} {v['overall']['HR@50']*100:5.2f} | {v['cold_targets']['HR@10']*100:5.2f} {v['cold_targets']['HR@50']*100:5.2f} | {v['warm_targets']['HR@10']*100:5.2f} {v['warm_targets']['HR@50']*100:5.2f}")
    print("reserved cold slots (top-10):  overall | cold | warm")
    for r, v in res["reserved_cold_slots_top10"].items():
        print(f"  r={r}  {v['overall']*100:5.2f} | {v['cold']*100:5.2f} | {v['warm']*100:5.2f}")

report["splits"] = out_split
Path("data/artifacts/baseline_eval/cold_start.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
