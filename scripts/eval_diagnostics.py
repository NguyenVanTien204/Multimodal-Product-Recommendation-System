"""Follow-up diagnostics for eval_baselines.py:
 1. tune item-kNN shrinkage on VALID (so the baseline is not a straw man), report the winner on TEST;
 2. does the User Tower actually use the user's history? (test with full / last-item-only / history swapped between users);
 3. paired bootstrap: reranker v2 vs raw tower vs RRF hybrid.
Usage: PYTHONPATH=src:scripts python scripts/eval_diagnostics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_baselines import D, P, bootstrap, item_sims, mask_scores, ranks_from_scores, sparse_hist  # noqa: E402

from datn.recommenders.reranker.inference import RerankerPipeline  # noqa: E402
from datn.recommenders.user_tower.content import ContentSource  # noqa: E402
from datn.recommenders.user_tower.dataset import PAD_IDX, build_eval_examples, build_item_vocab, load_user_sequences, pad_right  # noqa: E402

torch.set_num_threads(4)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = build_item_vocab(P / "items.parquet")
seqs = load_user_sequences(P / "train.parquet", P / "valid.parquet", P / "test.parquet", vocab)
V = vocab.vocab_size
counts = torch.zeros(V, device=dev)
for s in seqs.train.values():
    counts[torch.tensor(s, device=dev)] += 1
pop_norm = counts / (counts.max() + 1)


def prep(split):
    ex = build_eval_examples(seqs, 20, split)
    hist = [[int(i) for i in e.context if i != PAD_IDX] for e in ex]
    return ex, hist, [sorted(e.seen) for e in ex], torch.tensor([e.target for e in ex], device=dev)


def knn_ranks(S, decay, hist, seen, targets, bs=512):
    rk = np.empty(len(hist))
    for s in range(0, len(hist), bs):
        e = min(len(hist), s + bs)
        H = sparse_hist([{it: decay ** (len(h) - 1 - j) for j, it in enumerate(h)} for h in hist[s:e]], V, dev)
        sc = torch.sparse.mm(H, S).to_dense() + 1e-7 * pop_norm
        mask_scores(sc, seen[s:e], vocab.oov_idx)
        rk[s:e] = ranks_from_scores(sc, targets[s:e]).cpu().numpy()
    return rk


out: dict = {}
# ---- 1. item-kNN tuning on valid ----
exv, hv, sv, tv = prep("valid")
best = None
print("item-kNN tuning on VALID (HR@10 / HR@50 / HR@100):")
for mode in ("cooc", "trans"):
    for shrink in (1, 10, 50, 200, 1000):
        S = item_sims(seqs.train, V, dev, mode, shrink)
        rk = knn_ranks(S, 0.7, hv, sv, tv)
        h = [float(np.mean(rk < k)) for k in (10, 50, 100)]
        print(f"  {mode:5s} shrink={shrink:5d}  {h[0]*100:5.2f} {h[1]*100:5.2f} {h[2]*100:5.2f}", flush=True)
        if best is None or h[1] > best[0]:
            best = (h[1], mode, shrink)
out["knn_best_on_valid"] = {"HR@50": best[0], "mode": best[1], "shrink": best[2]}
ext, ht, st, tt = prep("test")
S = item_sims(seqs.train, V, dev, best[1], best[2])
rk_knn = knn_ranks(S, 0.7, ht, st, tt)
print(f"best={best[1]} shrink={best[2]}  TEST: " + " ".join(f"HR@{k}={np.mean(rk_knn<k)*100:.2f}" for k in (10, 50, 100)), flush=True)

# ---- 2. does the tower use the history? ----
src = [ContentSource("image", D / "embedding/image_embeddings.npy", D / "embedding/image_embedding_metadata.parquet"),
       ContentSource("text", D / "embedding/text_embeddings.npy", D / "embedding/text_embedding_metadata.parquet")]
pipe = RerankerPipeline.from_artifacts(
    user_tower_dir=D / "artifacts/user_tower_balanced_v1", reranker_dir=D / "artifacts/reranker_v2", items_path=P / "items.parquet",
    train_path=P / "train.parquet", valid_path=P / "valid.parquet", test_path=P / "test.parquet", content_sources=src, device="auto")
item_emb = pipe.tower.item_vectors()
n = len(ext)
rng = np.random.default_rng(3)
perm = rng.permutation(n)


@torch.no_grad()
def tower_ranks(contexts):
    rk = np.empty(n)
    ctx = torch.from_numpy(np.stack(contexts)).to(pipe.device)
    for s in range(0, n, 512):
        e = min(n, s + 512)
        sc = (pipe.tower.encode_user(ctx[s:e], item_emb) @ item_emb.T).to(dev)
        mask_scores(sc, st[s:e], vocab.oov_idx)
        rk[s:e] = ranks_from_scores(sc, tt[s:e]).cpu().numpy()
    return rk


variants = {
    "tower_full_history": [e.context for e in ext],
    "tower_last_item_only": [pad_right(h[-1:], 20) for h in ht],
    "tower_history_swapped_between_users": [ext[perm[i]].context for i in range(n)],
}
tr = {}
print("Does the tower use the user's history? (TEST)")
for name, ctxs in variants.items():
    tr[name] = tower_ranks(ctxs)
    print(f"  {name:38s} " + " ".join(f"HR@{k}={np.mean(tr[name]<k)*100:.2f}" for k in (10, 50, 100)), flush=True)

# ---- 3. paired comparisons among final candidates ----
saved = np.load("data/artifacts/baseline_eval/test_ranks.npz")
allr = {"pop": saved["popularity"], "rrf": saved["rrf(knn,content,pop)"], "tower_raw": saved["user_tower_raw"],
        "reranker_v2": saved["tower+reranker_v2"], "knn_tuned": rk_knn, "tower_swapped": tr["tower_history_swapped_between_users"],
        "tower_last1": tr["tower_last_item_only"]}
out["paired"] = {}
for K in (10, 50, 100):
    hits = {m: (r < K).astype(np.int8) for m, r in allr.items()}
    out["paired"][f"reranker_vs_tower_raw@{K}"] = bootstrap({m: hits[m] for m in ("tower_raw", "reranker_v2")}, "tower_raw", 2000)["reranker_v2"]
    out["paired"][f"reranker_vs_rrf@{K}"] = bootstrap({m: hits[m] for m in ("rrf", "reranker_v2")}, "rrf", 2000)["reranker_v2"]
    out["paired"][f"tower_raw_vs_swapped@{K}"] = bootstrap({m: hits[m] for m in ("tower_swapped", "tower_raw")}, "tower_swapped", 2000)["tower_raw"]
    out["paired"][f"knn_tuned_vs_pop@{K}"] = bootstrap({m: hits[m] for m in ("pop", "knn_tuned")}, "pop", 2000)["knn_tuned"]
    for key in ("reranker_vs_tower_raw", "reranker_vs_rrf", "tower_raw_vs_swapped", "knn_tuned_vs_pop"):
        v = out["paired"][f"{key}@{K}"]
        print(f"  {key}@{K}: diff {v['diff_vs_base']*100:+.2f}pp CI[{v['diff_ci95'][0]*100:+.2f},{v['diff_ci95'][1]*100:+.2f}] p={v['mcnemar_p']:.2g}")
out["history_ablation_test"] = {k: {f"HR@{q}": float(np.mean(v < q)) for q in (10, 50, 100)} for k, v in tr.items()}
Path("data/artifacts/baseline_eval/diagnostics.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
