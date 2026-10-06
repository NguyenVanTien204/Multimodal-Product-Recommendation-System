"""Two pre-declared analyses (all settings reported, nothing selected after the fact):
  A. Sampled protocol curve: expected HR@K of the raw tower for 1 positive + N uniform negatives, derived analytically from full ranks
     (a target with full rank r among M candidates beats each negative with prob 1 - r/M  ->  sampled rank ~ Binomial(N, r/M));
     validated against the measured 1+99 / 1+999 numbers.
  B. Product-type-level hit: k-means (spherical, seed fixed) on the CLIP image+text vectors, K in {30, 100} clusters;
     a hit = the top-10 / top-20 list contains an item from the cluster of the next purchase. Baselines included.
Usage: PYTHONPATH=src:scripts python scripts/eval_coarse.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_baselines import D, P, mask_scores  # noqa: E402

from datn.recommenders.reranker.candidates import CandidateExample, candidate_features  # noqa: E402
from datn.recommenders.reranker.inference import RerankerPipeline  # noqa: E402
from datn.recommenders.user_tower.content import ContentSource, load_content_matrix  # noqa: E402
from datn.recommenders.user_tower.dataset import PAD_IDX, build_eval_examples, build_item_vocab, load_user_sequences  # noqa: E402

torch.set_num_threads(4)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out: dict = {}

# ---------------- A. analytic sampled-protocol curve ----------------
ranks = np.load("data/artifacts/baseline_eval/test_ranks.npz")["user_tower_raw"]      # full-catalog rank (avg ties), 0-indexed
M = 32557 - 9                                                                           # ~ unseen, non-target candidates
p = np.clip(ranks / M, 0, 1)
from math import comb  # noqa: E402


def expected_hr(N: int, K: int) -> float:
    ks = np.arange(0, K)                                    # sampled rank j < K  <=> fewer than K negatives above the target
    logc = np.array([math.lgamma(N + 1) - math.lgamma(j + 1) - math.lgamma(N - j + 1) for j in ks])
    pp = np.clip(p, 1e-12, 1 - 1e-12)[:, None]
    pmf = np.exp(logc[None, :] + ks[None, :] * np.log(pp) + (N - ks[None, :]) * np.log1p(-pp))
    return float(pmf.sum(1).mean())


print("A. raw User Tower, 1 positive + N uniform negatives: expected HR@K (%)   [validation: measured 1+99 HR@10=45.1, HR@20=60.1; 1+999 HR@10=16.1]")
curve = {}
for N in (9, 19, 29, 49, 99, 199, 999):
    row = {K: expected_hr(N, K) for K in (1, 5, 10, 20) if K <= N}
    curve[N] = row
    print(f"  N={N:4d}  " + "  ".join(f"HR@{K}={v*100:5.1f}" for K, v in row.items()) + f"   random@10={min(1, 10/(N+1))*100:.1f}")
out["sampled_curve_tower_raw"] = {str(N): {str(K): v for K, v in row.items()} for N, row in curve.items()}

# ---------------- B. product-type-level hit ----------------
vocab = build_item_vocab(P / "items.parquet")
seqs = load_user_sequences(P / "train.parquet", P / "valid.parquet", P / "test.parquet", vocab)
V = vocab.vocab_size
ex = build_eval_examples(seqs, 20, "test")
n = len(ex)
hist = [[int(i) for i in e.context if i != PAD_IDX] for e in ex]
seen = [sorted(e.seen) for e in ex]
tgt = torch.tensor([e.target for e in ex], device=dev)
counts = torch.zeros(V, device=dev)
for s in seqs.train.values():
    counts[torch.tensor(s, device=dev)] += 1
src = [ContentSource("image", D / "embedding/image_embeddings.npy", D / "embedding/image_embedding_metadata.parquet"),
       ContentSource("text", D / "embedding/text_embeddings.npy", D / "embedding/text_embedding_metadata.parquet")]
cm = load_content_matrix(vocab, src)
half = cm.shape[1] // 2
cm = np.concatenate([cm[:, :half] / (np.linalg.norm(cm[:, :half], axis=1, keepdims=True) + 1e-9),
                     cm[:, half:] / (np.linalg.norm(cm[:, half:], axis=1, keepdims=True) + 1e-9)], axis=1) / math.sqrt(2)
X = torch.tensor(cm, device=dev)
real = torch.zeros(V, dtype=torch.bool, device=dev); real[1:vocab.num_items + 1] = True

g = torch.Generator(device="cpu").manual_seed(20260930)


def kmeans(k, iters=25):
    Xr = torch.nn.functional.normalize(X[real], dim=1)
    idx = torch.randperm(Xr.shape[0], generator=g)[:k].to(dev)
    C = Xr[idx].clone()
    for _ in range(iters):
        a = (Xr @ C.T).argmax(1)
        for j in range(k):
            m = a == j
            if m.any():
                C[j] = torch.nn.functional.normalize(Xr[m].mean(0), dim=0)
    lab = torch.zeros(V, dtype=torch.long, device=dev) - 1
    lab[real] = (Xr @ C.T).argmax(1)
    return lab


clusters = {k: kmeans(k) for k in (30, 100)}

pipe = RerankerPipeline.from_artifacts(
    user_tower_dir=D / "artifacts/user_tower_balanced_v1", reranker_dir=D / "artifacts/reranker_v2", items_path=P / "items.parquet",
    train_path=P / "train.parquet", valid_path=P / "valid.parquet", test_path=P / "test.parquet", content_sources=src, device="auto")
item_emb = pipe.tower.item_vectors()
ctx_all = torch.from_numpy(np.stack([e.context for e in ex])).to(pipe.device)
TOP = 20
names = ("popularity", "content_centroid", "user_tower_raw", "tower+reranker_v2")
tops = {m: np.zeros((n, TOP), dtype=np.int64) for m in names}
with torch.no_grad():
    for s in range(0, n, 512):
        e = min(n, s + 512)
        C = torch.stack([(torch.tensor([0.7 ** (len(h) - 1 - j) for j in range(len(h))], device=dev)[:, None] * X[h]).sum(0) for h in hist[s:e]])
        uv = pipe.tower.encode_user(ctx_all[s:e], item_emb)
        for name, sc in (("popularity", counts.expand(e - s, V) + 1e-6 * torch.rand(e - s, V, device=dev)), ("content_centroid", C @ X.T),
                         ("user_tower_raw", (uv @ item_emb.T).to(dev))):
            sc = sc.clone(); mask_scores(sc, seen[s:e], vocab.oov_idx)
            tops[name][s:e] = sc.topk(TOP, dim=1).indices.cpu().numpy()
    for s in range(0, n, 64):
        batch = [CandidateExample(context=x.context, seen=x.seen) for x in ex[s:s + 64]]
        rows = candidate_features(pipe.tower, batch, pipe.vocab, pipe.popularity, pipe.budget, pipe.device)
        for j, (cand, feats) in enumerate(rows):
            x = torch.from_numpy(((feats - pipe.feature_mean) / pipe.feature_std).astype(np.float32)).to(pipe.device)
            order = np.argsort(-pipe.ranker(x, blend=pipe.blend).cpu().numpy(), kind="stable")[:TOP]
            tops["tower+reranker_v2"][s + j, :len(order)] = cand[order]

t_np = tgt.cpu().numpy()
out["type_level"] = {}
print("\nB. product-type-level hit (cluster of next purchase appears in top-K list), %:")
for k, lab in clusters.items():
    lab_np = lab.cpu().numpy()
    sizes = np.bincount(lab_np[lab_np >= 0], minlength=k)
    major = float(np.mean(lab_np[t_np] == sizes.argmax()))
    print(f"  --- {k} clusters (largest cluster holds {sizes.max()/sizes.sum()*100:.1f}% of catalog; 'always the biggest cluster' would hit {major*100:.1f}% of targets)")
    out["type_level"][str(k)] = {"largest_cluster_share_targets": major}
    for name in names:
        res = {}
        for K in (10, 20):
            hit = (lab_np[tops[name][:, :K]] == lab_np[t_np][:, None]).any(1)
            res[f"K{K}"] = float(hit.mean())
        out["type_level"][str(k)][name] = res
        print(f"    {name:20s} top-10: {res['K10']*100:5.1f}   top-20: {res['K20']*100:5.1f}")
Path("data/artifacts/baseline_eval/coarse_analyses.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
