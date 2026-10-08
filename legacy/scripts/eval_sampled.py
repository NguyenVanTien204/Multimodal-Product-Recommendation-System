"""Sampled-ranking protocol (1 positive + N sampled negatives) for the balanced test set, all methods on the SAME negatives.

Variants: uniform-random negatives (N=99 and N=999) and popularity-weighted negatives (N=99); negatives exclude the user's seen items
and the target. Ties are counted as half. The reranker is turned into a full ranking (its score for candidates, tower score below them),
so it can be evaluated with the same protocol.
NOTE: sampled metrics are much higher than full-ranking metrics and are NOT comparable to them or to other papers' full-ranking numbers.
Usage: PYTHONPATH=legacy/src:legacy/scripts python legacy/scripts/eval_sampled.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_baselines import D, P, bootstrap  # noqa: E402

from datn_legacy.recommenders.reranker.candidates import CandidateExample, candidate_features  # noqa: E402
from datn_legacy.recommenders.reranker.inference import RerankerPipeline  # noqa: E402
from datn_legacy.recommenders.user_tower.content import ContentSource, load_content_matrix  # noqa: E402
from datn_legacy.recommenders.user_tower.dataset import PAD_IDX, build_eval_examples, build_item_vocab, load_user_sequences  # noqa: E402

torch.set_num_threads(4)
torch.manual_seed(20260930)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = build_item_vocab(P / "items.parquet")
seqs = load_user_sequences(P / "train.parquet", P / "valid.parquet", P / "test.parquet", vocab)
V = vocab.vocab_size
ex = build_eval_examples(seqs, 20, "test")
n = len(ex)
hist = [[int(i) for i in e.context if i != PAD_IDX] for e in ex]
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
pipe = RerankerPipeline.from_artifacts(
    user_tower_dir=D / "artifacts/user_tower_balanced_v1", reranker_dir=D / "artifacts/reranker_v2", items_path=P / "items.parquet",
    train_path=P / "train.parquet", valid_path=P / "valid.parquet", test_path=P / "test.parquet", content_sources=src, device="auto")
item_emb = pipe.tower.item_vectors()

SETTINGS = {"uniform_99": ("uniform", 99), "uniform_999": ("uniform", 999), "popweighted_99": ("pop", 99)}
METHODS = ("random", "popularity", "content_centroid", "user_tower_raw", "tower+reranker_v2")
R = {s: {m: np.zeros(n) for m in METHODS} for s in SETTINGS}   # 0-indexed rank of the positive among 1+N

BS = 64
with torch.no_grad():
    for s in range(0, n, BS):
        e = min(n, s + BS)
        b = e - s
        tgt = torch.tensor([x.target for x in ex[s:e]], device=dev)
        valid = torch.zeros(b, V, dtype=torch.bool, device=dev)
        valid[:, 1:vocab.num_items + 1] = True
        for i, x in enumerate(ex[s:e]):
            if x.seen:
                valid[i, torch.tensor(sorted(x.seen), device=dev)] = False
        valid[torch.arange(b), tgt] = False

        uv = pipe.tower.encode_user(torch.from_numpy(np.stack([x.context for x in ex[s:e]])).to(pipe.device), item_emb)
        tower = (uv @ item_emb.T).to(dev)
        C = torch.stack([(torch.tensor([0.7 ** (len(h) - 1 - j) for j in range(len(h))], device=dev)[:, None] * X[h]).sum(0) for h in hist[s:e]])
        content = C @ X.T
        batch = [CandidateExample(context=x.context, seen=x.seen) for x in ex[s:e]]
        rows = candidate_features(pipe.tower, batch, pipe.vocab, pipe.popularity, pipe.budget, pipe.device)
        rr = tower - 1e4                                   # non-candidates stay below every candidate, ordered by tower score
        for j, (cand, feats) in enumerate(rows):
            if len(cand):
                xs = torch.from_numpy(((feats - pipe.feature_mean) / pipe.feature_std).astype(np.float32)).to(pipe.device)
                rr[j, torch.from_numpy(cand).to(dev)] = pipe.ranker(xs, blend=pipe.blend).to(dev)
        scores = {"random": torch.rand(b, V, device=dev), "popularity": counts.expand(b, V), "content_centroid": content,
                  "user_tower_raw": tower, "tower+reranker_v2": rr}

        for name, (kind, N) in SETTINGS.items():
            w = valid.float() if kind == "uniform" else valid.float() * counts.clamp(min=1e-6)[None, :]
            negs = torch.multinomial(w, N, replacement=False)                 # (b, N) shared by every method
            for m in METHODS:
                sc = scores[m]
                ts = sc.gather(1, tgt[:, None])
                ns = sc.gather(1, negs)
                R[name][m][s:e] = ((ns > ts).sum(1).float() + 0.5 * (ns == ts).sum(1).float()).cpu().numpy()
        if (s // BS) % 60 == 0:
            print(f"  {s}/{n}", flush=True)

out: dict = {"n_users": n}
for name, (kind, N) in SETTINGS.items():
    out[name] = {}
    print(f"\n=== {name}: 1 positive + {N} sampled negatives ({kind}); random reference HR@10 = {min(1.0, 10 / (N + 1)) * 100:.1f}% ===")
    print(f"{'method':22s} HR@1   HR@5   HR@10  HR@20  NDCG@10  MRR")
    for m in METHODS:
        r = R[name][m]
        row = {f"HR@{k}": float(np.mean(r < k)) for k in (1, 5, 10, 20)}
        row["NDCG@10"] = float(np.where(r < 10, 1 / np.log2(r + 2), 0).mean())
        row["MRR"] = float(np.mean(1 / (r + 1)))
        out[name][m] = row
        print(f"{m:22s} " + "  ".join(f"{row[f'HR@{k}']*100:5.1f}" for k in (1, 5, 10, 20)) + f"   {row['NDCG@10']*100:5.1f}   {row['MRR']*100:5.1f}")
    h = {m: (R[name][m] < 10).astype(np.int8) for m in ("popularity", "user_tower_raw", "tower+reranker_v2")}
    bs = bootstrap(h, "popularity", 1000)
    out[name]["paired_HR@10_vs_popularity"] = {m: {k: bs[m][k] for k in ("hr", "ci95", "diff_vs_base", "diff_ci95", "mcnemar_p")} for m in ("user_tower_raw", "tower+reranker_v2")}
    v = bs["tower+reranker_v2"]
    print(f"tower+reranker HR@10 = {v['hr']*100:.1f}% CI[{v['ci95'][0]*100:.1f},{v['ci95'][1]*100:.1f}]; vs popularity {v['diff_vs_base']*100:+.1f}pp CI[{v['diff_ci95'][0]*100:+.1f},{v['diff_ci95'][1]*100:+.1f}]")
Path("data/artifacts/baseline_eval/sampled_protocol.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
