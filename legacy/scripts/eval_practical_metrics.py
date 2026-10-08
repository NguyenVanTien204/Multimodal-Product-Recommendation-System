"""Task-appropriate ("practical") metrics for the balanced test set, defined BEFORE looking at results:

  item HR@K        : the exact next purchase is in the top-K                          (already reported elsewhere)
  brand-hit@K      : top-K contains an item of the same brand as the next purchase    (brand loyalty; users with a known brand only)
  neighbor-hit@K   : top-K contains one of the 20 nearest content neighbours (CLIP image+text cosine) of the next purchase,
                     i.e. a near-substitute / variant of what the user actually bought
All methods are scored on the same users with the same seen-item masking; baselines included so lift is visible.
Usage: PYTHONPATH=legacy/src:legacy/scripts python legacy/scripts/eval_practical_metrics.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_baselines import D, P, bootstrap, mask_scores  # noqa: E402

from datn_legacy.recommenders.reranker.candidates import CandidateExample, candidate_features  # noqa: E402
from datn_legacy.recommenders.reranker.inference import RerankerPipeline  # noqa: E402
from datn_legacy.recommenders.user_tower.content import ContentSource, load_content_matrix  # noqa: E402
from datn_legacy.recommenders.user_tower.dataset import PAD_IDX, build_eval_examples, build_item_vocab, load_user_sequences  # noqa: E402

torch.set_num_threads(4)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KS = (10, 20, 50)
TOPN = 50
NEIGH = 20

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
pop_norm = counts / (counts.max() + 1)

src = [ContentSource("image", D / "embedding/image_embeddings.npy", D / "embedding/image_embedding_metadata.parquet"),
       ContentSource("text", D / "embedding/text_embeddings.npy", D / "embedding/text_embedding_metadata.parquet")]
cm = load_content_matrix(vocab, src)
half = cm.shape[1] // 2
cm = np.concatenate([cm[:, :half] / (np.linalg.norm(cm[:, :half], axis=1, keepdims=True) + 1e-9),
                     cm[:, half:] / (np.linalg.norm(cm[:, half:], axis=1, keepdims=True) + 1e-9)], axis=1) / math.sqrt(2)
X = torch.tensor(cm, device=dev)

items = pl.read_parquet(P / "items.parquet", columns=["brand"])
brands = [(b or "").strip().lower() for b in items["brand"].to_list()]
bmap = {b: i + 1 for i, b in enumerate(sorted({b for b in brands if b}))}
brand = torch.zeros(V, dtype=torch.long, device=dev)
brand[1:vocab.num_items + 1] = torch.tensor([bmap.get(b, 0) for b in brands], device=dev)
print(f"users={n} brands={len(bmap)} brand-known targets={(brand[tgt] > 0).float().mean().item()*100:.1f}%", flush=True)

pipe = RerankerPipeline.from_artifacts(
    user_tower_dir=D / "artifacts/user_tower_balanced_v1", reranker_dir=D / "artifacts/reranker_v2", items_path=P / "items.parquet",
    train_path=P / "train.parquet", valid_path=P / "valid.parquet", test_path=P / "test.parquet", content_sources=src, device="auto")
item_emb = pipe.tower.item_vectors()
ctx_all = torch.from_numpy(np.stack([e.context for e in ex])).to(pipe.device)
last_item = torch.tensor([h[-1] for h in hist], device=dev)
BS = 512

tops: dict[str, np.ndarray] = {m: np.zeros((n, TOPN), dtype=np.int64) for m in
                               ("popularity", "same_brand_as_last+pop", "content_centroid", "rrf(content,pop)", "user_tower_raw", "tower+reranker_v2")}


def top_of(sc, s, e, name):
    sc = sc.clone(); mask_scores(sc, seen[s:e], vocab.oov_idx)
    tops[name][s:e] = sc.topk(TOPN, dim=1).indices.cpu().numpy()
    return sc


neigh = np.zeros((n, NEIGH), dtype=np.int64)
with torch.no_grad():
    for s in range(0, n, BS):
        e = min(n, s + BS)
        pop = top_of(counts.expand(e - s, V) + 1e-6 * torch.rand(e - s, V, device=dev), s, e, "popularity")
        lb = brand[last_item[s:e]]
        top_of(2.0 * ((brand[None, :] == lb[:, None]) & (lb[:, None] > 0)).float() + pop_norm[None, :], s, e, "same_brand_as_last+pop")
        C = torch.stack([(torch.tensor([0.7 ** (len(h) - 1 - j) for j in range(len(h))], device=dev)[:, None] * X[h]).sum(0) for h in hist[s:e]])
        cs = top_of(C @ X.T, s, e, "content_centroid")
        rr = torch.zeros(e - s, V, device=dev)
        for sc in (C @ X.T, counts.expand(e - s, V) + 1e-6 * torch.rand(e - s, V, device=dev)):
            t = sc.clone(); mask_scores(t, seen[s:e], vocab.oov_idx)
            rr += 1.0 / (60.0 + t.argsort(dim=1, descending=True).argsort(dim=1).float())
        top_of(rr, s, e, "rrf(content,pop)")
        uv = pipe.tower.encode_user(ctx_all[s:e], item_emb)
        top_of((uv @ item_emb.T).to(dev), s, e, "user_tower_raw")
        # content neighbours of the true target (excluding itself and PAD/OOV)
        ns = X[tgt[s:e]] @ X.T
        ns[:, PAD_IDX] = -1; ns[:, vocab.oov_idx] = -1
        ns[torch.arange(e - s), tgt[s:e]] = -1
        neigh[s:e] = ns.topk(NEIGH, dim=1).indices.cpu().numpy()
    print("baselines done; running reranker pipeline...", flush=True)
    for s in range(0, n, 64):
        batch = [CandidateExample(context=e.context, seen=e.seen) for e in ex[s:s + 64]]
        rows = candidate_features(pipe.tower, batch, pipe.vocab, pipe.popularity, pipe.budget, pipe.device)
        for j, (cand, feats) in enumerate(rows):
            x = torch.from_numpy(((feats - pipe.feature_mean) / pipe.feature_std).astype(np.float32)).to(pipe.device)
            order = np.argsort(-pipe.ranker(x, blend=pipe.blend).cpu().numpy(), kind="stable")[:TOPN]
            tops["tower+reranker_v2"][s + j, :len(order)] = cand[order]

t_np = tgt.cpu().numpy()
b_np = brand.cpu().numpy()
brand_known = b_np[t_np] > 0
res: dict = {"n_users": n, "brand_known_users": int(brand_known.sum()), "neighbors_per_target": NEIGH, "metrics": {}}
hits_all: dict[str, dict[int, dict[str, np.ndarray]]] = {}
for name, top in tops.items():
    hits_all[name] = {}
    for K in KS:
        tk = top[:, :K]
        item_hit = (tk == t_np[:, None]).any(1)
        brand_hit = ((b_np[tk] == b_np[t_np][:, None]) & brand_known[:, None]).any(1)
        neigh_hit = (tk[:, :, None] == neigh[:, None, :]).any(2).any(1)
        hits_all[name][K] = {"item": item_hit, "brand": brand_hit, "neighbor": neigh_hit}
        res["metrics"].setdefault(name, {})[f"K{K}"] = {"item": float(item_hit.mean()), "brand_on_known": float(brand_hit[brand_known].mean()),
                                                          "neighbor": float(neigh_hit.mean())}

print(f"\n{'method':26s} " + " | ".join(f"K={K}: item  brand  neigh" for K in KS))
for name in tops:
    print(f"{name:26s} " + " | ".join(
        f"      {res['metrics'][name][f'K{K}']['item']*100:5.2f} {res['metrics'][name][f'K{K}']['brand_on_known']*100:6.2f} {res['metrics'][name][f'K{K}']['neighbor']*100:6.2f}" for K in KS))

res["paired_vs_popularity"] = {}
for K in KS:
    for kind in ("brand", "neighbor"):
        for name in ("tower+reranker_v2", "user_tower_raw", "same_brand_as_last+pop", "rrf(content,pop)"):
            mask = brand_known if kind == "brand" else np.ones(n, bool)
            h = {"popularity": hits_all["popularity"][K][kind][mask].astype(np.int8), name: hits_all[name][K][kind][mask].astype(np.int8)}
            b = bootstrap(h, "popularity", 1000)[name]
            res["paired_vs_popularity"][f"{name}|{kind}@{K}"] = {k: b[k] for k in ("hr", "ci95", "diff_vs_base", "diff_ci95", "mcnemar_p")}
print("\npaired vs popularity (tower+reranker_v2):")
for K in KS:
    for kind in ("brand", "neighbor"):
        v = res["paired_vs_popularity"][f"tower+reranker_v2|{kind}@{K}"]
        print(f"  {kind}@{K}: {v['hr']*100:.2f}% CI[{v['ci95'][0]*100:.2f},{v['ci95'][1]*100:.2f}]  diff {v['diff_vs_base']*100:+.2f}pp CI[{v['diff_ci95'][0]*100:+.2f},{v['diff_ci95'][1]*100:+.2f}]")
Path("data/artifacts/baseline_eval/practical_metrics.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
