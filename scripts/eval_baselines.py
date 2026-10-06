"""Baselines + paired-bootstrap significance audit for the balanced two-stage recommender.

Reproduces the repo's protocol exactly (same loaders, leave-last-out, full ranking over the 32,557-item catalog,
seen items masked) and adds: popularity, item-kNN, transition (Markov) kNN, content kNN, brand/category-of-last-item,
a reciprocal-rank-fusion hybrid, the raw User Tower and the final User Tower + Reranker v2 pipeline.

Usage (repo root, CPU is fine; GPU used automatically if present):
    python scripts/eval_baselines.py --split test --out data/artifacts/baseline_eval
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from datn.recommenders.reranker.candidates import candidate_features
from datn.recommenders.reranker.inference import RerankerPipeline
from datn.recommenders.user_tower.content import ContentSource, load_content_matrix
from datn.recommenders.user_tower.dataset import PAD_IDX, build_eval_examples, build_item_vocab, load_user_sequences

KS = (10, 50, 100)
D = Path("data")
P = D / "processed/balanced_u5_i2_v1"


def ranks_from_scores(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """0-indexed rank with average tie handling (ties do NOT favour the target)."""
    ts = scores.gather(1, targets[:, None])
    gt = (scores > ts).sum(1).float()
    eq = (scores == ts).sum(1).float() - 1.0
    return gt + 0.5 * eq


def mask_scores(scores: torch.Tensor, seen_lists: list[list[int]], oov_idx: int) -> None:
    scores[:, PAD_IDX] = float("-inf")
    scores[:, oov_idx] = float("-inf")
    for i, s in enumerate(seen_lists):
        if s:
            scores[i, torch.tensor(s, device=scores.device, dtype=torch.long)] = float("-inf")


def build_sparse(rows, cols, vals, n, device):
    idx = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(vals, dtype=torch.float32), (n, n)).coalesce().to(device)


def item_sims(train_seqs: dict[str, list[int]], V: int, device, mode: str, shrink: float = 1.0):
    """mode='cooc': cosine of binary co-occurrence. mode='trans': forward transitions within 3 steps, decay 1/d."""
    rows, cols, vals = [], [], []
    for seq in train_seqs.values():
        if mode == "cooc":
            u = np.unique(np.asarray(seq))
            a, b = np.meshgrid(u, u, indexing="ij")
            m = a != b
            rows.append(a[m]); cols.append(b[m]); vals.append(np.ones(m.sum(), dtype=np.float32))
        else:
            for d in (1, 2, 3):
                if len(seq) > d:
                    rows.append(np.asarray(seq[:-d])); cols.append(np.asarray(seq[d:]))
                    vals.append(np.full(len(seq) - d, 1.0 / d, dtype=np.float32))
    r, c, v = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    S = build_sparse(r, c, v, V, device)
    deg = torch.zeros(V, device=device)
    for seq in train_seqs.values():
        deg[torch.tensor(seq, device=device)] += 1
    idx = S.indices()
    w = S.values() / torch.sqrt((deg[idx[0]] + shrink) * (deg[idx[1]] + shrink)) if mode == "cooc" else S.values() / torch.sqrt(deg[idx[0]] + shrink)
    return torch.sparse_coo_tensor(idx, w, (V, V)).coalesce()


def sparse_hist(weights: list[dict[int, float]], V: int, device) -> torch.Tensor:
    r, c, v = [], [], []
    for i, w in enumerate(weights):
        for k, x in w.items():
            r.append(i); c.append(k); v.append(x)
    idx = torch.tensor([r, c], dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(v, dtype=torch.float32), (len(weights), V)).to(device)


def bootstrap(hits: dict[str, np.ndarray], base: str, n_boot: int, seed: int = 7):
    n = len(next(iter(hits.values())))
    rng = np.random.default_rng(seed)
    names = list(hits)
    means = {m: [] for m in names}
    for _ in range(n_boot // 100):
        idx = rng.integers(0, n, size=(100, n))
        for m in names:
            means[m].append(hits[m][idx].mean(axis=1))
    means = {m: np.concatenate(v) for m, v in means.items()}
    out = {}
    for m in names:
        lo, hi = np.percentile(means[m], [2.5, 97.5])
        d = means[m] - means[base]
        b = int(((hits[m] == 1) & (hits[base] == 0)).sum()); c = int(((hits[m] == 0) & (hits[base] == 1)).sum())
        k, tot = min(b, c), b + c
        p = min(1.0, 2 * sum(math.comb(tot, i) for i in range(k + 1)) / 2 ** tot) if tot < 1000 else float("nan")
        if tot >= 1000:  # normal approx of the exact McNemar test
            z = (abs(b - c) - 1) / math.sqrt(tot); p = math.erfc(z / math.sqrt(2))
        out[m] = dict(hr=float(hits[m].mean()), ci95=[float(lo), float(hi)], diff_vs_base=float(hits[m].mean() - hits[base].mean()),
                      diff_ci95=[float(x) for x in np.percentile(d, [2.5, 97.5])], only_this=b, only_base=c, mcnemar_p=p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="data/artifacts/baseline_eval")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N users")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip-pipeline", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    vocab = build_item_vocab(P / "items.parquet")
    seqs = load_user_sequences(P / "train.parquet", P / "valid.parquet", P / "test.parquet", vocab)
    ex = build_eval_examples(seqs, 20, args.split)
    if args.limit:
        ex = ex[: args.limit]
    n, V = len(ex), vocab.vocab_size
    print(f"split={args.split} users={n} catalog={vocab.num_items} device={dev}", flush=True)
    hist = [[int(i) for i in e.context if i != PAD_IDX] for e in ex]  # chronological, most recent last
    seen = [sorted(e.seen) for e in ex]
    targets = torch.tensor([e.target for e in ex], device=dev)
    train_pos_seqs = seqs.train

    counts = torch.zeros(V, device=dev)
    for s in train_pos_seqs.values():
        counts[torch.tensor(s, device=dev)] += 1
    pop_norm = counts / (counts.max() + 1)
    target_degree = counts[targets].cpu().numpy()
    hist_len = np.array([len(h) for h in hist])

    # content matrix (image+text, each L2-normalised)
    src = [ContentSource("image", D / "embedding/image_embeddings.npy", D / "embedding/image_embedding_metadata.parquet"),
           ContentSource("text", D / "embedding/text_embeddings.npy", D / "embedding/text_embedding_metadata.parquet")]
    cm = load_content_matrix(vocab, src)
    half = cm.shape[1] // 2
    cm = np.concatenate([cm[:, :half] / (np.linalg.norm(cm[:, :half], axis=1, keepdims=True) + 1e-9),
                         cm[:, half:] / (np.linalg.norm(cm[:, half:], axis=1, keepdims=True) + 1e-9)], axis=1) / math.sqrt(2)
    X = torch.tensor(cm, device=dev)

    items = pl.read_parquet(P / "items.parquet", columns=["category", "brand"])
    cat = torch.zeros(V, dtype=torch.long, device=dev); brand = torch.zeros(V, dtype=torch.long, device=dev)
    cat[1:vocab.num_items + 1] = torch.tensor(items["category"].fill_null("?").cast(pl.Categorical).to_physical().to_numpy().astype(np.int64) + 1, device=dev)
    brand_phys = items["brand"].fill_null("?").cast(pl.Categorical).to_physical().to_numpy().astype(np.int64) + 1
    brand_missing = (items["brand"].fill_null("").str.len_chars() == 0).to_numpy()
    brand[1:vocab.num_items + 1] = torch.tensor(np.where(brand_missing, 0, brand_phys), device=dev)

    S_cooc = item_sims(train_pos_seqs, V, dev, "cooc")
    S_trans = item_sims(train_pos_seqs, V, dev, "trans")
    print(f"prepared in {time.time()-t0:.0f}s; cooc nnz={S_cooc._nnz()} trans nnz={S_trans._nnz()}", flush=True)

    methods: dict[str, np.ndarray] = {}   # name -> per-user rank (float, inf if beyond what is measurable)
    tops: dict[str, np.ndarray] = {}      # name -> (n,10) top-10 item idx (for coverage)
    BS = 512

    def run(name, score_fn):
        rk = np.empty(n, dtype=np.float64); tp = np.empty((n, 10), dtype=np.int64)
        for s in range(0, n, BS):
            e = min(n, s + BS)
            sc = score_fn(s, e).clone()
            mask_scores(sc, seen[s:e], vocab.oov_idx)
            rk[s:e] = ranks_from_scores(sc, targets[s:e]).cpu().numpy()
            tp[s:e] = sc.topk(10, dim=1).indices.cpu().numpy()
        methods[name] = rk; tops[name] = tp
        print(f"  {name:28s} HR@10={np.mean(rk<10)*100:.3f} HR@50={np.mean(rk<50)*100:.3f} HR@100={np.mean(rk<100)*100:.3f}", flush=True)

    tie = 1e-7
    run("random", lambda s, e: torch.rand(e - s, V, device=dev))
    run("popularity", lambda s, e: counts.expand(e - s, V))

    def knn(S, decay, last_only_k=None):
        def f(s, e):
            w = []
            for h in hist[s:e]:
                hh = h[-last_only_k:] if last_only_k else h
                w.append({item: decay ** (len(hh) - 1 - j) for j, item in enumerate(hh)})
            H = sparse_hist(w, V, dev)
            return torch.sparse.mm(H, S).to_dense() + tie * pop_norm
        return f

    run("itemknn_cooc", knn(S_cooc, 1.0))
    run("itemknn_cooc_recency", knn(S_cooc, 0.7))
    run("transition_last3", knn(S_trans, 0.6, 3))

    def content_centroid(decay):
        def f(s, e):
            C = torch.stack([(torch.tensor([decay ** (len(h) - 1 - j) for j in range(len(h))], device=dev)[:, None] * X[h]).sum(0) for h in hist[s:e]])
            return C @ X.T
        return f
    run("content_centroid", content_centroid(0.7))
    run("content_last_item", lambda s, e: X[[h[-1] for h in hist[s:e]]] @ X.T)

    def same_attr(attr_vec_a, attr_vec_b):
        def f(s, e):
            last = torch.tensor([h[-1] for h in hist[s:e]], device=dev)
            a = (attr_vec_a[None, :] == attr_vec_a[last][:, None]).float()
            b = ((attr_vec_b[None, :] == attr_vec_b[last][:, None]) & (attr_vec_b[last][:, None] > 0)).float()
            return 2 * b + a + pop_norm[None, :]
        return f
    run("same_brand_cat_last+pop", same_attr(cat, brand))

    # RRF hybrid of three unrelated signals (no tuning)
    def rrf_hybrid(s, e):
        acc = torch.zeros(e - s, V, device=dev)
        for fn in (knn(S_cooc, 0.7), content_centroid(0.7), lambda a, b: counts.expand(b - a, V)):
            sc = fn(s, e).clone(); mask_scores(sc, seen[s:e], vocab.oov_idx)
            rank = sc.argsort(dim=1, descending=True).argsort(dim=1).float()
            acc += 1.0 / (60.0 + rank)
        return acc
    run("rrf(knn,content,pop)", rrf_hybrid)

    pipe = RerankerPipeline.from_artifacts(
        user_tower_dir=D / "artifacts/user_tower_balanced_v1", reranker_dir=D / "artifacts/reranker_v2",
        items_path=P / "items.parquet", train_path=P / "train.parquet", valid_path=P / "valid.parquet", test_path=P / "test.parquet",
        content_sources=src, device="auto")
    item_emb = pipe.tower.item_vectors()
    ctx = torch.from_numpy(np.stack([e.context for e in ex])).to(pipe.device)

    @torch.no_grad()
    def tower_scores(s, e):
        uv = pipe.tower.encode_user(ctx[s:e], item_emb)
        return (uv @ item_emb.T).to(dev)
    run("user_tower_raw", tower_scores)

    if not args.skip_pipeline:
        print("  running reranker pipeline (batched)...", flush=True)
        from datn.recommenders.reranker.candidates import CandidateExample
        rk = np.full(n, np.inf); tp = np.zeros((n, 10), dtype=np.int64)
        with torch.no_grad():
            for s in range(0, n, 64):
                batch = [CandidateExample(context=e.context, seen=e.seen) for e in ex[s:s + 64]]
                rows = candidate_features(pipe.tower, batch, pipe.vocab, pipe.popularity, pipe.budget, pipe.device)
                for j, (cand, feats) in enumerate(rows):
                    if len(cand) == 0:
                        continue
                    x = torch.from_numpy(((feats - pipe.feature_mean) / pipe.feature_std).astype(np.float32)).to(pipe.device)
                    sc = pipe.ranker(x, blend=pipe.blend).cpu().numpy()
                    order = np.argsort(-sc, kind="stable")
                    pos = np.where(cand[order] == ex[s + j].target)[0]
                    if len(pos):
                        rk[s + j] = float(pos[0])
                    top = cand[order][:10]; tp[s + j, :len(top)] = top
                if (s // 64) % 50 == 0:
                    print(f"    {s}/{n} ({time.time()-t0:.0f}s)", flush=True)
        methods["tower+reranker_v2"] = rk; tops["tower+reranker_v2"] = tp
        print(f"  {'tower+reranker_v2':28s} HR@10={np.mean(rk<10)*100:.3f} HR@50={np.mean(rk<50)*100:.3f} HR@100={np.mean(rk<100)*100:.3f}", flush=True)

    # ---- report ----
    report: dict = {"split": args.split, "n_users": n, "catalog": vocab.num_items, "seconds": round(time.time() - t0)}
    for K in KS:
        hits = {m: (r < K).astype(np.int8) for m, r in methods.items()}
        report[f"HR@{K}"] = bootstrap(hits, "popularity", args.n_boot)
    report["NDCG@10"] = {m: float(np.where(r < 10, 1 / np.log2(r + 2), 0).mean()) for m, r in methods.items()}
    report["coverage@10"] = {m: float(len(np.unique(t)) / vocab.num_items) for m, t in tops.items()}
    cold = target_degree == 0
    report["slice_target_cold"] = {"share": float(cold.mean())}
    for name, mask in (("warm", ~cold), ("cold", cold)):
        report["slice_target_cold"][name] = {m: {f"HR@{K}": float(np.mean(r[mask] < K)) for K in (10, 50)} for m, r in methods.items()}
    buckets = {"len<=4": hist_len <= 4, "len5-7": (hist_len >= 5) & (hist_len <= 7), "len8-11": (hist_len >= 8) & (hist_len <= 11), "len>=12": hist_len >= 12}
    report["slice_history_len"] = {b: {"users": int(m.sum()), **{name: {f"HR@{K}": float(np.mean(r[m] < K)) for K in (10, 50)} for name, r in methods.items()}} for b, m in buckets.items()}
    report["brand_missing_share_items"] = float(brand_missing.mean())
    report["category_counts"] = items["category"].value_counts().sort("count", descending=True).head(8).to_dicts()
    (out_dir / f"{args.split}_baselines.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out_dir / f"{args.split}_ranks.npz", **{k: v for k, v in methods.items()})
    print(f"done in {time.time()-t0:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
