"""Probe: are cold items pushed down by their own id_residual (learned only from negative samples)?
Inference-only test: zero id_residual for cold items (0 positive train interactions) and re-evaluate. No retraining.
Usage: PYTHONPATH=legacy/src:legacy/scripts python legacy/scripts/eval_cold_fix_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_baselines import D, P, mask_scores, ranks_from_scores  # noqa: E402

from datn_legacy.recommenders.reranker.inference import RerankerPipeline  # noqa: E402
from datn_legacy.recommenders.user_tower.content import ContentSource  # noqa: E402
from datn_legacy.recommenders.user_tower.dataset import build_eval_examples, build_item_vocab, load_user_sequences  # noqa: E402

torch.set_num_threads(4)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = build_item_vocab(P / "items.parquet")
seqs = load_user_sequences(P / "train.parquet", P / "valid.parquet", P / "test.parquet", vocab)
V = vocab.vocab_size
deg = torch.zeros(V, device=dev)
for s in seqs.train.values():
    deg[torch.tensor(s, device=dev)] += 1
real = torch.zeros(V, dtype=torch.bool, device=dev); real[1:vocab.num_items + 1] = True
cold, warm = (deg == 0) & real, (deg > 0) & real
src = [ContentSource("image", D / "embedding/image_embeddings.npy", D / "embedding/image_embedding_metadata.parquet"),
       ContentSource("text", D / "embedding/text_embeddings.npy", D / "embedding/text_embedding_metadata.parquet")]
pipe = RerankerPipeline.from_artifacts(
    user_tower_dir=D / "artifacts/user_tower_balanced_v1", reranker_dir=D / "artifacts/reranker_v2", items_path=P / "items.parquet",
    train_path=P / "train.parquet", valid_path=P / "valid.parquet", test_path=P / "test.parquet", content_sources=src, device="auto")
tower = pipe.tower
with torch.no_grad():
    idr = tower.id_residual.weight.to(dev)
    content_part = tower.content_proj(tower.content_matrix).to(dev)
    print("mean L2 norm  id_residual: cold %.3f  warm %.3f | content_proj: cold %.3f  warm %.3f" % (
        idr[cold].norm(dim=1).mean(), idr[warm].norm(dim=1).mean(), content_part[cold].norm(dim=1).mean(), content_part[warm].norm(dim=1).mean()))
    print("cold id_residual exactly zero rows:", int((idr[cold].abs().sum(1) == 0).sum()), "of", int(cold.sum()))
    full = tower.get_item_table().to(dev)
    fixed = full.clone(); fixed[cold] = content_part[cold]           # variant A: drop the learned id_residual of cold items only
    # variant B: also give every item with <=1 train interaction its content-only vector
    fixed2 = full.clone(); low = (deg <= 1) & real; fixed2[low] = content_part[low]
    # cosine of the id_residual of cold items with the mean user query direction (negative => pushed away from users)

out = {}
for split in ("valid", "test"):
    ex = build_eval_examples(seqs, 20, split)
    seen = [sorted(e.seen) for e in ex]
    tgt = torch.tensor([e.target for e in ex], device=dev)
    ctx = torch.from_numpy(np.stack([e.context for e in ex])).to(pipe.device)
    cold_t = cold[tgt].cpu().numpy()
    res = {}
    for name, table in (("as_trained", full), ("cold_idres_zeroed", fixed), ("idres_zeroed_deg<=1", fixed2)):
        rk = np.empty(len(ex)); gap = []; in_top100 = 0; distinct = set()
        with torch.no_grad():
            for s in range(0, len(ex), 512):
                e = min(len(ex), s + 512)
                uv = tower.encode_user(ctx[s:e], table.to(pipe.device))
                sc = (uv @ table.to(pipe.device).T).to(dev)
                gap.append(((sc[:, cold].mean(1) - sc[:, warm].mean(1)) / sc[:, warm].std(1)).cpu().numpy())
                mask_scores(sc, seen[s:e], vocab.oov_idx)
                rk[s:e] = ranks_from_scores(sc, tgt[s:e]).cpu().numpy()
                top = sc.topk(100, dim=1).indices
                in_top100 += int(cold[top].any(1).sum()); distinct.update(top[cold[top]].cpu().tolist())
        hr = lambda m, K: float(np.mean(rk[m] < K))
        allm = np.ones(len(ex), bool)
        res[name] = {"gap_std": float(np.mean(np.concatenate(gap))),
                     "overall": {f"HR@{K}": hr(allm, K) for K in (10, 50, 100)},
                     "cold_targets": {f"HR@{K}": hr(cold_t, K) for K in (10, 50, 100)},
                     "warm_targets": {f"HR@{K}": hr(~cold_t, K) for K in (10, 50, 100)},
                     "users_with_cold_in_top100": in_top100 / len(ex), "distinct_cold_in_top100": len(distinct)}
        v = res[name]
        print(f"[{split}] {name:22s} gap={v['gap_std']:+.2f}sd  overall HR@10/50/100 = " + "/".join(f"{v['overall'][f'HR@{k}']*100:.2f}" for k in (10, 50, 100))
              + "  cold = " + "/".join(f"{v['cold_targets'][f'HR@{k}']*100:.2f}" for k in (10, 50, 100))
              + "  warm = " + "/".join(f"{v['warm_targets'][f'HR@{k}']*100:.2f}" for k in (10, 50, 100))
              + f"  cold-in-top100 users={v['users_with_cold_in_top100']*100:.1f}% distinct={v['distinct_cold_in_top100']}", flush=True)
    out[split] = res
Path("data/artifacts/baseline_eval/cold_fix_probe.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
