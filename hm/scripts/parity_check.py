"""Parity check: hm/src/datn/recommenders/hm (port) vs the reference code in hm/notebooks/03 (cells 2-10).

Runs both on the same test users (history = train + selection + valid, shop clock = test cutoff, tower = refit,
ranker = refit LightGBM) on CPU and compares candidate unions, the 24-column feature matrix and the final top-K.

    python hm/scripts/parity_check.py --users 40
    python hm/scripts/parity_check.py --users 40 --tower "hm/checkpoints/best_retrieval_refit.pt"

Needs ~4 GB RAM (the notebook builds nn.Module buffers); uses CPU only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # both implementations on CPU: identical kernels, no GPU contention

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hm" / "src"))

NOTEBOOK = ROOT / "hm/notebooks/03_hm_reranking_evaluation.ipynb"
CELLS = (2, 3, 4, 6, 8, 10)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=40)
    ap.add_argument("--data", type=Path, default=ROOT / "data/hm")
    ap.add_argument("--ckpt-dir", type=Path, default=ROOT / "hm/checkpoints")
    ap.add_argument("--tower", type=Path, default=None, help="refit tower (default: <ckpt-dir>/best_retrieval_refit.pt)")
    ap.add_argument("--ranker", type=Path, default=None, help="LightGBM model (default: <ckpt-dir>/reranker_lgbm.txt)")
    ap.add_argument("--atol", type=float, default=1e-4)
    args = ap.parse_args()

    tower_path = args.tower or args.ckpt_dir / "best_retrieval_refit.pt"
    ranker_path = args.ranker or args.ckpt_dir / "reranker_lgbm.txt"
    os.environ.update(HM_DATA_DIR=str(args.data), HM_RETRIEVAL_DIR=str(args.ckpt_dir), HM_RERANK_DIR=str(ROOT / "data/_parity_tmp"))

    # ---- reference: execute the notebook's own cells ----------------------------------
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    src = "\n\n".join("".join(cells[i]["source"]) for i in CELLS)
    # the notebook loads best_retrieval.pt as both base and refit; point it at the refit tower we serve
    assert tower_path.parent == args.ckpt_dir, "--tower must live in --ckpt-dir (the notebook resolves it via RETRIEVAL_DIR)"
    src = src.replace('"best_retrieval.pt"', repr(tower_path.name))
    src = src.replace("torch.cuda.is_available()", "False")  # the notebook picks its device from this
    ns: dict = {"__name__": "nb03_reference"}
    print("running notebook cells", CELLS, "...")
    exec(compile(src, "nb03", "exec"), ns)  # noqa: S102

    import lightgbm as lgb
    import numpy as np

    from datn.recommenders.hm import HMRecommender

    test_ctx = {u: ns["train"].get(u, []) + ns["selection"].get(u, []) + ns["valid"].get(u, []) for u in ns["test"]}
    users = sorted(test_ctx)[: args.users]
    print(f"comparing {len(users)} test users | serving rule: {ns['SERVING']} | cutoff day {ns['CUTOFF']['test']}")

    # ---- port ---------------------------------------------------------------------------
    import datetime as dt

    port = HMRecommender.from_artifacts(
        tower_path=tower_path, ranker_path=ranker_path, items_path=args.data / "items.parquet",
        text_path=args.data / "text_embeddings.npy", image_path=args.data / "image_embeddings.npy",
        daily_counts_path=args.data / "item_daily_counts.parquet", as_of=dt.date(1970, 1, 1) + dt.timedelta(days=ns["CUTOFF"]["test"]), device="cpu",
    )
    assert port.as_of_day == ns["CUTOFF"]["test"]
    booster = lgb.Booster(model_file=str(ranker_path))

    stats_ref = ns["STATS"]["test"]
    tower_ref = ns["towers"]["refit"]
    bad = 0
    got = port.candidates_and_features([test_ctx[u] for u in users])
    for chunk_start in range(0, len(users), 32):
        chunk = users[chunk_start:chunk_start + 32]
        b = ns["batch_state"](tower_ref, test_ctx, chunk, stats_ref)
        for i, u in enumerate(chunk):
            hist = test_ctx[u]
            sources = ns["build_sources"](b, i, hist, stats_ref)
            union = list(dict.fromkeys(x for name in ns["SOURCES"] for x in sources[name] if x != 0))
            x_ref, _ = ns["featurize"](b, i, union, hist, sources, stats_ref, tower_ref)
            g = got[chunk_start + i]
            same_union = union == g["items"].tolist()
            max_diff = float(np.abs(x_ref - g["x"]).max()) if same_union else float("nan")
            ref_top = np.asarray(union)[np.argsort(-booster.predict(x_ref), kind="stable")][:12].tolist()
            new_top = g["items"][np.argsort(-booster.predict(g["x"]), kind="stable")][:12].tolist()
            ok = same_union and max_diff <= args.atol and ref_top == new_top
            bad += not ok
            if not ok or i == 0:
                print(f"  user {u[:10]}…: union_equal={same_union} (|ref|={len(union)}, |port|={len(g['items'])}) max|dX|={max_diff:.2e} top12_equal={ref_top == new_top}")
    print(f"\n{'PARITY OK' if not bad else f'PARITY FAILED for {bad}/{len(users)} users'} (atol={args.atol})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
