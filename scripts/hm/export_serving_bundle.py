"""Collect the files the H&M recommender service needs into one folder (default data/hm/serving).

The refit tower checkpoint carries two 105k x 512 embedding buffers (~430 MB) that the service reloads from the
.npy files anyway, so they are dropped here (same trick as hm_export/tower.pt in notebook 03). Upload the folder
(~0.5 GB) to the server; nothing else from data/hm is needed for recommendations.

    python scripts/hm/export_serving_bundle.py
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/hm")
    ap.add_argument("--ckpt-dir", type=Path, default=ROOT / "checkpoints/hm")
    ap.add_argument("--out", type=Path, default=ROOT / "data/hm/serving")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt_dir / "best_retrieval_refit.pt", map_location="cpu", weights_only=False, mmap=True)
    ckpt["state"] = {k: v.clone() for k, v in ckpt["state"].items() if k not in ("text", "image")}
    torch.save(ckpt, args.out / "tower.pt")
    for src, name in ((args.ckpt_dir / "reranker_lgbm.txt", None), (args.data / "items.parquet", None), (args.data / "text_embeddings.npy", None),
                      (args.data / "image_embeddings.npy", None), (args.data / "item_daily_counts.parquet", None)):
        shutil.copy2(src, args.out / (name or src.name))
    for f in sorted(args.out.iterdir()):
        print(f"{f.name:<28}{f.stat().st_size / 1e6:>9.1f} MB")


if __name__ == "__main__":
    main()
