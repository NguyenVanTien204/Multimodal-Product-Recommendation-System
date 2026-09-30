"""List the data files to upload for notebooks/kaggle_rag_reviews_and_eval.ipynb.

    python scripts/pack_kaggle_rag_inputs.py

The notebook needs exactly ONE Kaggle Dataset containing these 9 files, and nothing
else: all source code it uses is embedded in the notebook itself.
Light-weight: no model or GPU involved.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_FILES = [
    "data/items.parquet",
    "data/train.parquet",
    "data/valid.parquet",
    "data/test.parquet",
    "data/candidate_interactions.parquet",
    "data/embedding/image_embeddings.npy",
    "data/embedding/image_embedding_metadata.parquet",
    "data/embedding/text_embeddings.npy",
    "data/embedding/text_embedding_metadata.parquet",
]


def main() -> None:
    print("Upload these 9 files as ONE Kaggle dataset (e.g. `datn-rag-inputs`); sub-folders do not matter:")
    total, missing = 0, 0
    for rel in DATA_FILES:
        p = ROOT / rel
        if p.exists():
            total += p.stat().st_size
            print(f"  OK       {rel:52s} {p.stat().st_size / 2**20:9.1f} MiB")
        else:
            missing += 1
            print(f"  MISSING  {rel}")
    print(f"\ntotal {total / 2**30:.2f} GiB" + (f" ({missing} missing!)" if missing else ""))


if __name__ == "__main__":
    main()
