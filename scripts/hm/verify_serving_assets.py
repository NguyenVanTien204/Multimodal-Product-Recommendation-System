"""Check the H&M serving bundle (web migration, step G0) is complete and self-consistent.

    python scripts/hm/verify_serving_assets.py            # default: data/hm
    python scripts/hm/verify_serving_assets.py --root data/hm

Exit code 0 only when every required check passes; warnings never fail the run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

N_ITEMS = 105_542


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/hm"))
    ap.add_argument("--tower", type=Path, default=None, help="tower checkpoint to compare item order with")
    args = ap.parse_args()
    root: Path = args.root
    serving = root / "serving"

    failures: list[str] = []

    def check(ok: bool, label: str, detail: str = "", required: bool = True) -> None:
        tag = "OK  " if ok else ("FAIL" if required else "warn")
        print(f"[{tag}] {label}{(' — ' + detail) if detail else ''}")
        if not ok and required:
            failures.append(label)

    # --- files -------------------------------------------------------------------
    need = {
        "item_catalog.parquet": serving / "item_catalog.parquet",
        "personas.parquet": serving / "personas.parquet",
        "personas_history.parquet": serving / "personas_history.parquet",
        "personas_heldout.parquet": serving / "personas_heldout.parquet",
        "serving_manifest.json": serving / "serving_manifest.json",
        "item_daily_counts.parquet": serving / "item_daily_counts.parquet",
        "hm_export/tower.pt": serving / "hm_export" / "tower.pt",
        "hm_export/model_meta.json": serving / "hm_export" / "model_meta.json",
        "items.parquet": root / "items.parquet",
        "text_embeddings.npy": root / "text_embeddings.npy",
        "image_embeddings.npy": root / "image_embeddings.npy",
    }
    present = {k: p.is_file() for k, p in need.items()}
    for k, p in need.items():
        check(present[k], f"file {k}", str(p) if not present[k] else "")
    lgbm = sorted((serving / "hm_export").glob("*_lgbm.txt")) if (serving / "hm_export").is_dir() else []
    check(bool(lgbm), "hm_export/*_lgbm.txt (reranker)", ", ".join(p.name for p in lgbm))

    # --- catalog ------------------------------------------------------------------
    if present["item_catalog.parquet"] and present["items.parquet"]:
        cat = pl.read_parquet(need["item_catalog.parquet"])
        items = pl.read_parquet(need["items.parquet"], columns=["item_id"])
        check(cat.height == N_ITEMS, "catalog rows", f"{cat.height:,}")
        check(cat["item_id"].to_list() == items["item_id"].to_list(), "catalog order == items.parquet")
        check(cat["product_id"].to_list() == list(range(1, cat.height + 1)), "product_id == row + 1")
        for col in ("index_group_name", "garment_group_name", "perceived_colour_master_name", "product_code", "detail_desc", "n_28d", "price_median"):
            check(col in cat.columns, f"catalog column {col}")
        check(int(cat["n_all"].sum()) > 30_000_000 * 0.9, "catalog n_all sums to ~all transactions < AS_OF", f"{int(cat['n_all'].sum()):,}", required=False)
        print(f"       never-sold (global stats): {int((cat['n_all'] == 0).sum()):,} | selling in last 28d: {int((cat['n_28d'] > 0).sum()):,}")

        # item order must match the tower's vocabulary: product_id == tower index
        tower_path = args.tower or next((p for p in (root.parent.parent / "checkpoints" / "hm" / "best_retrieval_refit.pt",) if p.is_file()), None)
        if tower_path is not None:
            import torch

            ids = [str(x) for x in torch.load(tower_path, map_location="cpu", weights_only=False, mmap=True)["item_ids"]]
            check(ids == cat["item_id"].to_list(), f"catalog order == tower item_ids ({tower_path.name})")

        # --- daily counts agree with the catalog stats --------------------------------
        if present["item_daily_counts.parquet"]:
            dc = pl.read_parquet(need["item_daily_counts.parquet"], columns=["article_id", "t_dat", "n"])
            manifest = json.loads(need["serving_manifest.json"].read_text()) if present["serving_manifest.json"] else {}
            as_of = manifest.get("as_of", "2020-09-16")
            before = dc.filter(pl.col("t_dat") < pl.lit(as_of).str.to_date())
            check(int(before["n"].sum()) == int(cat["n_all"].sum()), "item_daily_counts (< AS_OF) total == catalog n_all", f"{int(before['n'].sum()):,} vs {int(cat['n_all'].sum()):,}")

        # --- images ------------------------------------------------------------------
        img_root = root / "images"
        on_disk = sum(1 for _ in img_root.glob("*/*.jpg")) if img_root.is_dir() else 0
        expect = int(cat["has_image_src"].sum())
        check(on_disk >= expect * 0.99, "images extracted to data/hm/images/", f"{on_disk:,} on disk / {expect:,} expected")

    # --- personas -----------------------------------------------------------------
    if present["personas.parquet"] and present["personas_history.parquet"] and present["personas_heldout.parquet"]:
        p = pl.read_parquet(need["personas.parquet"])
        h = pl.read_parquet(need["personas_history.parquet"])
        t = pl.read_parquet(need["personas_heldout.parquet"])
        check(p.height >= 10, "personas", f"{p.height}")
        check(set(h["customer_id"]) == set(p["customer_id"]) == set(t["customer_id"]), "personas: history/heldout cover the same customers")
        check(h["t_dat"].max() < t["t_dat"].min(), "personas: history strictly before held-out")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: " + "; ".join(failures))
        return 1
    print("All required checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
