"""H&M shop catalog: build it from the dataset files, then load it into Postgres (+ the backend's similarity collection).

    # 1. (needs the extracted images folder to decide which items are sellable)
    python -m datn.catalog.cli build-hm --serving data/hm/serving --images data/hm/images
    # 2. backend must have started once on the target DB so its tables exist (SQLAlchemy create_all)
    python -m datn.catalog.cli import-hm --serving data/hm/serving --pg-dsn postgresql://... --qdrant-url http://...
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import polars as pl

from .hm import CATEGORIES, build_shop_catalog

log = logging.getLogger("datn.catalog")
NOW = "2026-10-06 00:00:00"


def cmd_build(args: argparse.Namespace) -> None:
    serving: Path = args.serving
    items = pl.read_parquet(serving / "items.parquet")
    counts = pl.read_parquet(serving / "item_daily_counts.parquet", columns=["article_id", "t_dat", "n", "price_sum"])
    ids = items["item_id"].to_list()
    image_ids = {i for i in ids if (args.images / i[:3] / f"{i}.jpg").is_file()}
    log.info("catalog %d items, %d with a photo on disk", len(ids), len(image_ids))

    catalog, manifest = build_shop_catalog(
        items, counts, as_of=dt.date.fromisoformat(args.as_of), image_ids=image_ids, median_target_vnd=args.median_price
    )
    catalog.write_parquet(serving / "shop_catalog.parquet", compression="zstd")
    active = catalog.filter(pl.col("active"))["sku"].to_list()
    (serving / "shop_item_ids.txt").write_text("\n".join(active) + "\n", encoding="utf-8")
    (serving / "shop_catalog_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def cmd_import(args: argparse.Namespace) -> None:
    import psycopg
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm

    serving: Path = args.serving
    catalog = pl.read_parquet(serving / "shop_catalog.parquet").filter(pl.col("active"))
    log.info("importing %d active products", catalog.height)

    with psycopg.connect(args.pg_dsn) as conn, conn.cursor() as cur:
        # `create_all` never alters an existing table; these two columns were added for H&M.
        cur.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS audience VARCHAR(20)")
        cur.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS attributes JSON")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_products_audience ON products (audience)")
        for slug, name in CATEGORIES:
            cur.execute(
                "INSERT INTO categories (name, slug) VALUES (%s, %s) ON CONFLICT (slug) DO UPDATE SET name = EXCLUDED.name",
                (name, slug),
            )
        cur.execute("SELECT slug, id FROM categories")
        cat_id = dict(cur.fetchall())

        cur.execute("TRUNCATE TABLE products RESTART IDENTITY CASCADE")
        t0 = time.time()
        with cur.copy(
            "COPY products (id, sku, name, description, price, stock_quantity, image_url, category_id, is_active, created_at, audience, attributes) FROM STDIN"
        ) as copy:
            for row in catalog.iter_rows(named=True):
                copy.write_row(
                    (
                        row["product_id"], row["sku"], row["name"][:255], row["description"], row["price_vnd"], row["stock"],
                        row["image_url"], cat_id[row["category_slug"]], True, NOW, row["audience"], row["attributes"],
                    )
                )
        cur.execute("SELECT setval('products_id_seq', (SELECT max(id) FROM products))")
        conn.commit()
        log.info("postgres COPY done in %.1fs", time.time() - t0)

    # the backend's "similar products"/fallback collection: one unnamed text vector per product, id == product_id
    client = QdrantClient(url=args.qdrant_url, timeout=120)
    name = args.product_collection
    text = np.load(serving / "text_embeddings.npy", mmap_mode="r")
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(name, vectors_config=qm.VectorParams(size=int(text.shape[1]), distance=qm.Distance.COSINE, on_disk=True))
    pids = catalog["product_id"].to_list()
    skus = catalog["sku"].to_list()
    for start in range(0, len(pids), 1000):
        chunk = range(start, min(start + 1000, len(pids)))
        client.upsert(
            name,
            points=[
                qm.PointStruct(id=int(pids[i]), vector=np.asarray(text[pids[i] - 1], dtype=np.float32).tolist(), payload={"product_id": int(pids[i]), "sku": skus[i]})
                for i in chunk
            ],
            wait=True,
        )
    log.info("qdrant %s: %d vectors (%d-d)", name, len(pids), text.shape[1])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="datn-catalog")
    p.add_argument("--serving", type=Path, default=Path("data/hm/serving"))
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-hm")
    b.add_argument("--images", type=Path, default=Path("data/hm/images"))
    b.add_argument("--as-of", default="2020-09-16")
    b.add_argument("--median-price", type=int, default=350_000, help="catalog median price in VND (price convention, see catalog/hm.py)")
    i = sub.add_parser("import-hm")
    i.add_argument("--pg-dsn", default=os.environ.get("DATN_PG_DSN", "postgresql://mini_market:mini_market@localhost:5432/mini_market_hm"))
    i.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    i.add_argument("--product-collection", default="hm_product_embeddings")
    for name, helptext in (("mock-candidates", "nearest H&M products of every reviewed Amazon product (+ similarity profile)"),
                           ("mock-build", "assign Amazon reviews to H&M products and write hm_reviews_meta/embeddings")):
        m = sub.add_parser(name, help=helptext)
        m.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
        m.add_argument("--amazon-dir", type=Path, default=Path("data/embedding"), help="text_embeddings.npy + text_embedding_metadata.parquet of the Amazon catalog")
        if name == "mock-build":
            m.add_argument("--min-sim", type=float, required=True)
            m.add_argument("--cap-per-item", type=int, default=6)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.cmd in ("mock-candidates", "mock-build"):
        from qdrant_client import QdrantClient

        from . import mock_reviews_job as job

        client = QdrantClient(url=args.qdrant_url, timeout=300)
        if args.cmd == "mock-candidates":
            job.run_candidates(client, serving=args.serving, amazon_dir=args.amazon_dir)
        else:
            job.run_build(client, serving=args.serving, amazon_dir=args.amazon_dir, min_sim=args.min_sim, cap_per_item=args.cap_per_item)
        return
    {"build-hm": cmd_build, "import-hm": cmd_import}[args.cmd](args)


if __name__ == "__main__":
    main()
