from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from qdrant_client import QdrantClient

from . import schema as S
from .encoder import JinaClipEncoder
from .indexer import import_reviews, index_products, index_reviews

DEFAULT_DSN = "postgresql://mini_market:mini_market_secret_pass_2026@localhost:5432/mini_market"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="datn-retrieval", description="Build/inspect the multimodal retrieval indexes (Qdrant)")
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--pg-dsn", default=os.environ.get("DATN_PG_DSN", DEFAULT_DSN))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index-products", help="products collection: image+text vectors + shop payload")
    p.add_argument("--recreate", action="store_true")
    p.add_argument("--batch-size", type=int, default=256)

    h = sub.add_parser("index-hm", help="H&M products collection (512-d; needs DATN_VECTOR_SIZE=512 DATN_PRODUCTS_COLLECTION=hm_products)")
    h.add_argument("--serving", type=Path, default=Path("data/hm/serving"))
    h.add_argument("--recreate", action="store_true")

    hr = sub.add_parser("index-hm-reviews", help="H&M mock reviews -> reviews collection + review_count/avg_rating on products")
    hr.add_argument("--serving", type=Path, default=Path("data/hm/serving"))
    hr.add_argument("--keep-existing", action="store_true")

    r = sub.add_parser("index-reviews", help="reviews collection: Jina CLIP v2 text vectors (GPU recommended, resumable)")
    r.add_argument("--recreate", action="store_true")
    r.add_argument("--chunk-size", type=int, default=2048)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--device", default="auto")

    m = sub.add_parser("import-reviews", help="load review vectors computed on Kaggle (no GPU needed)")
    m.add_argument("--embeddings", type=Path, required=True, help="review_embeddings.npy from the notebook")
    m.add_argument("--meta", type=Path, required=True, help="reviews_meta.parquet from the notebook")
    m.add_argument("--keep-existing", action="store_true", help="do not recreate the collection first")

    sub.add_parser("info", help="print collection sizes")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = QdrantClient(url=args.qdrant_url, timeout=120)

    if args.cmd == "index-products":
        index_products(client, pg_dsn=args.pg_dsn, data_dir=args.data_dir, recreate=args.recreate, batch_size=args.batch_size)
    elif args.cmd == "index-hm":
        from .hm_indexer import index_hm_products

        index_hm_products(client, pg_dsn=args.pg_dsn, serving_dir=args.serving, recreate=args.recreate)
    elif args.cmd == "index-hm-reviews":
        from .hm_indexer import import_hm_reviews

        import_hm_reviews(client, serving_dir=args.serving, recreate=not args.keep_existing)
    elif args.cmd == "index-reviews":
        encoder = JinaClipEncoder(device=args.device)
        index_reviews(
            client,
            encoder,
            pg_dsn=args.pg_dsn,
            data_dir=args.data_dir,
            progress_file=args.data_dir / "rag" / "reviews_index_progress.json",
            recreate=args.recreate,
            chunk_size=args.chunk_size,
            limit=args.limit,
        )
    elif args.cmd == "import-reviews":
        import_reviews(client, embeddings_path=args.embeddings, meta_path=args.meta, recreate=not args.keep_existing)
    else:
        for name in (S.PRODUCTS_COLLECTION, S.REVIEWS_COLLECTION):
            if client.collection_exists(name):
                info = client.get_collection(name)
                print(name, "points:", info.points_count, "status:", info.status)
            else:
                print(name, "missing")


if __name__ == "__main__":
    main()
