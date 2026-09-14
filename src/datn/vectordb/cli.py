from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from .client import build_client
from .collection import ensure_collection
from .importer import import_embeddings
from .schema import VECTOR_IMAGE, VECTOR_TEXT

log = logging.getLogger(__name__)


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the DATN Qdrant `products` collection")
    parser.add_argument("--config", type=Path, default=Path("configs/qdrant.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create-collection", help="Create (or reuse) the products collection")
    create.add_argument("--recreate", action="store_true", help="Drop and recreate if it already exists")

    import_image = sub.add_parser("import-image", help="Upsert image embeddings into the collection")
    import_image.add_argument("--embeddings", type=Path, default=None)
    import_image.add_argument("--metadata", type=Path, default=None)

    import_text = sub.add_parser("import-text", help="Upsert text embeddings into the collection")
    import_text.add_argument("--embeddings", type=Path, default=None)
    import_text.add_argument("--metadata", type=Path, default=None)

    sub.add_parser("info", help="Print collection status")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = _load_config(args.config)
    client = build_client(config)

    if args.command == "create-collection":
        name = ensure_collection(client, config, recreate=args.recreate)
        print(f"Collection {name!r} ready")
    elif args.command == "import-image":
        ensure_collection(client, config)
        paths = config["paths"]
        embeddings = args.embeddings or Path(paths["image_embeddings"])
        metadata = args.metadata or Path(paths["image_embedding_metadata"])
        count = import_embeddings(
            client, config, VECTOR_IMAGE, embeddings, metadata, detect_fallback=True
        )
        print(f"Collection now holds {count} points with image vectors")
    elif args.command == "import-text":
        ensure_collection(client, config)
        paths = config["paths"]
        embeddings = args.embeddings or Path(paths["text_embeddings"])
        metadata = args.metadata or Path(paths["text_embedding_metadata"])
        count = import_embeddings(client, config, VECTOR_TEXT, embeddings, metadata)
        print(f"Collection now holds {count} points with text vectors")
    elif args.command == "info":
        name = config["collection"]["name"]
        info = client.get_collection(name)
        print(f"status={info.status} points={info.points_count} vectors={info.vectors_count}")


if __name__ == "__main__":
    main()
