from __future__ import annotations

import os

from qdrant_client import QdrantClient


def build_client(config: dict) -> QdrantClient:
    """Create a QdrantClient from the `connection` block of configs/qdrant.yaml.

    The API key is read from QDRANT_API_KEY rather than the yaml file so it never
    ends up committed to the repo.
    """
    conn = config.get("connection", {})
    return QdrantClient(
        url=conn.get("url", "http://localhost:6333"),
        api_key=os.environ.get("QDRANT_API_KEY") or conn.get("api_key"),
        timeout=conn.get("timeout_s", 30),
    )
