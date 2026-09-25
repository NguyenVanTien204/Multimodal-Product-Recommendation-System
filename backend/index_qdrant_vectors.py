import os
import sys
import time
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def index_vectors():
    print("=" * 60)
    print("INDEXING 152,086 VECTORS INTO QDRANT")
    print("=" * 60, flush=True)

    client = QdrantClient("http://localhost:6333", timeout=120)
    collection_name = "product_embeddings"

    # Recreate collection cleanly
    if client.collection_exists(collection_name):
        print(f"Deleting existing collection '{collection_name}'...", flush=True)
        client.delete_collection(collection_name)
        time.sleep(1)

    print(f"Creating collection '{collection_name}' (1024-dim, Cosine, on_disk=True)...", flush=True)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE, on_disk=True),
    )

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    npy_path = os.path.join(project_root, "data", "embedding", "text_embeddings.npy")
    parquet_path = os.path.join(project_root, "data", "items.parquet")

    print(f"Loading SKU mapping from {parquet_path}...", flush=True)
    df = pd.read_parquet(parquet_path, columns=["item_id"])
    sku_list = df["item_id"].astype(str).tolist()

    print(f"Loading embeddings from {npy_path}...", flush=True)
    embeddings = np.load(npy_path, mmap_mode="r")
    total = len(embeddings)
    print(f"Total vectors to index: {total:,}", flush=True)

    batch_size = 500
    t0 = time.time()

    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch_vecs = embeddings[start_idx:end_idx]

        points = [
            PointStruct(
                id=i + 1,
                vector=batch_vecs[i - start_idx].tolist(),
                payload={"product_id": i + 1, "sku": sku_list[i]},
            )
            for i in range(start_idx, end_idx)
        ]

        # Retry up to 3 times on transient network error
        for attempt in range(3):
            try:
                client.upsert(collection_name=collection_name, points=points, wait=False)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"Retry batch {start_idx}-{end_idx} due to: {e}", flush=True)
                time.sleep(1)

        if (end_idx % 10000 == 0) or (end_idx == total):
            elapsed = time.time() - t0
            speed = end_idx / max(elapsed, 0.001)
            print(f"Progress: {end_idx:,} / {total:,} ({end_idx*100/total:.1f}%) - Speed: {speed:.0f} vectors/s", flush=True)

    print(f"\nAll {total:,} vectors indexed successfully in {time.time() - t0:.1f}s!", flush=True)
    time.sleep(2)
    col_info = client.get_collection(collection_name)
    print(f"Qdrant collection '{collection_name}' status: {col_info.status}, points: {col_info.points_count}", flush=True)

    # Verification query
    test_pt = client.retrieve(collection_name, ids=[1], with_vectors=True)
    if test_pt and test_pt[0].vector:
        res = client.query_points(collection_name, query=test_pt[0].vector, limit=5).points
        sim_ids = [p.id for p in res if p.id != 1]
        print(f"[Verification] Top similar products for ID 1: {sim_ids}", flush=True)

if __name__ == "__main__":
    index_vectors()
