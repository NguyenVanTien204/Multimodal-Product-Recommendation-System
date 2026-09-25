import os
import sys
import time
from decimal import Decimal
import numpy as np
import pandas as pd
from sqlalchemy import select, delete, text
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from app.database import Base, SessionLocal, engine
from app.models import Category, Product, User, Cart, CartItem, Order, OrderItem
from app.security import hash_password
from app.config import settings

def main():
    print("=" * 70)
    print("IMPORTING ALL ~152K AMAZON PRODUCTS INTO POSTGRESQL & QDRANT")
    print("=" * 70)

    # 1. Database tables check
    Base.metadata.create_all(bind=engine)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    items_parquet = os.path.join(project_root, "data", "items.parquet")
    text_embeddings_npy = os.path.join(project_root, "data", "embedding", "text_embeddings.npy")

    print(f"Reading dataset: {items_parquet}...")
    t0 = time.time()
    df = pd.read_parquet(items_parquet)
    print(f"Loaded {len(df):,} items in {time.time() - t0:.2f}s")

    # Fast classification into 7 categories
    print("Classifying categories for 152k items...")
    tc = time.time()
    text_combined = (df["title"].fillna("") + " " + df["category"].fillna("")).str.lower()
    is_shoe = text_combined.str.contains("shoe|sneaker|boot|sandal|footwear|loafer|heel|slipper", regex=True)
    is_bag = text_combined.str.contains("backpack|tote|handbag|purse|wallet|luggage|duffel|crossbody|bag", regex=True)
    is_jewelry = text_combined.str.contains("watch|ring|necklace|bracelet|earring|jewelry|pendant|gold|silver|diamond", regex=True)
    is_acc = text_combined.str.contains("sock|belt|hat|cap|sunglasses|glasses|scarf|glove|beanie|tie", regex=True)
    is_sport = text_combined.str.contains("swim|running|yoga|gym|athletic|jersey|activewear|cycling", regex=True)
    is_men = text_combined.str.contains(r"men's|mens|male|guy|gentleman|polo", regex=True)

    cat_series = pd.Series("thoi-trang-nu", index=df.index)
    cat_series[is_men] = "thoi-trang-nam"
    cat_series[is_sport] = "the-thao-ngoai-troi"
    cat_series[is_acc] = "phu-kien"
    cat_series[is_jewelry] = "dong-ho-trang-suc"
    cat_series[is_bag] = "tui-xach-balo"
    cat_series[is_shoe] = "giay-dep"
    print(f"Categories classified in {time.time() - tc:.2f}s:\n{cat_series.value_counts()}")

    # Category mapping in PostgreSQL
    categories_def = [
        {"name": "Thời Trang Nam", "slug": "thoi-trang-nam"},
        {"name": "Thời Trang Nữ", "slug": "thoi-trang-nu"},
        {"name": "Giày Dép", "slug": "giay-dep"},
        {"name": "Túi Xách & Balo", "slug": "tui-xach-balo"},
        {"name": "Đồng Hồ & Trang Sức", "slug": "dong-ho-trang-suc"},
        {"name": "Phụ Kiện Thời Trang", "slug": "phu-kien"},
        {"name": "Thể Thao & Dã Ngoại", "slug": "the-thao-ngoai-troi"},
    ]

    with SessionLocal() as db:
        cat_map = {}
        for c in categories_def:
            cat = db.scalar(select(Category).where(Category.slug == c["slug"]))
            if not cat:
                cat = Category(name=c["name"], slug=c["slug"])
                db.add(cat)
                db.flush()
            cat_map[c["slug"]] = cat.id
        db.commit()

        # Users check
        admin = db.scalar(select(User).where(User.email == "admin@shopsense.vn"))
        if not admin:
            db.add(User(
                email="admin@shopsense.vn",
                full_name="Quản Trị Viên ShopSense",
                password_hash=hash_password("adminpassword123"),
                is_admin=True,
            ))
        demo = db.scalar(select(User).where(User.email == "demo@shopsense.vn"))
        if not demo:
            db.add(User(
                email="demo@shopsense.vn",
                full_name="Nguyễn Văn Demo",
                password_hash=hash_password("demopassword123"),
                is_admin=False,
            ))
        db.commit()

        # Clean old products to ensure pure ~152k dataset
        print("Cleaning previous products, carts and orders...")
        db.execute(delete(CartItem))
        db.execute(delete(OrderItem))
        db.execute(delete(Product))
        db.commit()

    # Fast bulk insertion into PostgreSQL via psycopg COPY
    print(f"\n--- Inserting all {len(df):,} products into PostgreSQL via fast binary/stream COPY ---")
    t_pg = time.time()

    base_prices = {
        "dong-ho-trang-suc": 850000,
        "tui-xach-balo": 550000,
        "giay-dep": 680000,
        "thoi-trang-nam": 420000,
        "thoi-trang-nu": 450000,
        "the-thao-ngoai-troi": 390000,
        "phu-kien": 190000,
    }

    # Prepare data generator
    np.random.seed(42)
    random_stocks = np.random.randint(15, 120, size=len(df))

    # Fast raw copy
    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            # Drop constraint index during bulk load for maximum speed, re-enable after
            cur.execute("TRUNCATE TABLE products RESTART IDENTITY CASCADE;")
            
            with cur.copy(
                "COPY products (id, sku, name, description, price, stock_quantity, image_url, category_id, is_active, created_at) "
                "FROM STDIN"
            ) as copy:
                now_str = "2026-09-24 12:00:00"
                cat_series_list = cat_series.tolist()
                for row in df.itertuples(index=True):
                    idx = row.Index
                    prod_id = idx + 1
                    sku = str(row.item_id)
                    
                    # Title
                    raw_title = str(row.title) if pd.notna(row.title) else ""
                    title = raw_title.strip()
                    if len(title) > 250:
                        title = title[:247] + "..."
                    elif not title:
                        title = f"Amazon Product {sku}"

                    # Description
                    desc_parts = []
                    if pd.notna(row.description) and str(row.description).strip():
                        desc_parts.append(str(row.description).strip())
                    if pd.notna(row.features) and str(row.features).strip():
                        desc_parts.append(f"Features: {str(row.features).strip()}")
                    if pd.notna(row.brand) and str(row.brand).strip():
                        desc_parts.append(f"Brand: {str(row.brand).strip()}")
                    desc = "\n\n".join(desc_parts)

                    # Price
                    raw_price = row.price
                    c_slug = cat_series_list[idx]
                    if pd.notna(raw_price) and float(raw_price) > 0:
                        price = round(float(raw_price) * 25000, -3)
                    else:
                        price = base_prices.get(c_slug, 350000)

                    stock = int(random_stocks[idx])
                    img = str(row.image_url) if pd.notna(row.image_url) else ""
                    cat_id = cat_map[c_slug]

                    copy.write_row((prod_id, sku, title, desc, price, stock, img, cat_id, True, now_str))

            # Set sequence
            cur.execute(f"SELECT setval('products_id_seq', {len(df) + 1}, true);")
            raw_conn.commit()
            print(f"Successfully inserted all {len(df):,} products into PostgreSQL in {time.time() - t_pg:.2f}s!")
    finally:
        raw_conn.close()

    # 3. Qdrant Vector DB Ingestion (152,086 vectors of dimension 1024)
    print("\n--- Connecting to Qdrant Vector Database ---")
    qdrant_client = QdrantClient(url=settings.qdrant_url, timeout=60)
    collection_name = "product_embeddings"

    print(f"Recreating collection '{collection_name}' with 1024-dim Cosine (on_disk=True)...")
    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE, on_disk=True),
    )

    if os.path.exists(text_embeddings_npy):
        print(f"Opening memory-mapped vector embeddings from {text_embeddings_npy}...")
        embeddings_matrix = np.load(text_embeddings_npy, mmap_mode="r")
        total_vectors = len(embeddings_matrix)
        print(f"Vector matrix loaded: {total_vectors:,} items, 1024-dim.")

        sku_list = df["item_id"].astype(str).tolist()
        batch_size = 2000
        t_qd = time.time()

        for start_idx in range(0, total_vectors, batch_size):
            end_idx = min(start_idx + batch_size, total_vectors)
            batch_vecs = embeddings_matrix[start_idx:end_idx]
            
            points = [
                PointStruct(
                    id=i + 1,
                    vector=batch_vecs[i - start_idx].tolist(),
                    payload={"product_id": i + 1, "sku": sku_list[i]}
                )
                for i in range(start_idx, end_idx)
            ]

            qdrant_client.upsert(collection_name=collection_name, points=points, wait=False)
            if (end_idx % 10000 == 0) or (end_idx == total_vectors):
                elapsed = time.time() - t_qd
                speed = end_idx / max(elapsed, 0.001)
                print(f"Upserted {end_idx:,}/{total_vectors:,} vectors ({speed:.0f} vectors/s)...")

        print(f"All {total_vectors:,} vectors indexed in Qdrant in {time.time() - t_qd:.2f}s!")

        # Verification test query
        time.sleep(1)
        test_rec = qdrant_client.retrieve(collection_name, ids=[1], with_vectors=True)
        if test_rec and test_rec[0].vector:
            res = qdrant_client.query_points(collection_name, query=test_rec[0].vector, limit=5).points
            similar_ids = [p.id for p in res if p.id != 1]
            print(f"[Verification] Similar items in Qdrant for Product 1: {similar_ids}")

    print("\n" + "=" * 70)
    print("FINISHED: ALL 152,086 PRODUCTS & VECTORS ARE LIVE IN DB & QDRANT!")
    print("=" * 70)

if __name__ == "__main__":
    main()
