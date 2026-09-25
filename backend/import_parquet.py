import os
import sys
import re
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

def categorize_item(title: str, desc: str, raw_cat: str) -> str:
    text_content = f"{title} {desc} {raw_cat}".lower()
    
    # 1. Shoes / Footwear
    if any(k in text_content for k in ["shoe", "sneaker", "boot", "sandal", "footwear", "loafer", "heel", "slipper", "cleat"]):
        return "giay-dep"
        
    # 2. Bags / Luggage
    if any(k in text_content for k in ["backpack", "tote", "handbag", "purse", "wallet", "luggage", "duffel", "crossbody", "sling bag", "briefcase", "satchel"]):
        return "tui-xach-balo"
        
    # 3. Watches & Jewelry
    if any(k in text_content for k in ["watch", "ring", "necklace", "bracelet", "earring", "jewelry", "pendant", "gold", "silver", "diamond", "zirconia", "chronograph"]):
        return "dong-ho-trang-suc"
        
    # 4. Accessories
    if any(k in text_content for k in ["sock", "belt", "hat", "cap", "sunglasses", "glasses", "scarf", "glove", "beanie", "tie", "headband"]):
        return "phu-kien"
        
    # 5. Sports & Outdoors
    if any(k in text_content for k in ["swimsuit", "swimwear", "running", "yoga", "gym", "athletic", "jersey", "activewear", "outdoor", "cycling"]):
        return "the-thao-ngoai-troi"
        
    # 6. Men's Fashion
    if any(k in text_content for k in ["men's", "mens", "for men", "guy", "male", "gentleman", "polo", "oxford shirt"]):
        return "thoi-trang-nam"
        
    # 7. Women's Fashion (Default apparel)
    if any(k in text_content for k in ["women's", "womens", "dress", "blouse", "skirt", "bra", "lingerie", "legging", "gown"]):
        return "thoi-trang-nu"

    # Default fallback by keyword
    if "men" in text_content:
        return "thoi-trang-nam"
    return "thoi-trang-nu"

def main():
    print("=" * 60)
    print("IMPORTING REAL PARQUET DATA INTO POSTGRESQL & QDRANT")
    print("=" * 60)

    # 1. Database tables
    Base.metadata.create_all(bind=engine)

    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    items_parquet = os.path.join(project_root, "data", "items.parquet")
    interactions_parquet = os.path.join(project_root, "data", "candidate_interactions.parquet")
    text_embeddings_npy = os.path.join(project_root, "data", "embedding", "text_embeddings.npy")
    text_meta_parquet = os.path.join(project_root, "data", "embedding", "text_embedding_metadata.parquet")

    print(f"Loading {items_parquet}...")
    df_items = pd.read_parquet(items_parquet)
    print(f"Total raw items: {len(df_items):,}")

    # Interaction popularity
    item_review_counts = {}
    if os.path.exists(interactions_parquet):
        print(f"Loading interaction counts from {interactions_parquet}...")
        df_inter = pd.read_parquet(interactions_parquet, columns=["item_id"])
        item_review_counts = df_inter["item_id"].value_counts().to_dict()
        print(f"Interaction counts loaded for {len(item_review_counts):,} unique items.")

    df_items["review_count"] = df_items["item_id"].map(lambda x: item_review_counts.get(x, 0))

    # Filter items with valid images
    valid_mask = (
        df_items["image_url"].notna() &
        df_items["image_url"].str.startswith("http") &
        df_items["title"].notna() &
        (df_items["title"].str.len() >= 5)
    )
    df_valid = df_items[valid_mask].copy()
    print(f"Valid items with photos: {len(df_valid):,}")

    # Re-order by review count descending to get best-reviewed items
    df_valid.sort_values(by=["review_count"], ascending=False, inplace=True)

    # 2. Categories setup
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
        # Create categories
        cat_map = {}
        for c in categories_def:
            cat = db.scalar(select(Category).where(Category.slug == c["slug"]))
            if not cat:
                cat = Category(name=c["name"], slug=c["slug"])
                db.add(cat)
                db.flush()
            cat_map[c["slug"]] = cat.id
        db.commit()
        print(f"Configured {len(cat_map)} categories in PostgreSQL.")

        # Ensure demo and admin accounts
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

        # Clean existing mock products so we only have real parquet data
        print("Cleaning previous mock products and items...")
        db.execute(delete(CartItem))
        db.execute(delete(OrderItem))
        db.execute(delete(Product))
        db.commit()

        # Target sample size: 500 top items per category (~3,500 real items)
        category_buckets = {slug: [] for slug in cat_map.keys()}
        max_per_cat = 500


        for row_idx, row in df_valid.iterrows():
            item_id = row["item_id"]
            title = str(row["title"]).strip()
            desc = str(row.get("description", "")) if pd.notna(row.get("description")) else ""
            feat = str(row.get("features", "")) if pd.notna(row.get("features")) else ""
            raw_cat = str(row.get("category", "")) if pd.notna(row.get("category")) else ""

            cat_slug = categorize_item(title, desc, raw_cat)
            if len(category_buckets[cat_slug]) < max_per_cat:
                category_buckets[cat_slug].append((row_idx, row, cat_slug))

            # Stop when all categories are filled
            if all(len(b) >= max_per_cat for b in category_buckets.values()):
                break

        selected_items = []
        for slug, items_list in category_buckets.items():
            print(f"Category '{slug}': {len(items_list)} items selected")
            selected_items.extend(items_list)

        print(f"Total curated real products to insert: {len(selected_items)}")

        # Insert products into PostgreSQL
        inserted_products = [] # list of (product_id, original_parquet_row_index)
        for original_idx, row, cat_slug in selected_items:
            # Price handling: convert USD to VND (1 USD = 25,000 VND)
            raw_price = row.get("price")
            if pd.notna(raw_price) and float(raw_price) > 0:
                price_vnd = round(float(raw_price) * 25000, -3)
            else:
                # Realistic price based on category
                base_prices = {
                    "dong-ho-trang-suc": 850000,
                    "tui-xach-balo": 550000,
                    "giay-dep": 680000,
                    "thoi-trang-nam": 420000,
                    "thoi-trang-nu": 450000,
                    "the-thao-ngoai-troi": 390000,
                    "phu-kien": 190000,
                }
                price_vnd = base_prices.get(cat_slug, 350000)

            title = str(row["title"]).strip()
            if len(title) > 250:
                title = title[:247] + "..."

            desc_parts = []
            if pd.notna(row.get("description")) and str(row.get("description")).strip():
                desc_parts.append(str(row.get("description")).strip())
            if pd.notna(row.get("features")) and str(row.get("features")).strip():
                desc_parts.append(f"Đặc điểm: {str(row.get('features')).strip()}")
            if pd.notna(row.get("brand")) and str(row.get("brand")).strip():
                desc_parts.append(f"Thương hiệu: {str(row.get('brand')).strip()}")

            description_text = "\n\n".join(desc_parts)

            stock = int(np.random.randint(15, 120))

            product = Product(
                sku=str(row["item_id"]),
                name=title,
                description=description_text,
                price=Decimal(str(price_vnd)),
                stock_quantity=stock,
                image_url=str(row["image_url"]),
                category_id=cat_map[cat_slug],
                is_active=True,
            )
            db.add(product)
            db.flush()
            inserted_products.append((product.id, original_idx, str(row["item_id"])))

        db.commit()
        print(f"Successfully inserted {len(inserted_products)} real products into PostgreSQL!")

    # 3. Qdrant Vector DB Ingestion (1024-dim real embeddings)
    print("\n--- Connecting to Qdrant Vector Database ---")
    qdrant_client = QdrantClient(url=settings.qdrant_url, timeout=20)
    collection_name = "product_embeddings"

    # Recreate collection with exact 1024 dimensions
    print(f"Recreating collection '{collection_name}' with 1024-dimension Cosine distance...")
    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )


    if os.path.exists(text_embeddings_npy):
        print(f"Opening memory-mapped vector embeddings from {text_embeddings_npy}...")
        embeddings_matrix = np.load(text_embeddings_npy, mmap_mode="r")
        print(f"Vector matrix loaded: shape {embeddings_matrix.shape}, dtype {embeddings_matrix.dtype}")

        batch_size = 100
        points = []
        indexed_count = 0

        for prod_id, original_idx, sku in inserted_products:
            vec = embeddings_matrix[original_idx].astype(float).tolist()
            point = PointStruct(
                id=prod_id,
                vector=vec,
                payload={
                    "product_id": prod_id,
                    "sku": sku,
                    "parquet_index": int(original_idx),
                }
            )
            points.append(point)

            if len(points) >= batch_size:
                qdrant_client.upsert(collection_name=collection_name, points=points, wait=True)
                indexed_count += len(points)
                print(f"Indexed {indexed_count}/{len(inserted_products)} vectors in Qdrant...")
                points = []

        if points:
            qdrant_client.upsert(collection_name=collection_name, points=points, wait=True)
            indexed_count += len(points)
            print(f"Indexed {indexed_count}/{len(inserted_products)} vectors in Qdrant.")

        # Test query in Qdrant
        first_prod_id = inserted_products[0][0]
        test_rec = qdrant_client.retrieve(collection_name, ids=[first_prod_id], with_vectors=True)
        if test_rec and test_rec[0].vector:
            res = qdrant_client.query_points(collection_name, query=test_rec[0].vector, limit=4).points
            similar_ids = [p.id for p in res if p.id != first_prod_id]
            print(f"\n[Test] Query similar for Product ID {first_prod_id} in Qdrant: {similar_ids}")

    print("\n" + "=" * 60)
    print("SUCCESS: REAL PARQUET DATA LOADED INTO POSTGRESQL & QDRANT!")
    print("=" * 60)

if __name__ == "__main__":
    main()
