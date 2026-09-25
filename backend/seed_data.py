import random
from sqlalchemy import select
from app.database import Base, SessionLocal, engine
from app.models import Category, Product, User
from app.security import hash_password
from app.services import PRODUCT_COLLECTION, upsert_product_vector, qdrant_health

def seed():
    print("Creating tables in database...")
    Base.metadata.create_all(bind=engine)

    with SessionLocal() as db:
        # 1. Admin user
        admin = db.scalar(select(User).where(User.email == "admin@shopsense.vn"))
        if not admin:
            admin = User(
                email="admin@shopsense.vn",
                full_name="Quản Trị Viên ShopSense",
                password_hash=hash_password("adminpassword123"),
                is_admin=True,
            )
            db.add(admin)
            print("Created admin user: admin@shopsense.vn / adminpassword123")

        # 2. Demo customer user
        demo_user = db.scalar(select(User).where(User.email == "demo@shopsense.vn"))
        if not demo_user:
            demo_user = User(
                email="demo@shopsense.vn",
                full_name="Nguyễn Văn Demo",
                password_hash=hash_password("demopassword123"),
                is_admin=False,
            )
            db.add(demo_user)
            print("Created demo user: demo@shopsense.vn / demopassword123")

        db.commit()

        # 3. Categories
        categories_data = [
            {"name": "Thời Trang Nam", "slug": "thoi-trang-nam"},
            {"name": "Thời Trang Nữ", "slug": "thoi-trang-nu"},
            {"name": "Giày Dép Thể Thao", "slug": "giay-dep-the-thao"},
            {"name": "Túi Xách & Balo", "slug": "tui-xach-balo"},
            {"name": "Đồng Hồ & Trang Sức", "slug": "dong-ho-trang-suc"},
        ]

        cat_map = {}
        for c in categories_data:
            existing = db.scalar(select(Category).where(Category.slug == c["slug"]))
            if not existing:
                existing = Category(name=c["name"], slug=c["slug"])
                db.add(existing)
                db.flush()
            cat_map[c["slug"]] = existing.id
        db.commit()
        print(f"Categories seeded: {len(cat_map)}")

        # 4. Products
        products_data = [
            {
                "sku": "MEN-SHIRT-001",
                "name": "Áo Sơ Mi Nam Oxford Premium Slim Fit",
                "description": "Áo sơ mi nam chất liệu vải Oxford 100% cotton cao cấp, thoáng mát, đứng form, chống nhăn nhẹ. Thích hợp đi làm công sở hoặc đi tiệc lịch sự.",
                "price": 450000,
                "stock_quantity": 45,
                "image_url": "https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=800&auto=format&fit=crop&q=80",
                "category_slug": "thoi-trang-nam",
            },
            {
                "sku": "MEN-POLO-002",
                "name": "Áo Polo Nam Dệt Kim Cổ Bẻ Thấm Hút Mồ Hôi",
                "description": "Chất liệu Pique dệt mắt chim mềm mịn, co giãn 4 chiều. Phối viền cổ và bo tay sang trọng, phong cách thể thao thanh lịch.",
                "price": 320000,
                "stock_quantity": 60,
                "image_url": "https://images.unsplash.com/photo-1581655353564-df123a1eb820?w=800&auto=format&fit=crop&q=80",
                "category_slug": "thoi-trang-nam",
            },
            {
                "sku": "MEN-BLAZER-003",
                "name": "Áo Blazer Nam Form Rộng Hàn Quốc Hiện Đại",
                "description": "Blazer dáng suông thời thượng, lót lụa cao cấp, đường may tỉ mỉ. Dễ dàng mix-match cùng áo phông hoặc sơ mi.",
                "price": 890000,
                "stock_quantity": 25,
                "image_url": "https://images.unsplash.com/photo-1507679799987-c73779587ccf?w=800&auto=format&fit=crop&q=80",
                "category_slug": "thoi-trang-nam",
            },
            {
                "sku": "WOMEN-DRESS-001",
                "name": "Đầm Xòe Dáng Dài Voan Hoa Vintage Dịu Dàng",
                "description": "Váy voan 2 lớp bồng bềnh, họa tiết hoa nhí vintage trang nhã. Thiết kế chiết eo tôn dáng, thích hợp dạo phố, du lịch.",
                "price": 550000,
                "stock_quantity": 35,
                "image_url": "https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=800&auto=format&fit=crop&q=80",
                "category_slug": "thoi-trang-nu",
            },
            {
                "sku": "WOMEN-SUIT-002",
                "name": "Set Vest Công Sở Nữ Thanh Lịch Tone Be Tối Giản",
                "description": "Bộ suit nữ gồm áo vest và quần tây cạp cao, tôn chân thon dài. Chất vải tuyết mưa nhập khẩu không xù lông.",
                "price": 950000,
                "stock_quantity": 20,
                "image_url": "https://images.unsplash.com/photo-1548767797-d8c844163c4c?w=800&auto=format&fit=crop&q=80",
                "category_slug": "thoi-trang-nu",
            },
            {
                "sku": "WOMEN-KNIT-003",
                "name": "Áo Cardigan Len Dệt Kim Họa Tiết Cúc Gỗ",
                "description": "Len lông cừu mềm mại, giữ ấm tốt, cúc gỗ mộc mạc cổ điển. Phù hợp diện trong những ngày se lạnh.",
                "price": 380000,
                "stock_quantity": 50,
                "image_url": "https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=800&auto=format&fit=crop&q=80",
                "category_slug": "thoi-trang-nu",
            },
            {
                "sku": "SHOES-RUN-001",
                "name": "Giày Chạy Bộ Sneaker Đệm Khí Êm Ái AirFlex",
                "description": "Đế đệm khí êm ái đàn hồi cao, thân giày dệt lưới thoáng khí tản nhiệt tốt, bảo vệ khớp gối khi vận động cường độ cao.",
                "price": 850000,
                "stock_quantity": 40,
                "image_url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=800&auto=format&fit=crop&q=80",
                "category_slug": "giay-dep-the-thao",
            },
            {
                "sku": "SHOES-RETRO-002",
                "name": "Giày Thể Thao Sneaker Cổ Điển White Classic",
                "description": "Sneaker da màu trắng basic kinh điển, phù hợp mọi phong cách trang phục từ quần jeans, kaki đến chân váy.",
                "price": 620000,
                "stock_quantity": 70,
                "image_url": "https://images.unsplash.com/photo-1525966222134-fcfa99b8ae77?w=800&auto=format&fit=crop&q=80",
                "category_slug": "giay-dep-the-thao",
            },
            {
                "sku": "BAG-TOTE-001",
                "name": "Túi Tote Da Thật Đựng Laptop 14 Inch Chống Nước",
                "description": "Túi tote da bò sáp tự nhiên bền bỉ theo thời gian, có ngăn chống sốc chuyên dụng đựng laptop, quai đeo êm ái.",
                "price": 750000,
                "stock_quantity": 30,
                "image_url": "https://images.unsplash.com/photo-1590874103328-eac38a683ce7?w=800&auto=format&fit=crop&q=80",
                "category_slug": "tui-xach-balo",
            },
            {
                "sku": "BAG-BACKPACK-002",
                "name": "Balo Du Lịch Đa Năng Cổng Sạc USB Chống Trộm",
                "description": "Vải Oxford phủ màng chống thấm nước trượt nước, nhiều ngăn phân loại thông minh, đệm lưng tổ ong thoáng khí.",
                "price": 490000,
                "stock_quantity": 55,
                "image_url": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=800&auto=format&fit=crop&q=80",
                "category_slug": "tui-xach-balo",
            },
            {
                "sku": "WATCH-CHRONO-001",
                "name": "Đồng Hồ Nam Dây Da Sapphire Chronograph Thể Thao",
                "description": "Mặt kính Sapphire chống xước hoàn hảo, vỏ thép không gỉ 316L, chịu nước 5ATM. Bộ máy quartz Nhật Bản chính xác tuyệt đối.",
                "price": 1850000,
                "stock_quantity": 15,
                "image_url": "https://images.unsplash.com/photo-1524805444758-089113d48a6d?w=800&auto=format&fit=crop&q=80",
                "category_slug": "dong-ho-trang-suc",
            },
            {
                "sku": "JEWELRY-RING-002",
                "name": "Nhẫn Bạc S925 Đính Đá Zirconia Tinh Xảo Lấp Lánh",
                "description": "Bạc Ý S925 phủ bạch kim sáng bóng không xỉn màu. Đính đá Zirconia giác cắt kim cương tỏa sáng rực rỡ dưới ánh sáng.",
                "price": 350000,
                "stock_quantity": 80,
                "image_url": "https://images.unsplash.com/photo-1605100804763-247f67b3557e?w=800&auto=format&fit=crop&q=80",
                "category_slug": "dong-ho-trang-suc",
            },
        ]

        created_count = 0
        product_entities = []
        for p in products_data:
            cat_id = cat_map.get(p["category_slug"])
            existing_prod = db.scalar(select(Product).where(Product.sku == p["sku"]))
            if not existing_prod:
                prod = Product(
                    sku=p["sku"],
                    name=p["name"],
                    description=p["description"],
                    price=p["price"],
                    stock_quantity=p["stock_quantity"],
                    image_url=p["image_url"],
                    category_id=cat_id,
                    is_active=True,
                )
                db.add(prod)
                db.flush()
                product_entities.append(prod)
                created_count += 1
            else:
                product_entities.append(existing_prod)

        db.commit()
        print(f"Products seeded: {created_count} created (total {len(product_entities)})")

        # 5. Optional Qdrant embeddings index
        if qdrant_health():
            print("Indexing product vectors in Qdrant...")
            random.seed(42)
            for prod in product_entities:
                # Generate a reproducible pseudo-vector for similarity demo
                vector = [random.uniform(-1.0, 1.0) for _ in range(64)]
                try:
                    upsert_product_vector(prod.id, vector)
                except Exception as e:
                    print(f"Failed to index vector for product {prod.id}: {e}")
            print("Indexed product vectors successfully in Qdrant!")
        else:
            print("Qdrant not ready or unreachable, skipping vector indexing.")

if __name__ == "__main__":
    seed()
