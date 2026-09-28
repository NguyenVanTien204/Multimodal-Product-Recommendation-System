# Mini Marketplace Backend

Backend độc lập cho web bán hàng: FastAPI + PostgreSQL + Qdrant. Thư mục này
không import `src/datn/`, không đọc Parquet/checkpoint và không can thiệp tiến
trình train. Model retrieval (User Tower) + reranking chạy trong service riêng
tại `apps/recommender/`; marketplace chỉ gọi qua `DATN_RECOMMENDER_URL`, dịch
`Product.id` (Postgres) sang `Product.sku` (ASIN, khớp vocab của model) ở hai
chiều, và tự rơi về Qdrant/catalog popularity khi service đó không sẵn sàng.

## Chức năng

- Đăng ký, đăng nhập JWT và phân quyền admin.
- Hồ sơ cá nhân (`PATCH /auth/me`) và đổi mật khẩu (`POST /auth/change-password`).
- Quản trị category/product; duyệt catalog công khai.
- Giỏ hàng theo user, kiểm tra tồn kho.
- Checkout atomic: lock product, trừ tồn kho, snapshot giá/tên vào order item.
- Lịch sử đơn hàng.
- Gợi ý cá nhân hóa (`/recommendations/for-you`) và gateway tuần tự
  (`/recommendations/sequential`) qua model retrieval+reranking thật
  (`apps/recommender/`), fallback Qdrant (vector nội dung thật, xem
  `scripts/index_qdrant_vectors.py`) rồi fallback catalog popularity.
- Qdrant index/search sản phẩm tương tự (`/recommendations/products/{id}/similar`).

## Chạy bằng Docker

```powershell
Copy-Item .env.example .env
# Đổi POSTGRES_PASSWORD và JWT_SECRET trong .env
docker compose up --build
```

API: `http://localhost:8000/docs`; Qdrant dashboard: `http://localhost:6333/dashboard`.

## API chính

| Nhóm | Endpoint |
|---|---|
| Auth | `POST /auth/register`, `POST /auth/login`, `GET /auth/me`, `PATCH /auth/me`, `POST /auth/change-password` |
| Catalog | `GET /categories`, `GET /products`, `GET /products/{id}` |
| Admin | `POST /categories`, `POST/PUT/DELETE /products`, `PATCH /orders/{id}/status`, `PUT /recommendations/vectors` |
| Cart | `GET /cart`, `POST /cart/items`, `DELETE /cart/items/{product_id}` |
| Order | `POST /orders/checkout`, `GET /orders` |
| Recommendation | `GET /recommendations/health`, `GET /recommendations/for-you`, `GET /recommendations/products/{id}/similar`, `POST /recommendations/sequential` |

Tài khoản admin được đánh dấu trực tiếp trong PostgreSQL cho môi trường demo.
Trước khi public deployment, thay `create_all` bằng Alembic migration, dùng secret
manager và bổ sung payment provider/webhook xác thực; checkout hiện tạo đơn hàng
`PENDING`, không mô phỏng thanh toán thành công.
