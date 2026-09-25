# Hợp đồng API và tích hợp frontend

## 1. Nguyên tắc tích hợp

- UI không đọc Parquet, checkpoint hay FAISS trực tiếp.
- Backend là nguồn chân lý cho rank, price, metadata, session và evidence.
- Tất cả request có `session_id`; `user_id` là tùy chọn để phục vụ demo history.
- Next.js client gọi same-origin route handler (`/api/*`); handler chuyển tiếp request tới FastAPI qua biến server-only `DATN_API_BASE_URL`. Không đưa URL nội bộ, API key hoặc secret vào `NEXT_PUBLIC_*`.
- Khi model/index chưa sẵn, frontend chạy `demo mode` bằng fixture cục bộ. Demo mode không gửi request sang backend và luôn hiện nhãn rõ ràng.

## 2. Response chung

```json
{
  "request_id": "uuid",
  "session_id": "uuid",
  "source": "model | popularity_fallback | catalog_lexical | demo_mock",
  "model_version": "optional-string",
  "products": [],
  "understood_preferences": {},
  "message": "optional user-safe message"
}
```

Mỗi product tối thiểu có `item_id`, `title`, `rank`; trường `price`, `brand`, `image_url`, `category`, `score` là optional. UI phải chịu được giá trị thiếu.

## 3. Endpoint frontend dùng

| Endpoint | Request chính | Response/ghi chú |
|---|---|---|
| `POST /recommend` | `session_id`, `user_id?`, `preferences?`, `k` | Top-K cá nhân hóa hoặc fallback |
| `POST /search` | `session_id`, `query?`, `reference_item_id?`, `filters`, `k` | hard filters; hiện tại có `catalog_lexical` fallback cho text query, semantic encoder sẽ thay thế ở mốc retrieval |
| `POST /refine` | `session_id`, `preference_delta`, `k` | preference merged + reranked products |
| `POST /explain` | `session_id`, `item_id` | explanation có `evidence` |
| `POST /compare` | `session_id`, `item_ids[2]` | shared attributes, missing field explicit |
| `POST /chat` | `session_id`, `message` | `reply`, `tool_calls?`, `products?`, `evidence?` |
| `GET /health` | — | service/model/index availability |

Next.js giữ cùng đường dẫn public (`/api/recommend`, `/api/search`, …) và proxy 1:1 tới FastAPI (`/recommend`, `/search`, …). FastAPI vẫn là API contract chính thức, có OpenAPI/Swagger phục vụ kiểm thử độc lập.

## 4. Quy ước lỗi

| HTTP | UI xử lý |
|---|---|
| 400/422 | Giữ input, hiển thị lỗi validation cụ thể |
| 404 | Báo sản phẩm/phiên không còn hợp lệ; đề xuất tìm lại |
| 409 | Refresh session/result theo payload nếu có |
| 503 | Hiển thị service không sẵn; direct mock chỉ dùng khi người dùng bật demo mode |
| timeout/network | Không tự gửi lại chat; cho nút thử lại |

## 5. Bảo mật và dữ liệu

- Không gửi API key sang browser/UI source; API key LLM chỉ tồn tại trong backend environment.
- Production/demo ưu tiên proxy same-origin. Nếu cần browser gọi FastAPI trực tiếp khi phát triển, đặt allowlist `DATN_CORS_ORIGINS` (không dùng `*` mặc định).
- Không log raw chat/image trừ khi có policy nghiên cứu được phê duyệt.
- Giới hạn upload ảnh theo MIME, kích thước và kích thước pixel ở cả client/backend.
- `user_id` Amazon là định danh đã ẩn danh; không nhập PII vào phiên demo.
