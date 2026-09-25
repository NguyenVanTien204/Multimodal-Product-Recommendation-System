# Kế hoạch giao hàng web

## Backlog theo mốc

| Mốc | Hạng mục | Điều kiện hoàn thành |
|---|---|---|
| W0 | Phân tích nghiệp vụ, UML, API contract | Bốn tài liệu trong `docs/web/` hoàn chỉnh |
| W1 | Next.js + Tailwind storefront chạy demo mode | Có home, filter, product card, compare, chat layout; responsive |
| W2 | FastAPI contract | Endpoint có schema/Pydantic, health và fallback |
| W3 | Kết nối recommender artifact | `/recommend`, `/search`, `/refine` dùng model/index version rõ ràng |
| W4 | Agent/RAG | Explain/compare trả evidence đúng item; agent lỗi không ảnh hưởng API lõi |
| W5 | QA và báo cáo | Smoke tests, latency log, demo scripts, screenshots/luận văn |

## Definition of Done cho W1

- Chạy được bằng `npm run dev` khi chưa có model.
- Có badge **Demo data**; không gắn kết quả mock với metric research.
- Frontend gọi same-origin `/api/*`; Next.js route handler dùng `DATN_API_BASE_URL` server-side khi biến này được đặt.
- Lỗi API không làm crash UI; báo trạng thái rõ ràng.
- Có TypeScript strict, ESLint, responsive layout cho mobile/desktop và kiểm tra accessibility cơ bản.
- Không sửa `data/train.parquet`, `data/valid.parquet`, `data/test.parquet` hay artifact huấn luyện.

## Bàn giao từ quá trình train

Để chuyển từ mock sang model, phía training/API cần cung cấp:

1. Catalog item gồm `item_id`, title, image URL, category, brand, price (có thể null) và mapping index row.
2. Endpoint theo contract, kèm `model_version`, `index_version`, fallback reason.
3. Session store/TTL và schema preference thống nhất.
4. Context evidence có provenance cho `/explain`, `/compare`, `/chat`.
5. Một `user_id` demo có lịch sử hợp lệ hoặc route demo fallback.

## Tiêu chí demo bảo vệ

1. Mở web → gọi gợi ý; quan sát source model/fallback.
2. Tìm “black casual shoes under 100” → thấy filter đã hiểu.
3. Nói “rẻ hơn” → kết quả thay đổi mà không retrain.
4. Hỏi vì sao sản phẩm #2 → evidence hiện diện hoặc thông báo thiếu evidence.
5. So sánh hai sản phẩm → cùng schema, field thiếu không bị bịa.
