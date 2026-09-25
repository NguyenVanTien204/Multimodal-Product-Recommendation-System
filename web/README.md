# ShopSense Web

Frontend của đồ án được chốt là **Next.js (App Router, TypeScript) + Tailwind CSS**. Source sẽ được khởi tạo tại thư mục này ở mốc W1 của [kế hoạch giao hàng](../docs/web/04-delivery-plan.md).

Frontend gọi same-origin `/api/*`; Next.js Route Handlers chuyển tiếp request tới FastAPI thông qua biến server-only `DATN_API_BASE_URL`. Chi tiết contract và cấu trúc dự án tại:

- [Hợp đồng API](../docs/web/03-api-and-integration.md)
- [Kiến trúc frontend](../docs/web/06-frontend-architecture.md)
- [SRS và ma trận truy vết](../docs/web/05-srs-and-traceability.md)

Không dùng Streamlit cho implementation chính thức.
