# Thiết kế Web Demo — ShopSense

Thư mục này lưu các artefact phân tích và thiết kế cho web demo của đồ án.
Chúng bổ sung cho đặc tả nghiệp vụ tổng thể tại [04-business-requirements.md](../04-business-requirements.md), không thay thế các ràng buộc dữ liệu hoặc nghiên cứu trong `AGENT.md`.

## Mục lục

1. [01-domain-and-user-flow.md](01-domain-and-user-flow.md) — phạm vi, tác nhân, quy trình nghiệp vụ và user flow.
2. [02-uml.md](02-uml.md) — use-case, sequence, state và component UML bằng Mermaid.
3. [03-api-and-integration.md](03-api-and-integration.md) — hợp đồng API cho frontend và quy tắc fallback.
4. [04-delivery-plan.md](04-delivery-plan.md) — backlog, tiêu chí nghiệm thu và lộ trình tích hợp model.
5. [05-srs-and-traceability.md](05-srs-and-traceability.md) — SRS web, yêu cầu kiểm thử được và ma trận truy vết.
6. [06-frontend-architecture.md](06-frontend-architecture.md) — kiến trúc triển khai FastAPI + Next.js + Tailwind CSS.
7. [07-marketplace-backend.md](07-marketplace-backend.md) — backend độc lập, PostgreSQL, Qdrant và ranh giới với recommender.

## Quyết định nền tảng

- Frontend chốt: **Next.js (App Router, TypeScript) + Tailwind CSS** trong `web/`.
- Backend đích: FastAPI trong `src/datn/api/`.
- Khi chưa có artifact model/index: frontend dùng fixture catalog cục bộ, được gắn nhãn **Demo data**; không mô phỏng số liệu đánh giá mô hình.
- Khi API sẵn sàng: Next.js route handler chuyển tiếp request tới FastAPI qua `DATN_API_BASE_URL`; browser không truy cập checkpoint, Parquet hay index.

## Không thuộc phạm vi MVP

Hệ thống gồm mini marketplace: tài khoản JWT, catalog, giỏ hàng, tồn kho và tạo đơn hàng `PENDING`. Không gồm cổng thanh toán thật, giao vận, hoàn tiền hoặc lưu PII ngoài thông tin tối thiểu người dùng chủ động nhập trong môi trường demo.
