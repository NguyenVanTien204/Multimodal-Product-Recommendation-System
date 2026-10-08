# Bộ tài liệu đồ án

## Tên đề tài

**Xây dựng hệ thống gợi ý sản phẩm đa phương thức ứng dụng RAG**  
**Developing a Multimodal Product Recommendation System using RAG**

## Trạng thái bộ dữ liệu (cập nhật 05/10/2026)

| Bộ dữ liệu | Vai trò hiện tại | Tài liệu |
|---|---|---|
| **H&M Personalized Fashion Recommendations** (`hm_v1`) | **Bộ dữ liệu chính mới** cho thực nghiệm khuyến nghị và đánh giá; luận văn đang được chuyển sang bộ này | [`hm/`](../hm/docs/README.md) |
| Amazon Reviews 2023 — Clothing, Shoes and Jewelry (`balanced_u5_i2_v1`) | Bộ ban đầu của đề tài; **được giữ nguyên** làm đối chứng, và là nguồn của kho RAG (có review) và hệ thống web hiện tại | các mục 1–9 bên dưới, `LuanVan/`, `logs/` (dòng 1–7) |
| Coveo (session e-commerce) | Pipeline notebook phụ | [`COVEO_NOTEBOOK_GUIDE.md`](../legacy/docs/COVEO_NOTEBOOK_GUIDE.md) |

> Các tài liệu mô tả bộ Amazon có ghi chú phạm vi ở đầu file. Khi luận văn chuyển sang H&M, đọc [`hm/05_thesis_plan.md`](../hm/docs/05_thesis_plan.md) trước (luận điểm được phép khẳng định, thực nghiệm còn thiếu, ánh xạ sang từng chương, các quyết định cần chốt).

> **Cấu trúc thư mục (06/10/2026):** code và tài liệu bộ H&M nằm trong [`hm/`](../hm/README.md); code và báo cáo bộ Amazon/Coveo nằm trong [`legacy/`](../legacy/README.md). Thư mục `docs/` này chỉ còn tài liệu cấp dự án (tầm nhìn, tổng quan, luận văn `LuanVan/`, mục lục nhật ký). Đường dẫn trong các nhật ký theo ngày là đường dẫn cũ — xem bảng đổi tên ở [`hm/README.md`](../hm/README.md).

## Danh mục tài liệu

### A. Kế hoạch và đặc tả (viết theo bộ Amazon ban đầu)
1. [Tầm nhìn sản phẩm](./01-vision.md)
2. [Tổng quan đề tài và hệ thống](./02-overview.md)
3. [Roadmap triển khai 12 tuần](./03-roadmap.md)
4. [Đặc tả nghiệp vụ và yêu cầu hệ thống](./04-business-requirements.md)

### B. H&M — bộ dữ liệu chính mới ([`hm/`](../hm/docs/README.md))
* [`hm/01_dataset_and_protocol.md`](../hm/docs/01_dataset_and_protocol.md) — dữ liệu, cửa sổ thời gian, giao thức đánh giá
* [`hm/02_methods.md`](../hm/docs/02_methods.md) — phương pháp (tower, luật phục vụ, ứng viên, reranker, kênh cold)
* [`hm/03_experiments_and_results.md`](../hm/docs/03_experiments_and_results.md) — toàn bộ thực nghiệm và kết quả
* [`hm/04_notebook_guide.md`](../hm/docs/04_notebook_guide.md) — cách chạy và tái lập
* [`hm/05_thesis_plan.md`](../hm/docs/05_thesis_plan.md) — kế hoạch chuyển luận văn sang H&M
* [`hm/results/`](../hm/docs/results/) — kết quả gốc (JSON)

### C. Amazon — dữ liệu, mô hình, thực nghiệm (giữ nguyên)
5. [Báo cáo phân tích tập dữ liệu](../legacy/docs/dataset_analysis.md)
6. [Báo cáo trích xuất đặc trưng đa phương thức (Embeddings)](../legacy/docs/multimodal_embeddings_report.md)
7. [Thiết kế User Tower (Attention/Transformer trên chuỗi hành vi)](../legacy/docs/user_tower_design.md)
8. [Báo cáo thực nghiệm & Ablation Study - User Tower](../legacy/docs/user_tower_experiments.md)
9. [Đánh giá thực nghiệm User Tower (bản nhận xét thô)](../legacy/docs/evaluate/user_tower.md)

### D. Hệ thống: vector DB, RAG, web (hiện dựng trên dữ liệu Amazon)
10. [Thiết kế cơ sở dữ liệu vector (Qdrant)](../hm/docs/qdrant_vector_db_design.md)
11. [Chatbot RAG gợi ý & tìm kiếm sản phẩm](../hm/docs/rag_chatbot_design.md)
12. [Thiết kế web / marketplace](../hm/docs/web/README.md)

### E. Nhật ký kiểm chứng và phản biện
13. [Nhật ký kiểm chứng & phản biện học thuật (Logs)](./logs/README.md) — gồm biên bản Amazon (14/09–30/09/2026) và nhật ký H&M (05/10/2026)

### F. Luận văn
14. [`LuanVan/`](./LuanVan/README.md) — bộ khung luận văn (hiện theo bộ Amazon; kế hoạch chuyển sang H&M ở `hm/05_thesis_plan.md`)

## Nguyên tắc phạm vi

- Phần nghiên cứu cốt lõi là đánh giá **Interaction only vs Content only vs Multimodal** trên gợi ý thông thường và sản phẩm ít tương tác. *(Trên H&M, ablation này chưa được chạy; xem `hm/05_thesis_plan.md`, mục 4.)*
- Conversational recommendation, RAG, explanation và lightweight agent là phần mở rộng hệ thống.
- RAG không huấn luyện và không thay thế recommender; RAG chỉ tạo câu trả lời dựa trên bằng chứng sau khi hệ thống đã có Top-K.
- Agent chỉ điều phối các công cụ có sẵn, không tự quyết định sản phẩm ngoài kết quả của recommender.
- Nếu chậm tiến độ, ưu tiên theo thứ tự: dữ liệu và đánh giá → multimodal và cold-start → hội thoại → RAG → agent.
- Mọi so sánh phải kèm baseline phù hợp và khoảng tin cậy; không so trực tiếp số của hai bộ dữ liệu khác nhau.
