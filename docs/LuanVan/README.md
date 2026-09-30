# HỆ THỐNG TÀI LIỆU ĐỒ ÁN TỐT NGHIỆP

> **Tên đề tài**: Xây dựng hệ thống gợi ý sản phẩm đa phương thức ứng dụng RAG  
> **Sinh viên thực hiện**: Nguyễn Văn Tiến – MSV: 2221050201 – Lớp: DCCTKH67A  
> **Giảng viên hướng dẫn**: Nguyễn Duy Huy  
> **Khoa / Trường**: Khoa Công nghệ Thông tin – Trường Đại học Mỏ - Địa chất (HUMG)  
> **Năm thực hiện**: 2026

---

## 1. Cấu trúc bộ tài liệu luận văn

Toàn bộ nội dung báo cáo Đồ án Tốt nghiệp được tổ chức dạng module hóa theo các chương nội dung lớn, tương ứng với chuẩn cấu trúc luận văn đại học:

| Tệp tài liệu | Tên chương / Nội dung | Tóm tắt trọng tâm |
| :--- | :--- | :--- |
| [00_Mo_dau.md](file:///d:/WorkSpace/Work/DATN/docs/LuanVan/00_Mo_dau.md) | **Mở đầu & Thông tin đồ án** | Thông tin sinh viên, mục tiêu cụ thể đo lường được, phạm vi dữ liệu (`balanced_u5_i2_v1`), lý do chọn đề tài, tính cấp thiết và bố cục luận văn. |
| [01_Chuong_1_Tong_quan_ly_thuyet.md](file:///d:/WorkSpace/Work/DATN/docs/LuanVan/01_Chuong_1_Tong_quan_ly_thuyet.md) | **Chương 1: Tổng quan lý thuyết về lĩnh vực nghiên cứu** | Cơ sở lý luận về RS, dữ liệu thưa, cold-start; thuật toán lọc K-core, SASRec Self-Attention, Contrastive Learning (CLIP), kiến trúc Two-Stage, nguyên lý RAG, các chỉ số đánh giá (HR@K, NDCG@K) và hệ sinh thái công nghệ. |
| [02_Chuong_2_Quy_trinh_xay_dung_he_thong.md](file:///d:/WorkSpace/Work/DATN/docs/LuanVan/02_Chuong_2_Quy_trinh_xay_dung_he_thong.md) | **Chương 2: Quy trình xây dựng hệ thống gợi ý đa phương thức và Chatbot RAG** | Chi tiết quy trình xử lý dữ liệu Amazon Reviews 2023, kiến trúc User Tower kết hợp biểu diễn lai CLIP, chiến lược sinh ứng viên 4 nguồn (2000 candidates), mạng Residual Listwise Reranker, kho vector Qdrant, cơ chế Grounding Guardrails, thiết kế kiến trúc phần mềm, CSDL PostgreSQL, API và lược đồ UML. |
| [03_Chuong_3_Thuc_nghiem_va_danh_gia.md](file:///d:/WorkSpace/Work/DATN/docs/LuanVan/03_Chuong_3_Thuc_nghiem_va_danh_gia.md) | **Chương 3: Thực nghiệm và đánh giá hệ thống** | Cấu hình phần cứng (AMD Ryzen 7 6800H, GPU RTX 3050 4GB), siêu tham số, bảng kết quả Full-Ranking (Candidate Recall 34.444%, HitRate@10 3.518%, NDCG@10 2.489%), Ablation study, kết quả kiểm thử tự động Chatbot RAG 47/47 tests, kiểm thử tích hợp web/API và khai báo minh bạch AI. |
| [04_Ket_luan.md](file:///d:/WorkSpace/Work/DATN/docs/LuanVan/04_Ket_luan.md) | **Kết luận và hướng phát triển** | Đối chiếu các mục tiêu đã hoàn thành, bảng so sánh yêu cầu tối thiểu trong cẩm nang đồ án, phân tích hạn chế khách quan, bảng khai báo sử dụng AI chi tiết và đề xuất các hướng mở rộng tương lai. |
| [Bao_cao.md](file:///d:/WorkSpace/Work/DATN/docs/LuanVan/Bao_cao.md) | **Báo cáo tổng hợp (Master Draft)** | Tài liệu khung gốc tổng hợp mục lục, danh mục hình vẽ/bảng biểu và đề cương chi tiết của đồ án. |

---

## 2. Nguồn dữ liệu và tài liệu đối chiếu kỹ thuật

Các tài liệu trên được biên soạn dựa trên sự đối chiếu trực tiếp với mã nguồn và các báo cáo kỹ thuật đã được kiểm chứng trong dự án:

- **Bộ dữ liệu chuẩn hóa**: `data/processed/balanced_u5_i2_v1/` và `data/processed/balanced_u5_i2_v1/dataset_manifest.json`.
- **Trích xuất đặc trưng đa phương thức**: [multimodal_embeddings_report.md](file:///d:/WorkSpace/Work/DATN/docs/multimodal_embeddings_report.md) và `data/embedding/`.
- **Thiết kế & Thực nghiệm User Tower**: [user_tower_design.md](file:///d:/WorkSpace/Work/DATN/docs/user_tower_design.md), [user_tower_experiments.md](file:///d:/WorkSpace/Work/DATN/docs/user_tower_experiments.md) và [2026-09-22_balanced_retrieval_reranker_results.md](file:///d:/WorkSpace/Work/DATN/docs/logs/2026-09-22_balanced_retrieval_reranker_results.md).
- **Cơ sở dữ liệu Vector Qdrant**: [qdrant_vector_db_design.md](file:///d:/WorkSpace/Work/DATN/docs/qdrant_vector_db_design.md).
- **Thiết kế Chatbot RAG**: [rag_chatbot_design.md](file:///d:/WorkSpace/Work/DATN/docs/rag_chatbot_design.md) và bộ kiểm thử `tests/test_rag_units.py`.
- **Thiết kế Kiến trúc Web & API**: Thư mục [docs/web/](file:///d:/WorkSpace/Work/DATN/docs/web/).
