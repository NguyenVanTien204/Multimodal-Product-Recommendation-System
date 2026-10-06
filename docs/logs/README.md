# Nhật ký Kiểm chứng Thực nghiệm & Phản biện Hệ thống (Audit & Verification Logs)

Thư mục này lưu trữ các biên bản kiểm chứng kỹ thuật, nhật ký phản biện học thuật, và bằng chứng thực nghiệm độc lập cho đề tài *"Xây dựng hệ thống gợi ý sản phẩm đa phương thức ứng dụng RAG"*.

> **Phạm vi bộ dữ liệu (cập nhật 05/10/2026):** các biên bản từ 14/09 đến 30/09/2026 là của bộ **Amazon Reviews 2023** (`balanced_u5_i2_v1`) và được giữ nguyên làm tài liệu đối chứng. Bộ dữ liệu chính mới của đề tài là **H&M**; nội dung chuẩn hoá nằm ở [`../hm/`](../hm/README.md), biên bản diễn tiến ở dòng 8 bên dưới.

## Danh sách Biên bản Kiểm chứng (Audit Logs)

| STT | Thời gian | Bộ dữ liệu | Tên tài liệu / Biên bản | Nội dung chính | Trạng thái |
|:---:|:---:|:---:|---|---|:---:|
| 1 | 14/09/2026 | Amazon | [`2026-09-14_user_tower_verification.md`](./2026-09-14_user_tower_verification.md) | Phản biện P0: Kiểm chứng Evaluation Protocol, Leave-Last-Out Split, Candidate Retrieval 152k, Bóc tách nguyên nhân CF-only sụp đổ (Sparsity 99.995%, Zero-init), Phân rã thực nghiệm Warm vs Cold. | **ĐÃ HOÀN THÀNH** |
| 2 | 14/09/2026 | Amazon | [`2026-09-14_user_tower_p1_difficulty_and_popularity_bias.md`](./2026-09-14_user_tower_p1_difficulty_and_popularity_bias.md) | Phản biện P1: Kiểm chứng Độ khó bài toán (Sparsity 99.995%, 82% singleton items), Đánh giá Popularity Baseline vs User Tower, Phân tích Overlap (Disjoint Head vs Tail), Kiểm chứng Popularity Bias. | **ĐÃ HOÀN THÀNH** |
| 3 | 14/09/2026 | Amazon | [`2026-09-14_item_degree_sparsity_root_cause.md`](./2026-09-14_item_degree_sparsity_root_cause.md) | Phản biện Nguồn gốc Dữ liệu: Phân tích nguyên nhân 80% sản phẩm chỉ có 1 tương tác (Bẻ gãy đồ thị do User-Hash 1/32, bộ lọc helpful_vote >= 3, và chủ đích tắt MIN_ITEM_DEGREE=1). | **ĐÃ HOÀN THÀNH** |
| 4 | 14/09/2026 | Amazon | [`2026-09-14_dataset_sparsity_critique_and_defense.md`](./2026-09-14_dataset_sparsity_critique_and_defense.md) | Đánh giá Rủi ro & Chiến lược Phản biện: Tính thực tế của dữ liệu siêu thưa, giải trình trước Hội đồng về sự đánh đổi RAG vs CF, và bộ kịch bản bảo vệ luận văn (Defense Playbook). | **ĐÃ HOÀN THÀNH** |
| 5 | 14/09/2026 | Amazon | [`2026-09-14_enriched_dataset_user_tower_results.md`](./2026-09-14_enriched_dataset_user_tower_results.md) | Báo cáo Đánh giá Toàn diện: Kết quả Thực nghiệm User Tower trên Dữ liệu Làm giàu (26.428 Users, 152k Catalog), So sánh Tăng trưởng vs Dữ liệu Cũ, Phân tích Overlap Trực giao với Popularity Baseline. | **ĐÃ HOÀN THÀNH** |
| 6 | 22/09/2026 | Amazon | [`2026-09-22_balanced_retrieval_reranker_results.md`](./2026-09-22_balanced_retrieval_reranker_results.md) | Quyết định bộ dữ liệu `balanced_u5_i2_v1` và công thức ứng viên 4 nguồn (≤ 2.000) cho baseline luận văn Amazon; kết quả retrieval + reranker. | **ĐÃ HOÀN THÀNH** |
| 7 | 30/09/2026 | Amazon | [`2026-09-30_baseline_and_significance_audit.md`](./2026-09-30_baseline_and_significance_audit.md) | Kiểm toán baseline và ý nghĩa thống kê của `balanced_two_stage_v2` (paired bootstrap, McNemar), cold-start Amazon, sampled 1+99 (mục 8), tổng hợp chỉ số thực tiễn. | **ĐÃ HOÀN THÀNH** |
| 8 | 05/10/2026 | **H&M** | [`2026-10-05_hm_coldstart_sampled_eval.md`](./2026-10-05_hm_coldstart_sampled_eval.md) | Nhật ký phiên: chẩn đoán dữ liệu H&M, sửa cold-start (ID-dropout), thêm 1+99, luật phục vụ nhận thức thời gian, reranker LightGBM, kênh cold CLIP, 4 lần chạy Kaggle; kèm sai sót và điều chỉnh. Kết quả chuẩn hoá: [`../hm/03_experiments_and_results.md`](../hm/03_experiments_and_results.md). | **ĐÃ HOÀN THÀNH** (còn việc ở `docs/hm/05`) |

---

## Nguyên tắc Ghi chép Nhật ký Kiểm chứng (Protocol)

Mỗi biên bản kiểm chứng tiếp theo trong thư mục này phải tuân thủ nghiêm ngặt cấu trúc:
1. **Tóm tắt điều hành (Executive Summary)**: Kết luận nhanh các nghi vấn / giả thuyết.
2. **Kiểm chứng mã nguồn & Tính toàn vẹn dữ liệu (Code & Data Evidence)**: Dẫn chứng đường dẫn file, dòng mã, câu truy vấn SQL/Polars.
3. **Phân tích bản chất kỹ thuật (Root Cause Analysis)**: Giải thích cơ chế toán học / thuật toán / kiến trúc sâu bên dưới.
4. **Thực nghiệm độc lập & Trace mẫu (Empirical Verification & Traces)**: Có bảng số liệu phân rã chi tiết, trace log từng user/item cụ thể.
5. **Đề xuất hoàn thiện (Actionable Recommendations)**: Hướng dẫn cập nhật báo cáo luận văn hoặc tối ưu mã nguồn tương ứng.

Với nhật ký diễn tiến của một phiên làm việc (như dòng 8), cấu trúc rút gọn gồm: tóm tắt điều hành, dòng thời gian (bước – bằng chứng – quyết định), sai sót và điều chỉnh, việc còn lại; kết quả đầy đủ đặt trong tài liệu chuẩn hoá tương ứng.
