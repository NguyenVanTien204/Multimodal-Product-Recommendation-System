# H&M — Bộ dữ liệu chính mới của đề tài

> Cập nhật 05/10/2026. Thư mục này là **nguồn duy nhất, đồng bộ** cho mọi nội dung liên quan đến thực nghiệm trên bộ **H&M Personalized Fashion Recommendations** (`hm_v1`): dữ liệu, phương pháp, kết quả, cách chạy, và kế hoạch chuyển luận văn. Đây là tiền đề để viết lại `docs/LuanVan/` (hiện dựng trên bộ Amazon, không bị xoá — xem `05_thesis_plan.md`).

## Tóm tắt một đoạn

Trên bộ H&M (mẫu 50.000 khách, catalog 105.542 sản phẩm, dự đoán giỏ hàng 7 ngày kế tiếp), baseline mạnh nhất không phải collaborative filtering mà là **"bán chạy tuần trước"** (HitRate@12 = 0,076, gấp ~4 lần popularity toàn thời gian), vì mỗi tuần chỉ ~4,5 nghìn / 105 nghìn sản phẩm đang bán. Hệ thống cuối cùng — User Tower đa phương thức (Jina CLIP v2) có **ID-dropout**, **luật phục vụ nhận thức thời gian**, ứng viên 7 nguồn với thống kê bán hàng theo ngày, **LightGBM LambdaRank 24 đặc trưng** và **kênh cold dựa trên CLIP** — đạt HitRate@12 = **0,128**, HitRate@100 = **0,365** trên test (vượt baseline mạnh nhất +0,052 HitRate@12, CI 95% [+0,037, +0,066]). Cold-start gồm hai loại: *mới-bán-gần-đây* giải được bằng thống kê bán hàng; *chưa-từng-bán* chỉ còn tín hiệu nội dung (CLIP) và được xử lý bằng chính sách chèn vị trí với đánh đổi định lượng. **Chưa làm**: ablation modality (id/text/image/multimodal), nhiều seed, cold-start người dùng.

## Danh mục tài liệu

| File | Nội dung |
|---|---|
| [`01_dataset_and_protocol.md`](01_dataset_and_protocol.md) | Dữ liệu `hm_v1`, lấy mẫu, cửa sổ thời gian, chẩn đoán dữ liệu, thống kê item không rò rỉ, định nghĩa metric, giao thức sampled 1+99, so sánh với Amazon |
| [`02_methods.md`](02_methods.md) | User Tower, ID-dropout, cold-fix, luật phục vụ, refit, sinh ứng viên, kênh cold CLIP, 24 đặc trưng, LightGBM, chèn vị trí, siêu tham số |
| [`03_experiments_and_results.md`](03_experiments_and_results.md) | Toàn bộ thực nghiệm E0–E13 có thứ tự (v1 vs v2, các thăm dò, 3 lần chạy Kaggle, kết quả cuối, hạn chế, bài học) |
| [`04_notebook_guide.md`](04_notebook_guide.md) | Cách chạy 4 notebook trên Kaggle, biến môi trường, công thức tái lập từng cấu hình, quy ước bảo trì |
| [`05_thesis_plan.md`](05_thesis_plan.md) | Luận điểm được/không được khẳng định, thí nghiệm còn thiếu, bảng/hình đề xuất, quyết định cần chốt, ánh xạ sang các chương luận văn, Q&A phản biện |
| [`06_web_migration.md`](06_web_migration.md) | Chuyển web, RAG và recommender sang H&M: catalog cửa hàng (72.811 sản phẩm, giá/nhóm khách là quy ước), mô hình phục vụ khớp notebook, review minh hoạ mượn từ Amazon (phương pháp, số liệu, hạn chế), kiểm chứng và giới hạn |
| [`results/`](results/) | Kết quả gốc: `retrieval_metrics.json`, `reranker_metrics.json`, `retrieval_history.json`, `dataset_stats.json`, `reranker_lgbm.txt` |

Nhật ký diễn tiến theo ngày: [`../logs/2026-10-05_hm_coldstart_sampled_eval.md`](logs/2026-10-05_hm_coldstart_sampled_eval.md).

## Bản đồ artifact

| Loại | Vị trí | Trạng thái |
|---|---|---|
| Notebook | `hm/notebooks/00_hm_data_preparation.ipynb` … `03_hm_reranking_evaluation.ipynb` | v2, bảo trì thủ công (nguồn chính thức) |
| Tham số tham chiếu | `hm/configs/hm.yaml` | Chỉ để tra cứu (notebook đọc biến môi trường `HM_*`) |
| Script thăm dò / thống kê | `hm/scripts/dataset_stats.py`, `probe_cold_clip.py`, `probe_serving_rule.py` | Tái lập các số E0, E5, E7, E9 |
| Checkpoint | `hm/checkpoints/` (tower base + refit ~490 MB/bản, LightGBM, JSON) | Không đưa vào git (`.gitignore`); JSON nhỏ nằm ở `results/` |
| Dữ liệu mẫu cục bộ | `data/hm/` (Parquet, embedding, ảnh) | Không theo dõi trong git |
| Bộ sinh notebook cũ | `hm/scripts/build_hm_notebooks.py` | **Đã vô hiệu hoá** (chạy sẽ ghi đè v2 bằng v1) |
| Nhật ký kỹ thuật | `engineering_logs/{CHANGELOG,DECISION_LOG,EVALUATION_LOG,TASK_LEDGER}.md` | Cập nhật 05/10/2026 |

## Chạy lại nhanh
Xem `04_notebook_guide.md`. Tóm tắt: (0) xuất `item_daily_counts.parquet` một lần từ notebook 00; (1) chạy 02 rồi 03 trong cùng phiên Kaggle (GPU T4, tổng ≈ 30 phút); (2) lưu bằng *Save Version* và tải `hm_export.zip` + hai file JSON.

## Tài liệu đã gỡ khỏi repo (còn trong lịch sử git)

| File | Lý do |
|---|---|
| `docs/hm_retrieval_diagnosis_and_business_metrics.md` | Chẩn đoán một phiên bản notebook trước đó (lỗi Log-Q, `PairData`, position embedding) — các lỗi đó không còn ở notebook hiện hành hoặc không áp dụng; chứa "kết quả kỳ vọng" chưa từng đo (cold HitRate 8–15%, lift 150–250×, HitRate@100 28–42%) và một số đã bị số liệu thực bác bỏ; kịch bản thuyết trình điền sẵn con số chưa có. Nội dung còn giá trị đã được đưa vào `01`, `02`, `03` với số đo thật |
| `docs/HM_NOTEBOOK_GUIDE.md` | Mô tả pipeline v1; thay bằng `04_notebook_guide.md` (giữ lại các ghi chú kỹ thuật về embedding Jina ở mục 9) |

Tài liệu của bộ **Amazon** và **Coveo** được giữ nguyên; chúng chỉ được thêm ghi chú phạm vi (xem `docs/README.md`).
