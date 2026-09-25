# Roadmap triển khai 12 tuần

## 1. Mốc tổng thể

| Tuần | Trọng tâm | Deliverable / Exit criteria |
|---:|---|---|
| 1 | Dataset, schema, EDA | `data_report.md`, raw Parquet, báo cáo missing/schema |
| 2 | Filtering, K-core, sampling, split | Dataset đóng băng và manifest |
| 3 | Baseline recommendation | Popularity + BPR/MF + 3 metric |
| 4 | Image/Text embeddings | Feature store và cache hoàn chỉnh |
| 5 | Multimodal fusion | Các model content/hybrid chạy end-to-end |
| 6 | Experiments và cold-start | Bảng kết quả nghiên cứu chính |
| 7 | Conversational recommendation | Extract/refine preference và rerank |
| 8 | RAG explanation/comparison | Explain/compare có evidence |
| 9 | Lightweight agent và tích hợp | Demo end-to-end |
| 10 | Error analysis và hardening | Release candidate |
| 11 | Báo cáo, biểu đồ, rerun | Kết quả cuối tái lập được |
| 12 | Buffer, slide, bảo vệ | Thesis-ready, demo/slide hoàn chỉnh |

## 2. Kế hoạch chi tiết

### Tuần 1 — Data foundation

- Tải interaction, review và metadata cần thiết.
- Inspect và xác nhận schema/kiểu dữ liệu.
- Chuyển JSONL sang Parquet bằng Polars.
- Liên kết interaction với metadata qua `parent_asin`.
- Phân tích missing, duplicate, phân bố rating và interaction.
- Ghi rõ nguồn, phiên bản và quy trình xử lý.

Deliverable:

```text
data_report.md
data/interim/interactions_raw.parquet
data/interim/items_raw.parquet
```

### Tuần 2 — Đóng băng dataset

- Chọn positive interaction với `rating >= 4`.
- Chạy iterative K-core.
- Lấy subset theo target tài nguyên.
- Chạy lại K-core sau sampling.
- Sort theo timestamp và leave-last-out theo user.
- Kiểm tra không trùng/rò rỉ giữa train, validation, test.
- Lưu seed, filter, counts và checksum vào manifest.

Deliverable:

```text
data/processed/train.parquet
data/processed/valid.parquet
data/processed/test.parquet
data/processed/items.parquet
data/processed/reviews.parquet
data/processed/dataset_manifest.json
```

Sau mốc này không thay dataset tùy tiện. Mọi thay đổi phải tạo dataset version/manifest mới và chỉ thực hiện khi có lỗi dữ liệu được ghi nhận.

### Tuần 3 — Safety milestone

- Cấu hình protocol đánh giá Top-K thống nhất.
- Huấn luyện/evaluate Popularity và BPR hoặc MF.
- Lưu config, seed, checkpoint và kết quả.
- Có bảng Recall@10, NDCG@10, HitRate@10.

Exit criteria: pipeline `train → recommend → evaluate` chạy end-to-end và tái lập được.

### Tuần 4 — Multimodal feature extraction

- Chỉ tải ảnh của item đã lọc.
- Retry, kiểm tra ảnh hỏng và ghi trạng thái download.
- Chuẩn hóa product text từ title, features, description.
- Encode image/text theo batch.
- Chuẩn hóa vector và lưu mapping `item_id ↔ row index`.
- Cache embedding kèm model/config/version.

Exit criteria: mọi item hợp lệ có embedding hoặc có missing policy được ghi rõ; không encode lại giữa các experiment giống nhau.

### Tuần 5 — Model chính

- Xây dựng projection và late fusion.
- Hoàn thành Image-only, Text-only, Image+Text, CF+Text, CF+Image và Full Multimodal.
- Xây dựng candidate retrieval/ranking nhất quán.
- Smoke test FAISS và kiểm tra loại item đã tương tác khỏi recommendation nếu protocol yêu cầu.

Exit criteria: toàn bộ model matrix chạy bằng cùng một evaluation command/config family.

### Tuần 6 — Research milestone

- Chạy main comparison.
- Chạy ablation `-CF`, `-Image`, `-Text`.
- Quét `λ = 0, 0.25, 0.5, 0.75, 1` trên validation.
- Đánh giá toàn bộ test, item `<= 10` và `<= 5` train interactions.
- Phân tích thống kê mô tả và error cases.
- Chốt bảng/biểu đồ dùng cho luận văn.

Exit criteria: trả lời được RQ1–RQ3 bằng kết quả thực nghiệm. Đây là mốc hoàn thành thesis core.

### Tuần 7 — Conversational recommendation

- Xây dựng schema preference và validation.
- Extract category, color, style, price và negative preference.
- Lưu/cập nhật trạng thái theo phiên.
- Hỗ trợ “rẻ hơn”, “màu tối hơn”, “ít thể thao hơn”, “giống sản phẩm #2”.
- Rerank, không retrain.

### Tuần 8 — RAG

- Lập product/review chunks có `item_id` và provenance.
- Tạo FAISS index cho context.
- Truy xuất context bị giới hạn theo sản phẩm trong Top-K/compare request.
- Thực hiện Why recommended và Compare A vs B.
- Hiển thị bằng chứng hoặc nguồn review/metadata trong câu trả lời.

### Tuần 9 — Agent và tích hợp

- Bọc các chức năng thành tool rõ input/output.
- Tạo một agent điều phối recommend/search/refine/explain/compare.
- Thêm timeout, fallback và validation tool result.
- Tích hợp FastAPI và Next.js + Tailwind CSS thành demo end-to-end.

### Tuần 10 — Hoàn thiện

- Error analysis, edge case, cold user/fallback.
- Kiểm tra seed, config, artifact version và reproducibility.
- Đo latency retrieval/ranking/RAG.
- Viết test cho data split, filters, API và các flow demo.
- Hoàn thiện UI và demo scenario.
- Không thêm model mới.

### Tuần 11–12 — Báo cáo và buffer

- Rerun experiment cuối cùng trên dataset/config đã chốt.
- Xuất bảng, biểu đồ và ví dụ định tính.
- Hoàn thiện báo cáo, slide và demo recording.
- Fix bug, diễn tập bảo vệ và chuẩn bị câu hỏi phản biện.
- Không thêm feature mới.

## 3. Thứ tự ưu tiên khi trễ tiến độ

| Ưu tiên | Phạm vi | Quyết định |
|---|---|---|
| P0 | Dataset, evaluation, collaborative baseline | Không cắt |
| P1 | Image, text, multimodal, cold-start, ablation | Không cắt nếu muốn hoàn thành research core |
| P2 | Conversational recommendation | Giảm số intent nếu cần |
| P3 | RAG explanation/comparison | Giảm số review/chức năng trước khi ảnh hưởng core |
| P4 | Lightweight agent | Cắt đầu tiên; giữ API trực tiếp |

## 4. Quản trị mốc và thay đổi

- Mỗi tuần kết thúc bằng một artifact hoặc bảng kết quả có thể kiểm tra, không chỉ bằng mã nguồn.
- Dataset, protocol đánh giá và metric phải ổn định trước khi so sánh model.
- Chỉ dùng validation để chọn model/hyperparameter; không điều chỉnh theo test.
- Kết quả experiment phải lưu config, dataset version, seed, commit và thời gian chạy.
- Từ tuần 10 chỉ sửa lỗi, phân tích và hoàn thiện; mọi đề xuất feature mới đưa vào future work.

## 5. Definition of Done tổng thể

- Dữ liệu đã đóng băng và có manifest.
- Baseline và multimodal variants được đánh giá bằng cùng protocol.
- Có ablation và sparse-item evaluation ở hai threshold.
- RQ1–RQ3 có kết luận dựa trên số liệu và phân tích sai số.
- API/demo thể hiện ít nhất recommend, refine, explain và compare; agent là tùy chọn cuối.
- Mã nguồn, config, artifact và hướng dẫn chạy đủ để tái lập kết quả chính.
