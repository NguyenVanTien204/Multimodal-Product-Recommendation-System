# Multimodal Product Recommendation — Phase 1

> **Cập nhật 05/10/2026 — bộ dữ liệu chính mới là H&M.** Thực nghiệm khuyến nghị (User Tower + luật phục vụ nhận thức thời gian + LightGBM + kênh cold CLIP) được chuẩn hoá ở [`hm/docs/`](hm/docs/README.md); hướng dẫn chạy notebook: [`hm/docs/04_notebook_guide.md`](hm/docs/04_notebook_guide.md). Phần còn lại của README này mô tả pipeline **Amazon** (giữ nguyên làm đối chứng, đồng thời là nguồn của kho RAG và hệ thống web hiện tại). Mục lục toàn bộ tài liệu: [`docs/README.md`](docs/README.md).

## Cấu trúc thư mục (từ 06/10/2026)

| Thư mục | Nội dung |
|---|---|
| [`hm/`](hm/README.md) | **Toàn bộ code hệ thống dùng bộ H&M**: `apps/` (backend, recommender, rag, web), `src/datn/` (thư viện), `notebooks/`, `scripts/`, `configs/`, `tests/`, `docs/`, `reports/`, `checkpoints/`, `deploy/azure/` |
| [`legacy/`](legacy/README.md) | Code bộ Amazon + Coveo (đối chứng): package `datn_legacy`, notebook, script `eval_*`, test, báo cáo |
| `docs/` | Tài liệu cấp dự án (tầm nhìn, tổng quan, luận văn `LuanVan/`, mục lục nhật ký) |
| `data/` | Dữ liệu chạy, ngoài git (`data/hm/` = H&M) |
| `docker-compose.yml`, `pyproject.toml`, `.env*` | Cấu hình chung của cả stack; `pyproject.toml` cài hai package `datn` và `datn_legacy` |

Chạy mọi lệnh từ gốc repo, ví dụ `python -m pytest` (cả hai nhánh) hoặc `python hm/scripts/parity_check.py`. Bảng đổi đường dẫn cũ → mới: [`hm/README.md`](hm/README.md#bảng-đổi-đường-dẫn-trước--sau-06102026).

Pipeline notebook Coveo hai tầng mới: [hướng dẫn Retrieval → Reranker](legacy/docs/COVEO_NOTEBOOK_GUIDE.md).

Giai đoạn 1 xây nền tảng dữ liệu cho Amazon Reviews 2023
`Clothing_Shoes_and_Jewelry`. Pipeline đọc JSONL/JSONL.GZ theo lô, ghi các part
Parquet rồi hợp nhất bằng Polars streaming; thống kê và kiểm tra liên kết được chạy
out-of-core bằng DuckDB.

## Chạy

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
# Đặt hai file nguồn vào data/raw theo legacy/configs/phase1.yaml
.venv\Scripts\datn-data --config legacy/configs/phase1.yaml
```

Nguồn chính thức (các file rất lớn, không được pipeline tự tải ngầm):

- Reviews: `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Clothing_Shoes_and_Jewelry.jsonl.gz`
- Metadata: `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl.gz`

Hãy kiểm tra dung lượng đĩa trước khi tải. Category này có khoảng 66 triệu review;
việc không tự tải là chủ ý để một lệnh ETL không bất ngờ chiếm hết ổ đĩa/băng thông.

Kết quả:

- `data/interim/interactions_raw.parquet`
- `data/interim/items_raw.parquet`
- `data_report.md`

Pipeline đo RAM khả dụng ngay lúc bắt đầu, giữ phần dự phòng cho hệ điều hành,
chọn ngân sách nhỏ nhất trong các giới hạn cấu hình, và giảm batch khi có áp lực
bộ nhớ. `max_memory_gb` là trần tùy chọn, không phải lượng RAM được giả định là có.
Nếu RAM khả dụng không đủ cho cả `reserve_gb` và `min_budget_mb`, pipeline dừng
trước khi đọc dữ liệu thay vì đánh đổi sự ổn định của toàn máy.
Chạy lại với `--overwrite` nếu chủ ý thay thế artifact hiện có.

## Balanced recommendation benchmark

Dataset chính cho recommender được tạo thành một version mới, không ghi đè các
Parquet nguồn. Pipeline chạy positive iterative k-core (`user >= 5`, `item >= 2`),
chọn hai positive event cuối làm validation/test và giữ rating `<= 2` thành strong
negative riêng:

```powershell
.venv\Scripts\datn-balanced-data --config legacy/configs/balanced_dataset.yaml
```

Output mặc định là `data/processed/balanced_u5_i2_v1/`, kèm
`dataset_manifest.json` chứa tham số, checksum và các kiểm tra leakage. Thư mục
version đã tồn tại sẽ không bị ghi đè; hãy đổi `paths.output_dir` khi tạo bản mới.

Huấn luyện User Tower bằng sampled-softmax mixed negatives có log-Q correction:

```powershell
.venv\Scripts\datn-user-tower --config legacy/configs/user_tower.balanced.yaml train
```

Notebook cũng tự nhận dataset local mới. Trên Kaggle có thể chỉ định các mount bằng
hai biến môi trường `DATN_DATA_DIR` và `DATN_EMB_DIR`.

Đánh giá và lưu full-ranking test metrics, sau đó chạy residual listwise reranker:

```powershell
.venv\Scripts\datn-user-tower --config legacy/configs/user_tower.balanced.yaml evaluate `
  --split test --mode full `
  --output data/artifacts/user_tower_balanced_v1/test_metrics.json
# Chạy legacy/notebooks/reranker_training.ipynb
```

Đóng gói model, scaler, configs, notebook snapshots, source snapshot, metrics,
training history, môi trường và SHA256 thành checkpoint bất biến:

```powershell
.venv\Scripts\datn-checkpoint --destination data/checkpoints/balanced_two_stage_v2
```

## Chatbot RAG gợi ý và tìm kiếm sản phẩm đa phương thức

Kiến trúc, quyết định thiết kế, số liệu đo và giới hạn: [hm/docs/rag_chatbot_design.md](hm/docs/rag_chatbot_design.md).

```powershell
docker compose up -d postgres qdrant
pip install -e ".[rag,dev]"
datn-retrieval index-products            # collection `products` (vector đã có sẵn)
datn-retrieval import-reviews --embeddings data/embedding/review_embeddings.npy --meta data/embedding/reviews_meta.parquet
# review_embeddings.npy được tạo bằng hm/notebooks/kaggle_rag_reviews_and_eval.ipynb (GPU Kaggle)

copy .env.example .env                   # điền RAG_LLM_API_KEY (Gemini); để trống = chạy không LLM
python hm/scripts/check_llm.py              # tự kiểm LLM: kết nối, câu trả lời có căn cứ, chống prompt-injection
docker compose --profile ai up -d --build  # Bỏ comment DATN_RECOMMENDER_URL và DATN_RAG_URL trong .env trước khi chạy
python hm/scripts/benchmark_rag.py --label gpu --reps 10   # đo độ trễ dịch vụ đang chạy
pytest                                   # 78 test (không cần GPU, Qdrant hay LLM)
```

### Tiết kiệm RAM khi chạy đủ stack

Qdrant dùng vector INT8, payload on-disk và graph HNSW nhỏ hơn để giảm RAM khi cùng chạy Jina CLIP. Cấu hình này chỉ có hiệu lực khi tạo collection mới, vì vậy sau khi cập nhật mã cần tái tạo index một lần (lệnh này xóa hai collection Qdrant cũ rồi nạp lại từ các file local):

```powershell
datn-retrieval index-products --recreate
datn-retrieval import-reviews --embeddings data/embedding/review_embeddings.npy --meta data/embedding/reviews_meta.parquet
```

Đổi lại, truy vấn vector có thể tăng nhẹ độ trễ và recall xấp xỉ của HNSW thấp hơn cấu hình `m=32`; RRF và reranker vẫn xử lý lại tập ứng viên trước khi trả kết quả.

LLM dùng Gemini qua endpoint tương thích OpenAI (`https://generativelanguage.googleapis.com/v1beta/openai`, model `gemini-3.5-flash-lite` hoặc `gemini-3.5-flash`). Khoá API chỉ đặt trong `.env` (đã nằm trong `.gitignore`), không bao giờ commit.
