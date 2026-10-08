# `hm/` — Hệ thống dùng bộ dữ liệu H&M

> Thư mục tổng cho **toàn bộ code của hệ thống chạy trên bộ H&M Personalized Fashion Recommendations** (bộ dữ liệu chính của đề tài, `hm_v1`): thực nghiệm, thư viện, dịch vụ, web, test, tài liệu, triển khai. Cấu trúc mô phỏng một repo nhỏ: mỗi thư mục con làm đúng một việc.
> Bộ Amazon/Coveo (đối chứng) nằm ở [`../legacy/`](../legacy/README.md). Cập nhật 06/10/2026 khi tách thư mục.

## Bản đồ

| Thư mục | Vai trò | Nội dung chính |
|---|---|---|
| [`apps/`](apps/) | **Dịch vụ chạy được** (4 container) | `backend/` marketplace FastAPI (auth, catalog, giỏ, đơn, preferences, cổng `/chat`) · `recommender/` dịch vụ gợi ý `:8100` · `rag/` chatbot RAG `:8200` · `web/` Next.js |
| [`src/datn/`](src/datn/) | **Thư viện Python dùng chung** (package `datn`) | `recommenders/hm/` (User Tower + luật phục vụ + LightGBM, port từ notebook) · `catalog/` (dựng catalog cửa hàng, review minh hoạ) · `retrieval/` (Jina CLIP, Qdrant, tìm kiếm lai) · `rag/` (câu trả lời có căn cứ, LLM) · `agent/` (ý định, phiên, bộ nhớ sở thích) · `vectordb/` · `evaluation/` |
| [`notebooks/`](notebooks/) | **Nghiên cứu** — pipeline huấn luyện/đánh giá trên Kaggle | `00` chuẩn bị dữ liệu → `01` embedding Jina → `02` retrieval (User Tower) → `03` reranking + đánh giá → `04` serving assets; `export_hm_images_zip` |
| [`scripts/`](scripts/) | **Công cụ dòng lệnh** | thăm dò / thống kê (`dataset_stats`, `probe_*`), kiểm chứng cổng (`parity_check`, `eval_port`, `verify_serving_assets`), `export_serving_bundle`, `sim_preference_feedback`; công cụ RAG (`check_llm`, `benchmark_rag`, `build_rag_notebooks`) |
| [`configs/`](configs/) | Tham số | `hm.yaml` (tham chiếu; notebook đọc biến môi trường `HM_*`), `qdrant.yaml` |
| [`tests/`](tests/) | Test (pytest) | `test_hm_*`, `test_preferences`, `test_mock_reviews`, `test_rag_*`, `test_chat_gateway`, `test_compact_weights` |
| [`docs/`](docs/README.md) | Tài liệu H&M | `01`–`07` (dữ liệu, phương pháp, thực nghiệm, notebook, kế hoạch luận văn, chuyển web, bộ nhớ sở thích) · `web/` (thiết kế web) · `rag_chatbot_design.md` · `qdrant_vector_db_design.md` · `results/` · `logs/` |
| [`reports/`](reports/) | Kết quả chạy script (JSON/log) | `feedback_sim_*.json` |
| [`checkpoints/`](checkpoints/) | Checkpoint (`*.pt`/`*.zip` ngoài git) và số liệu nhỏ | tower, LightGBM, `*_metrics.json` |
| [`deploy/azure/`](deploy/azure/README.md) | Triển khai VM Azure | script `vm.sh`, `01-setup.sh`, overlay compose, `Dockerfile.tools` |

## Nằm ngoài `hm/` (có chủ ý)

| Ở gốc repo | Lý do |
|---|---|
| `docker-compose.yml`, `.env`, `.env.example`, `.dockerignore` | Một stack duy nhất; VM Azure chạy `docker compose` từ gốc repo |
| `pyproject.toml` | Cài **cả hai** package: `datn` (từ `hm/src`) và `datn_legacy` (từ `legacy/src`); cấu hình pytest cho cả hai nhánh |
| `data/` | Dữ liệu chạy (ngoài git, được compose mount): `data/hm/` (H&M), phần còn lại là Amazon |
| `docs/` | Tài liệu cấp dự án: tầm nhìn, tổng quan, `LuanVan/` (luận văn), `logs/README.md` (mục lục nhật ký) |
| `engineering_logs/`, `engineering_artifacts/`, `.agents/`, `AGENT.md`, `ROADMAP.md` | Quy trình và nhật ký kỹ thuật |
| [`../legacy/`](../legacy/README.md) | Code Amazon + Coveo (đối chứng) |

## Chạy

```powershell
# test toàn bộ (hm + legacy), chạy từ gốc repo
.venv\Scripts\python -m pytest
.venv\Scripts\python -m pytest hm/tests          # chỉ nhánh H&M

# script (chạy từ gốc repo)
.venv\Scripts\python hm/scripts/parity_check.py --users 40
.venv\Scripts\python hm/scripts/sim_preference_feedback.py --users 300

# stack dịch vụ (từ gốc repo)
docker compose --profile ai up -d --build

# VM Azure
hm/deploy/azure/vm.sh start | stop | tunnel
```

## Code dùng chung giữa H&M và Amazon

`hm/src/datn` là package duy nhất mà các dịch vụ import. Ba chỗ vẫn mang dấu vết Amazon vì H&M tái sử dụng chúng, nên **không** chuyển sang `legacy/`:

- `retrieval/indexer.py`, `retrieval/reviews.py` — nạp sản phẩm/review Amazon vào Qdrant; `retrieval/hm_indexer.py` import hằng số và hàm từ đây.
- `evaluation/retrieval_eval.py` — đo chất lượng tìm kiếm của kho vector.
- `apps/recommender` với `DATN_ENGINE=amazon` — nạp `datn_legacy.recommenders.{reranker,user_tower}`; `DATN_ENGINE=hm` (dùng cho H&M) không đụng tới `datn_legacy`.

## Bảng đổi đường dẫn (trước → sau 06/10/2026)

Các tài liệu hiện hành đã được cập nhật theo bảng này. **Nhật ký theo ngày** (`engineering_logs/`, `docs/logs/20*`, `hm/docs/logs/20*`, `legacy/docs/logs/20*`) giữ nguyên văn bản gốc nên còn đường dẫn cũ; đối chiếu bằng bảng.

| Trước | Sau |
|---|---|
| `apps/…` | `hm/apps/…` |
| `src/datn/{agent,rag,retrieval,vectordb,catalog,evaluation,recommenders/hm}` | `hm/src/datn/…` |
| `src/datn/{data,features,experiments,recommenders/{user_tower,reranker,coveo}}` | `legacy/src/datn_legacy/…` (import: `datn_legacy.…`) |
| `notebooks/hm/…` | `hm/notebooks/…` |
| `scripts/hm/…`, `scripts/{build_hm_*,benchmark_rag,check_llm,pack_kaggle_rag_inputs,build_rag_notebooks}.py` | `hm/scripts/…` |
| `configs/{hm,qdrant}.yaml` | `hm/configs/…` |
| `checkpoints/hm/…`, `reports/hm/…` | `hm/checkpoints/…`, `hm/reports/…` |
| `docs/hm/…`, `docs/web/…`, `docs/{rag_chatbot_design,qdrant_vector_db_design}.md`, `docs/logs/2026-10-05_hm_*` | `hm/docs/…`, `hm/docs/web/…`, `hm/docs/…`, `hm/docs/logs/…` |
| `tests/test_{hm_*,mock_reviews,preferences,chat_gateway,rag_*,compact_weights}.py` | `hm/tests/…` |
| `deploy/azure/…` | `hm/deploy/azure/…` |
| `notebooks/*`, `scripts/*` (còn lại), `configs/*` (còn lại), `tests/*` (còn lại), `checkpoints/*` (còn lại), `docs/{COVEO…,dataset_analysis,evaluate,multimodal…,user_tower_*}`, `docs/logs/2026-09-*` | `legacy/…` |
