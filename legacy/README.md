# `legacy/` — Bộ Amazon Reviews 2023 và Coveo (đối chứng)

> Code của các bộ dữ liệu **không còn là bộ chính**. Được giữ nguyên để tái lập kết quả cũ và làm đối chứng cho H&M; không xoá. Bộ chính hiện tại: [`../hm/`](../hm/README.md). Cập nhật 06/10/2026 khi tách thư mục.

| Thư mục | Nội dung |
|---|---|
| [`src/datn_legacy/`](src/datn_legacy/) | Package Python **`datn_legacy`** (tách từ `datn`): `data/` (pipeline Amazon, `balanced`, Coveo) · `features/` · `experiments/` (checkpoint bất biến) · `recommenders/{user_tower,reranker,coveo}/` |
| [`notebooks/`](notebooks/) | Notebook Amazon (EDA, embedding, User Tower, reranker) và `coveo/` (00–04) |
| [`scripts/`](scripts/) | `eval_*` (baseline, cold-start, sampled, chẩn đoán…), `build_coveo_notebooks.py` |
| [`configs/`](configs/) | `phase1`, `balanced_dataset`, `user_tower*.yaml`, `coveo*.yaml` |
| [`tests/`](tests/) | `test_phase1`, `test_balanced_dataset`, `test_user_tower`, `test_coveo_pipeline`, `test_experiment_checkpoint` |
| [`docs/`](docs/) | Báo cáo Amazon/Coveo: thiết kế User Tower, embedding, phân tích dữ liệu, `logs/2026-09-*` |
| [`checkpoints/`](checkpoints/) | Snapshot checkpoint Amazon cũ |

## Điều cần biết

- `datn_legacy` **không** được `hm/src/datn` import. Chiều ngược lại có một chỗ: `hm/apps/recommender` (nhánh `DATN_ENGINE=amazon`) import `datn_legacy.recommenders.{reranker,user_tower}`.
- Lệnh CLI cũ vẫn dùng được sau khi cài `pip install -e ".[dev]"`: `datn-data`, `datn-balanced-data`, `datn-user-tower`, `datn-checkpoint` (trỏ tới `datn_legacy.*`).
- Script `legacy/scripts/eval_*.py` chạy từ gốc repo: `PYTHONPATH=legacy/src:legacy/scripts python legacy/scripts/eval_sampled.py`.
- Test nhánh này: `python -m pytest legacy/tests`. Có 1 test lỗi sẵn từ trước khi tách thư mục: `test_coveo_pipeline.py::test_dense_temporal_slice_keeps_coherent_active_catalog` (DuckDB không cho tham số có chuẩn bị ở câu lệnh không phải câu cuối trong `coveo_dense.py`).
- Dữ liệu Amazon vẫn ở `data/` (gốc repo), không chuyển vào đây.
