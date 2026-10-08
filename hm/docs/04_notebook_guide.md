# H&M — Hướng dẫn chạy notebook (Kaggle) và tái lập kết quả

> Cập nhật 05/10/2026. Thay thế `docs/HM_NOTEBOOK_GUIDE.md` (đã gỡ). Bốn notebook trong `hm/notebooks/` độc lập với pipeline Amazon và Coveo; mặc định ghi vào `/kaggle/working/`. **Các notebook được bảo trì thủ công** — `hm/scripts/build_hm_notebooks.py` (bộ sinh notebook v1) đã bị vô hiệu hoá để không ghi đè bản v2 (xem mục 8).

## 1. Thứ tự chạy

| # | Notebook | Việc | GPU | Đầu ra chính |
|---|---|---|---|---|
| 00 | `00_hm_data_preparation.ipynb` | Lấy mẫu 50k khách, chia cửa sổ, ETL Parquet; **v2: xuất `item_daily_counts.parquet`** | không | `train/rerank_train/valid/test.parquet`, `items.parquet`, `run_config.json`, `dataset_manifest.json`, `item_daily_counts.parquet` |
| 01 | `01_hm_jina_embeddings.ipynb` | Jina CLIP v2 → embedding text/ảnh 512 chiều | 2×T4 | `text_embeddings.npy`, `image_embeddings.npy`, `embedding_manifest.json` |
| 02 | `02_hm_retrieval.ipynb` | Train tower, chọn epoch, cold-fix, **luật phục vụ**, refit, đánh giá full-ranking + 1+99 | 1 GPU | `best_retrieval.pt`, `best_retrieval_refit.pt`, `retrieval_metrics.json`, `retrieval_history.json` |
| 03 | `03_hm_reranking_evaluation.ipynb` | Ứng viên 7 nguồn, 24 đặc trưng, LightGBM, kênh cold, đánh giá, **xuất model** | GPU nhẹ + CPU | `reranker_lgbm.txt`, `reranker_metrics.json`, `hm_export/` + `hm_export.zip` |

Notebook 00 và 01 đã chạy cho `hm_v1` (dữ liệu đã được đưa lên Kaggle Dataset `hoho0111/hm-dataset-filled`). Chỉ cần **chạy lại 02 và 03** khi thay đổi mô hình; riêng `item_daily_counts.parquet` cần bổ sung một lần (mục 2).

## 2. Bước 0 (một lần): xuất thống kê item từ toàn bộ giao dịch
Cần để các số báo cáo cuối dùng thống kê toàn cục (`01`, mục 5). Nếu bỏ qua, 02 và 03 chạy với thống kê từ mẫu 50k (kết quả khác, xem `03` E3 so với E8) và in dòng cảnh báo.

1. Mở `00_hm_data_preparation.ipynb` trên Kaggle, thêm dữ liệu cuộc thi H&M vào Input.
2. **Không Run All** (cell ETL sẽ ghi đè các Parquet đã dùng cho mọi kết quả). Chỉ chạy: cell cấu hình đầu tiên → cell import/seed → cell đọc `transactions_train.csv` (tạo `schema`) → cell "Thống kê item theo ngày".
3. Tải `/kaggle/working/hm_v1/item_daily_counts.parquet` (cột `article_id, t_dat, n, price_sum`; một dòng cho mỗi cặp item–ngày có giao dịch, cỡ vài triệu dòng) rồi thêm vào Kaggle Dataset `hm-dataset-filled` (version mới), cùng thư mục với `items.parquet`. Hoặc đặt đường dẫn bằng `HM_GLOBAL_STATS`.

## 3. Chạy 02 rồi 03
* Bật GPU (T4). Input: Dataset `hm-dataset-filled`. Chạy **02 xong rồi 03 trong cùng phiên** (03 đọc `/kaggle/working/hm_retrieval/`), hoặc tải thư mục `hm_retrieval` thành Dataset và đặt `HM_RETRIEVAL_DIR`.
* Kiểm tra sau khi 02 bắt đầu: dòng `sanity 1+99: oracle HR@10=1.000 | random HR@10=0.10x` và dòng `thống kê bán hàng từ: global (...)`.
* Sau 02 đọc: lưới cold-fix, 5 cấu hình tốt nhất của luật phục vụ cùng dòng "không luật", và dòng `chọn active_days=…, cold_bonus=…, pop_weight=…`. Nếu `pop_weight` hoặc `cold_bonus` rơi vào biên lưới (`0,5` / `0,2`), mở rộng `HM_POP_GRID`/`HM_BONUS_GRID`.
* Sau 03 đọc: bảng `TEST (tower refit + ranker refit)`, các dòng `reranker+cold_slot/*` và lát `never_sold_items`, bootstrap ghép cặp.
* **Lưu kết quả**: dùng *Save Version → Save & Run All* và lấy file ở tab Output; thư mục `/kaggle/working` của phiên tương tác sẽ mất khi phiên kết thúc.

## 4. Biến môi trường (đặt bằng `os.environ[...]` ở cell đầu, hoặc môi trường cục bộ)

### Chung
| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `HM_PROFILE` | `quick` | `quick` = 50k khách, 8 epoch; `full` = 300k khách (chưa chạy), 20 epoch |
| `HM_DATA_DIR` | `/kaggle/input/datasets/hoho0111/hm-dataset-filled` | Thư mục dữ liệu (Parquet, `run_config.json`, embedding; 02/03 tìm `.npy` ở gốc, `embedding/hm/`, `embedding/`) |
| `HM_GLOBAL_STATS` | `<HM_DATA_DIR>/item_daily_counts.parquet` | Bảng đếm theo (item, ngày) toàn bộ khách |
| `HM_DEBUG_USERS` | `0` | Chỉ để chạy thử nhanh: giới hạn số khách (02: số khách train; 03: số khách mỗi cửa sổ) |
| `HM_SAMPLED_NEG` | `99` | Số âm trong giao thức 1+99 |

### Notebook 02
| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `HM_OUTPUT_DIR` | `/kaggle/working/hm_retrieval` | Thư mục ra |
| `HM_MODALITY` | `multimodal` | `id` / `text` / `image` / `multimodal` (ablation) |
| `HM_EPOCHS` | 8 (`quick`) | Số epoch tối đa |
| `HM_NUM_NEGATIVES` | 128 | Số âm khi train |
| `HM_UNIFORM_NEG_RATIO` | 0.0 | Tỉ lệ âm lấy đều (còn lại theo popularity^0,75) |
| `HM_ID_DROPOUT` | 0.3 | Xác suất bỏ ID mỗi hàng mỗi bước; 0 = tắt |
| `HM_COLD_FIX` / `HM_COLD_GRID` | 1 / `1,2,3,5,10` | Quét ngưỡng tắt ID lúc suy luận (0 luôn được thử) |
| `HM_TUNE_SERVING` | 1 | 0 = bỏ luật phục vụ |
| `HM_ACTIVE_GRID` / `HM_BONUS_GRID` / `HM_POP_GRID` | `0,14,28` / `0,0.05,0.1,0.2` / `0,0.02,0.05,0.1,0.2,0.3,0.5` | Lưới luật phục vụ (quét trên valid) |
| `HM_REFIT` | 1 | Train lại trên train+selection+valid để chấm test |
| `HM_NUM_WORKERS` | 2 (Linux) / 0 (Windows) | DataLoader |

### Notebook 03
| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `HM_RETRIEVAL_DIR` / `HM_RERANK_DIR` / `HM_EXPORT_DIR` | `/kaggle/working/hm_retrieval` / `hm_reranking` / `hm_export` | Thư mục vào/ra |
| `HM_RANKER` | `lgbm` | `lgbm` hoặc `mlp` (tự rơi về `mlp` nếu không có lightgbm) |
| `HM_FEATURES` | `v2` | `v2` (24 đặc trưng, kèm ablation 7 đặc trưng) hoặc `v1_like` |
| `HM_TOWER_FOR_TEST` | `refit` | `refit` hoặc `base` |
| `HM_RANKER_REFIT` | 1 | Refit ranker trên selection+valid cho test |
| `HM_COLD_K` | 100 | Số ứng viên của kênh cold (0 = tắt) |
| `HM_COLD_EVERY` | `0,10,5` | Chèn 1 item cold sau mỗi N vị trí; báo cáo từng N |
| `HM_RERANK_EPOCHS` | 6 (`quick`) | Chỉ cho `HM_RANKER=mlp` |

## 5. Công thức tái lập các cấu hình trong báo cáo

| Mục đích | Cấu hình |
|---|---|
| Kết quả cuối cùng (E8, E12) | Mặc định; có `item_daily_counts.parquet` |
| Tái lập tower v1 | 02 với `HM_ID_DROPOUT=0 HM_COLD_FIX=0 HM_REFIT=0 HM_TUNE_SERVING=0` (E1) |
| Không luật phục vụ | `HM_TUNE_SERVING=0` (khi đó báo cáo vẫn in `tower_without_serving_rule`) |
| Reranker 7 đặc trưng | Luôn được train song song dưới tên `reranker_v1_features` khi `HM_FEATURES=v2` |
| Không kênh cold | 03 với `HM_COLD_K=0` |
| Ablation modality (chưa chạy) | Chạy 02 + 03 bốn lần với `HM_MODALITY=id`, `text`, `image`, `multimodal`; mỗi lần đổi `HM_OUTPUT_DIR`, `HM_RETRIEVAL_DIR`, `HM_RERANK_DIR` |
| Chạy thử nhanh | `HM_DEBUG_USERS=2000 HM_EPOCHS=1` (02), `HM_DEBUG_USERS=300` (03) — mất 1–3 phút |

## 6. Chạy cục bộ (Windows, RTX 3050 4GB)
Môi trường: `.venv` của dự án (có `polars`, `torch+cu121`); `lightgbm` chưa cài sẵn — `pip install lightgbm`, nếu không 03 dùng MLP. Dữ liệu cục bộ ở `data/hm/` (không có `item_daily_counts.parquet` → thống kê mẫu). Chạy notebook bằng Jupyter/VS Code sau khi đặt biến môi trường `HM_DATA_DIR=data/hm`, `HM_OUTPUT_DIR=...`; hoặc nạp các cell mã như `hm/scripts/probe_serving_rule.py` (nạp phần định nghĩa của notebook 02 mà không huấn luyện). Mỗi epoch tower ≈ 99 giây; đầy đủ 8 epoch + refit ≈ 30 phút GPU — hãy báo trước khi chạy trên máy cá nhân.

Script thăm dò kèm theo (đều có `--help`):
* `hm/scripts/dataset_stats.py` — thống kê dữ liệu (CPU, ~1 phút).
* `hm/scripts/probe_cold_clip.py` — thăm dò cold thật sự bằng CLIP (CPU, ~1 phút).
* `hm/scripts/probe_serving_rule.py --ckpt …` — quét luật phục vụ lúc suy luận (GPU nhỏ, 1–2 phút).

## 7. Đầu ra và nơi lưu kết quả

| File | Nơi | Ghi chú |
|---|---|---|
| `best_retrieval.pt` (tower base), `best_retrieval_refit.pt` | `hm/checkpoints/` (không đưa vào git) | ≈ 490 MB/bản vì chứa hai buffer embedding; `hm_export/tower.pt` là bản gọn (~58 MB) |
| `reranker_lgbm.txt`, `reranker_v1_features_lgbm.txt` | `hm/checkpoints/`, bản chính cũng ở `hm/docs/results/` | Mô hình LightGBM dạng text; **không đủ** để suy luận nếu thiếu pipeline đặc trưng (mục 4 của notebook 03) và `item_daily_counts.parquet` |
| `retrieval_metrics.json`, `reranker_metrics.json`, `retrieval_history.json` | `hm/docs/results/` (có git) | Nguồn của mọi bảng trong `03_experiments_and_results.md` |
| `model_meta.json` | trong `hm_export/` | Thứ tự đặc trưng, luật phục vụ, hằng số ứng viên, cutoff |

`hm/checkpoints/*.pt` và `*.zip` đã được thêm vào `.gitignore`; các file kết quả nhỏ nằm ở `hm/docs/results/`.

## 8. Quy ước bảo trì
* Notebook `hm/notebooks/00…03` là **nguồn chính thức** (không sinh tự động). `hm/scripts/build_hm_notebooks.py` là bộ sinh của phiên bản v1 và đã bị chặn bằng cờ `--force`; không chạy nó để "đồng bộ".
* `hm/configs/hm.yaml` chỉ là bảng tham chiếu tham số; notebook đọc **biến môi trường**, không đọc YAML.
* Mọi thay đổi phương pháp phải kèm: cập nhật `hm/docs/02_methods.md`, một mục thực nghiệm trong `03`, và dòng trong `engineering_logs/`.
* Thống kê item chỉ được dùng ngày `< cutoff` của cửa sổ đích; không bao giờ chèn nhãn vào ứng viên.

## 9. Ghi chú về notebook embedding (01)
Giữ nguyên các quyết định kỹ thuật đã kiểm chứng ở lần chạy đầu:
* Cần bật Internet Kaggle lần đầu để tải `jinaai/jina-clip-v2`; cache tại `/kaggle/working/hm_v1/hf_cache`.
* Chạy **FP32** (code ảnh từ xa của Jina tạo tensor Float32 bên trong; ép FP16/BF16 trên T4 gây lỗi ghi chỉ mục khác dtype). Mỗi T4 giữ một bản FP32; batch text/ảnh 64/16.
* Văn bản item embed với `task=None`; checkpoint này chỉ nhận `retrieval.query` cho truy vấn, không hỗ trợ `retrieval.passage`.
* Ghim bộ thư viện nhất quán (`numpy==2.0.2`, `scipy==1.14.1`, `scikit-learn==1.5.2`, `transformers==4.51.3`, `einops==0.8.1`, `timm==1.0.19`, `Pillow==11.3.0`); **khởi động lại phiên Kaggle một lần** sau cell cài đặt (Pillow 12 bỏ `_Ink` làm hỏng import ảnh).
* Cell ảnh có kiểm tra thử 1 ảnh, ghi mỗi batch vào `.npy` memmap, tự tiếp tục sau gián đoạn, cô lập JPEG hỏng vào `image_embedding_failures.json`. Chỉ chạy 02 khi `embedding_manifest.json` báo độ phủ ảnh mong đợi (99,58%).

## 10. Đọc metric đúng
Báo cáo HitRate/Recall/NDCG/MAP@12 trên cửa sổ 7 ngày, kèm Recall@100 và recall của tập ứng viên (cận trên của reranker), luôn cạnh `recent_popularity` và `popularity`; so sánh bằng bootstrap ghép cặp. Không so số của H&M với số Amazon leave-last-out (`01`, mục 7).
