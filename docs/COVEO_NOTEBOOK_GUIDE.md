# Hướng dẫn chạy pipeline Coveo: Retrieval → Reranker

## 1. Mục tiêu và phạm vi

Bộ notebook này thêm **Coveo SIGIR eCommerce 2021** mà vẫn giữ pipeline Amazon để đối chiếu. Coveo có khoảng 36 triệu event, gần 5 triệu session, tín hiệu `detail`, `add`, `purchase`, search click và vector 50 chiều cho mô tả/ảnh/query. Dữ liệu đầy đủ yêu cầu đăng ký và chấp nhận điều khoản tại [repository chính thức của Coveo](https://github.com/coveooss/SIGIR-ecom-data-challenge); dự án không lưu URL riêng tư và không vượt qua cổng cấp quyền.

Hai model được train:

1. **Action-aware multimodal two-tower**: Transformer mã hóa session; item tower dùng ID residual cộng projection text/image; loss retrieval đa dương tính cộng auxiliary purchase loss.
2. **Residual listwise reranker**: học phần hiệu chỉnh có giới hạn trên retrieval score từ feature session, category, price, popularity và similarity text/image.

Target không được chèn cưỡng bức vào candidate pool. Chọn checkpoint bằng validation; test chỉ đọc sau khi khóa model.

## 2. Chuẩn bị môi trường

```powershell
python -m pip install -e ".[dev,coveo]"
```

Khuyến nghị Kaggle GPU T4/P100 hoặc máy có CUDA. Hai profile trong `configs/coveo.yaml`:

- `quick`: xác nhận pipeline, tối đa 10.000 session retrieval và 1.000 session mỗi split reranker.
- `full`: tối đa 500.000 session retrieval và 10.000 session mỗi split reranker. Mức này đủ cho đồ án mà không biến bài toán thành benchmark hạ tầng.

Override profile mà không sửa notebook:

```powershell
$env:COVEO_PROFILE = "full"
```

## 3. Lấy dữ liệu hợp lệ

### Cách A — Kaggle

Gắn một Kaggle Dataset riêng chứa đúng ba file. Notebook tự tìm đệ quy trong `/kaggle/input`:

- `browsing_train.csv`
- `search_train.csv`
- `sku_to_content.csv`

### Cách B — ZIP tải từ Coveo

Sau khi đăng ký và chấp nhận điều khoản:

```powershell
$env:COVEO_TERMS_ACCEPTED = "1"
$env:COVEO_ZIP_PATH = "D:\Downloads\coveo_sigir.zip"
```

### Cách C — smoke test bằng sample chính thức

```powershell
$env:COVEO_USE_SAMPLE = "1"
```

Sample chỉ kiểm tra schema và ETL; quá nhỏ để train/đánh giá có ý nghĩa. Bộ test tự động dùng fixture tổng hợp lớn hơn sample này.

## 4. Thứ tự chạy notebook

Chạy đúng thứ tự trong `notebooks/coveo/`:

1. `00_get_data.ipynb`: phát hiện Kaggle/ZIP/URL được cấp quyền hoặc sample; kiểm tra ba file; lưu SHA-256 và nguồn.
2. `01_prepare_sessions.ipynb`: explode search click theo batch, hợp nhất browsing/search, deduplicate, giữ các positive event `detail/search_click/add/purchase` và session ≥2 event, split toàn cục theo thời gian kết thúc session 80/10/10 rồi lưu Parquet ZSTD. `remove` không được dùng làm positive target.
3. `02_prepare_embeddings.ipynb`: parse vector 50 chiều, căn hàng theo `item_idx`, chuẩn hóa L2, lưu mask thiếu dữ liệu và checksum.
4. `03_train_retrieval.ipynb`: train hai tower; chọn `best_retrieval.pt` duy nhất bằng validation `HitRate@1000`.
5. `04_train_reranker.ipynb`: tạo candidate union retrieval + popularity, tính feature không dùng tương lai, train listwise reranker, đánh giá và đóng băng toàn bộ run.

## 5. Cấu trúc output

```text
data/
├── raw/coveo/acquisition_manifest.json
├── processed/coveo_v1/
│   ├── train.parquet
│   ├── valid.parquet
│   ├── test.parquet
│   ├── items.parquet
│   └── dataset_manifest.json
├── processed/coveo_embeddings_v1/
│   ├── text_embeddings.npy
│   ├── image_embeddings.npy
│   ├── embedding_metadata.parquet
│   └── embedding_manifest.json
└── artifacts/
    ├── coveo_retrieval_v1/
    │   ├── best_retrieval.pt
    │   ├── history.json
    │   └── metrics.json
    └── coveo_reranker_v1/
        ├── best_reranker.pt
        ├── history.json
        ├── metrics.json
        └── candidates/

checkpoints/coveo_<RUN_ID>/checkpoint_manifest.json
```

Processed dataset và checkpoint là bất biến. Muốn thay filter, split hoặc hyperparameter quan trọng, đổi suffix `v1` thành `v2`; không xóa rồi ghi đè kết quả cũ.

## 6. Đọc metric đúng cách

- `HitRate@1000` retrieval là candidate recall riêng của two-tower ở K=1000.
- `candidate_recall` reranker là recall của union candidate thực tế.
- `ConditionalHR@10`: trong số target đã lọt candidate, reranker đưa vào top 10 bao nhiêu lần.
- `HitRate@10 = candidate_recall × ConditionalHR@10` là kết quả end-to-end.

Mốc 70% candidate recall và HR@10 10–15% là mục tiêu stretch, không phải số được bảo đảm trước khi chạy. Luôn báo cùng K, catalog size, rule loại seen item, split thời gian và số session. Kết quả thuyết phục là cải thiện có kiểm soát so với popularity, content-only và retrieval-only trên cùng split.

## 7. Ablation tối thiểu

Giữ seed/split, chạy bốn cấu hình retrieval:

| Thí nghiệm | ID | Text | Image | Action/Purchase auxiliary |
|---|---:|---:|---:|---:|
| Popularity baseline | — | — | — | — |
| Collaborative | ✓ | — | — | ✓ |
| CF + Text | ✓ | ✓ | — | ✓ |
| Full multimodal | ✓ | ✓ | ✓ | ✓ |

Với reranker, so sánh retrieval-only với full features; thêm lát cắt session 2–3 event, session dài, item sparse và target purchase. Chỉ khóa test sau khi chọn cấu hình trên validation.

## 8. Kiểm thử và xử lý lỗi

```powershell
python -m pytest tests/test_coveo_pipeline.py -q
python -m pytest -q
```

- `PermissionError`: chưa đặt `COVEO_TERMS_ACCEPTED=1` cho ZIP/URL full data.
- `FileExistsError`: output version đã tồn tại; đổi version thay vì ghi đè.
- CUDA OOM: giảm `batch_size`; không giảm candidate K trước khi đo ảnh hưởng recall.
- Candidate generation lâu: giảm `max_sessions_per_split_*` khi phát triển, nhưng giữ tập test cuối cố định khi so sánh.
- Candidate recall thấp: ưu tiên retrieval và K trước khi tăng độ phức tạp reranker.
