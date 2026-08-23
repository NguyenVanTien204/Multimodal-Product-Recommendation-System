# Multimodal Product Recommendation — Phase 1

Giai đoạn 1 xây nền tảng dữ liệu cho Amazon Reviews 2023
`Clothing_Shoes_and_Jewelry`. Pipeline đọc JSONL/JSONL.GZ theo lô, ghi các part
Parquet rồi hợp nhất bằng Polars streaming; thống kê và kiểm tra liên kết được chạy
out-of-core bằng DuckDB.

## Chạy

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
# Đặt hai file nguồn vào data/raw theo configs/phase1.yaml
.venv\Scripts\datn-data --config configs/phase1.yaml
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
