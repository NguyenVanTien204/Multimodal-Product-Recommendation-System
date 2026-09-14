# Thiết kế Cơ sở dữ liệu Vector (Qdrant)

Tài liệu này ghi nhận thiết kế collection Qdrant phục vụ truy hồi đa phương thức (multimodal retrieval) cho Đồ án Tốt nghiệp: **"Xây dựng hệ thống gợi ý sản phẩm đa phương thức ứng dụng RAG"**, thay thế phương án FAISS song song (`products_image.faiss` / `products_text.faiss`) được phác thảo ban đầu tại [multimodal_embeddings_report.md](./multimodal_embeddings_report.md) mục 4.

## 1. Quyết định thiết kế

### 1.1. Một collection duy nhất, named vectors thay vì hai collection tách rời

Thay vì `products_image` và `products_text` riêng biệt, hệ thống dùng **một collection `products`** với hai *named vector* dùng chung không gian biểu diễn Jina CLIP v2 ($D=1024$):

- `image`: vector thị giác (EVA-02 ViT-L/14 qua Jina CLIP v2).
- `text`: vector văn bản (Jina CLIP v2 text branch), **sẽ nạp sau** khi notebook `kaggle_text_embeddings.ipynb` hoàn tất.

Lý do:
- Một point có thể **thiếu** một named vector (Qdrant chấp nhận partial vectors trên một point). Vector ảnh được nạp trước, vector text nạp sau qua `upsert` sẽ **hợp nhất vào đúng point cũ** thay vì tạo bản ghi mới — không cần migrate schema khi Pha text hoàn thành.
- Cùng payload metadata (`title`, `brand`, `category`, `price`...) chỉ lưu một lần thay vì trùng lặp ở hai collection.
- Hỗ trợ trực tiếp truy vấn hợp nhất đa phương thức (late fusion / RRF) qua Query API của Qdrant (`prefetch` + `fusion`) mà không cần tự ráp kết quả từ hai chỉ mục FAISS riêng như thiết kế FAISS ban đầu.

### 1.2. Point ID = vị trí dòng trong `items.parquet`

ID điểm dữ liệu là số nguyên không dấu (`0..152085`), **không dùng `item_id` (ASIN 10 ký tự) trực tiếp** vì Qdrant chỉ chấp nhận ID là số nguyên không dấu hoặc UUID.

Thay vì tin tưởng thứ tự dòng có sẵn trong `image_embedding_metadata.parquet` (dù báo cáo đã xác nhận khớp 100%), importer tự xây bảng tra cứu `item_id -> point_id` từ chính `items.parquet` tại thời điểm chạy ([`src/datn/vectordb/importer.py`](../src/datn/vectordb/importer.py)). Điều này đảm bảo:
- Import ảnh và import text (chạy ở hai thời điểm khác nhau, có thể trên máy khác) luôn map về **cùng một point**, bất kể thứ tự nội tại của từng file metadata.
- Item nào có mặt trong file embedding nhưng biến mất khỏi `items.parquet` (dữ liệu catalog thay đổi) sẽ bị bỏ qua có cảnh báo, thay vì ghi nhầm point.

`item_id` gốc vẫn được lưu trong payload (indexed keyword) để tra cứu ngược và join lại với `items.parquet` khi cần các trường dài (`description`, `features`) không lưu trong Qdrant.

### 1.3. Payload: chỉ lưu trường phục vụ filter/hiển thị nhanh

| Field | Kiểu | Index | Ghi chú |
| :--- | :--- | :---: | :--- |
| `item_id` | keyword | ✅ | ASIN, tra cứu ngược về `items.parquet` |
| `title` | text | ❌ | hiển thị nhanh kết quả, không search full-text |
| `brand` | keyword | ✅ | facet/filter thương hiệu; bỏ qua nếu null (0.34% items) |
| `category` | keyword | ✅ | 30 giá trị duy nhất trong catalog |
| `price` | float | ✅ (range) | bỏ qua nếu null (~46.7% items không có giá) thay vì ghi `0` để filter theo khoảng giá không bị lệch |
| `image_url` | keyword | ❌ | phục vụ hiển thị |
| `has_image` / `has_text` | bool | ✅ | lọc nhanh item nào đã có vector nào — quan trọng trong giai đoạn quá độ khi chỉ có ảnh |
| `is_image_fallback` | bool | ✅ | đánh dấu 557 item dùng vector ảnh đen fallback (ảnh gốc lỗi/mất) — cho phép loại trừ hoặc hạ ưu tiên khi retrieval thuần ảnh |

`description` và `features` (văn bản dài) **không** lưu trong Qdrant — giữ payload nhẹ để HNSW build/scan nhanh; khi cần hiển thị đầy đủ, tra cứu lại `items.parquet` bằng `item_id`.

`is_image_fallback` được tính lại từ dữ liệu thực tế (không hard-code số 557): importer nhóm các vector trùng khớp byte-for-byte, cụm trùng lặp lớn nhất (nếu > 1 phần tử) được đánh dấu fallback — cùng phương pháp hash MD5 mô tả tại báo cáo mục 2.3.D, nhưng chạy lại mỗi lần import để không phụ thuộc vào con số cố định.

### 1.4. Distance & HNSW

- **Distance:** Cosine — khớp với vector đã chuẩn hóa $L_2 = 1.0$ (report mục 2.3.A), tương đương dot-product nhưng tránh sai số nếu tương lai có vector chưa chuẩn hóa.
- **HNSW:** `m=32`, `ef_construct=128` — khớp cấu hình `faiss.IndexHNSWFlat(1024, 32)` mà báo cáo đã đề xuất cho môi trường production (mục 4.2), giữ recall > 98% với latency thấp.
- Với quy mô 152k items (~594MB/vector field), toàn bộ giữ **in-memory** (`on_disk_vectors: false`). Có thể bật `on_disk_vectors: true` trong [`configs/qdrant.yaml`](../configs/qdrant.yaml) nếu chạy trên máy RAM hạn chế — đánh đổi latency.

## 2. Kiến trúc thư mục

```
docker-compose.yml              # service Qdrant (image chính thức, volume named, healthcheck)
configs/qdrant.yaml              # kết nối, cấu hình collection, đường dẫn dữ liệu, batch import
src/datn/vectordb/
  schema.py                      # tên collection, named vectors, tên field payload, field cần index
  client.py                      # dựng QdrantClient từ config (+ QDRANT_API_KEY env)
  collection.py                  # ensure_collection(): tạo collection + payload index (idempotent)
  importer.py                    # đọc items.parquet + embeddings .npy/.parquet, upsert theo batch
  cli.py                         # datn-vectordb create-collection | import-image | import-text | info
```

## 3. Cách chạy

```bash
# 1. Cài thêm nhóm dependency vectordb
pip install -e ".[vectordb]"

# 2. Khởi động Qdrant (REST :6333, gRPC :6334, dashboard tại /dashboard)
docker compose up -d

# 3. Tạo collection (idempotent, an toàn chạy lại)
datn-vectordb create-collection

# 4. Nạp embedding ảnh đã có (152,086 điểm, ~594MB, mất khoảng 1 phút)
datn-vectordb import-image

# 5. Khi text_embeddings.npy sẵn sàng (Pha kế tiếp), nạp bổ sung vào đúng các point đã có
datn-vectordb import-text

# Kiểm tra trạng thái
datn-vectordb info
```

`docker-compose.yml` pin phiên bản `qdrant/qdrant:v1.12.4` để tái lập được kết quả; nâng phiên bản qua `docker compose pull qdrant` khi cần, đồng thời cập nhật `qdrant-client` trong `pyproject.toml` (nhóm `vectordb`) khớp minor version với server để tránh cảnh báo tương thích.

## 4. Đã kiểm chứng

- Import `image_embeddings (1).npy` + `image_embedding_metadata (1).parquet` → 152,086/152,086 điểm, khớp `items.parquet`.
- Số lượng item gắn cờ `is_image_fallback=True`: **557**, khớp chính xác số liệu báo cáo tại mục 2.3.D.
- Truy vấn tương đồng ảnh lặp lại thử nghiệm "Easy USA Sneaker" (mục 2.4.1 báo cáo): Top-5 kết quả và điểm cosine (0.9324, 0.9323, 0.9084, 0.9068, 0.9031) khớp tuyệt đối với báo cáo gốc, xác nhận vector nạp vào Qdrant không bị lệch/hỏng so với `image_embeddings.npy` nguồn.

## 5. Việc còn lại khi có text embeddings

1. Chạy `kaggle_text_embeddings.ipynb` → sinh `data/embedding/text_embeddings.npy` + `text_embedding_metadata.parquet` (schema `index`, `item_id`, giống ảnh).
2. Chạy `datn-vectordb import-text` — không cần `--recreate`, các point ảnh hiện có sẽ được bổ sung vector `text`.
3. Cân nhắc dùng Query API `prefetch` (image + text) với `fusion=rrf` để hiện thực công thức late-fusion $S_{final} = \alpha S_{CF} + \beta S_{text} + \gamma S_{image}$ mô tả tại báo cáo mục 4.3, với $S_{CF}$ tính riêng ngoài Qdrant từ dữ liệu tương tác (`candidate_interactions.parquet`).
