# Báo cáo Trích xuất Đặc trưng Đa phương thức (Multimodal Embeddings Report)

Tài liệu này ghi nhận chi tiết kiến trúc, quy trình xử lý kỹ thuật, kết quả kiểm định chất lượng và thực nghiệm truy vấn tương đồng của các vector đặc trưng đa phương thức (**Image Embeddings** & **Text Embeddings**) phục vụ cho Pha 2 (Trích xuất đặc trưng) và Pha 3 (FAISS Indexing & Hybrid Retrieval) của Đồ án Tốt nghiệp: **"Xây dựng hệ thống gợi ý sản phẩm đa phương thức ứng dụng RAG"**.

---

## 1. Tổng quan Kiến trúc Vector Đa phương thức

Hệ thống gợi ý đa phương thức tích hợp thông tin từ nhiều nguồn ngữ nghĩa độc lập nhằm giải quyết triệt để bài toán **Cold-start Items** (sản phẩm mới chưa có lịch sử tương tác, chiếm tới ~61% - 62% ở tập Valid/Test theo [dataset_analysis.md](./dataset_analysis.md)):

```
                             [Catalog: items.parquet (152,086 items)]
                                         /              \
                                        /                \
                     [Image Pipeline]                      [Text Pipeline]
             Vision Encoder: Jina CLIP v2 (EVA-02)    Text Encoder: Jina CLIP v2 / SBERT
                           |                                      |
                           v                                      v
              image_embeddings.npy (1024-d)          text_embeddings.npy (1024-d)
              image_embedding_metadata.parquet       text_embedding_metadata.parquet
                           \                                      /
                            \                                    /
                             v                                  v
                        [FAISS Indexing: products_image.faiss / products_text.faiss]
                                         |
                                         v
                         [Multimodal Hybrid Candidate Retrieval]
```

---

## 2. Trích xuất Đặc trưng Hình ảnh (Image Embeddings)

### 2.1. Cấu hình Mô hình & Môi trường Thực thi

- **Mô hình thị giác (Vision Encoder):** `jinaai/jina-clip-v2`
  - **Kiến trúc cốt lõi:** EVA-02 ViT-L/14 (kích thước ảnh chuẩn 224x224).
  - **Số chiều vector (Embedding Dimension):** $D = 1024$.
  - **Cơ chế chuẩn hóa:** Vector được chuẩn hóa $L_2\text{-norm} = 1.0$ (Cosine Similarity = Dot Product).
- **Môi trường huấn luyện/trích xuất:** Kaggle GPU (Hỗ trợ 2x NVIDIA T4 hoặc 1x NVIDIA P100).
- **Notebook nguồn:** [`notebooks/imageembedding.ipynb`](../notebooks/imageembedding.ipynb).

### 2.2. Các Giải pháp Kỹ thuật & Khắc phục Lỗi Môi trường (Patches)

Trong quá trình chạy trên môi trường Kaggle với thư viện `transformers==5.3.0`, notebook đã áp dụng các giải pháp kỹ thuật tối ưu hóa:

1. **Vá lỗi tương thích Jina CLIP v2 (`transformers` core):**
   - `patch_dot_natural_key`: Sửa hàm sort tham số state_dict, ngăn lỗi `TypeError: '<' not supported between instances of 'str' and 'int'`.
   - `meta_init_safe_load`: Ép khởi tạo tensor về `cpu` thay vì thiết bị ảo `meta`, đảm bảo các buffer được tính giá trị thực ngay từ đầu.
   - `restore_non_persistent_buffers`: Khắc phục lỗi nghiêm trọng nhất khi `transformers` ghi đè các buffer non-persistent (RoPE `freqs_cos/sin`, `inv_freq`...) bằng vùng nhớ rác (`torch.empty_like`), ngăn hiện tượng model sinh vector `NaN` hoặc suy hao âm thầm.
2. **Vá môi trường Kaggle Runtime:**
   - Bổ sung định nghĩa `PIL._typing._Ink` trong RAM và file vật lý tránh lỗi import của Pillow mới.
   - Vá decorator `torch.library.register_fake` và `Library.impl` tránh lỗi xung đột đăng ký kernel của `torchvision::nms` và `roi_align`.
3. **Cơ chế song song hóa & Chống gián đoạn (Checkpointing):**
   - **Tự động cân bằng GPU:** Tự phát hiện số GPU khả dụng (`cuda:0`, `cuda:1`...), chia đều dữ liệu qua `split_df`, nạp model sang từng thiết bị và chạy trên luồng song song qua `ThreadPoolExecutor`.
   - **DataLoader tối ưu:** Thiết lập `persistent_workers=True`, `pin_memory=True`, phân bổ số worker giải mã ảnh theo số CPU thực tế (`os.cpu_count() // n_devices`).
   - **Checkpoint nguyên tử (Atomic Checkpoint):** Lưu định kỳ 20 batches ra file `.tmp` rồi đổi tên bằng `os.replace` thành `chunk_gpu{id}_{batch:04d}.npz`. Nếu phiên làm việc Kaggle bị ngắt (hết giờ 9h/12h hoặc rớt mạng), notebook tự phát hiện số ảnh đã xử lý và tiếp tục chạy mà không bị mất tiến trình.
   - **Cơ chế Fallback ảnh lỗi:** Sản phẩm thiếu ảnh hoặc URL hỏng được thay bằng ảnh đen chuẩn `(224, 224, color=0)` để tránh làm crash cả mini-batch.

---

### 2.3. Thẩm định Chi tiết Dữ liệu Thực tế (Validation Results)

Hai file dữ liệu được sinh ra và lưu trữ tại thư mục `data/embedding/`:

#### A. File Vector: `image_embeddings.npy`
- **Kích thước file:** **622,944,384 bytes** (~594.09 MB).
- **Kích thước ma trận (Shape):** `(152086, 1024)`.
- **Kiểu dữ liệu (Dtype):** `float32`.
- **Kiểm tra tính toàn vẹn (Integrity Checks):**
  - Số lượng giá trị `NaN`: **0** (Hoàn toàn không có).
  - Số lượng giá trị `Inf`: **0** (Hoàn toàn không có).
  - Số lượng vector toàn 0 (Zero vectors): **0**.
  - Phân phối độ dài $L_2\text{-norm}$: $\text{Min} = 1.000000, \text{Max} = 1.000000, \text{Mean} = 1.000000$ (100% vector đã được chuẩn hóa đơn vị tuyệt đối).

#### B. File Siêu dữ liệu: `image_embedding_metadata.parquet`
- **Kích thước file:** **1,100,042 bytes** (~1.05 MB).
- **Kích thước bảng:** **152,086 dòng, 2 cột**.
- **Cấu trúc Schema:**
  - `index`: `Int64` (Đánh số liên tục từ `0` đến `152085`).
  - `item_id`: `String` (Mã định danh sản phẩm Amazon, 10 ký tự).
- **Kiểm tra tính toàn vẹn:**
  - Số lượng giá trị `Null`: **0 null** trên cả hai cột.
  - Số lượng `item_id` duy nhất: **152,086 / 152,086** (100% unique, không trùng lặp).

#### C. Mức độ đồng bộ với Siêu dữ liệu gốc (`items.parquet`)
- **Tổng số sản phẩm trong `items.parquet`:** 152,086 sản phẩm.
- **Tính đồng bộ thứ tự:** **Khớp chính xác 100% từng dòng theo thứ tự** (`(items_df['item_id'] == meta_df['item_id']).all() == True`).
- **Ý nghĩa:** Vector tại hàng thứ $i$ trong ma trận `image_embeddings.npy` đại diện chính xác cho sản phẩm tại dòng thứ $i$ của `items.parquet`.

#### D. Thống kê Ảnh Fallback (Ảnh đen thay thế)
- Phân tích băm MD5 toàn bộ 152,086 vector phát hiện:
  - Chỉ có **557 items** (~0.37% tổng dataset) chia sẻ cùng một vector ảnh đen fallback (do ảnh không tồn tại trên CDN hoặc lỗi mạng lúc tải).
  - **151,529 items** (> 99.63%) còn lại đều sở hữu vector đặc trưng ảnh thực tế phân biệt.

---

### 2.4. Thực nghiệm Kiểm thử: Truy vấn Tương đồng Hình ảnh (Visual Similarity)

Do toàn bộ vector đã có $L_2\text{-norm} = 1.0$, độ tương đồng Cosine giữa vector truy vấn $\mathbf{q}$ và toàn bộ ma trận $\mathbf{E}$ được tính nhanh bằng phép nhân ma trận:

$$\text{Cosine\_Similarity}(\mathbf{q}, \mathbf{e}_i) = \mathbf{q} \cdot \mathbf{e}_i$$

Dưới đây là kết quả kiểm thử thực nghiệm trên 4 nhóm ngành hàng tiêu biểu của catalog Amazon Fashion:

#### 1. Nhóm Giày Sneaker cổ thấp buộc dây
- **Sản phẩm truy vấn:** `Easy USA Womens Lace Up Canvas Plimsol Sneakers Shoes` (Item ID: `B07C1Z3TKP`, Brand: `Easy USA`)

| Thứ hạng | Cosine Sim | Item ID | Thương hiệu | Tên sản phẩm tương đồng |
| :---: | :---: | :---: | :--- | :--- |
| **Top 1** | **0.9324** | `B0816RW9DY` | FRACORA | Womens Canvas Sneakers Low Top Lace Up Canvas Shoes Fashion... |
| **Top 2** | **0.9323** | `B08F72KXF4` | Epic Step | Sneakers for Women Fashion Sneakers Tennis Shoes Women Sneak... |
| **Top 3** | **0.9084** | `B0B2TD9LDV` | Keds | Keds Women's Champion CVO Seasonals |
| **Top 4** | **0.9068** | `B07NK96N1D` | Soda | Soda Flat Women Shoes Linen Canvas Slip On Sneakers Lace Up... |
| **Top 5** | **0.9031** | `B006V99KQE` | Keds | Keds Women's Champion Seasonal Spring 2015 Sneaker |

> **Nhận xét:** Model học được chính xác phom dáng giày plimsol/canvas cổ thấp, dây buộc trắng và đế cao su mỏng đặc trưng.

---

#### 2. Nhóm Đồng hồ kim loại nam NIXON
- **Sản phẩm truy vấn:** `Nixon Sentry SS Stainless Steel Day/Date 42mm WR 100M Mens Watch` (Item ID: `B00S33DIGY`, Brand: `NIXON`)

| Thứ hạng | Cosine Sim | Item ID | Thương hiệu | Tên sản phẩm tương đồng |
| :---: | :---: | :---: | :--- | :--- |
| **Top 1** | **0.9144** | `B001L8N4GW` | NIXON | NIXON Cannon Watch |
| **Top 2** | **0.9139** | `B079Z62HZN` | NIXON | NIXON Time Teller A045. 100m Water Resistant Watch (37mm Sta... |
| **Top 3** | **0.9105** | `B00EO5WHKI` | NIXON | Nixon Women's Mod Stainless Steel Watch with Fabric Band |
| **Top 4** | **0.9016** | `B013VNW9VU` | NIXON | Nixon Men's '51-30 SW, Vader' Quartz Stainless Steel Casual... |
| **Top 5** | **0.8953** | `B00S33C0HC` | NIXON | Nixon Men's A105 Sentry 42mm Stainless Steel Leather Quartz... |

> **Nhận xét:** Toàn bộ Top 5 đều bắt trúng các dòng đồng hồ kim loại cùng hãng NIXON với cấu trúc mặt số tròn và dây đeo kim loại cùng ngôn ngữ thiết kế.

---

#### 3. Nhóm Váy hoa bé gái (Floral Dress)
- **Sản phẩm truy vấn:** `2-9T Flower Girls Floral Dresses Toddler Pageant Striped Dress` (Item ID: `B07CTDY9NV`)

| Thứ hạng | Cosine Sim | Item ID | Thương hiệu | Tên sản phẩm tương đồng |
| :---: | :---: | :---: | :--- | :--- |
| **Top 1** | **0.9959** | `B07CT9K3QY` | Cichic | Cichic Elegant Girls' Special Occasion Dress Princess Dress... |
| **Top 2** | **0.9332** | `B07D7D9W2K` | Flofallzique | Flofallzique Summer Little Girls Dress Vintage Floral Toddler... |
| **Top 3** | **0.9234** | `B0836G6R28` | ASTRILL | Toddler Baby Girls Summer Dress Ruffled Sleeve Top and Floral... |
| **Top 4** | **0.9216** | `B07T43GB52` | aibeiboutique | Toddler Girls Sleeveless Dresses Floral Summer Cute Tutu... |
| **Top 5** | **0.9168** | `B07PQM79S8` | HILEELANG | HILEELANG Girl Summer Beach Dress Halter Neck Sleeveless... |

> **Nhận xét:** Độ tương đồng cực kỳ cao (lên đến 0.9959 ở Top 1), nhận diện chính xác hoạ tiết hoa mùa hè trên thân váy xoè trẻ em.

---

#### 4. Nhóm Kính mát thời trang (Sunglasses)
- **Sản phẩm truy vấn:** `O'Neill Offshore Polarized Sunglasses` (Item ID: `B07Q3SHKCD`, Brand: `O'Neill`)

| Thứ hạng | Cosine Sim | Item ID | Thương hiệu | Tên sản phẩm tương đồng |
| :---: | :---: | :---: | :--- | :--- |
| **Top 1** | **0.9141** | `B01LZ03L49` | Knockaround | Knockaround Classics Non-Polarized Sunglasses, Frosted... |
| **Top 2** | **0.9134** | `B083Q64Y3Q` | MEETSUN | MEETSUN Polarized Sunglasses for Women Men Classic Retro... |
| **Top 3** | **0.9036** | `B00B4U0000` | Fitovers | Fitovers Eyewear Aurora Sunglasses |
| **Top 4** | **0.9024** | `B001L1P2C0` | Flying Fisherman | Flying Fisherman Viper Polarized Sunglasses |
| **Top 5** | **0.9023** | `B07V2QJ4RN` | Kursan | Kids Polarized Sunglasses for Boys Girls TPEE Rubber... |

---

## 3. Trích xuất Đặc trưng Văn bản (Text Embeddings)

> [!NOTE]
> **Khu vực chờ cập nhật (Placeholder):** Phần này sẽ được ghi nhận và đối soát chi tiết ngay sau khi hoàn thành chạy notebook trích xuất đặc trưng văn bản sản phẩm (`textembedding.ipynb` / `kaggle_text_embeddings.ipynb`).

### 3.1. Cấu hình Mô hình & Chiến lược Ghép Văn bản (Dự kiến)

- **Mô hình văn bản (Text Encoder):** `jinaai/jina-clip-v2` (Text branch) hoặc `BAAI/bge-m3` / `sentence-transformers/all-MiniLM-L6-v2`.
  - *Gợi ý ưu tiên:* Dùng chung `jinaai/jina-clip-v2` (text branch) sẽ cho phép vector text và vector image nằm chung một không gian biểu diễn đa phương thức (Shared Latent Space, $D=1024$), hỗ trợ trực tiếp tính toán Cross-modal (truy vấn văn bản ra hình ảnh và ngược lại).
- **Template ghép chuỗi văn bản sản phẩm:**
  ```python
  text_input = (
      f"Title: {title} | "
      f"Category: {category} | "
      f"Brand: {brand} | "
      f"Features: {features} | "
      f"Description: {description}"
  )
  ```
- **Xử lý giá trị rỗng/dài:** Cắt ngắn theo số token tối đa của model (ví dụ 512 hoặc 8192 tokens đối với Jina), điền placeholder cho các trường null (brand: 0.34% null, description: 39.77% null theo bảng phân tích tại `dataset_analysis.md`).

---

### 3.2. Bảng Thẩm định Kỹ thuật Text Embeddings *(Sẽ cập nhật số liệu)*

| Chỉ số kiểm tra | Kết quả thực tế | Kỳ vọng thiết kế | Trạng thái |
| :--- | :--- | :--- | :---: |
| **Đường dẫn tệp** | `data/embedding/text_embeddings.npy` | `text_embeddings.npy` | ⏳ Pending |
| **Dung lượng tệp** | *TBD (Dự kiến ~594 MB nếu 1024-d)* | Phù hợp số chiều $D$ | ⏳ Pending |
| **Kích thước ma trận (Shape)** | `(152086, D)` *(D = 1024)* | Đúng 152,086 items | ⏳ Pending |
| **Kiểu dữ liệu (dtype)** | `float32` | `np.float32` | ⏳ Pending |
| **Kiểm tra NaN / Inf** | *TBD (bắt buộc = 0)* | 0 NaN, 0 Inf | ⏳ Pending |
| **Độ dài L2 Norm** | *TBD (bắt buộc = 1.0)* | Normalized unit vector | ⏳ Pending |
| **File Metadata** | `data/embedding/text_embedding_metadata.parquet` | Schema: `index`, `item_id` | ⏳ Pending |
| **Khớp thứ tự với `items.parquet`** | *TBD* | Khớp 100% theo thứ tự | ⏳ Pending |

---

### 3.3. Thực nghiệm Kiểm thử Truy vấn Tương đồng Văn bản *(Sẽ cập nhật số liệu)*

- **Thử nghiệm 1: Text-to-Text Similarity (Văn bản - Văn bản)**
  - *[Sẽ cập nhật câu truy vấn, Top 5 kết quả tương đồng ngữ nghĩa nhất, và điểm Cosine Similarity]*
- **Thử nghiệm 2: Text-to-Image Cross-modal Retrieval (Nếu dùng chung không gian Jina CLIP v2)**
  - *[Sẽ cập nhật truy vấn bằng câu miêu tả tự nhiên, ví dụ "vintage floral summer dress for little girls", và kiểm tra ảnh tương ứng trả về trong Top-K]*

---

## 4. Kế hoạch Kết nối Pha 3 (FAISS Indexing & Hybrid Retrieval)

Sau khi hoàn tất cả 2 tệp embedding (`image_embeddings.npy` và `text_embeddings.npy`), bước triển khai tiếp theo bao gồm:

1. **Chuẩn hóa quy ước đặt tên tệp:**
   - Đổi tên tệp hiện tại `image_embeddings (1).npy` thành `image_embeddings.npy`.
   - Đổi tên `image_embedding_metadata (1).parquet` thành `image_embedding_metadata.parquet`.
2. **Xây dựng chỉ mục FAISS Vector:**
   - **Lựa chọn Index:**
     - Môi trường thử nghiệm / Đánh giá độ chính xác tuyệt đối: `faiss.IndexFlatIP(1024)` (Tìm kiếm vét cạn tích vô hướng chính xác 100%).
     - Môi trường phục vụ thực tế (Inference Latency < 10ms): `faiss.IndexHNSWFlat(1024, 32)` (Đồ thị xấp xỉ phân cấp, tốc độ cao, độ thu hồi > 98%).
3. **Chiến lược Late Fusion đa phương thức:**
   - Tính toán điểm kết hợp giữa đặc trưng thị giác ($S_{\text{image}}$), đặc trưng ngữ nghĩa văn bản ($S_{\text{text}}$), và điểm hành vi người dùng ($S_{\text{CF}}$):
   
   $$S_{\text{final}}(u, i) = \alpha \cdot S_{\text{CF}}(u, i) + \beta \cdot S_{\text{text}}(u, i) + \gamma \cdot S_{\text{image}}(u, i)$$
   
   - Đối với **Cold-start Items** (chưa có $S_{\text{CF}}$), hệ thống tự động gán $\alpha = 0$ và tái phân bổ trọng số cho $\beta$ và $\gamma$, kích hoạt cơ chế gợi ý thuần nội dung đa phương thức.
