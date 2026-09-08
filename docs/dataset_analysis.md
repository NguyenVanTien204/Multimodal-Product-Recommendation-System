# Báo cáo Phân tích So sánh Bộ dữ liệu Cũ & Mới

Báo cáo này so sánh chi tiết giữa **Bộ dữ liệu Cũ** (phiên bản đã commit trên Git HEAD) và **Bộ dữ liệu Mới** (phiên bản vừa được tạo ra sau khi bạn thay đổi các tham số lọc hoặc modulus).

## 1. So sánh Tổng quan (General Comparison)

Dưới đây là so sánh trực quan các thông số tổng quát giữa hai bộ dữ liệu:

| Chỉ số (Metric) | Bộ dữ liệu Cũ (HEAD) | Bộ dữ liệu Mới (Vừa lấy) | Sự thay đổi (Change) |
| --- | --- | --- | --- |
| **Tổng số Users** | 12,134 | 29,096 | +16,962 (+139.79%) |
| **Tương tác Train (Rows)** | 89,387 | 171,788 | +82,401 (+92.18%) |
| **Số lượng Items (Train)** | 30,226 | 122,378 | +92,152 (+304.88%) |
| **Tổng số Items trong Metadata** | 31,368 | 152,086 | +120,718 (+384.84%) |
| **Độ thưa của ma trận (Sparsity)** | 99.9756% | 99.9952% | +0.0195% |
| **Tương tác trung bình/User (Train)** | 7.37 | 5.90 | -1.46 |
| **Tổng kích thước 4 files Parquet** | 13.10 MB | 96.35 MB | +83.25 MB (+635.65%) |

> [!NOTE]
> **Lý do thay đổi:** Ở bộ dữ liệu mới, số lượng user được lấy mẫu đã tăng khoảng **139.8%** (từ 12,134 lên 29,096).
> Điều này trực tiếp kéo theo số lượng tương tác tập Train tăng từ **89,387 lên 171,788** dòng (92.2%), và số lượng sản phẩm liên quan trong Metadata tăng từ **31,368 lên 152,086** sản phẩm.

## 2. Phân tích Chi tiết Bộ dữ liệu tương tác Mới (New Interaction Splits)

Phân tích chi tiết cấu trúc schema của file tương tác (`train.parquet`, `valid.parquet`, `test.parquet` có schema đồng nhất):

| Tên cột (Column) | Kiểu dữ liệu | Số lượng Null | Tỷ lệ Null (%) | Số lượng Unique | Giá trị Min / Độ dài Min | Giá trị Max / Độ dài Max | Giá trị Trung bình / Độ dài TB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `user_id` | `object` | 0 | 0.00% | 29,096 | 28 | 32 | 28.00 |
| `item_id` | `object` | 0 | 0.00% | 122,378 | 10 | 10 | 10.00 |
| `rating` | `float32` | 0 | 0.00% | 5 | 1.00 | 5.00 | 4.18 |
| `timestamp` | `int64` | 0 | 0.00% | 171,757 | 1094124284000.00 | 1690251917520.00 | 1524139172685.62 |
| `verified_purchase` | `bool` | 0 | 0.00% | 1 | 1.00 | 1.00 | 1.00 |
| `is_positive` | `int32` | 0 | 0.00% | 2 | 0.00 | 1.00 | 0.78 |
| `review_title` | `object` | 0 | 0.00% | 137,219 | 1 | 218 | 28.10 |
| `review_text` | `object` | 0 | 0.00% | 171,204 | 0 | 16102 | 554.67 |
| `helpful_vote` | `int32` | 0 | 0.00% | 679 | 3.00 | 13182.00 | 16.32 |

### Phân phối điểm đánh giá (Rating Distribution) - Cũ vs Mới:

| Điểm (Rating) | Số lượng review Cũ | Tỷ lệ Cũ (%) | Số lượng review Mới | Tỷ lệ Mới (%) | Thay đổi tỷ lệ (%) |
| --- | --- | --- | --- | --- | --- |
| 1.0 | 0 | 0.00% | 10,401 | 6.05% | +6.05% |
| 2.0 | 0 | 0.00% | 9,515 | 5.54% | +5.54% |
| 3.0 | 0 | 0.00% | 17,900 | 10.42% | +10.42% |
| 4.0 | 16,597 | 18.57% | 34,165 | 19.89% | +1.32% |
| 5.0 | 72,790 | 81.43% | 99,807 | 58.10% | -23.33% |

### Đặc trưng hoạt động của User - Cũ vs Mới:

| Thống kê độ hoạt động User | Bộ dữ liệu Cũ | Bộ dữ liệu Mới |
| --- | --- | --- |
| **User hoạt động nhiều nhất** | 209 | 285 |
| **Trung bình số tương tác/User** | 7.37 | 5.90 |
| **Phân vị 25% (Q1)** | 3 | 3 |
| **Phân vị 50% (Median)** | 5 | 4 |
| **Phân vị 75% (Q3)** | 9 | 6 |

### Đặc trưng độ phổ biến của Item - Cũ vs Mới:

| Thống kê độ phổ biến Item | Bộ dữ liệu Cũ | Bộ dữ liệu Mới |
| --- | --- | --- |
| **Item được tương tác nhiều nhất** | 595 | 81 |
| **Trung bình số tương tác/Item** | 2.96 | 1.40 |
| **Số lượng Items chỉ có 1 tương tác** | 5,891 (19.49%) | 98,954 (80.86%) |

## 3. Phân tích Siêu dữ liệu sản phẩm Mới (New Item Metadata - `items.parquet`)

Bảng phân tích thuộc tính chi tiết sản phẩm của file `items.parquet` mới:

| Tên cột (Column) | Kiểu dữ liệu | Số lượng Null | Tỷ lệ Null (%) | Số lượng Unique | Giá trị Min / Độ dài Min | Giá trị Max / Độ dài Max | Giá trị Trung bình / Độ dài TB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `item_id` | `object` | 0 | 0.00% | 152,086 | 10 | 10 | 10.00 |
| `title` | `object` | 0 | 0.00% | 145,253 | 1 | 1213 | 78.29 |
| `description` | `object` | 0 | 0.00% | 60,483 | 0 | 23862 | 277.75 |
| `features` | `object` | 0 | 0.00% | 127,367 | 0 | 9578 | 543.97 |
| `category` | `object` | 0 | 0.00% | 30 | 0 | 25 | 13.57 |
| `brand` | `object` | 514 | 0.34% | 29,072 | N/A | N/A | N/A |
| `price` | `float64` | 71,019 | 46.70% | 7,764 | 0.01 | 4599.00 | 40.10 |
| `image_url` | `object` | 46 | 0.03% | 147,905 | N/A | N/A | N/A |

### Phân phối Danh mục sản phẩm (Top 5 Categories) - Cũ vs Mới:

| Danh mục (Category) | Số lượng Cũ | Tỷ lệ Cũ (%) | Số lượng Mới | Tỷ lệ Mới (%) |
| --- | --- | --- | --- | --- |
| AMAZON FASHION | 29,399 | 93.72% | 141,510 | 93.05% |
| Empty/Unknown | 724 | 2.31% | 4,937 | 3.25% |
| Sports & Outdoors | 288 | 0.92% | 1,301 | 0.86% |
| Amazon Home | 271 | 0.86% | 1,043 | 0.69% |
| All Beauty | 88 | 0.28% | 706 | 0.46% |

## 4. Sự nhất quán giữa các Splits và hiện tượng Cold-start - Cũ vs Mới

| Hiện tượng Cold-start Items | Bộ dữ liệu Cũ | Bộ dữ liệu Mới |
| --- | --- | --- |
| **Item mới ở Valid (chưa có ở Train)** | 833 (9.20%) | 15,568 (61.04%) |
| **Item mới ở Test (chưa có ở Train)** | 908 (10.21%) | 15,354 (62.57%) |

> [!TIP]
> Mặc dù số lượng sản phẩm tăng lên rất nhiều ở bộ dữ liệu Mới, tỷ lệ **Cold-start Items ở Valid và Test vẫn ở mức cao (~61% - 62%)**.
> Điều này phản ánh đặc thù rất lớn của tập Amazon Fashions khi catalog sản phẩm rất rộng và lượng tương tác của mỗi sản phẩm thường rất ít (long-tail). Khi huấn luyện mô hình, bạn cần áp dụng các kỹ thuật khuyến nghị dựa trên Content (như văn bản mô tả, title, hoặc hình ảnh) để xử lý lượng cold-start items khổng lồ này.

## 5. Mức độ giao thoa và Nguyên nhân khác biệt hoàn toàn (Overlap & Discrepancy Analysis)

Vì cả hai bộ dữ liệu được lấy mẫu bằng phương pháp **băm ổn định (stable-hash)** dựa trên cùng một SEED (`20260813`):

- **Số lượng User chung:** 573 (chỉ bằng **4.72%** số user của tập cũ).
- **Số lượng Item chung:** 19,135 (bằng **61.00%** số item của tập cũ).

> [!IMPORTANT]
> **Giải thích nguyên nhân tập User giao thoa cực kỳ thấp (chỉ 4.72%):**
> 1. **Sự thay đổi nghiêm ngặt về Bộ lọc reviews:**
>    - **Bộ dữ liệu Cũ:** Chỉ lấy reviews có `rating >= 4.0` (tích cực), không lọc theo lượt bình chọn hữu ích (`helpful_vote >= 0`), và cho phép giao dịch chưa xác thực.
>    - **Bộ dữ liệu Mới:** Chấp nhận toàn bộ dải điểm `rating >= 1.0`, nhưng lại áp bộ lọc rất nghiêm ngặt: **chỉ giữ lại reviews có `helpful_vote >= 3`** và bắt buộc phải là giao dịch đã xác thực (`verified_purchase = True`).
> 2. **Tác động trực tiếp đến điều kiện chọn User (K-core):**
>    - Vì hầu hết các đánh giá trên Amazon không có hoặc có rất ít lượt vote hữu ích, bộ lọc `helpful_vote >= 3` đã loại bỏ phần lớn số reviews của tập gốc.
>    - Do đó, phần lớn các user trong bộ dữ liệu cũ (dù có $\ge 5$ reviews tổng cộng ở tập cũ) bị mất đi các reviews không đủ vote hữu ích. Khi số lượng review hợp lệ của họ giảm xuống dưới 5 (`MIN_USER_DEGREE < 5`), họ lập tức bị loại khỏi danh sách eligible.
>    - Chỉ có **573 user cũ** giữ được đủ số lượng reviews hữu ích để tiếp tục xuất hiện trong bộ dữ liệu mới. Nhóm user còn lại của bộ dữ liệu mới là các user hoàn toàn mới được tuyển chọn thêm để bù đắp chỉ tiêu `TARGET_USERS`.

## 6. Phân tích các file Tạm / Trung gian (Candidate & Intermediate Files)

Trong quá trình xử lý stream và trích xuất dữ liệu, notebook sinh ra hai file trung gian đóng vai trò lưu trữ tạm thời:

1. **`candidate_interactions.parquet`** (Dung lượng: ~60.31 MB):
   - **Số dòng tương tác thô:** `237,588` dòng.
   - **Số lượng Users thô:** `30,000` users (đạt chính xác chỉ tiêu băm ban đầu `TARGET_USERS = 30,000`).
   - **Số lượng Items thô:** `153,431` items.
   - **Vai trò:** Lưu cache toàn bộ tương tác của nhóm user được băm trúng tuyển (thỏa mãn bước lọc review thô ban đầu) nhằm phục vụ các phép tính toán cục bộ trong DuckDB mà không cần tải lại file nén khổng lồ từ UCSD.
   - **Nguyên nhân hao hụt (từ 237,588 dòng xuống còn 229,980 dòng ở tập chia tách cuối cùng):** Có **7,608** dòng tương tác đã bị loại bỏ thông qua các bước xử lý kế tiếp:
     - **Khử trùng lặp:** Giữ lại review mới nhất nếu một user review một sản phẩm nhiều lần.
     - **Khớp nối Metadata:** Loại bỏ tương tác của các sản phẩm không tìm thấy metadata trên máy chủ UCSD.
     - **Thuật toán K-core:** Loại bỏ các user bị giảm số lượng tương tác hợp lệ xuống dưới 5 (làm giảm số user từ 30,000 xuống còn 29,096 user hợp lệ cuối cùng).

2. **`items_candidate.parquet`** (Dung lượng: ~63.08 MB):
   - **Số dòng (sản phẩm):** `152,090` sản phẩm.
   - **Vai trò:** Lưu trữ siêu dữ liệu (Metadata) thô của toàn bộ các sản phẩm liên quan xuất hiện trước bước K-core cuối.
   - **Sự khác biệt so với `items.parquet` cuối cùng (152,086 sản phẩm):** Chỉ có **4 sản phẩm** bị loại bỏ ở bước lọc K-core cuối cùng do không còn bất kỳ tương tác nào từ nhóm 29,096 user hợp lệ.

> [!TIP]
> **Khuyến nghị dọn dẹp bộ nhớ:**
> Hai file `.parquet` có chữ `candidate` này chỉ đóng vai trò là cache trung gian trong quá trình chạy notebook. 
> Sau khi bạn đã thu được 4 file sạch cuối cùng (`train.parquet`, `valid.parquet`, `test.parquet`, và `items.parquet`), bạn hoàn toàn **có thể xóa hai file candidate này** cùng file DuckDB tạm (`processing.duckdb`) để giải phóng hơn **120 MB** dung lượng ổ đĩa.
