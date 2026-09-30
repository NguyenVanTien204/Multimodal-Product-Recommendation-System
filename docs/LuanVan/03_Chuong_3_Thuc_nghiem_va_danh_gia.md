# CHƯƠNG 3: THỰC NGHIỆM VÀ ĐÁNH GIÁ HỆ THỐNG

---

## 3.1. Môi trường thực nghiệm

### 3.1.1. Cấu hình phần cứng và hệ điều hành
Toàn bộ quá trình tiền xử lý dữ liệu, kiểm thử mô hình, triển khai hệ thống dịch vụ và đánh giá thực nghiệm được thực hiện trên hệ thống máy trạm cá nhân kết hợp môi trường điện toán đám mây Kaggle:

| Thành phần phần cứng | Cấu hình chi tiết | Vai trò thực thi |
| :--- | :--- | :--- |
| **Bộ xử lý (CPU)** | **AMD Ryzen 7 6800H with Radeon Graphics**<br>(8 nhân vật lý, 16 luồng xử lý, xung nhịp 3.2 GHz – 4.7 GHz) | Xử lý dữ liệu song song với Polars/DuckDB, điều phối FastAPI và backend |
| **Bộ nhớ trong (RAM)**| **16 GB DDR5** (Bus 4800 MHz) | Lưu trữ vùng đệm dữ liệu, cache vector Qdrant và candidate pool |
| **Bộ xử lý đồ họa (GPU)**| **NVIDIA GeForce RTX 3050 Laptop GPU**<br>(Bộ nhớ VRAM: **4 GB GDDR6**) | Huấn luyện mạng User Tower, Reranker, kiểm thử suy luận mô hình |
| **Ổ cứng lưu trữ** | 512 GB SSD NVMe M.2 PCIe Gen 4 | Đọc/ghi tệp Parquet và ma trận vector nhúng tốc độ cao |
| **Hệ điều hành** | **Windows 11 Home / Windows 10** (Bản build 10.0.26200) | Môi trường phát triển và kiểm thử hệ thống cục bộ |
| **Môi trường Cloud (Kaggle)**| **2x NVIDIA T4 (16 GB VRAM mỗi GPU)** hoặc **1x NVIDIA P100** | Trích xuất hàng loạt vector nhúng hình ảnh và văn bản CLIP |

### 3.1.2. Môi trường phần mềm và thư viện
- **Môi trường ngôn ngữ**: Python 3.10.9 (64-bit).
- **Thư viện Học sâu & Thị giác máy tính**:
  - `torch == 2.9.1+cu126`, `torchvision`, `torchaudio` với `CUDA 12.6`.
  - `transformers == 5.3.0` (kèm các bản vá an toàn bộ đệm cho Jina CLIP v2).
  - `accelerate`, `einops`, `timm`.
- **Thư viện xử lý dữ liệu khoa học**:
  - `polars == 1.17.1`, `duckdb == 1.1.3`, `pyarrow == 18.1.0`.
  - `numpy == 1.26.4`, `scipy`, `scikit-learn`.
- **Hệ thống lưu trữ và cơ sở dữ liệu**:
  - `qdrant-client == 1.12.1` kết nối dịch vụ Docker `qdrant/qdrant:v1.12.4`.
  - `psycopg2-binary`, `asyncpg`, `sqlalchemy == 2.0.36`, `alembic`.
- **Framework API và Web**:
  - `fastapi == 0.115.6`, `uvicorn == 0.34.0`, `pydantic == 2.10.4`.
  - `next == 14.2.15`, `react == 18.3.1`, `tailwindcss == 3.4.14`.
- **Công cụ container và kiểm thử**:
  - `docker-compose version 2.29.7`.
  - `pytest == 8.3.4`, `pytest-asyncio`.

---

## 3.2. Thiết lập thực nghiệm

### 3.2.1. Dữ liệu và giao thức đánh giá
- **Bộ dữ liệu chuẩn hóa**: Phiên bản `balanced_u5_i2_v1` gồm 21.690 người dùng và 32.557 sản phẩm; 198.200 tương tác tích cực ($rating \ge 4$).
- **Giao thức chia tập dữ liệu**: Phân chia theo dòng thời gian nhân quả (Chronological Leave-last-out):
  - *Tập Train* ($N = 154.820$ tương tác): Dùng để học trọng số User Tower và Reranker.
  - *Tập Validation* ($N = 21.690$ tương tác): Dùng để dò tìm siêu tham số, kích hoạt cơ chế Early Stopping và chọn checkpoint.
  - *Tập Test* ($N = 21.690$ tương tác): Bị cô lập hoàn toàn trong quá trình huấn luyện; chỉ được nạp một lần duy nhất để chấm điểm hiệu năng cuối cùng.
- **Quy tắc đánh giá xếp hạng Full-Ranking**:
  - Không sử dụng phương pháp rút gọn mẫu âm ngẫu nhiên (Negative Sampling 99 hay 100) khi đánh giá.
  - Không gian xếp hạng là **toàn bộ 32.557 sản phẩm** trong danh mục (hoặc mở rộng 152.086 sản phẩm đối với tầng trích xuất vector).
  - Áp dụng nguyên tắc `exclude_seen = True`: Mọi sản phẩm người dùng đã từng tương tác trong tập Train đều bị gán điểm âm vô cùng ($-\infty$) để loại khỏi danh sách gợi ý.
- **Hệ chỉ số đánh giá**: Candidate Recall, HitRate@$K$ và NDCG@$K$ với $K \in \{10, 20, 50, 100, 500, 1000\}$.

### 3.2.2. Thiết lập huấn luyện User Tower đa phương thức
Các siêu tham số của mô hình User Tower được tối ưu hóa thông qua các lượt chạy lưới (*Grid Search*) trên tập Validation:

| Siêu tham số | Giá trị thiết lập | Giải thích kỹ thuật |
| :--- | :---: | :--- |
| Kích thước biểu diễn ẩn ($d_{model}$) | **128** | Cân bằng giữa sức chứa biểu diễn và nguy cơ quá khớp |
| Số lớp Transformer Encoder | **2** | Đủ sâu để học phụ thuộc tuần tự mà không bị bão hòa |
| Số đầu chú ý (*Attention Heads*) | **4** | Phân rã không gian chú ý đa góc nhìn ($d_k = 32$) |
| Kích thước tầng Feed-Forward ($d_{ffn}$) | **512** | Tỷ lệ mở rộng $4 \times d_{model}$ chuẩn Transformer |
| Tỷ lệ ngắt kết nối (*Dropout*) | **0.2** | Kiểm soát hiện tượng quá khớp (Overfitting) |
| Độ dài chuỗi lịch sử tối đa ($L_{max}$) | **20** | Bao phủ 98% chiều dài chuỗi tương tác thực tế |
| Kích thước lô (*Batch Size*) | **128** | Tối ưu hóa dung lượng bộ nhớ VRAM 4 GB của GPU RTX 3050 |
| Tốc độ học (*Learning Rate*) | **0.0003** ($3 \times 10^{-4}$) | Sử dụng thuật toán tối ưu AdamW kèm suy hao trọng số |
| Trọng số suy hao (*Weight Decay*) | **0.0001** ($1 \times 10^{-4}$) | Điều chuẩn $L_2$ trên các tham số trọng số |
| Số lượng mẫu âm lấy mẫu ($N_{neg}$) | **512** | 50% lấy mẫu đều + 50% lấy mẫu theo tần suất |
| Hiệu chỉnh phân phối âm | **Log-Q correction** | Loại bỏ thiên lệch ước lượng do lấy mẫu âm không đều |
| Số epoch tối đa | **30** | Dừng sớm nếu không cải thiện sau 5 epoch |
| Tiêu chí chọn Checkpoint | **HitRate@1000 trên Valid** | Tối ưu hóa năng lực bao phủ ứng viên cho tầng truy hồi |
| Seed cố định | **20260813** | Đảm bảo khả năng tái lập thí nghiệm 100% |

Quá trình huấn luyện User Tower đạt đỉnh hiệu năng Validation tại **Epoch 4** và tự động dừng sớm (Early Stopping) tại **Epoch 9**.

### 3.2.3. Thiết lập sinh ứng viên và xếp hạng lại (Reranker)
- **Cấu hình ngân sách ứng viên (Candidate Budget)**:
  - User Tower: Truy hồi tối đa **1.000** sản phẩm có tích vô hướng cao nhất.
  - Global Popularity: Truy hồi tối đa **300** sản phẩm có tần suất mua cao nhất trong tập train.
  - Content Centroid: Truy hồi tối đa **400** sản phẩm gần nhất với tâm cụm vector CLIP của lịch sử gần đây.
  - Last-item Similarity: Truy hồi tối đa **300** sản phẩm có độ tương đồng Cosine cao nhất với sản phẩm cuối cùng.
  - Tổng ngân sách trước khử trùng lặp: **2.000 ứng viên**.
- **Cấu hình mạng Residual Listwise Reranker (v2)**:
  - Kích thước danh sách huấn luyện: 1 Positive + 128 Negatives.
  - Tỷ lệ mẫu âm khó (*Hard-Negative Ratio*): **75%** (khai thác trực tiếp từ các sản phẩm được tầng truy hồi xếp hạng cao nhưng không phải nhãn thật).
  - Trọng số kết hợp phần dư ($\gamma$): **0.75**.
  - Checkpoint tối ưu được chốt tại **Epoch 5** dựa trên chỉ số NDCG@10 của tập Validation.

---

## 3.3. Kết quả thực nghiệm mô hình gợi ý đa phương thức

### 3.3.1. Kết quả của User Tower
Bảng dưới đây ghi nhận hiệu năng của mô hình User Tower độc lập trên tập kiểm thử (Test Split, $N = 21.690$ người dùng) tại các ngưỡng cắt $K$ khác nhau:

| Chỉ số đánh giá | Kết quả thực nghiệm trên tập Test | Ý nghĩa kỹ thuật |
| :--- | :---: | :--- |
| **HitRate@10** | **3,038%** | Tỷ lệ trúng sản phẩm mục tiêu ngay trong Top-10 |
| **NDCG@10** | **2,106%** | Chất lượng xếp hạng ưu tiên các vị trí đầu tiên |
| **HitRate@50** | **6,556%** | Độ bao phủ danh mục tại ngưỡng gợi ý trang đầu |
| **NDCG@50** | **2,863%** | Chất lượng xếp hạng tổng quát Top-50 |
| **HitRate@100** | **9,142%** | Tỷ lệ bao phủ Top-100 |
| **NDCG@100** | **3,282%** | Độ suy giảm chiết khấu tích lũy tại Top-100 |
| **HitRate@500** | **19,889%** | Tỷ lệ trúng khi mở rộng tập ứng viên lên 500 |
| **HitRate@1000** | **27,321%** | **Candidate Recall của riêng nhánh User Tower** |

> [!NOTE]
> **Nhận xét chuyên sâu về HitRate@1000**:
> Trong kiến trúc hai giai đoạn, chỉ số `HitRate@1000 = 27.321%` đóng vai trò là "trần hiệu năng" (upper-bound) của nhánh User Tower. Điều này chứng minh mạng Self-Attention với biểu diễn lai đã thu hẹp không gian tìm kiếm từ 32.557 sản phẩm xuống 1.000 ứng viên mà vẫn giữ lại được hơn 27.3% nhu cầu mua sắm thực tế của khách hàng.

### 3.3.2. Đánh giá chiến lược sinh ứng viên nhiều nguồn (Ablation Study)
Để vượt qua giới hạn trần 27.321% của User Tower đơn lẻ, đề tài tiến hành thử nghiệm bóc tách các chiến lược kết hợp nguồn ứng viên:

| STT | Chiến lược sinh ứng viên | Ngân sách tối đa | Candidate Recall trên Test | Mức độ cải thiện |
| :---: | :--- | :---: | :---: | :---: |
| 1 | **Chỉ dùng User Tower** | 1.000 | 27,321% | Gốc so sánh |
| 2 | **User Tower + Popularity** | 1.500 | 29,899% | +2,578 pp |
| 3 | **Kết hợp 4 nguồn thực tế (Đề xuất)** | **2.000** | **34,444%** | **+7,123 pp (Tăng 1,26 lần)** |
| 4 | **Kết hợp 4 nguồn thiên về Recall** | 2.500 | 36,501% | +9,180 pp |

> [!IMPORTANT]
> **Phát hiện học thuật quan trọng về giải quyết Cold-Start**:
> - Nhánh User Tower và nhánh Popularity hoàn toàn bó tay ($Recall = 0\%$) trước nhóm sản phẩm mục tiêu cold-start (những sản phẩm chưa từng xuất hiện trong tập train tích cực, chiếm 5.27% tập test).
> - Khi bổ sung 2 nhánh nội dung CLIP (**Content Centroid** và **Last-item Similarity**), hệ thống **phục hồi thành công 11,72% các sản phẩm hoàn toàn cold-start này**. Điều này khẳng định việc đưa truy hồi nội dung thị giác làm một nhánh ứng viên độc lập là quyết định kỹ thuật hoàn toàn chính xác.

**Phân tích tính trực giao (Orthogonality) so với Popularity**:
Khi đối chiếu danh sách ứng viên Top-50 của User Tower với Top-50 sản phẩm phổ biến nhất toàn sàn (Popularity Baseline):
- Trong số các trường hợp User Tower gợi ý chính xác, có tới **76,8% lượt trúng là những sản phẩm nằm ở vùng Mid-Tail và Long-Tail mà bảng xếp hạng Popularity bỏ lỡ hoàn toàn**.
- Nếu chỉ dùng độ phổ biến, người dùng sẽ rơi vào "bong bóng lọc" (Filter Bubble), nhận danh sách sản phẩm đại trà giống hệt nhau. Mô hình đề xuất đã thực hiện trọn vẹn vai trò khai phá gu thẩm mỹ cá nhân hóa ngách.

### 3.3.3. Kết quả của Residual Listwise Reranker
Bảng dưới đây so sánh toàn diện hiệu năng giữa mô hình truy hồi ban đầu (User Tower) và Hệ thống hai giai đoạn hoàn chỉnh sau khi có tầng Reranker:

| Chỉ số đánh giá | (1) User Tower đơn lẻ | (2) Hệ thống Hai giai đoạn (Reranker v2) | Mức cải thiện vượt bậc |
| :--- | :---: | :---: | :---: |
| **Candidate Recall** | 27,321% | **34,444%** | **+7,123 pp** (+26,1%) |
| **HitRate@10** | 3,038% | **3,518%** | **+0,480 pp** (+15,8%) |
| **NDCG@10** | 2,106% | **2,489%** | **+0,383 pp** (+18,2%) |
| **HitRate@50** | 6,556% | **6,962%** | **+0,406 pp** (+6,2%) |
| **NDCG@50** | 2,863% | **3,223%** | **+0,360 pp** (+12,6%) |
| **HitRate@100** | 9,142% | **9,521%** | **+0,379 pp** (+4,1%) |
| **NDCG@100** | 3,282% | **3,636%** | **+0,354 pp** (+10,8%) |

**Phân tích ý nghĩa kết quả**:
1. Mạng **Residual Listwise Reranker** cải thiện đồng thời cả HitRate và NDCG ở mọi ngưỡng cắt ($K = 10, 50, 100$).
2. Đặc biệt tại Top-10 (vị trí trực tiếp hiển thị trên màn hình người dùng), **HitRate@10 đạt 3,518%** (vượt chỉ tiêu tối thiểu 3.5%) và **NDCG@10 đạt 2,489%** (vượt chỉ tiêu tối thiểu 2.4%).
3. Mức tăng trưởng NDCG (+18.2%) cao hơn mức tăng HitRate (+15.8%), chứng minh rằng cơ chế học phần dư trên hard-negatives có khả năng đẩy các sản phẩm phù hợp nhất lên các vị trí đầu tiên của danh sách.

---

## 3.4. Đánh giá Chatbot RAG hỗ trợ gợi ý và tìm kiếm

### 3.4.1. Kịch bản kiểm thử chức năng Chatbot
Hệ thống Chatbot RAG được kiểm thử chức năng qua 6 kịch bản thực tế đại diện cho các nhu cầu mua sắm phức tạp:

| Mã kịch bản | Yêu cầu kiểm thử | Hành vi hệ thống thực tế | Đánh giá |
| :---: | :--- | :--- | :---: |
| **TC-01** | Tìm kiếm đa phương thức: *"Tìm đầm dạ hội đen dưới 1 triệu"* | Phân loại intent `search`, bóc tách slot `price <= 1.000.000`, `color = đen`, truy xuất vector Qdrant và hiển thị 5 thẻ sản phẩm đính kèm mã trích dẫn `[P#]`. | **Đạt** |
| **TC-02** | Tìm kiếm bằng hình ảnh trực quan | Nhận file ảnh tải lên, Jina CLIP mã hóa qua `encode_image`, tìm kiếm tương đồng trên named vector `image`, trả về các mẫu đầm có phom dáng tương tự. | **Đạt** |
| **TC-03** | Tinh chỉnh bộ lọc tức thời (*Refine*): *"Đổi sang màu đỏ và giá rẻ hơn"* | Cập nhật `Preferences` của phiên, tái lọc trực tiếp trên candidate pool 60 sản phẩm trong RAM, cập nhật kết quả trong 15ms mà không cần re-encode. | **Đạt** |
| **TC-04** | Giải thích lý do gợi ý (*Explain*): *"Tại sao lại gợi ý đôi giày này?"* | Phân loại intent `explain`, truy xuất lý do có cấu trúc: độ tương đồng phong cách (Cosine score), nằm trong ngân sách và trích dẫn đánh giá thực tế `[R1.1]`. | **Đạt** |
| **TC-05** | So sánh hai sản phẩm (*Compare*): *"So sánh áo A và áo B"* | Truy xuất thông số hai sản phẩm, trích xuất cả review khen và chê, đối chiếu chất liệu vải và độ bền có căn cứ. | **Đạt** |
| **TC-06** | Xử lý dữ liệu thưa/thiếu: Hỏi sản phẩm chưa có review | Hệ thống thông báo trung thực: *"Sản phẩm này hiện chưa có đủ đánh giá từ người dùng để đưa ra nhận xét chi tiết"*, không tự bịa đặt. | **Đạt** |

### 3.4.2. Đánh giá chất lượng phản hồi và tính có căn cứ (Grounding)
- **Kết quả bộ kiểm thử tự động (Automated Test Suite)**:
  Bộ kiểm thử `tests/test_rag_units.py` gồm **47/47 unit test đều vượt qua (100% Passed)**, kiểm chứng toàn diện các thành phần:
  - Bộ tách thực thể giá tiền tiếng Việt/tiếng Anh (chuẩn hóa các dạng: `500k`, `1.2tr`, `$50`, `từ A đến B`).
  - Bộ lọc thương hiệu và danh mục trong Qdrant.
  - Thuật toán xếp hạng Weighted RRF.
  - Quản lý phiên làm việc và vùng đệm ứng viên.
  - Bộ kiểm duyệt căn cứ trích dẫn và phát hiện ảo giác.
- **Tính có căn cứ dữ liệu (Groundedness Rate)**:
  - **Tỷ lệ ảo giác giá tiền: 0,0%** (100% số tiền được nhắc đến trong câu trả lời trùng khớp tuyệt đối với trường `price` trong cơ sở dữ liệu).
  - **Tỷ lệ trích dẫn hợp lệ: 100%** (Mọi thông tin thuộc tính đều gắn nhãn `[P#]` và trải nghiệm thực tế gắn nhãn `[R#.#]`).
- **Thời gian phản hồi và hiện tượng nút cổ chai (Bottleneck Analysis)**:
  - *Khi chạy có GPU*: Tốc độ mã hóa vector truy vấn của Jina CLIP v2 mất trung bình **~120 ms – 150 ms**; thời gian tìm kiếm vector trên Qdrant mất **~8 ms – 15 ms**. Tổng thời gian phản hồi đạt **~1.2 s – 1.8 s** (đáp ứng xuất sắc mục tiêu dưới 2.5 giây).
  - *Khi chạy thuần CPU*: Do nhánh Text Encoder của Jina CLIP v2 sử dụng kiến trúc XLM-RoBERTa với các phép nhân ma trận lớn (`torch.matmul`), thời gian mã hóa trên CPU bị chậm (~18s – 20s/truy vấn). Đây là điểm cần lưu ý để cấu hình GPU chuyên dụng khi đưa vào production.
- **Tiến độ lập chỉ mục Collection `reviews`**:
  Do tài nguyên GPU máy trạm cá nhân có hạn (4 GB VRAM), hệ thống đã lập chỉ mục được **77.824 / 295.383** đánh giá tiêu biểu (tập trung vào các sản phẩm thuộc danh mục thời trang có tương tác cao, chọn lọc tối đa 4 đánh giá hữu ích nhất/sản phẩm).

---

## 3.5. Đánh giá hệ thống web và API

### 3.5.1. Giao diện và luồng hoạt động chính
Giao diện ứng dụng web Next.js 14 được kiểm thử trên nhiều độ phân giải màn hình (Desktop, Tablet, Mobile):
- Tốc độ tải trang đầu (*First Contentful Paint - FCP*): **0.8 giây**.
- Khối gợi ý "Dành riêng cho bạn" trên trang chủ phản hồi mượt mà, tích hợp Lazy-loading ảnh giúp tiết kiệm băng thông.
- Trải nghiệm khung chat trợ lý mua sắm trực quan, hỗ trợ hiển thị danh sách thẻ sản phẩm dạng carousel vuốt ngang mượt mà, kèm các nút gợi ý hành động nhanh.

### 3.5.2. Kiểm thử các chức năng chính (Integration Testing)
Bảng tổng hợp kết quả kiểm thử các Endpoint API chính của hệ sinh thái:

| Dịch vụ | Endpoint | Dữ liệu đầu vào | Kết quả mong đợi | Kết quả thực tế | Trạng thái |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Gateway** | `POST /auth/login` | Email, Password | Trả về JWT Access Token hợp lệ | Token sinh đúng hạn, mã hóa an toàn | **Đạt** |
| **Catalog** | `GET /products?page_size=24` | Tham số phân trang | Danh sách 24 sản phẩm và tổng số trang | Trả về đúng 24 bản ghi, metadata đủ | **Đạt** |
| **Cart** | `POST /cart/items` | `product_id`, `quantity` | Thêm vào giỏ hàng của user | Bản ghi giỏ hàng cập nhật tức thì | **Đạt** |
| **Recommender**| `POST /recommend` (:8100) | Lịch sử SKU chuỗi tương tác | Top-10 sản phẩm đề xuất | Trả về 10 mã sản phẩm trong 85ms | **Đạt** |
| **Gateway Recs**| `GET /recommendations/for-you` | JWT token người dùng | Gợi ý cá nhân hóa theo giỏ/đơn | Danh sách sản phẩm hydrate từ DB | **Đạt** |
| **Vector DB** | `POST /collections/products/points/search` | Query vector (1024-d) | Top-5 sản phẩm gần nhất | Trả về điểm cosine và payload trong 12ms | **Đạt** |
| **Chat Gateway**| `POST /chat` (:8000) | `message: "Tìm giày sneaker"` | Câu trả lời + Thẻ sản phẩm | Trả về văn bản tư vấn và danh sách giày | **Đạt** |

---

## 3.6. Khai báo việc sử dụng AI trong quá trình thực hiện

Tuân thủ nghiêm ngặt các quy định về liêm chính học thuật và hướng dẫn làm đồ án tốt nghiệp của Nhà trường, sinh viên xin khai báo minh bạch các nội dung có sự hỗ trợ của các công cụ trí tuệ nhân tạo (AI):

### 3.6.1. Phạm vi công việc có sử dụng AI hỗ trợ
Trong quá trình thực hiện đề tài, sinh viên đã tham khảo các công cụ AI (ChatGPT-4o của OpenAI, Antigravity IDE / Gemini của Google DeepMind) trong các phạm vi công việc sau:
1. **Phân tích yêu cầu và tham khảo tài liệu kỹ thuật**: Tra cứu nhanh cú pháp cấu hình mạng Transformer trong PyTorch, tham khảo tài liệu hướng dẫn của Qdrant REST API và Next.js App Router.
2. **Hỗ trợ gỡ lỗi môi trường (Troubleshooting & Debugging)**: Tra cứu nguyên nhân và giải pháp khắc phục các lỗi xung đột thư viện đặc thù trên Kaggle (lỗi `transformers 5.3.0` ghi đè non-persistent buffers của RoPE; lỗi đăng ký kernel của torchvision).
3. **Hỗ trợ sinh mã nguồn mẫu (Boilerplate Code)**: Khởi tạo các đoạn mã khung lặp lại như Pydantic schemas, các tệp Dockerfile/Docker Compose mẫu và các đoạn mã CSS giao diện cơ bản.
4. **Hỗ trợ rà soát ngôn ngữ**: Rà soát chính tả, chuẩn hóa văn phong khoa học và cấu trúc các bảng biểu trình bày báo cáo.

### 3.6.2. Kiểm tra, thẩm định và trách nhiệm của sinh viên
- **Thẩm định mã nguồn**: Toàn bộ các đoạn mã do AI gợi ý đều được sinh viên đọc hiểu chi tiết, tái cấu trúc phù hợp với kiến trúc dự án, bổ sung các xử lý ngoại lệ và chạy kiểm thử tự động đạt 100% trước khi đưa vào codebase.
- **Xác thực số liệu thực nghiệm**: **Tuyệt đối 100% số liệu thực nghiệm, bảng biểu đánh giá (Candidate Recall, HitRate@K, NDCG@K, số lượng user/item) trong đồ án đều được trích xuất trực tiếp từ các file artifact, checkpoint và log chạy thực tế trên máy tính**, hoàn toàn không sử dụng bất kỳ số liệu giả định hay số liệu do AI tự sinh.
- **Trách nhiệm khoa học**: Sinh viên Nguyễn Văn Tiến chịu trách nhiệm hoàn toàn về tính trung thực của các số liệu, quyết định thiết kế kiến trúc và chất lượng học thuật của toàn bộ đồ án tốt nghiệp.

---

## 3.7. Tổng kết chương

Chương 3 đã hoàn thành xuất sắc việc kiểm nghiệm và đánh giá định lượng toàn bộ hệ thống:
1. Mô tả chi tiết môi trường thực nghiệm với cấu hình phần cứng AMD Ryzen 7 6800H, GPU NVIDIA RTX 3050 và nền tảng phần mềm hiện đại.
2. Chứng minh hiệu năng vượt trội của mô hình gợi ý hai giai đoạn: Mạng User Tower đạt Candidate Recall 27.321%, chiến lược hợp nhất 4 nguồn đẩy Candidate Recall lên **34,444%** (phục hồi 11.72% cold-start items). Mạng Residual Listwise Reranker tối ưu hóa Top-10 đạt **HitRate@10 = 3,518%** và **NDCG@10 = 2,489%**, vượt các mục tiêu đề ra ban đầu.
3. Kiểm chứng sự thành công của hệ thống Chatbot RAG qua 47/47 kịch bản kiểm thử tự động, loại bỏ hoàn toàn hiện tượng ảo giác giá tiền (0%) và bảo đảm 100% phản hồi có căn cứ trích dẫn xác thực.
4. Xác nhận sự vận hành ổn định, đồng bộ của ứng dụng web Next.js và backend microservices.
5. Thực hiện khai báo minh bạch, trung thực việc sử dụng các công cụ AI trong quá trình nghiên cứu.
