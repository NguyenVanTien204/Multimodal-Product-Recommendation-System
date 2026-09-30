# CHƯƠNG 2: QUY TRÌNH XÂY DỰNG HỆ THỐNG GỢI Ý SẢN PHẨM ĐA PHƯƠNG THỨC VÀ CHATBOT RAG

---

## 2.1. Phát biểu bài toán và phạm vi xây dựng hệ thống

### 2.1.1. Bài toán gợi ý sản phẩm đa phương thức
Hệ thống gợi ý sản phẩm được xây dựng nhằm giải quyết bài toán xếp hạng cá nhân hóa quy mô lớn trên nền tảng thương mại điện tử thời trang, trong bối cảnh dữ liệu tương tác thưa thớt và danh mục sản phẩm có tính thẩm mỹ trực quan cao.

- **Mô tả đầu vào**:
  - Lịch sử tương tác tuần tự của người dùng $u$: $S_u = (i_1, i_2, \dots, i_t)$, trong đó mỗi $i_k$ là một sản phẩm tương tác tích cực được sắp theo mốc thời gian thực tế tăng dần.
  - Ma trận đặc trưng trực quan hình ảnh $V \in \mathbb{R}^{|\mathcal{I}| \times D}$ và ma trận đặc trưng ngữ nghĩa văn bản $T \in \mathbb{R}^{|\mathcal{I}| \times D}$ được trích xuất từ mô hình liên kết CLIP ($D = 1024$).
  - Không gian toàn bộ danh mục sản phẩm $\mathcal{I}$ có quy mô từ $32.557$ sản phẩm (trong tập dữ liệu rút gọn chuẩn) đến $152.086$ mặt hàng (toàn bộ catalog thương mại).
- **Mô tả đầu ra**:
  - Danh sách Top-$K$ ($K \in \{10, 50\}$) sản phẩm được xếp hạng theo xác suất người dùng $u$ sẽ tương tác tích cực tại thời điểm kế tiếp $t+1$: $\mathcal{R}_K(u) = [i_{(1)}, i_{(2)}, \dots, i_{(K)}]$, đảm bảo loại bỏ toàn bộ các sản phẩm người dùng đã từng tương tác trong lịch sử huấn luyện ($i_{(k)} \notin S_u$).
- **Cơ chế hai giai đoạn (Two-Stage Pipeline)**:
  - *Tầng 1 - Truy hồi ứng viên (Candidate Retrieval)*: Thu hẹp không gian từ toàn bộ danh mục $\mathcal{I}$ xuống tập ứng viên tiềm năng $\mathcal{C}_u$ với ngân sách tối đa $2.000$ ứng viên trước khi khử trùng lặp, đảm bảo độ trễ thấp và độ bao phủ (Candidate Recall) cao.
  - *Tầng 2 - Xếp hạng lại (Reranking)*: Áp dụng mạng nơ-ron học phần dư listwise (**Residual Listwise Reranker**) chấm điểm lại tập $\mathcal{C}_u$, tối ưu hóa độ chính xác và tính đa dạng ở các vị trí đầu danh sách hiển thị.

### 2.1.2. Bài toán Chatbot RAG hỗ trợ mua sắm
Trợ lý đàm thoại mua sắm được thiết kế nhằm nâng cao trải nghiệm khách hàng thông qua giao tiếp ngôn ngữ tự nhiên hai chiều:

- **Đầu vào**:
  - Truy vấn ngôn ngữ tự nhiên từ người dùng bằng tiếng Việt hoặc tiếng Anh (có thể kèm theo ảnh tải lên).
  - Ngữ cảnh phiên làm việc (*Session Context*): Lịch sử đối thoại, trạng thái bộ lọc ưu tiên (*Preferences*), giỏ hàng hiện tại, và vùng đệm ứng viên gần nhất (*Candidate Pool* gồm 60 sản phẩm).
- **Đầu ra**:
  - Phản hồi văn bản tự nhiên, có cấu trúc rõ ràng.
  - Danh sách sản phẩm tương ứng hiển thị trực quan dưới dạng thẻ (thông tin tên, ảnh, giá bán VND, nhãn danh mục, điểm đánh giá).
  - Trích dẫn bằng chứng xác thực (`[P#]` đối với thông tin sản phẩm, `[R#.#]` đối với nhận xét đánh giá).
- **Ranh giới trách nhiệm kiến trúc**:
  - *Bộ máy gợi ý (Recommender Service)*: Chịu trách nhiệm tính toán vector User Tower và mạng Reranker, xếp hạng số học tối ưu giữa chuỗi hành vi người dùng và danh mục sản phẩm.
  - *Chatbot RAG (RAG Service)*: Đóng vai trò là lớp giao diện tương tác thông minh, bóc tách ý định, điều phối truy xuất tri thức thực tế trên Qdrant, tinh chỉnh bộ lọc tức thời trên candidate pool, giải thích lý do đề xuất và kiểm soát tính trung thực của thông tin.

---

## 2.2. Nguồn dữ liệu và mô tả dữ liệu

### 2.2.1. Nguồn dữ liệu Amazon Reviews 2023
Đề tài sử dụng tập dữ liệu chuẩn mực quốc tế **Amazon Reviews 2023** do nhóm nghiên cứu của Giáo sư Julian McAuley tại Đại học California San Diego (UCSD) thu thập và công bố:
- **Danh mục lựa chọn**: `Clothing_Shoes_and_Jewelry` – nhánh dữ liệu thời trang phong phú và phức tạp nhất trên Amazon, phản ánh đầy đủ đặc trưng thẩm mỹ thị giác và thuộc tính văn bản.
- **Cấu trúc hai nguồn dữ liệu gốc**:
  1. *Dữ liệu tương tác (User Reviews & Interactions)*: Chứa hàng triệu lượt đánh giá chi tiết gồm mã người dùng `user_id`, mã sản phẩm `parent_asin`, điểm số `rating` (1–5 sao), tiêu đề `title`, văn bản `text`, mốc thời gian `timestamp`, số lượt bình chọn hữu ích `helpful_vote` và cờ xác thực mua hàng `verified_purchase`.
  2. *Siêu dữ liệu sản phẩm (Product Metadata)*: Chứa thông tin mô tả chi tiết của từng `parent_asin` gồm tên gọi, danh mục phân cấp đa tầng, thương hiệu, khoảng giá bán, danh sách thuộc tính chi tiết (`features`) và danh sách liên kết hình ảnh phân giải cao.

### 2.2.2. Bộ dữ liệu thực nghiệm chốt của đề tài
Để phục vụ quá trình huấn luyện và đánh giá nghiêm ngặt, đề tài đã xây dựng và đóng băng phiên bản dữ liệu thực nghiệm chính thức: **`balanced_u5_i2_v1`** (cấu hình tại `configs/balanced_dataset.yaml`).

| Đặc trưng thống kê | Giá trị định lượng | Ghi chú kỹ thuật |
| :--- | :---: | :--- |
| **Số lượng người dùng hợp lệ ($|\mathcal{U}|$ )** | **21.690** | Đã lọc qua K-core ($K_u \ge 5$) |
| **Số lượng sản phẩm trong catalog ($\mathcal{I}$)** | **32.557** | Đã lọc qua K-core ($K_i \ge 2$) |
| **Tổng số lượt tương tác tích cực ($rating \ge 4$)** | **198.200** | verified_purchase = True |
| **Số tương tác tích cực dùng để huấn luyện (Train)** | **154.820** | Phân chia tuần tự theo thời gian |
| **Số tương tác tích cực thẩm định (Validation)** | **21.690** | Tương tác áp chót ($t_{last-1}$) |
| **Số tương tác tích cực kiểm thử (Test)** | **21.690** | Tương tác cuối cùng ($t_{last}$) |
| **Độ dài lịch sử tương tác trung bình ở tập Train** | **7,14** | Đảm bảo đủ chiều dài cho Self-Attention |
| **Tương tác âm mạnh dùng cho huấn luyện ($rating \le 2$)** | **12.886** | Khai thác hard-negative |
| **Tỷ lệ sản phẩm mục tiêu Test hoàn toàn mới (Cold-target)** | **5,27%** | Thách thức thực tế đánh giá cold-start |
| **Quy mô danh mục mở rộng lập chỉ mục vector (Qdrant)** | **152.086** | Catalog đầy đủ phục vụ tìm kiếm |

Tập dữ liệu sau tiền xử lý được lưu trữ dưới các tệp Parquet bất biến tại `data/processed/balanced_u5_i2_v1/`:
- `items.parquet`: Danh mục 32.557 sản phẩm kèm đầy đủ thông tin chuẩn hóa.
- `train.parquet`: 154.820 tương tác lịch sử phục vụ huấn luyện User Tower và Reranker.
- `valid.parquet`: 21.690 tương tác liền kề cuối phục vụ chọn siêu tham số và checkpoint.
- `test.parquet`: 21.690 tương tác cuối cùng để đánh giá hiệu năng khách quan độc lập.
- `test_context.parquet`: Toàn bộ lịch sử trước đó của các người dùng trong tập test.
- `train_strong_negatives.parquet`: 12.886 tương tác đánh giá thấp ($rating \le 2$) để khai thác mẫu hard-negative.
- `dataset_manifest.json`: Bảng kê cấu hình lọc, kích thước dữ liệu và mã kiểm tra toàn vẹn SHA256/MD5 nhằm bảo đảm khả năng tái lập thí nghiệm 100%.

---

## 2.3. Quy trình tổng thể xây dựng hệ thống

Kiến trúc tổng thể của hệ thống được chia làm hai luồng nghiệp vụ lớn vận hành song song và tương hỗ:

```
[ DỮ LIỆU GỐC: Amazon Reviews 2023 ]
                  │
                  ▼
   [ Tiền xử lý & Iterative K-core ] ──► Cố định Dataset: balanced_u5_i2_v1
                  │
        ┌─────────┴────────────────────────────────────────┐
        ▼                                                  ▼
[ LUỒNG GỢI Ý HAI GIAI ĐOẠN ]                     [ LUỒNG CHATBOT RAG ]
        │                                                  │
  Jina CLIP v2 (1024-d Embeddings)                  Qdrant Vector DB (Named Vectors)
        │                                           Collections: `products` & `reviews`
        ├──────────────────────────┐                       │
        ▼                          ▼                       ▼
[ User Tower (SASRec) ]   [ Multi-source Union ]   [ User Request: Text / Image ]
  Top-1000 Ứng viên        Budget: 2000 Ứng viên           │
        │                          │                       ▼
        └─────────────┬────────────┘               [ Intent Classifier (Rule-based) ]
                      ▼                                    │
          [ Residual Listwise Reranker ]                   ├─ Search / Recommend / Refine
            Top-K Xếp hạng tối ưu                          ├─ Explain / Compare
                      │                                    │
                      ▼                                    ▼
               Personalization                     [ Hybrid Retrieval & RRF ]
                      │                             Vector similarity + Payload filter
                      │                                    │
                      └─────────────────┬──────────────────┘
                                        ▼
                         [ Grounding & Hallucination Guard ]
                           Kiểm duyệt [P#], [R#.#], Giá tiền
                                        │
                                        ▼
                         [ Phản hồi hoàn chỉnh tới Web UI ]
```

### 2.3.1. Pipeline xử lý dữ liệu và huấn luyện mô hình gợi ý
1. **Trích xuất đặc trưng**: Mã hóa văn bản và hình ảnh sản phẩm bằng Jina CLIP v2 ($D = 1024$).
2. **Huấn luyện User Tower**: Tiếp nhận chuỗi tương tác (được đệm phải `pad_right`), nén qua mạng Self-Attention, tối ưu hàm mất mát Sampled-Softmax kết hợp phân phối âm Popularity và hiệu chỉnh log-Q.
3. **Khai thác ứng viên đa nguồn**: Kết hợp đồng thời 4 bộ sinh ứng viên (User Tower, Popularity, Content Centroid, Last-item) theo cấu hình `CandidateBudget` để tối đa hóa Candidate Recall.
4. **Huấn luyện Reranker**: Huấn luyện mạng `ResidualListwiseRanker` với 15 đặc trưng trên các mẫu hard-negative khai thác từ các nguồn ứng viên bị xếp sai.

### 2.3.2. Pipeline hoạt động của Chatbot RAG
1. **Tiếp nhận & Phân tích ý định**: Tiếp nhận câu hỏi hoặc ảnh từ người dùng, chuẩn hóa văn phong tiếng Việt (bỏ dấu kiểm tra `fold`, glossary tiếng Anh), bóc tách thực thể (thương hiệu, khoảng giá VND, màu sắc) và phân loại ý định.
2. **Truy xuất lai (Hybrid Retrieval)**: Mã hóa truy vấn bằng Jina CLIP v2, tìm kiếm song song trên named vector `text` và `image` trong Qdrant, hợp nhất bằng Weighted Reciprocal Rank Fusion (RRF), đồng thời áp dụng các bộ lọc số/chuỗi cứng trong Qdrant.
3. **Cá nhân hóa theo phiên**: Kết hợp điểm số từ Recommender service (`PERSONAL_WEIGHT = 0.6`) nếu người dùng có lịch sử giỏ hàng hoặc các mặt hàng vừa xem trong phiên.
4. **Xây dựng ngữ cảnh & Kiểm chứng**: Trích xuất tối đa 4 đánh giá tiêu biểu nhất cho mỗi sản phẩm liên quan, đánh mã trích dẫn `[P#]`, `[R#.#]`, kiểm duyệt tính trung thực về giá bán và sinh câu trả lời (qua LLM hoặc Template Engine dự phòng).

---

## 2.4. Tiền xử lý và xây dựng dữ liệu thực nghiệm

### 2.4.1. Làm sạch và lọc dữ liệu tương tác
- **Bộ lọc mua hàng xác thực**: Loại bỏ toàn bộ đánh giá có `verified_purchase = False` nhằm bảo đảm dữ liệu phản ánh hành vi tiêu dùng có cam kết tài chính thực tế.
- **Khử trùng lặp**: Trong trường hợp một người dùng tương tác nhiều lần với cùng một sản phẩm (mua lại hoặc cập nhật đánh giá), hệ thống chỉ lưu lại tương tác gần nhất tại mốc thời gian lớn nhất để tránh làm lệch trọng số chú ý trong chuỗi tuần tự.
- **Gán nhãn tương tác**:
  - Tương tác tích cực: Điểm đánh giá $rating \ge 4.0$ (chiếm 198.200 bản ghi).
  - Tương tác âm mạnh: Điểm đánh giá $rating \le 2.0$ (chiếm 12.886 bản ghi), được lưu riêng để làm nguồn khai thác negative mẫu mực cho tầng reranker.
  - Điểm trung tính: $rating = 3.0$ bị loại bỏ khỏi tập mục tiêu.

### 2.4.2. Lọc Iterative K-core và xử lý dữ liệu thưa
Để khắc phục tình trạng ma trận tương tác có mật độ quá thấp làm suy giảm khả năng hội tụ của mạng Self-Attention, quy trình lọc lặp $K$-core được hiện thực qua giải thuật:

```python
def iterative_k_core(df: pl.DataFrame, k_user: int = 5, k_item: int = 2) -> pl.DataFrame:
    """Lọc lặp đồng thời user và item cho đến khi hội tụ."""
    while True:
        initial_len = len(df)
        user_counts = df.group_by("user_id").len()
        valid_users = user_counts.filter(pl.col("len") >= k_user).select("user_id")
        df = df.join(valid_users, on="user_id")
        
        item_counts = df.group_by("item_id").len()
        valid_items = item_counts.filter(pl.col("len") >= k_item).select("item_id")
        df = df.join(valid_items, on="item_id")
        
        if len(df) == initial_len:
            break
    return df
```

Nhờ quy trình này, toàn bộ 21.690 người dùng đều có ít nhất 5 tương tác (đảm bảo ít nhất 3 tương tác trong tập train sau khi giữ lại 2 cho valid và test), độ dài chuỗi hành vi trung bình đạt 7,14, tạo điều kiện lý tưởng cho cơ chế Attention học biểu diễn sở thích.

### 2.4.3. Chia tập dữ liệu theo thời gian và kiểm tra rò rỉ dữ liệu
- **Giao thức Chronological Leave-last-out**:
  Đối với mỗi người dùng $u$, sắp xếp toàn bộ chuỗi tương tác tích cực theo `timestamp`:
  $$(i_1, t_1), (i_2, t_2), \dots, (i_{m-2}, t_{m-2}), (i_{m-1}, t_{m-1}), (i_m, t_m)$$
  - Target Kiểm thử (Test Target): $i_m$
  - Target Thẩm định (Validation Target): $i_{m-1}$
  - Tập huấn luyện (Train Sequence): $(i_1, i_2, \dots, i_{m-2})$
- **Kiểm tra chống rò rỉ dữ liệu (Anti-leakage Checks)**:
  - Kiểm tra vi phạm thời gian: Bảo đảm $t_{train} < t_{valid} < t_{test}$ trên 100% người dùng.
  - Không bao giờ sử dụng dữ liệu từ Validation hay Test để tính toán độ phổ biến hay tạo embedding hành vi trong quá trình huấn luyện.

### 2.4.4. Chuẩn bị đặc trưng văn bản và hình ảnh
- **Mô hình trích xuất**: `jinaai/jina-clip-v2` chạy trên môi trường Kaggle GPU với 2x NVIDIA T4.
- **Đặc trưng thị giác (Image Embeddings)**:
  - Ảnh sản phẩm được tiền xử lý về kích thước chuẩn $224 \times 224$ và đưa qua mạng EVA-02 ViT-L/14.
  - Xử lý ngoại lệ ảnh lỗi: Đối với 557 sản phẩm (~0.37%) có URL ảnh hỏng hoặc mất trên CDN của Amazon, hệ thống kích hoạt cơ chế ảnh đen chuẩn `(224, 224, color=0)` và gắn cờ `is_image_fallback = True` trong metadata để nhận diện.
  - Ma trận vector `image_embeddings.npy` có kích thước $(152086, 1024)$, chuẩn hóa $L_2 = 1.0$, hoàn toàn không chứa giá trị `NaN` hay `Inf`.
- **Đặc trưng ngữ nghĩa văn bản (Text Embeddings)**:
  - Văn bản đầu vào được chuẩn hóa theo mẫu: `"Title: {title} | Brand: {brand} | Category: {category} | Description: {features}"`.
  - Đưa qua nhánh Text Transformer của Jina CLIP v2, trích xuất vector 1024 chiều, chuẩn hóa $L_2 = 1.0$.

---

## 2.5. Xây dựng mô hình gợi ý đa phương thức

### 2.5.1. Xây dựng User Tower và Item Representation
- **Quyết định thiết kế đệm phải (Right-Padding) thay vì Left-Padding**:
  Trong `src/datn/recommenders/user_tower/dataset.py`, hàm `pad_right` được lựa chọn có chủ đích:
  ```python
  def pad_right(seq: list[int], max_len: int) -> np.ndarray:
      trimmed = seq[-max_len:]
      out = np.full(max_len, PAD_IDX, dtype=np.int64)
      if trimmed:
          out[: len(trimmed)] = trimmed
      return out
  ```
  *Lý do kỹ thuật*: Khi kết hợp với mặt nạ nhân quả (*Causal Attention Mask*), đệm phải đảm bảo mọi vị trí truy vấn luôn có ít nhất vị trí khóa 0 không bị che (vị trí 0 không bao giờ là padding vì chuỗi luôn có độ dài $\ge 1$). Nếu dùng đệm trái (*Left-padding*), các vị trí đệm đầu chuỗi sẽ bị mặt nạ nhân quả che toàn bộ các khóa hợp lệ, dẫn đến hàng softmax chứa toàn bộ giá trị $-\infty$ (sinh ra `NaN`), làm nhiễm độc các tầng attention kế tiếp.

- **Kiến trúc mạng User Tower**:
  - Độ dài chuỗi cực đại: $L_{max} = 20$.
  - Số lớp Attention: 2 lớp Transformer Encoder (`norm_first = True` - Pre-LN).
  - Số đầu chú ý (*Attention Heads*): 4 heads.
  - Kích thước không gian biểu diễn ẩn: $d_{model} = 128$.
  - Kích thước tầng Feed-Forward: $d_{ff} = 512$.
  - Tỷ lệ Dropout: 0.2.
  - Activation: GELU.

- **Biểu diễn sản phẩm lai (Item Tower)**:
  Định nghĩa bảng vector sản phẩm duy nhất trong `UserTower`:
  
  $$e_i = e_i^{id} + W_{proj} \cdot \mathbf{v}_{clip}(i)$$
  
  Trong đó:
  - $e_i^{id}$ là vector ID khởi tạo hoàn toàn bằng 0 (`nn.init.zeros_`), có `padding_idx = PAD_IDX (0)`.
  - $\mathbf{v}_{clip}(i) \in \mathbb{R}^{D}$ là vector CLIP đã ghép nối (ảnh và chữ).
  - $W_{proj}$ là tầng chiếu tuyến tính không bias (`nn.Linear(content_dim, d_model, bias=False)`).
  
  *Ý nghĩa*: Khi sản phẩm chưa từng xuất hiện trong tập train, $e_i^{id} = 0$, vector biểu diễn suy giảm một cách duyên dáng (*graceful degradation*) về chính hình chiếu nội dung CLIP, giải quyết triệt để cold-start.

- **Hàm mất mát Sampled-Softmax có hiệu chỉnh log-Q**:
  Huấn luyện với 512 mẫu âm (50% lấy mẫu đều, 50% lấy mẫu theo tần suất với lũy thừa 0.75), áp dụng hiệu chỉnh log-Q để loại bỏ thiên lệch ước lượng xác suất:
  
  $$\mathcal{L} = -\sum_{u} \log \frac{\exp(h_u \cdot e_{pos} - \log Q(pos))}{\exp(h_u \cdot e_{pos} - \log Q(pos)) + \sum_{j \in \mathcal{N}_u} \exp(h_u \cdot e_j - \log Q(j))}$$

### 2.5.2. Sinh ứng viên từ nhiều nguồn (Multi-Source Candidate Generation)
Được định nghĩa tại `src/datn/recommenders/reranker/candidates.py`, lớp `CandidateBudget` phân bổ hạn mức trước khi khử trùng lặp:

| Nguồn ứng viên | Hạn mức (`CandidateBudget`) | Cơ chế trích xuất |
| :--- | :---: | :--- |
| **1. User Tower** | **1.000** | Tích vô hướng $h_{user} \cdot e_i$ trên toàn bộ catalog |
| **2. Global Popularity** | **300** | Top sản phẩm có tần suất cao nhất trong tập train tích cực |
| **3. Content Centroid** | **400** | Khoảng cách Cosine với tâm cụm vector CLIP của lịch sử gần đây |
| **4. Last-item Similarity**| **300** | Khoảng cách Cosine với vector CLIP của sản phẩm tương tác cuối cùng |
| **Tổng ngân sách trước dedup** | **2.000** | **Candidate Recall đạt 34,444% trên tập Test** |

Sau khi tổng hợp, hệ thống loại bỏ các sản phẩm trùng lặp và loại trừ các sản phẩm nằm trong tập tương tác quá khứ (`seen`).

### 2.5.3. Xây dựng Residual Listwise Reranker
Được hiện thực tại `src/datn/recommenders/reranker/model.py`, lớp `ResidualListwiseRanker` xây dựng 15 đặc trưng đại diện cho từng ứng viên:

- **15 đặc trưng đầu vào (`FEATURE_NAMES`)**:
  1. `tower_score`: Điểm User Tower đã chuẩn hóa Z-score.
  2. `popularity_z`: Điểm log-popularity chuẩn hóa Z-score.
  3. `rr_tower`: Nghịch đảo thứ hạng trong User Tower ($1 / (1 + rank)$).
  4. `rr_popularity`: Nghịch đảo thứ hạng trong danh sách phổ biến.
  5. `content_centroid_score`: Điểm tương đồng với tâm cụm nội dung.
  6. `last_item_content_score`: Điểm tương đồng với sản phẩm cuối.
  7. `rr_content`: Nghịch đảo thứ hạng trong nhánh content centroid.
  8. `rr_last_item`: Nghịch đảo thứ hạng trong nhánh last-item.
  9. `tower_sim_last`: Độ tương đồng User Tower với sản phẩm cuối.
  10. `tower_sim_mean`: Độ tương đồng trung bình với các sản phẩm trong lịch sử.
  11. `tower_sim_max`: Độ tương đồng cực đại với lịch sử.
  12. `history_length`: Chiều dài chuỗi lịch sử người dùng.
  13. `from_tower`: Cờ nhị phân (1 nếu ứng viên đến từ User Tower, 0 nếu không).
  14. `from_content`: Cờ nhị phân từ nhánh Content Centroid.
  15. `from_last_item`: Cờ nhị phân từ nhánh Last-item.

- **Kiến trúc mạng nơ-ron học phần dư**:
  ```python
  self.net = nn.Sequential(
      nn.Linear(15, 64), nn.GELU(), nn.Dropout(0.10),
      nn.Linear(64, 32), nn.GELU(),
      nn.Linear(32, 1),
  )
  nn.init.zeros_(self.net[-1].weight)
  nn.init.zeros_(self.net[-1].bias)
  ```
  Công thức tính điểm:
  
  $$\text{Score}_{final} = x[..., 0] + \text{blend} \cdot \text{MLP}(x)$$
  
  Với $x[..., 0]$ là `tower_score` và $\text{blend} = 0.75$. Việc khởi tạo tầng cuối bằng 0 đảm bảo mô hình khi chưa huấn luyện hoặc khi $\text{blend} = 0$ tái lập chính xác 100% thứ tự của User Tower ban đầu.
- **Học Listwise Softmax trên Hard-Negatives**: Huấn luyện trên danh sách 1 Positive + 128 Negatives, trong đó 75% là hard-negatives được khai thác từ chính các ứng viên được tầng truy hồi xếp điểm cao nhưng không phải là mục tiêu thực sự.

---

## 2.6. Xây dựng Chatbot RAG hỗ trợ gợi ý và tìm kiếm sản phẩm

### 2.6.1. Xây dựng kho tri thức và cơ chế truy xuất
- **Cơ sở dữ liệu vector Qdrant (`src/datn/vectordb/`)**:
  - Collection **`products`** ($152.086$ points):
    - Named vectors: `image` (1024-d, Cosine) và `text` (1024-d, Cosine).
    - Point ID = số nguyên không dấu = `items.parquet` dòng index + 1 = `products.id` trong PostgreSQL.
    - Payload fields: `product_id`, `item_id` (ASIN), `title`, `brand`, `category`, `category_id`, `category_slug`, `price` (VND, quy đổi theo `USD_TO_VND = 25000`), `price_estimated`, `image_url`, `description`, `features`, `review_count`, `avg_rating`, `has_image`, `has_text`, `is_image_fallback`.
    - Payload Indexes: keyword (`item_id`, `brand`, `category_slug`), float (`price`), bool (`has_image`, `has_text`, `is_image_fallback`).
  - Collection **`reviews`**:
    - Vector 1024-d từ Jina CLIP v2 biểu diễn `title + text`.
    - Payload: `product_id`, `item_id`, `rating`, `helpful_vote`, `verified_purchase`, `title`, `text`. Mỗi sản phẩm được chọn lọc tối đa **4 đánh giá hữu ích nhất**.
- **Thuật toán tìm kiếm lai Weighted RRF (`src/datn/retrieval/search.py`)**:
  Hợp nhất thứ hạng tìm kiếm trên vector văn bản và vector hình ảnh:
  
  $$\text{RRF\_Score}(d) = w_{text} \cdot \frac{1}{60 + r_{text}(d)} + w_{img} \cdot \frac{1}{60 + r_{img}(d)}$$

### 2.6.2. Xử lý hội thoại và tạo phản hồi có căn cứ
Được điều phối tại `src/datn/agent/orchestrator.py`:
1. **Bộ phân loại ý định tiếng Việt / tiếng Anh (`src/datn/agent/intent.py`)**:
   - Sử dụng hàm chuẩn hóa `fold` loại bỏ dấu tiếng Việt để bóc tách từ khóa nhưng giữ nguyên chuỗi gốc cho bộ mã hóa đa ngôn ngữ.
   - Từ điển thuật ngữ `GLOSSARY` ánh xạ các danh từ thời trang tiếng Việt sang tiếng Anh của Amazon (ví dụ: `ao khoac` $\to$ `jacket`, `giay the thao` $\to$ `sneakers`, `dam` $\to$ `dress`).
   - Bóc tách thực thể giá tiền `parse_price`: hỗ trợ `duoi 500k`, `tu 300k den 700k`, `under $50`, cùng các chế độ tương đối (`cheaper`, `pricier`).
   - Nhận diện 9 hành động: `search`, `refine`, `recommend`, `similar`, `explain`, `compare`, `reset`, `greet`, `help`.
2. **Vùng đệm ứng viên và Tinh chỉnh tức thời (`POOL = 60`)**:
   Hệ thống duy trì 60 sản phẩm ứng viên trong bộ nhớ phiên. Khi người dùng yêu cầu `refine` (ví dụ: "đổi màu đỏ", "giá rẻ hơn"), hệ thống tái lọc trực tiếp trên 60 ứng viên này trong 15ms mà không cần gọi lại mô hình nhúng vector.
3. **Cá nhân hóa kết hợp Recommender (`PERSONAL_WEIGHT = 0.6`)**:
   Nếu người dùng có lịch sử giỏ hàng hoặc xem hàng trong phiên, hệ thống gọi Recommender Service (:8100) để lấy thứ hạng mô hình và cộng điểm theo trọng số $0.6 \times \text{rank}_{model} + 0.4 \times \text{rank}_{semantic}$.
4. **Kiểm duyệt căn cứ và chống ảo giác (`src/datn/rag/context.py`)**:
   - Bắt buộc đánh mã trích dẫn `[P#]` cho sản phẩm và `[R#.#]` cho review.
   - Hàm `check_grounding`: Quét toàn bộ số tiền xuất hiện trong câu trả lời; nếu có bất kỳ số tiền nào không khớp với giá trong cơ sở dữ liệu, câu trả lời lập tức bị hủy bỏ và chuyển sang chế độ mẫu chuẩn (*Template Fallback*).

---

## 2.7. Thiết kế kiến trúc hệ thống và triển khai ứng dụng

### 2.7.1. Các tác nhân của hệ thống
1. **Khách vãng lai (Guest)**: Xem danh mục, tìm kiếm đa phương thức, xem chi tiết và chat thử nghiệm.
2. **Khách hàng (Customer)**: Đăng ký/đăng nhập, nhận gợi ý Top-K "Dành cho bạn", thêm giỏ hàng, đặt hàng mô phỏng và chat trợ lý ảo có cá nhân hóa theo giỏ hàng.
3. **Quản trị viên (Admin)**: Quản lý sản phẩm, danh mục và kích hoạt lập chỉ mục vector.
4. **Recommender Service**: Dịch vụ độc lập chạy trên cổng `8100`, nạp checkpoint User Tower và Reranker.
5. **RAG Service**: Dịch vụ độc lập chạy trên cổng `8200`, điều phối hội thoại và truy xuất Qdrant.

### 2.7.2. Yêu cầu chức năng của hệ thống (Functional Requirements)
- **FR-01: Quản lý xác thực**: Đăng ký, đăng nhập JWT, đổi mật khẩu, xem thông tin cá nhân.
- **FR-02: Quản lý danh mục & sản phẩm**: Xem danh mục, danh sách sản phẩm phân trang, tìm kiếm theo tên, xem chi tiết sản phẩm.
- **FR-03: Quản lý giỏ hàng**: Thêm sản phẩm vào giỏ, cập nhật số lượng, xóa khỏi giỏ.
- **FR-04: Đặt hàng mô phỏng**: Tạo đơn hàng từ giỏ hàng, lưu vết đơn hàng để làm giàu lịch sử tương tác.
- **FR-05: Gợi ý cá nhân hóa**: Cung cấp danh sách đề xuất "Dành cho bạn" dựa trên User Tower và Reranker.
- **FR-06: Trợ lý mua sắm RAG**: Hội thoại thời gian thực bằng chữ hoặc ảnh, trả về câu trả lời có trích dẫn và thẻ sản phẩm trực quan.
- **FR-07: Tinh chỉnh & Giải thích**: Hỗ trợ lọc lại trên pool 60 sản phẩm và giải thích lý do gợi ý kèm trích dẫn đánh giá thực tế.
- **FR-08: So sánh sản phẩm**: Đối chiếu trực quan thuộc tính và ưu/nhược điểm giữa 2 sản phẩm.

### 2.7.3. Yêu cầu phi chức năng (Non-Functional Requirements)
- **NFR-01: Tính tái lập thực nghiệm**: Toàn bộ dữ liệu, checkpoint và kết quả đo đạc được cố định bằng manifest, seed 20260813 và checksum.
- **NFR-02: Tính có căn cứ dữ liệu (Grounding)**: Không sinh ảo giác giá cả (tỷ lệ ảo giác giá tiền bằng 0%).
- **NFR-03: Suy giảm có kiểm soát (Graceful Degradation)**: Khi LLM hoặc RAG service gặp sự cố, hệ thống tự động kích hoạt chế độ tìm kiếm từ khóa fallback và template dựng sẵn mà không làm sập giao diện web.
- **NFR-04: Hiệu năng phản hồi**: API gợi ý phản hồi dưới 150ms; Chatbot phản hồi dưới 2.0s trên môi trường có GPU.
- **NFR-05: An toàn & Bảo mật**: Băm mật khẩu bằng bcrypt, xác thực stateless bằng JWT.

### 2.7.4. Kiến trúc phân tầng và phân bổ cổng dịch vụ
Hệ thống được đóng gói hoàn chỉnh bằng Docker Compose với các dịch vụ và cổng mạng thực tế:

```
                      [ Người dùng / Trình duyệt ]
                                   │
                                   ▼
                   [ Frontend Web: Next.js 14 / React ]
                            (Cổng :3000)
                                   │
                                   ▼ (HTTP / JSON)
             [ Backend Marketplace Gateway: FastAPI ] (Cổng :8000)
             │
             ├──► [ CSDL PostgreSQL 16 ] (Cổng :5432)
             │     Database: `mini_market`
             │
             ├──► [ Recommender Service: FastAPI ] (Cổng :8100)
             │     Checkpoint: `data/artifacts/`
             │
             └──► [ RAG Chatbot Service: FastAPI ] (Cổng :8200)
                   │  Jina CLIP v2, Orchestrator, SessionStore
                   │
                   └──► [ Vector Database: Qdrant 1.12.4 ] (Cổng :6333)
                         Collections: `products`, `reviews`
```

### 2.7.5. Thiết kế dữ liệu nghiệp vụ (PostgreSQL) và Vector (Qdrant)
- **Lược đồ CSDL quan hệ PostgreSQL (`apps/backend/app/*/models.py`)**:
  - `users`: `id` (PK), `email` (Unique), `password_hash`, `full_name`, `is_admin` (Bool), `created_at`.
  - `categories`: `id` (PK), `name` (Unique), `slug` (Unique, Index).
  - `products`: `id` (PK), `sku` (ASIN, Unique, Index), `name`, `description`, `price` (Numeric(12,2)), `stock_quantity`, `image_url`, `category_id` (FK $\to$ `categories.id`), `is_active` (Bool), `created_at`.
  - `carts`: `id` (PK), `user_id` (FK $\to$ `users.id`, Unique).
  - `cart_items`: `id` (PK), `cart_id` (FK), `product_id` (FK), `quantity` (Int). Unique constraint (`cart_id`, `product_id`).
  - `orders`: `id` (PK), `user_id` (FK), `status`, `shipping_address`, `total_amount` (Numeric(12,2)), `created_at`.
  - `order_items`: `id` (PK), `order_id` (FK), `product_id` (FK), `product_name`, `unit_price`, `quantity`.

- **Lược đồ CSDL Vector Qdrant**:
  - Point ID = `products.id` trong PostgreSQL (1-indexed).
  - Collection `products`: Named vectors `image` (1024-d, Cosine) và `text` (1024-d, Cosine).

### 2.7.6. Thiết kế API của các dịch vụ

#### A. Backend Gateway (`apps/backend/app/`, Cổng :8000)
| Nhóm API | Endpoint | Phương thức | Quyền truy cập | Chức năng nghiệp vụ |
| :--- | :--- | :---: | :---: | :--- |
| **Auth** | `/auth/register` | POST | Public | Đăng ký tài khoản người dùng mới |
| | `/auth/login` | POST | Public | Xác thực thông tin, trả về JWT Access Token |
| | `/auth/me` | GET/PATCH | Customer | Xem hoặc cập nhật thông tin cá nhân |
| | `/auth/change-password` | POST | Customer | Đổi mật khẩu tài khoản |
| **Catalog** | `/categories` | GET | Public | Lấy danh sách toàn bộ danh mục |
| | `/products` | GET | Public | Lấy danh sách sản phẩm phân trang (`page`, `page_size`, `q`, `category_id`) |
| | `/products/{product_id}` | GET | Public | Xem thông tin chi tiết một sản phẩm |
| **Cart** | `/cart` | GET | Customer | Xem thông tin giỏ hàng hiện tại |
| | `/cart/items` | POST | Customer | Thêm sản phẩm vào giỏ hàng |
| | `/cart/items/{item_id}` | PATCH/DELETE | Customer | Cập nhật số lượng hoặc xóa mặt hàng |
| **Orders** | `/orders` | GET/POST | Customer | Xem danh sách đơn hàng hoặc tạo đơn mới |
| | `/orders/{order_id}` | GET | Customer | Xem chi tiết đơn hàng đã tạo |
| **Recs** | `/recommendations/health` | GET | Public | Kiểm tra trạng thái Qdrant và Recommender |
| | `/recommendations/for-you` | GET | Optional User| Gợi ý cá nhân hóa (gọi Recommender $\to$ fallback Qdrant) |
| | `/recommendations/products/{id}/similar` | GET | Public | Tìm sản phẩm tương đồng về vector |
| **Chat** | `/chat` | POST | Optional User| Cổng chat chính: tích hợp giỏ hàng người dùng $\to$ gọi RAG service |
| | `/chat/health` | GET | Public | Kiểm tra kết nối tới RAG service |

#### B. Recommender Service (`apps/recommender/`, Cổng :8100)
- `GET /health`: Trả về `model_loaded`, `model_version`, `catalog_size`.
- `POST /recommend`: Nhận `history` (danh sách SKU) và `k`, chạy qua `RerankerPipeline` trả về danh sách gợi ý và điểm số.

#### C. RAG Service (`apps/rag/`, Cổng :8200)
- `GET /health`: Trả về trạng thái encoder, LLM, collections Qdrant và số sessions đang hoạt động.
- `POST /chat`: Tiếp nhận tin nhắn/ảnh, điều phối đầy đủ pipeline RAG.
- `POST /search`: Tìm kiếm ngữ nghĩa có kèm bộ lọc cứng.
- `POST /recommend`: Gợi ý sản phẩm dựa trên lịch sử SKU của session.
- `POST /refine`: Tinh chỉnh điều kiện trên vùng đệm 60 ứng viên của session.
- `POST /explain`: Giải thích lý do gợi ý sản phẩm cụ thể.
- `POST /compare`: So sánh 2 sản phẩm dựa trên thông số và review.
- `DELETE /sessions/{session_id}`: Xóa phiên làm việc trong RAM.

### 2.7.7. Use Case và Sequence Diagram

#### A. Sơ đồ Use Case tổng quát
```
               ┌────────────────────────────────────────────────────────┐
               │              HỆ THỐNG SHOPPING ASSISTANT               │
               │                                                        │
               │   (1) Xem danh mục & tìm kiếm từ khóa / ảnh            │
  Khách vãng lai ──┼──► (2) Xem chi tiết sản phẩm                       │
       (Guest) │   │                                                    │
               │   │   (3) Đăng ký / Đăng nhập tài khoản                │
               └───┼────────────────────────────────────────────────────┤
                   │   (4) Nhận danh sách gợi ý Top-K "Dành cho bạn"    │
    Khách hàng ────┼──► (5) Quản lý giỏ hàng & Đặt hàng mô phỏng         │
   (Customer)      │   (6) Chat đàm thoại với Trợ lý ảo RAG             │
                   │   (7) Tinh chỉnh tức thời (refine) trên bộ đệm     │
                   │   (8) Xem giải thích / So sánh sản phẩm            │
                   └────────────────────────────────────────────────────┘
```

#### B. Sequence Diagram: Luồng gợi ý cá nhân hóa (`/recommendations/for-you`)
```
User (Browser)        Next.js Web           Backend Gateway       Recommender (:8100)     PostgreSQL
      │                    │                       │                      │                   │
      │── Xem trang chủ ──►│                       │                      │                   │
      │                    │── GET /for-you ──────►│                      │                   │
      │                    │   (kèm JWT User)      │── Lấy lịch sử giỏ ──►│                   │
      │                    │                       │◄── Danh sách SKUs ───┼──────────────────►│
      │                    │                       │                      │                   │
      │                    │                       │── POST /recommend ──►│                   │
      │                    │                       │   (history=SKUs, k)  │ [User Tower]      │
      │                    │                       │                      │  Truy hồi ứng viên
      │                    │                       │                      │ [Reranker]        │
      │                    │                       │                      │  Xếp hạng phần dư │
      │                    │                       │◄── Top-K SKUs ───────┤                   │
      │                    │                       │                      │                   │
      │                    │                       │── SELECT products ───┼──────────────────►│
      │                    │                       │◄── Rows metadata ────┼───────────────────│
      │                    │◄── JSON Top-K items ──┤                      │                   │
      │◄── Hiển thị Card ──│                       │                      │                   │
```

#### C. Sequence Diagram: Luồng tương tác Chatbot RAG (`/chat`)
```
User (Browser)        Next.js Chat          Backend Gateway          RAG Service (:8200)     Qdrant (:6333)
      │                    │                       │                         │                     │
      │── Nhập tin nhắn ──►│                       │                         │                     │
      │   ("Đầm dạ hội     │── POST /chat ────────►│                         │                     │
      │    dưới 1 triệu")  │                       │── Lấy history SKUs      │                     │
      │                    │                       │── POST /chat ──────────►│                     │
      │                    │                       │   (body + history_skus) │ [Intent: search]    │
      │                    │                       │                         │ [Slot: price<=1tr]  │
      │                    │                       │                         │                     │
      │                    │                       │                         │── Encode text query │
      │                    │                       │                         │── Hybrid search ───►│
      │                    │                       │                         │   (Named vectors)   │
      │                    │                       │                         │◄── Top-60 pool ─────┤
      │                    │                       │                         │                     │
      │                    │                       │                         │── Lấy top reviews ─►│
      │                    │                       │                         │◄── Reviews data ────┤
      │                    │                       │                         │                     │
      │                    │                       │                         │ [check_grounding]   │
      │                    │                       │                         │  Kiểm duyệt giá/mã  │
      │                    │                       │                         │ [Sinh phản hồi]     │
      │                    │                       │◄── JSON ChatResponse ───┤  (LLM / Template)   │
      │                    │                       │    (reply, products,    │                     │
      │                    │                       │     evidence, citations)│                     │
      │                    │                       │── Hydrate DB product    │                     │
      │                    │◄── ChatOut payload ───┤                         │                     │
      │◄── Hiển thị thẻ ───│                       │                         │                     │
      │    sản phẩm + trích dẫn                    │                         │                     │
```

---

## 2.8. Tổ chức mã nguồn, môi trường phát triển và quản lý phiên bản

### 2.8.1. Cấu trúc mã nguồn và quản lý dữ liệu
Mã nguồn được tổ chức theo cấu trúc phân hệ chuẩn mực công nghiệp:

```
DATN/
├── apps/
│   ├── backend/              # Gateway API chính (:8000) (Auth, Catalog, Cart, Orders, Chat proxy)
│   ├── rag/                  # RAG Service chuyên biệt (:8200) (FastAPI, Jina CLIP, Orchestrator)
│   ├── recommender/          # Recommender Service (:8100) (FastAPI, User Tower, RerankerPipeline)
│   └── web/                  # Giao diện người dùng Next.js 14 / React (:3000)
├── configs/                  # Tệp cấu hình YAML chuẩn
│   ├── balanced_dataset.yaml # Cấu hình dataset balanced_u5_i2_v1
│   ├── qdrant.yaml           # Cấu hình collection Qdrant HNSW
│   └── user_tower.balanced.yaml # Siêu tham số User Tower
├── data/
│   ├── embedding/            # image_embeddings.npy, text_embeddings.npy, metadata parquet
│   └── processed/
│       └── balanced_u5_i2_v1/ # Bộ dữ liệu thực nghiệm bất biến, manifest và checksum
├── docs/                     # Tài liệu thiết kế hệ thống và báo cáo thực nghiệm
│   └── LuanVan/              # Tài liệu đồ án tốt nghiệp chính thức
├── notebooks/                # Jupyter notebooks trích xuất embedding và EDA
├── src/
│   └── datn/
│       ├── agent/            # Xử lý hội thoại: intent.py, orchestrator.py, session.py
│       ├── rag/              # Trích xuất bằng chứng: context.py, evidence.py, answer.py, llm.py
│       ├── recommenders/     # user_tower/ (dataset, model), reranker/ (candidates, model, inference)
│       ├── retrieval/        # encoder.py (Jina CLIP), search.py (RRF), filters.py, schema.py
│       └── vectordb/         # schema.py, importer.py, client.py, collection.py
├── tests/                    # Bộ kiểm thử tự động (test_rag_units.py gồm 47/47 tests)
├── docker-compose.yml        # Điều phối 5 container: postgres, qdrant, recommender, rag, backend
├── pyproject.toml            # Khai báo phụ thuộc [recommender], [vectordb], [rag]
└── README.md                 # Hướng dẫn khởi chạy hệ thống
```

### 2.8.2. Môi trường phát triển, Docker và Git
- **Quản lý môi trường**: Cấu hình tách biệt các nhóm phụ thuộc trong `pyproject.toml` giúp hệ thống nhẹ nhàng và linh hoạt khi triển khai trên các môi trường phần cứng khác nhau.
- **Tự động hóa với Docker Compose**: Thiết lập sẵn các dịch vụ phụ trợ gồm `postgres:16-alpine` và `qdrant/qdrant:v1.12.4` có cấu hình volume lưu trữ bền vững và cơ chế kiểm tra sức khỏe (*Healthcheck*).
- **Quy chuẩn Git & Khả năng tái lập**: Mọi thay đổi mã nguồn được quản lý qua Git với thông điệp cam kết (*commit messages*) rõ ràng; các siêu tham số huấn luyện và seed cố định (`seed = 20260813`) được lưu chi tiết trong tệp cấu hình để mọi nhà nghiên cứu khác đều có thể tái lập kết quả một cách độc lập.

---

## 2.9. Tổng kết chương

Chương 2 đã trình bày toàn bộ quy trình kỹ thuật hiện thực hóa hệ thống gợi ý sản phẩm đa phương thức kết hợp Chatbot RAG dựa trên sự đối chiếu trực tiếp với mã nguồn thực tế của dự án:
1. Xác định bài toán gợi ý hai giai đoạn và bài toán hội thoại RAG có căn cứ dữ liệu.
2. Thiết lập bộ dữ liệu thực nghiệm chuẩn `balanced_u5_i2_v1` từ Amazon Reviews 2023 qua giải thuật $K$-core và phân chia tuần tự theo thời gian nghiêm ngặt.
3. Chi tiết hóa kiến trúc User Tower đệm phải (`pad_right`) tích hợp biểu diễn sản phẩm lai giải quyết cold-start, chiến lược sinh ứng viên 4 nguồn (`CandidateBudget = 2000`) và mạng `ResidualListwiseRanker` với 15 đặc trưng chuyên sâu.
4. Trình bày kho tri thức vector Qdrant, cơ chế tìm kiếm lai Weighted RRF, vùng đệm 60 ứng viên để tinh chỉnh tức thời và bộ quy tắc kiểm duyệt chống ảo giác thông tin của Chatbot RAG.
5. Thiết kế toàn diện kiến trúc hệ thống phần mềm, cơ sở dữ liệu quan hệ PostgreSQL, danh mục API thực tế của 3 dịch vụ backend, lược đồ tuần tự use-case và cấu trúc tổ chức mã nguồn chuẩn mực.

Đây là cơ sở hoàn chỉnh để tiến hành các thử nghiệm định lượng chuyên sâu và đánh giá hiệu năng hệ thống được trình bày trong **Chương 3**.
