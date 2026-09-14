# Báo cáo Kiểm chứng & Phản biện Học thuật: User Tower Recommender

- **Ngày thực hiện**: 14/09/2026
- **Đối tượng kiểm chứng**: Module User Tower ([`src/datn/recommenders/user_tower/`](../../src/datn/recommenders/user_tower/)), các tập dữ liệu tương tác ([`data/*.parquet`](../../data/)), và báo cáo thực nghiệm ([`docs/user_tower_experiments.md`](../user_tower_experiments.md)).
- **Mục tiêu**: Làm rõ và kiểm chứng thực nghiệm độc lập các nghi vấn P0 của chuyên gia về tính hợp lệ của Evaluation Protocol, Hiện tượng rò rỉ dữ liệu (Leakage), Cơ chế Candidate Generation, và sự chênh lệch hiệu năng 33–42 lần giữa Collaborative Filtering (CF) thuần và Multimodal.

---

## 1. Tóm tắt Kết luận Kiểm chứng (Executive Summary)

1. **Evaluation Protocol & Data Integrity: HỢP LỆ & KHÔNG CÓ GIAN LẬN**
   - Protocol chuẩn **Leave-Last-Out theo thời gian của từng user**: Mỗi user có đúng 1 ground-truth positive item ở Valid và Test.
   - **$Recall@K \equiv HitRate@K$ là đẳng thức toán học chuẩn xác** khi số lượng ground-truth $|Relevant| = 1$, không phải lỗi thuật toán hay trùng lặp chỉ số.
   - Đánh giá **Full-Catalog Retrieval trên toàn bộ 152.086 sản phẩm**, không lấy mẫu ứng viên (no candidate sampling) khi test.
   - Cơ chế loại trừ item đã xem (`exclude_seen: true`) hoạt động chuẩn xác với cơ chế bảo vệ `seen.discard(target)`, không lọc nhầm target.
   - **Không có rò rỉ thời gian (Data Leakage)**: Đã kiểm chứng $\max(Train) \le \min(Valid) \le \min(Test)$ và mặt nạ nhân quả (Causal mask) chặn hoàn toàn việc nhìn về tương lai.

2. **Nguyên nhân gốc rễ của việc CF-only sụp đổ (thấp hơn Multimodal 33–42 lần)**:
   - **Đặc thù dữ liệu siêu thưa (Extreme Sparsity 99.995%)**: Thống kê thực tế cho thấy trong tập Train, **82.17% sản phẩm chỉ xuất hiện ĐÚNG 1 LẦN DUY NHẤT**. Toàn bộ catalog 152k sản phẩm chỉ có 230 sản phẩm có trên 10 tương tác. CF dựa vào đồng xuất hiện (co-occurrence) bị **chết đói dữ liệu**.
   - **Tác động phụ của khởi tạo `nn.init.zeros_`**: Để phục vụ Multimodal (giúp cold-start tự động suy thoái về CLIP embedding), bảng `id_residual` được khởi tạo bằng 0. Khi chạy CF-only (không có content), tại bước 0 toàn bộ embedding đều bằng 0, khiến gradient truyền ngược về Transformer encoder bị triệt tiêu hoàn toàn, mô hình bị nghẽn hội tụ ngay từ đầu.

3. **Phát hiện bước ngoặt về Cold-Start giữa CF và Multimodal**:
   - Khi phân rã thực nghiệm kiểm chứng: Đối với **13.214 user có target là Cold Item** (chưa từng có trong train), **cả CF lẫn Multimodal đều đạt 0.0% HitRate@50** (việc đưa 1 item chưa từng có tương tác vào Top 50 giữa 152k ứng viên là cực kỳ thách thức).
   - Toàn bộ 128 hits của Multimodal trên tập Test đều đến từ **9.200 user có target là Warm Item** (đạt HitRate@50 = **1.39%**, trong khi CF chỉ đạt **0.043%** — 4 hits).
   - **Kết luận**: Multimodal vượt trội CF không phải vì "giải quyết triệt để bài toán retrieval cold-start thuần", mà vì nó cung cấp **Semantic Representation Transfer**: ngay cả khi 2 sản phẩm chỉ xuất hiện 1 lần trong train, CLIP vẫn biết chúng là cùng loại sản phẩm thông qua thị giác và ngôn ngữ, điều mà CF thuần ID hoàn toàn bất lực.

---

## 2. Chi tiết Kiểm chứng P0: Evaluation Protocol, Data Split & Candidates

### A. Evaluation Protocol

| Câu hỏi kiểm chứng | Kết quả | Bằng chứng mã nguồn & Kiểm tra dữ liệu |
|---|:---:|---|
| **1. Mỗi user chỉ có 1 positive item ở val/test?** | **XÁC NHẬN** | Trong [`kaggle_streaming_smart_subset.ipynb:L316-L318`](file:///d:/WorkSpace/Work/DATN/notebooks/kaggle_streaming_smart_subset.ipynb#L316-L318), dữ liệu được chia bằng `row_number() OVER (PARTITION BY user_id ORDER BY timestamp DESC)`. Rank 1 là Test, Rank 2 là Valid, còn lại là Train. Tại [`dataset.py:L61-L69`](file:///d:/WorkSpace/Work/DATN/src/datn/recommenders/user_tower/dataset.py#L61-L69), hàm `_positive_sorted` chỉ giữ lại tương tác `is_positive == 1` (rating $\ge 4$) làm ground-truth. |
| **2. Recall@K = HitRate@K do evaluator chỉ có 1 item?** | **XÁC NHẬN** | Theo toán học: Khi $|Relevant| = 1$, nếu target nằm trong Top-K thì $Recall = 1/1 = 1 = HitRate$; nếu không thì $Recall = 0/1 = 0 = HitRate$. Trung bình toàn bộ user: $Recall@K \equiv HitRate@K$. Được ghi chú trực tiếp tại [`evaluate.py:L20-L25`](file:///d:/WorkSpace/Work/DATN/src/datn/recommenders/user_tower/evaluate.py#L20-L25). |
| **3. Đánh giá trên toàn bộ 152.086 items hay subset?** | **TOÀN BỘ CATALOG** | [`evaluate.py:L27-L43`](file:///d:/WorkSpace/Work/DATN/src/datn/recommenders/user_tower/evaluate.py#L27-L43) tính tương đồng qua `scores = user_vecs @ item_emb.T` với `item_emb` shape `(152088, d_model)`. Sau khi gán `-inf` cho PAD (0) và OOV (152087), đúng **152.086 sản phẩm** được xếp hạng đầy đủ. |
| **4. Có filter item user đã tương tác trong train không?** | **CÓ** | [`evaluate.py:L44-L48`](file:///d:/WorkSpace/Work/DATN/src/datn/recommenders/user_tower/evaluate.py#L44-L48): khi `exclude_seen: true` ([`configs/user_tower.yaml:L31`](file:///d:/WorkSpace/Work/DATN/configs/user_tower.yaml#L31)), toàn bộ các item trong `seen` được gán điểm `-inf`. |
| **5. Có vô tình filter nhầm cả positive val/test không?** | **KHÔNG** | Tại [`dataset.py:L169`](file:///d:/WorkSpace/Work/DATN/src/datn/recommenders/user_tower/dataset.py#L169), mã nguồn gọi lệnh tường minh: `seen.discard(target)`. Đồng thời pipeline tiền xử lý đã assert khử trùng `(user_id, item_id)`. |
| **6. Có item nào không có embedding trong candidate?** | **KHÔNG** | 100% 152.086 item đều có vector CLIP (ảnh 1024-d, text 1024-d) trong [`data/embedding/`](file:///d:/WorkSpace/Work/DATN/data/embedding/) và đều có một hàng trong ma trận `id_residual`. |
| **7. Có user nào không đủ history vẫn được đưa vào eval?** | **KHÔNG** | Đã kiểm tra độ dài lịch sử trên toàn bộ 22.414 user test và 22.268 user valid: **100% user có context length $\ge 1$** (trung bình ~5-6 items, tối đa 20 items). Số user có history rỗng bằng 0. |

---

### B. Train / Validation / Test Split

| Câu hỏi kiểm chứng | Kết quả | Bằng chứng mã nguồn & Kiểm tra dữ liệu |
|---|:---:|---|
| **8. Split theo user hay theo interaction?** | **Interaction** | Temporal Split trên chuỗi tương tác của từng user (Leave-Last-Out per User), không phải user-level split ngẫu nhiên. Mọi user hợp lệ đều có mặt ở Train, Valid và Test. |
| **9. Có bị distribution shift mạnh không?** | **RẤT MẠNH (Cold-Start)** | **Có tới 58.95% item ở Test (13.214/22.414) hoàn toàn chưa từng xuất hiện trong Train.** Đây là đặc trưng tự nhiên của Amazon Fashion khi 80.86% sản phẩm chỉ có duy nhất 1 review trong toàn bộ lịch sử. |
| **10. Tương tác có xuất hiện đồng thời ở train và val/test?** | **KHÔNG** | Phân chia bằng hàm cửa sổ `row_number()`, mỗi dòng tương tác chỉ thuộc đúng một partition duy nhất. |
| **11. Có leakage từ tương lai vào user history?** | **KHÔNG** | Pipeline đã assert kiểm tra: $\max(train\_ts) \le \min(valid\_ts) \le \min(test\_ts)$. Tháp User Tower áp dụng causal mask (`torch.triu`) chặn hoàn toàn việc nhìn về tương lai. |
| **12. Positive ở test user đã thấy trong train chưa?** | **CHƯA TỪNG** | Đã được deduplicate nghiêm ngặt: $count(rows) = count(distinct(user\_id, item\_id))$. |
| **13. Phân bố interaction/user giữa các tập?** | **CHUẨN** | Test: đúng 1 tương tác; Valid: đúng 1 tương tác; Train: phần còn lại ($\ge 3$ tương tác do bộ lọc K-core $\ge 5$). |

---

### C. Candidate Generation & Negative Sampling

- **Số lượng item rank**: Đúng **152.086 sản phẩm** (Full Catalog Retrieval).
- **Candidate Sampling**: **KHÔNG sampling khi evaluation**. Toàn bộ 152k sản phẩm đều được tính điểm dot-product.
- **Negative sampling khi Train**: Dùng $K=4$ negative items cho mỗi vị trí bước thời gian, lấy mẫu theo phân phối tần suất $P(i) \propto freq(i)^{0.75}$ với 1-smoothing (kiểu word2vec/BPR), định nghĩa tại [`train.py:L42-L65`](file:///d:/WorkSpace/Work/DATN/src/datn/recommenders/user_tower/train.py#L42-L65).

---

## 3. Phân tích Chuyên sâu: Vì sao CF-only quá yếu (33–42 lần)?

### 3.1. Cấu trúc mô hình & Luồng biểu diễn
- User Tower sử dụng kiến trúc **SASRec (Sequential Recommendation)**: User embedding $h_{user}$ được sinh ra động từ chuỗi hành vi của user thông qua Transformer Causal Self-Attention.
- Mô hình **không sử dụng bảng User ID embedding**. Đây là thiết kế inductive chuẩn để hỗ trợ suy luận cho user mới từ hành vi gần nhất.
- Indexing và Dimension hoàn toàn khớp nhau: `d_model = 128`, không có lỗi lệch hàng hay nhầm index.

### 3.2. Thống kê tần suất sản phẩm trong Train (Gốc rễ sụp đổ co-occurrence)
Kiểm tra thực tế phân phối tần suất của các item trong `train.parquet`:
- **Tổng số item trong catalog**: 152.086 sản phẩm.
- **Số item từng xuất hiện trong train**: 99.315 sản phẩm.
- **Số item chỉ xuất hiện ĐÚNG 1 LẦN trong train**: **81.608 sản phẩm (82.17%)**.
- **Số item xuất hiện $\le 2$ lần trong train**: **92.255 sản phẩm (92.89%)**.
- **Số item xuất hiện $\le 5$ lần trong train**: **98.065 sản phẩm (98.74%)**.
- **Số item xuất hiện $> 10$ lần trong toàn bộ train**: **Chỉ có 230 sản phẩm!**
- **Tần suất tương tác trung bình**: Chỉ đạt $99.464 / 152.086 \approx 0.65$ bước chuyển dịch/item.

> **Nhận xét**: Collaborative Filtering học dựa trên quy luật đồng xuất hiện ($P(i_B \mid i_A)$). Với ma trận thưa 99.995% và hơn 82% sản phẩm chỉ xuất hiện 1 lần duy nhất, việc mong đợi CF học được vector không gian có ý nghĩa để phân biệt giữa 152.086 sản phẩm là bất khả thi về mặt thống kê.

### 3.3. Tác động của cơ chế khởi tạo Zero (`nn.init.zeros_`)
- Trong [`model.py:L54-L55`](file:///d:/WorkSpace/Work/DATN/src/datn/recommenders/user_tower/model.py#L54-L55):
  ```python
  self.id_residual = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
  nn.init.zeros_(self.id_residual.weight)
  ```
- Với Multimodal: $e_i = 0 + content\_proj(clip(i))$, việc khởi tạo zero giúp cold-start item suy thoái hoàn hảo về vector nội dung CLIP đã chuẩn hóa.
- Với CF-only: Không có content, tất cả item embedding ban đầu bằng 0.
  - Tại epoch 0, logits giữa user vector (chỉ chứa `position_embedding`) và item embedding bằng 0.
  - Gradient ngược về Transformer encoder bị triệt tiêu: $\frac{\partial \mathcal{L}}{\partial hidden} = \mathbf{0}$.
  - Mô hình gặp khó khăn ngay từ pha phá vỡ đối xứng (symmetry breaking) và bị Early Stopping ở ngay **Epoch 5**.
  - Các cold item (chưa từng có ở train) chỉ nhận gradient âm từ negative sampling $\to$ điểm số bị dìm xuống mức rất âm (từ $-4.0$ đến $-5.4$).

---

## 4. Dữ liệu Thực nghiệm Kiểm chứng Độc lập (Empirical Evidence)

### 4.1. Bảng phân rã Warm vs Cold (Test Split - 22.414 Users)

Toàn bộ 22.414 user trên tập test được phân tách độc lập thành 2 nhóm dựa trên việc target item đã từng xuất hiện trong tập Train hay chưa:

| Phân nhóm dữ liệu | Số lượng Users | Mô hình | HitRate@10 | HitRate@20 | HitRate@50 | NDCG@10 | Số Hits Top-50 |
|---|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **Toàn bộ Test Split** | **22.414** | **Collaborative Filtering (CF)** | 0.0045% | 0.0134% | 0.0178% | 0.0017% | 4 / 22.414 |
| | | **Full Multimodal (Đề xuất)** | **0.1428%** | **0.2811%** | **0.5711%** | **0.0721%** | **128 / 22.414** |
| **Warm Targets**<br>*(Đã xuất hiện trong Train)* | **9.200** | **Collaborative Filtering (CF)** | 0.0109% | 0.0326% | 0.0435% | 0.0042% | 4 / 9.200 |
| | | **Full Multimodal (Đề xuất)** | **0.3478%** | **0.6848%** | **1.3913%** | **0.1756%** | **128 / 9.200** |
| **Cold Targets**<br>*(Chưa từng có trong Train)* | **13.214** | **Collaborative Filtering (CF)** | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0 / 13.214 |
| | | **Full Multimodal (Đề xuất)** | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0 / 13.214 |

### 4.2. Trace mẫu từng User cụ thể

#### Mẫu 1: User có Target là Warm Item (`AG5BSWLA4JVCUYA2KKQQBYKK2CZA`)
- **Lịch sử Context (3 items)**: `['B00007JTZ7', 'B00KJGKZ7I', 'B09WYX2KPH']`.
- **Target Item**: `B07JPYSNDP` (Index 32706, đã từng xuất hiện ở Train).
- **Kết quả CF-only**:
  - Norm của User vector: `26.7419`.
  - Target score: `1.3369` (Top 5 scores: `5.55` – `6.29`).
  - **Thứ hạng Target: #9.016 / 152.086**.
- **Kết quả Multimodal**:
  - Norm của User vector: `17.8181`.
  - Target score: `1.6851` (Top 5 scores: `4.32` – `4.59`).
  - **Thứ hạng Target: #740 / 152.086** (tiến bộ hơn 12 lần so với CF).

#### Mẫu 2: User có Target là Cold Item (`AEQU6BMGSQWEPU2IBG5J7CGLBB5A`)
- **Lịch sử Context (4 items)**: `['B0000AZ0ZO', 'B0000AZ0ZR', 'B000B7PNAE', 'B000M9MZVG']`.
- **Target Item**: `B0002X4HTK` (Index 92230, hoàn toàn chưa từng có trong Train).
- **Kết quả CF-only**:
  - Norm của User vector: `26.1852`.
  - Target score: `-4.0679` (bị dìm sâu do negative gradient).
  - **Thứ hạng Target: #67.142 / 152.086**.
- **Kết quả Multimodal**:
  - Norm của User vector: `17.6065`.
  - Target score: `0.7131` (nhờ chiếu từ đặc trưng CLIP).
  - **Thứ hạng Target: #2.790 / 152.086** (nhảy vọt từ hạng 67.142 lên hạng 2.790).

---

## 5. Kiến nghị Hoàn thiện Luận văn & Báo cáo

1. **Khẳng định tính liêm chính học thuật**:
   - Toàn bộ pipeline kiểm thử được bảo toàn nghiêm ngặt: không có data leakage thời gian, không gian lận candidate sampling, protocol $Recall@K \equiv HitRate@K$ là chính xác về mặt toán học.
2. **Hiệu chỉnh lại phát biểu về Cold-Start trong Báo cáo**:
   - **Cách viết cũ**: "Multimodal giải quyết triệt để vấn đề Cold-start" $\to$ dễ gây hiểu lầm là Multimodal đưa được các sản phẩm cold-start vào thẳng Top 10/Top 50.
   - **Cách viết chuẩn xác học thuật**: Trên không gian tìm kiếm cực rộng (152.086 items), việc đưa một sản phẩm chưa từng có tương tác vào Top 50 là rào cản quá lớn đối với tầng Retrieval đơn thuần (cả 2 đều 0 hit). Tuy nhiên, Multimodal đã nâng thứ hạng của Cold-item từ **vùng đáy (#60.000 – #140.000)** lên **vùng tiệm cận ứng viên (#2.000 – #5.000)**.
3. **Làm rõ vai trò thực sự của Multimodal**:
   - Sự vượt trội 33–42 lần về mặt chỉ số HitRate@10/NDCG@10 thực chất đến từ việc **Multimodal giải quyết thành công bài toán dữ liệu siêu thưa (Extreme Sparsity) trên nhóm Warm items (từ 4 hits lên 128 hits, gấp 32 lần)** bằng cơ chế Semantic Transfer thông qua hình ảnh và văn bản.
