# CHƯƠNG 1: TỔNG QUAN LÝ THUYẾT VỀ LĨNH VỰC NGHIÊN CỨU

---

## 1.1. Tổng quan bài toán

### 1.1.1. Bài toán gợi ý sản phẩm trong thương mại điện tử
Trong kỷ nguyên số, các nền tảng thương mại điện tử (E-commerce) phải đối mặt với sự gia tăng theo cấp số nhân của số lượng hàng hóa và thông tin mô tả. Khái niệm **Hệ thống gợi ý sản phẩm (Product Recommendation System)** được định nghĩa là một hệ thống phần mềm thông minh có chức năng tự động ước lượng mức độ quan tâm của người dùng đối với các sản phẩm chưa từng tương tác, từ đó đề xuất một danh sách hữu hạn các mặt hàng phù hợp nhất nhằm tối ưu hóa trải nghiệm khách hàng và gia tăng doanh số cho doanh nghiệp.

- **Vai trò của gợi ý cá nhân hóa**: Gợi ý cá nhân hóa đóng vai trò là "người định hướng" (navigator) trên sàn thương mại điện tử, giúp rút ngắn hành trình tìm kiếm sản phẩm từ hàng giờ xuống vài giây, giảm tỷ lệ thoát trang (*bounce rate*), gia tăng giá trị đơn hàng trung bình (*Average Order Value - AOV*) và củng cố lòng trung thành của khách hàng.
- **Đầu vào và đầu ra của bài toán**:
  - *Đầu vào*: Tập hợp người dùng $\mathcal{U} = \{u_1, u_2, \dots, u_{|\mathcal{U}|}\}$, danh mục sản phẩm $\mathcal{I} = \{i_1, i_2, \dots, i_{|\mathcal{I}|}\}$, lịch sử chuỗi tương tác tuần tự của từng người dùng $S_u = (i_1^{(u)}, i_2^{(u)}, \dots, i_{|S_u|}^{(u)})$ được sắp xếp theo mốc thời gian tăng dần, cùng với tập siêu dữ liệu đa dạng của sản phẩm gồm văn bản mô tả $T_i$ (tiêu đề, thuộc tính, thương hiệu) và hình ảnh trực quan $V_i$.
  - *Đầu ra*: Một danh sách xếp hạng Top-$K$ sản phẩm $\mathcal{R}_K(u) = [i_{(1)}, i_{(2)}, \dots, i_{(K)}] \subset \mathcal{I} \setminus S_u$ có điểm số dự đoán phù hợp cao nhất dành cho người dùng $u$.
- **Các đặc trưng thách thức chính**:
  1. *Dữ liệu cực thưa (Data Sparsity)*: Ma trận tương tác giữa người dùng và sản phẩm thường có mật độ phản hồi thực tế dưới $0.1\%$, khiến các mô hình phân rã ma trận truyền thống không thể thu thập đủ tín hiệu đồng xuất hiện.
  2. *Phân phối đuôi dài (Long-tail Distribution)*: Phần lớn các tương tác tập trung vào một nhóm nhỏ các sản phẩm cực kỳ phổ biến (*Head items*), trong khi đại đa số danh mục sản phẩm (*Tail items*) nhận được rất ít hoặc không có tương tác, dẫn đến hiện tượng thiên lệch phổ biến (*Popularity Bias*).
  3. *Vấn đề khởi đầu lạnh (Cold-start Problem)*: Xảy ra đối với những sản phẩm mới được đăng bán hoặc người dùng mới tham gia hệ thống, nơi hoàn toàn thiếu vắng lịch sử tương tác quá khứ để mô hình hóa hành vi.

### 1.1.2. Bài toán gợi ý đa phương thức và Chatbot RAG
Nhằm vượt qua các rào cản trên, hướng nghiên cứu hiện đại chuyển dịch mạnh mẽ sang tích hợp đa phương thức và tương tác đàm thoại thông minh:

- **Gợi ý đa phương thức (Multimodal Recommendation)**: Là hướng tiếp cận đồng thời khai thác ba luồng tín hiệu bổ trợ lẫn nhau:
  - *Tín hiệu hành vi cộng tác (Collaborative Signal)*: Phản ánh thị hiếu ngầm qua chuỗi hành vi mua sắm trong quá khứ.
  - *Tín hiệu ngữ nghĩa văn bản (Textual Signal)*: Cung cấp thông tin chi tiết về chất liệu vải, kích cỡ, công năng, xuất xứ thương hiệu.
  - *Tín hiệu thị giác hình ảnh (Visual Signal)*: Cung cấp cảm nhận trực quan về phom dáng, hoa văn, màu sắc thực tế và phong cách thời trang.
  Việc kết hợp đa phương thức cho phép hệ thống biểu diễn chính xác giá trị của một món đồ ngay cả khi món đồ đó chưa hề có tương tác mua hàng trong quá khứ.
- **Chatbot hỗ trợ mua sắm thông minh**: Trong bối cảnh mua sắm trực tuyến hiện đại, khách hàng không chỉ muốn xem một danh sách đề xuất tĩnh, mà mong muốn được tư vấn chủ động thông qua đối thoại tự nhiên:
  - Tìm kiếm linh hoạt (*Search*): Diễn đạt nhu cầu tự do ("Áo sơ mi lụa đi tiệc tối dưới 500k").
  - Gợi ý cá nhân hóa (*Recommend*): Nhận đề xuất theo gu thời trang cá nhân.
  - Tinh chỉnh linh hoạt (*Refine*): Yêu cầu điều chỉnh bộ lọc ("Đổi sang tông màu pastel hoặc giá rẻ hơn").
  - Giải thích thấu đáo (*Explain*): Hiểu rõ lý do vì sao hệ thống lại gợi ý sản phẩm này.
  - So sánh chi tiết (*Compare*): Đối chiếu ưu nhược điểm giữa hai sản phẩm dựa trên phản hồi của những người mua trước.
- **Vai trò của Retrieval-Augmented Generation (RAG)**: Các mô hình ngôn ngữ lớn (LLM) nếu hoạt động độc lập rất dễ mắc lỗi "bịa đặt tri thức" (*Hallucination*). Bằng cách ứng dụng RAG, mọi câu trả lời của Chatbot đều được neo chặt (*grounded*) vào các thông tin sản phẩm và đánh giá thực tế được truy xuất trực tiếp từ cơ sở dữ liệu vector.
- **Ranh giới trách nhiệm kiến trúc**:
  - *Bộ máy gợi ý (Recommender System)*: Chịu trách nhiệm tính toán xếp hạng toán học tối ưu trên toàn bộ catalog để đề xuất danh sách Top-$K$ ứng viên phù hợp với người dùng.
  - *Chatbot RAG*: Đóng vai trò là lớp giao diện giao tiếp thông minh, giải thích ngữ cảnh, hỗ trợ người dùng tinh chỉnh yêu cầu và trích xuất bằng chứng xác thực để trả lời câu hỏi.

---

## 1.2. Cơ sở lý thuyết về dữ liệu và tiền xử lý

### 1.2.1. Dữ liệu tương tác và dữ liệu sản phẩm
Trong các hệ thống thương mại điện tử, dữ liệu thu thập được chia làm hai nhóm chính:
- **Tín hiệu phản hồi của người dùng**:
  - *Phản hồi rõ ràng (Explicit Feedback)*: Người dùng chủ động chấm điểm sản phẩm thông qua thang đo định lượng (ví dụ: rating từ 1 đến 5 sao). Ưu điểm là phản ánh chính xác mức độ hài lòng, nhưng nhược điểm là tỷ lệ người dùng để lại đánh giá thường rất thấp.
  - *Phản hồi ngầm (Implicit Feedback)*: Thu thập thụ động qua các hành vi click chuột, xem trang chi tiết, thêm vào giỏ hàng hoặc thời gian dừng trang. Dữ liệu này rất dồi dào nhưng mang tính nhiễu cao.
- **Quy ước tương tác trong đề tài**: Để xây dựng tập dữ liệu chất lượng cao từ các bản ghi đánh giá Amazon, đề tài quy định:
  - *Tương tác tích cực (Positive Interaction)*: Các bản ghi có $rating \ge 4$ (sản phẩm thực sự làm người dùng hài lòng, thể hiện xu hướng ưa chuộng).
  - *Tương tác âm mạnh (Strong Negative Interaction)*: Các bản ghi có $rating \le 2$ (sản phẩm gây thất vọng, hỗ trợ khai thác hard-negative khi huấn luyện reranker).
- **Siêu dữ liệu sản phẩm (Product Metadata)**:
  - Định danh sản phẩm: Mã chuẩn Amazon Standard Identification Number (`parent_asin` hay `item_id`).
  - Dữ liệu văn bản: Tiêu đề (*title*), danh mục phân cấp (*category*), thương hiệu (*brand*), mô tả chi tiết (*description*), thuộc tính cấu tạo (*features*).
  - Dữ liệu trực quan: Đường dẫn ảnh sản phẩm đại diện độ phân giải cao (*image_url*).
  - Đánh giá của khách hàng (*Customer Reviews*): Tiêu đề nhận xét, nội dung văn bản đánh giá và số lượt bình chọn hữu ích (*helpful_vote*).

### 1.2.2. Tiền xử lý và xây dựng tập dữ liệu thực nghiệm
Một quy trình tiền xử lý nghiêm ngặt là điều kiện tiên quyết để đảm bảo tính hợp lệ của nghiên cứu:
- **Làm sạch và khử trùng lặp**:
  - Chỉ giữ lại các tương tác có cờ `verified_purchase = True` nhằm loại bỏ các đánh giá ảo hoặc spam.
  - Xử lý các cặp $(user, item)$ trùng lặp trong dữ liệu gốc: gộp về một bản ghi duy nhất tại thời điểm tương tác mới nhất.
- **Lọc lặp Iterative K-core**:
  - Nhằm loại bỏ các nút biên (thực thể có quá ít tương tác không đủ để học đặc trưng), giải thuật $K$-core được áp dụng lặp: loại bỏ các user có ít hơn $K_u = 5$ tương tác tích cực và các item có ít hơn $K_i = 2$ tương tác tích cực.
  - Quá trình này được thực thi lặp tuần tự cho đến khi số lượng user và item đạt trạng thái ổn định (hội tụ). Điều này vừa đảm bảo người dùng có đủ độ dài chuỗi hành vi lịch sử để huấn luyện mô hình tuần tự, vừa giữ lại được phân phối đuôi dài tự nhiên của dữ liệu thời trang.
- **Phân chia tập dữ liệu theo thời gian (Chronological Leave-last-out)**:
  - Thay vì chia ngẫu nhiên (dễ gây ra rò rỉ dữ liệu từ tương lai về quá khứ), hệ thống sắp xếp chuỗi tương tác của từng người dùng theo thứ tự thời gian tăng dần:
    - Tương tác tích cực cuối cùng $(t_{last})$ được giữ lại cho tập **Kiểm thử (Test Split)**.
    - Tương tác tích cực liền trước $(t_{last-1})$ được dành cho tập **Thẩm định (Validation Split)**.
    - Toàn bộ các tương tác trước đó $(t_1, \dots, t_{last-2})$ được dùng để **Huấn luyện (Train Split)**.
  - Cơ chế này mô phỏng trung thực quy trình vận hành thực tế: dùng dữ liệu quá khứ để dự đoán hành vi trong tương lai.

---

## 1.3. Cơ sở lý thuyết về hệ thống gợi ý đa phương thức

### 1.3.1. Gợi ý tuần tự và mô hình User Tower
Các mô hình gợi ý truyền thống thường giả định sở thích người dùng là tĩnh. Tuy nhiên, hành vi mua sắm thực tế mang tính tuần tự cao (ví dụ: mua giày chạy bộ thường dẫn đến việc mua tất thể thao hoặc quần áo thể thao).

- **Kiến trúc SASRec (Self-Attention Sequential Recommendation)**:
  Được Kang và McAuley đề xuất năm 2018, SASRec áp dụng cơ chế tự chú ý (Self-Attention) từ Transformer để mô hình hóa chuỗi tương tác tuần tự $S_u = (i_1, i_2, \dots, i_t)$:
  
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$
  
  Trong đó ma trận truy vấn $Q$, khóa $K$ và giá trị $V$ được biến đổi tuyến tính từ ma trận nhúng chuỗi sản phẩm cộng với vector vị trí (*Positional Embedding*). Mặt nạ nhân quả (*Causal Mask*) được áp dụng để đảm bảo dự đoán tại bước $t$ chỉ dựa vào các hành vi từ bước $t$ trở về trước.

- **Kiến trúc Two-Tower (Hai tháp)**:
  ```
  User Action Sequence (i_1, i_2, ..., i_t) ──► [ User Tower (SASRec) ] ──► h_user (128-d)
                                                                                  │ (Dot Product)
  Candidate Item i                           ──► [ Item Representation ] ──► e_i    (128-d)
                                                                                  ▼
                                                                             Score(u, i)
  ```
  - **User Tower**: Tiếp nhận chuỗi sản phẩm người dùng đã tương tác, nén thông tin qua các khối Self-Attention để tạo ra vector đại diện người dùng $h_{user}(u) \in \mathbb{R}^{d_{model}}$.
  - **Item Representation**: Đại diện cho từng sản phẩm $e_i \in \mathbb{R}^{d_{model}}$.
  - Điểm tương thích giữa người dùng $u$ và sản phẩm $i$ được tính bằng tích vô hướng: $\text{Score}(u, i) = h_{user}(u) \cdot e_i$.

### 1.3.2. Biểu diễn đa phương thức và mô hình CLIP
Mô hình **CLIP (Contrastive Language-Image Pre-training)** do Radford et al. (OpenAI, 2021) giới thiệu đã tạo nên một chuẩn mực mới trong học biểu diễn thị giác - ngôn ngữ liên kết:
- **Nguyên lý của CLIP**: Sử dụng cơ chế học tương phản (*Contrastive Learning*) trên hàng trăm triệu cặp ảnh - văn bản trên Internet. Mô hình tối đa hóa độ tương đồng cosine giữa vector ảnh và vector văn bản của cùng một đối tượng, đồng thời giảm thiểu độ tương đồng giữa các cặp không khớp.
- **Kiến trúc Jina CLIP v2**: Đề tài ứng dụng mô hình `jinaai/jina-clip-v2` sử dụng mạng thị giác EVA-02 ViT-L/14 kết hợp bộ mã hóa văn bản đa ngôn ngữ hiện đại, ánh xạ mọi sản phẩm thành các vector nhúng chuẩn hóa $D = 1024$ chiều trên mặt cầu siêu cầu $L_2 = 1.0$.
- **Cơ chế biểu diễn sản phẩm lai giải quyết Cold-start**:
  Trong SASRec truyền thống, ma trận nhúng sản phẩm được khởi tạo ngẫu nhiên theo ID: $e_i = \text{Embedding}(i)$. Khi một sản phẩm không có tương tác trong tập huấn luyện (cold-start), vector của nó sẽ mãi là nhiễu ngẫu nhiên.
  
  Để khắc phục triệt để, đề tài thiết kế cơ chế nhúng lai kết hợp phần dư:
  
  $$e_i = e_i^{id} + W_{proj} \cdot \mathbf{v}_{clip}(i)$$
  
  Trong đó:
  - $e_i^{id}$ là vector phần dư khởi tạo bằng $0$, chỉ cập nhật khi sản phẩm có tương tác mua hàng trong tập train.
  - $\mathbf{v}_{clip}(i) \in \mathbb{R}^{1024}$ là vector đặc trưng nội dung (ảnh và mô tả) cố định từ CLIP.
  - $W_{proj} \in \mathbb{R}^{d_{model} \times 1024}$ là ma trận chiếu tuyến tính được huấn luyện chung với mạng User Tower.
  
  Khi sản phẩm hoàn toàn mới ($e_i^{id} = 0$), biểu diễn $e_i$ chính là hình chiếu nội dung ngữ nghĩa của nó, giúp hệ thống gợi ý chính xác mà không cần chờ dữ liệu tương tác lịch sử.

### 1.3.3. Kiến trúc gợi ý hai giai đoạn (Two-Stage Recommendation)
Trong môi trường công nghiệp với hàng trăm nghìn hoặc hàng triệu sản phẩm, việc chấm điểm toàn bộ danh mục bằng một mạng nơ-ron phức tạp là bất khả thi về mặt độ trễ thời gian thực. Do đó, kiến trúc hai giai đoạn là tiêu chuẩn vàng:
1. **Giai đoạn 1: Truy hồi ứng viên (Candidate Retrieval)**:
   - Mục tiêu: Từ toàn bộ danh mục sản phẩm lớn ($N \ge 32.557 \dots 152.086$), sử dụng các thuật toán tính toán nhanh ($O(1)$ hoặc $O(N \cdot d)$ qua Approximate Nearest Neighbor / tích vô hướng song song) để trích xuất ra một tập nhỏ ứng viên tiềm năng ($K \approx 1000 \dots 2000$).
   - Các nguồn ứng viên: Kết hợp đa nguồn gồm User Tower (cá nhân hóa chuỗi), Global Popularity (xu hướng đại chúng), Content Centroid (tâm cụm nội dung CLIP gần đây) và Last-item Similarity (sản phẩm tương tự món đồ vừa xem).
2. **Giai đoạn 2: Xếp hạng lại (Reranking)**:
   - Mục tiêu: Tiếp nhận danh sách ứng viên nhỏ được truy hồi, áp dụng mô hình phức tạp hơn (**Residual Listwise Reranker**) kết hợp nhiều tín hiệu tương quan sâu để xếp hạng lại chính xác danh sách Top-10 / Top-50 hiển thị cho người dùng.

---

## 1.4. Cơ sở lý thuyết về Retrieval-Augmented Generation (RAG)

### 1.4.1. Kiến trúc RAG
Được Lewis et al. công bố năm 2020, kiến trúc **Retrieval-Augmented Generation** kết hợp sức mạnh của hệ thống truy xuất thông tin (*Dense Retrieval*) và mô hình sinh văn bản (*Parametric Generator*):

```
User Query ──► [ Query Encoder ] ──► Search Vector
                                           │
                                           ▼
                                [ Qdrant Vector DB ]
                                (Metadata + Reviews)
                                           │
                                           ▼
                           Retrieved Top-k Documents
                                           │
                                           ▼
[ Prompt Construction ] ◄── Context: Metadata, Specs, Reviews [P#], [R#.#]
        │
        ▼
   [ Generator ] (LLM / Validated Template) ──► Grounded Response + Evidence
```

- **Nguyên lý hoạt động**: Khi nhận câu hỏi, hệ thống chuyển câu hỏi thành vector truy vấn, tìm kiếm trong cơ sở dữ liệu vector các đoạn văn bản có độ tương đồng ngữ nghĩa cao nhất, ghép các đoạn trích này vào ngữ cảnh (*Context*) của Prompt để yêu cầu mô hình sinh câu trả lời.
- **Ưu điểm vượt trội**:
  - Không cần tái huấn luyện (*Fine-tuning*) mô hình nền tảng đắt đỏ mỗi khi có sản phẩm mới.
  - Ngăn ngừa hiện tượng ảo giác (*Hallucination*).
  - Có khả năng kiểm chứng nguồn gốc thông tin thông qua việc trích dẫn rõ ràng tài liệu tham khảo.

### 1.4.2. Chatbot RAG hỗ trợ gợi ý và tìm kiếm sản phẩm
Ứng dụng RAG trong trợ lý mua sắm thời trang đòi hỏi phải mở rộng từ hỏi đáp thông thường sang hệ thống điều phối đa tác vụ:
- **Phân loại ý định người dùng (Intent Classification)**:
  - *Search*: Tìm sản phẩm theo thuộc tính, màu sắc, khoảng giá.
  - *Recommend*: Yêu cầu đề xuất theo ngữ cảnh hoặc lịch sử mua sắm.
  - *Refine*: Tinh chỉnh điều kiện tìm kiếm trên tập ứng viên hiện có mà không cần gọi lại toàn bộ pipeline tìm kiếm vector.
  - *Explain*: Yêu cầu lý giải vì sao một sản phẩm cụ thể lại được gợi ý.
  - *Compare*: Đối chiếu thuộc tính, ưu điểm và nhược điểm giữa hai sản phẩm dựa trên nhận xét thực tế của người dùng khác.
- **Kiểm soát ảo giác nghiêm ngặt (Groundedness Guardrails)**:
  - Mọi thông tin về giá thành phải khớp 100% với giá lưu trong cơ sở dữ liệu.
  - Mỗi nhận định về chất lượng sản phẩm phải được bảo chứng bằng mã trích dẫn review cụ thể (ví dụ: `[R1.2]`).
  - Khi dữ liệu không đủ (ví dụ sản phẩm chưa có review), hệ thống phải thành thật trả lời "Chưa có đủ thông tin nhận xét" thay vì tự suy đoán.

---

## 1.5. Phương pháp đánh giá

### 1.5.1. Đánh giá mô hình gợi ý sản phẩm
Để đảm bảo tính khách quan và chuẩn mực học thuật, mô hình được đánh giá theo giao thức **Full-Ranking trên toàn bộ danh mục sản phẩm** (không dùng lấy mẫu ngẫu nhiên 99 hay 100 negative mẫu giả tạo). Trước khi xếp hạng, toàn bộ sản phẩm người dùng đã tương tác trong tập huấn luyện đều bị loại bỏ (`exclude_seen = True`).

Các chỉ số định lượng bao gồm:
1. **Candidate Recall**:
   Đo lường tỷ lệ sản phẩm mục tiêu (Ground Truth item) nằm trong tập $K$ ứng viên được truy hồi từ tầng 1:
   
   $$\text{Candidate Recall} = \frac{1}{|\mathcal{U}_{test}|} \sum_{u \in \mathcal{U}_{test}} \mathbb{I}(y_u \in \mathcal{C}_u)$$
   
   Trong đó $y_u$ là sản phẩm thực tế người dùng tương tác, $\mathcal{C}_u$ là tập ứng viên của người dùng $u$.

2. **Hit Rate tại Top-$K$ (HR@$K$)**:
   Đo lường tỷ lệ người dùng có sản phẩm mục tiêu xuất hiện trong Top-$K$ vị trí đầu tiên của danh sách gợi ý:
   
   $$\text{HR@}K = \frac{1}{|\mathcal{U}_{test}|} \sum_{u \in \mathcal{U}_{test}} \mathbb{I}(\text{rank}(u, y_u) \le K)$$

3. **Normalized Discounted Cumulative Gain (NDCG@$K$)**:
   Đo lường chất lượng xếp hạng có tính đến vị trí xuất hiện của sản phẩm mục tiêu (sản phẩm đúng xuất hiện ở vị trí càng cao thì điểm số càng lớn):
   
   $$\text{DCG@}K = \sum_{r=1}^K \frac{2^{\text{rel}_r} - 1}{\log_2(r + 1)}$$
   
   $$\text{NDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}$$
   
   Với bài toán Leave-one-out ($\text{rel} \in \{0, 1\}$), $\text{IDCG@}K = 1$, do đó nếu sản phẩm đúng nằm ở vị trí thứ $r \le K$ thì $\text{NDCG@}K = \frac{1}{\log_2(r + 1)}$, ngược lại bằng 0.

### 1.5.2. Đánh giá Chatbot RAG
Hệ thống Chatbot RAG được đánh giá trên cả phương diện kỹ thuật và trải nghiệm đàm thoại:
- **Độ chính xác phân loại ý định (Intent Accuracy)**: Tỷ lệ các câu truy vấn tự nhiên tiếng Việt và tiếng Anh được gán đúng hành động nghiệp vụ.
- **Hiệu năng truy xuất ngữ cảnh (Context Retrieval Quality)**: Khả năng lọc đúng các sản phẩm liên quan và các đánh giá giàu thông tin nhất.
- **Tính có căn cứ (Groundedness / Faithfulness)**: Tỷ lệ các câu khẳng định trong phản hồi có bằng chứng hỗ trợ trực tiếp từ tài liệu trích dẫn; tỷ lệ ảo giác thông tin giá cả phải bằng $0\%$.
- **Độ trễ thời gian phản hồi (Response Latency)**: Đo lường thời gian trích xuất vector, tìm kiếm trên Qdrant và sinh câu trả lời trên các cấu hình phần cứng khác nhau (GPU vs CPU).
- **Cơ chế suy giảm có kiểm soát (Graceful Degradation)**: Khả năng hệ thống tự động kích hoạt bộ sinh phản hồi dựa trên mẫu khuôn định sẵn (Template Fallback) khi dịch vụ LLM gặp sự cố hoặc thời gian chờ quá tải.

---

## 1.6. Công nghệ và công cụ sử dụng

### 1.6.1. Công nghệ xử lý dữ liệu, huấn luyện và triển khai mô hình
- **Ngôn ngữ lập trình**: **Python 3.10.9** – ngôn ngữ tiêu chuẩn công nghiệp trong khoa học dữ liệu và học máy.
- **Framework Học sâu**: **PyTorch 2.9.1+cu126** kết hợp CUDA 12.6, cung cấp nền tảng tính toán tensor hiệu năng cao trên GPU, tự động tính đạo hàm và tối ưu hóa mạng nơ-ron sâu.
- **Công cụ xử lý dữ liệu lớn**:
  - **Polars & DuckDB**: Các công cụ xử lý dữ liệu thế hệ mới dựa trên kiến trúc Apache Arrow, đa luồng song song vượt trội so với Pandas truyền thống khi xử lý hàng triệu bản ghi đánh giá Amazon.
  - **Apache Parquet**: Định dạng lưu trữ dữ liệu dạng cột nén cao cấp, tối ưu hóa tốc độ I/O và đảm bảo toàn vẹn kiểu dữ liệu.
- **Cơ sở dữ liệu Vector**: **Qdrant 1.12.4** – hệ quản trị cơ sở dữ liệu vector chuyên dụng viết bằng Rust, hỗ trợ cấu trúc Named Vectors, thuật toán tìm kiếm láng giềng gần xấp xỉ HNSW (*Hierarchical Navigable Small World*) và lọc thuộc tính thời gian thực với độ trễ phần nghìn giây.
- **Backend Framework**: **FastAPI** – framework xây dựng RESTful API bất đồng bộ (*Async I/O*) hiệu năng cực cao, tích hợp chuẩn hóa dữ liệu Pydantic và tự động sinh tài liệu OpenAPI/Swagger.

### 1.6.2. Công nghệ phát triển giao diện và quản lý hệ thống
- **Giao diện người dùng (Frontend)**: **Next.js 14** (React Framework) kết hợp **Tailwind CSS**, xây dựng giao diện sàn thương mại điện tử hiện đại, tối ưu hóa hiển thị hình ảnh sản phẩm và hỗ trợ luồng chat thời gian thực.
- **Hệ quản trị cơ sở dữ liệu nghiệp vụ**: **PostgreSQL 16** kết hợp **SQLAlchemy ORM** và **Alembic**, quản lý an toàn dữ liệu người dùng, tài khoản, danh mục, giỏ hàng và đơn hàng mô phỏng.
- **Đóng gói và Triển khai**: **Docker & Docker Compose**, đóng gói đồng bộ toàn bộ các service (Backend, Recommender, RAG, Qdrant, PostgreSQL, Web Frontend) thành các container độc lập, đảm bảo khả năng triển khai tức thì trên mọi môi trường máy chủ.
- **Quản lý mã nguồn và phiên bản**: **Git & GitHub**, tuân thủ nghiêm ngặt quy trình quản lý phiên bản mã nguồn, lưu trữ manifest, checksum và cấu hình thí nghiệm.

---

## 1.7. Các công trình nghiên cứu liên quan

### 1.7.1. Nghiên cứu về gợi ý tuần tự và gợi ý đa phương thức
- **SASRec (Kang & McAuley, 2018)**: Đặt nền móng cho việc đưa kiến trúc Self-Attention vào bài toán gợi ý tuần tự, vượt trội hoàn toàn so với các mạng nơ-ron hồi quy RNN (GRU4Rec) và mạng tích chập CNN (Caser) cả về độ chuẩn xác lẫn tốc độ tính toán song song. Tuy nhiên, SASRec nguyên bản chỉ khai thác ID sản phẩm, dẫn đến bất lợi lớn trước bài toán cold-start.
- **CLIP (Radford et al., 2021)**: Mở ra kỷ nguyên biểu diễn liên kết đa phương thức không giám sát quy mô lớn, chứng minh rằng không gian vector kết hợp giữa ảnh và văn bản có tính chất ngữ nghĩa phong phú và khả năng tổng quát hóa zero-shot vượt bậc.
- **Các nghiên cứu gợi ý đa phương thức gần đây (MMRec, VBPR, multimodal SASRec)**: Đã có những nỗ lực đưa đặc trưng ảnh vào hệ thống gợi ý, song phần lớn tiếp cận theo hướng ghép nối vector muộn (*Late Fusion*) ở mức vector hoặc chỉ sử dụng các mạng thị giác cũ như ResNet-50. Đề tài kế thừa các bài học này, khắc phục lỗi logic trong ghép nối vector, và ứng dụng trực tiếp mô hình thị giác hiện đại EVA-02 ViT-L/14 qua Jina CLIP v2 kết hợp cơ chế phần dư $id\_residual$.

### 1.7.2. Nghiên cứu về Retrieval-Augmented Generation
- **RAG (Lewis et al., 2020)**: Công trình tiên phong khẳng định việc kết hợp bộ truy xuất thông tin không gian vector với mô hình tạo sinh ngôn ngữ giúp giải quyết căn bệnh ảo giác của các mô hình nơ-ron tạo sinh, cho phép cập nhật tri thức động mà không cần tái huấn luyện.
- **Các hệ thống trợ lý mua sắm hội thoại (Conversational Recommender Systems - CRS)**: Các hệ thống CRS thời kỳ đầu thường dựa trên đồ thị tri thức cứng (*Knowledge Graph*) hoặc câu hỏi dạng cây quyết định đóng, gây cảm giác gò bó cho khách hàng. Sự xuất hiện của LLM kết hợp RAG tạo điều kiện xây dựng trải nghiệm đối thoại mở tự nhiên, vừa thấu hiểu ngôn từ phức tạp, vừa đảm bảo câu trả lời được neo chặt vào catalog thực tế.

---

## 1.8. Tổng kết chương

Chương 1 đã thiết lập toàn diện bức tranh cơ sở khoa học và lý luận vững chắc cho đề tài:
1. Phân tích chi tiết bài toán gợi ý sản phẩm cá nhân hóa trong thương mại điện tử, chỉ rõ các thách thức cố hữu về dữ liệu thưa, phân phối đuôi dài và khởi đầu lạnh.
2. Trình bày nền tảng lý thuyết về xử lý dữ liệu tuần tự, giải thuật lọc lặp $K$-core và chiến lược chia tập dữ liệu chống rò rỉ theo dòng thời gian.
3. Làm rõ các nguyên lý cốt lõi của mạng gợi ý tuần tự SASRec, mô hình biểu diễn đa phương thức CLIP và giải pháp kiến trúc hai giai đoạn kết hợp truy hồi đa nguồn.
4. Tổng quan kiến trúc Retrieval-Augmented Generation (RAG) và các tiêu chuẩn kiểm soát ảo giác thông tin trong trợ lý đàm thoại mua sắm.
5. Xác định hệ thống chỉ số đánh giá định lượng khoa học và tổng hợp hệ sinh thái công nghệ hiện đại được ứng dụng trong đồ án.

Những cơ sở lý luận này là tiền đề trực tiếp để bước sang **Chương 2**, nơi toàn bộ quy trình thiết kế, hiện thực hóa kỹ thuật và triển khai kiến trúc hệ thống sẽ được trình bày tường minh.
