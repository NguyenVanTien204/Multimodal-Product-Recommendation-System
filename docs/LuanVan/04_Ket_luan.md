# KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

> **Phạm vi bộ dữ liệu (cập nhật 05/10/2026):** bộ khung luận văn này viết theo bộ **Amazon**. Bộ dữ liệu chính mới là **H&M**; luận điểm được phép khẳng định, thực nghiệm còn thiếu và ánh xạ sửa từng chương nằm ở [`../hm/05_thesis_plan.md`](../../hm/docs/05_thesis_plan.md). Các số liệu Amazon bên dưới vẫn đúng cho Amazon.

---

## 1. Đánh giá kết quả đạt được

### 1.1. Đánh giá theo mục tiêu đề tài
Đối chiếu với các mục tiêu cụ thể, đo lường được đã đề ra tại phần Mở đầu, đề tài **"Xây dựng hệ thống gợi ý sản phẩm đa phương thức ứng dụng RAG"** đã hoàn thành toàn diện và vượt mức các chỉ tiêu đề ra:

1. **Mục tiêu 1: Xây dựng mô hình gợi ý hai giai đoạn đa phương thức**:
   - *Chỉ tiêu đề ra*: Candidate Recall $\ge 30\%$, HitRate@10 $\ge 3.5\%$, NDCG@10 $\ge 2.4\%$.
   - *Kết quả thực tế đạt được*:
     - **Candidate Recall đạt 34,444%** trên tập kiểm thử độc lập (vượt mục tiêu 4,444 điểm phần trăm), phục hồi thành công **11,72%** các sản phẩm hoàn toàn cold-start nhờ cơ chế sinh ứng viên kết hợp 4 nguồn.
     - **HitRate@10 đạt 3,518%** (vượt chỉ tiêu 3,5%).
     - **NDCG@10 đạt 2,489%** (vượt chỉ tiêu 2,4%).
     - Mạng **Residual Listwise Reranker (v2)** chứng minh sự vượt trội toàn diện so với mô hình User Tower đơn lẻ ở mọi ngưỡng cắt ($K \in \{10, 50, 100\}$).
   - *Đánh giá*: **Hoàn thành xuất sắc (Vượt chỉ tiêu)**.

2. **Mục tiêu 2: Xây dựng hệ thống Chatbot RAG hỗ trợ mua sắm thông minh**:
   - *Chỉ tiêu đề ra*: Vận hành theo cơ chế Closed-domain RAG, hỗ trợ 5 nhóm ý định mua sắm, 100% câu trả lời có căn cứ trích xuất từ dữ liệu (không ảo giác về giá bán hay thuộc tính), thời gian phản hồi dưới 2.5 giây khi có tăng tốc phần cứng.
   - *Kết quả thực tế đạt được*:
     - Triển khai kho tri thức Qdrant với hai collection `products` (152.086 points) và `reviews` (77.824 points hữu ích nhất).
     - Bộ phân loại ý định hỗ trợ chuẩn xác 9 tác vụ hội thoại và trích xuất thực thể giá, màu, thương hiệu.
     - Cơ chế kiểm duyệt căn cứ (*Grounding Guardrails*) đảm bảo **100% câu trả lời có mã trích dẫn hợp lệ** (`[P#]`, `[R#.#]`), **tỷ lệ ảo giác giá tiền đạt 0,0%**.
     - Vượt qua **47/47 kịch bản kiểm thử tự động** trong bộ test suite `hm/tests/test_rag_units.py`.
     - Độ trễ mã hóa và tìm kiếm vector trên GPU chỉ mất **~150 ms**, tổng thời gian phản hồi đạt **~1.5 giây**.
   - *Đánh giá*: **Hoàn thành xuất sắc**.

3. **Mục tiêu 3: Tích hợp ứng dụng web và hệ sinh thái microservices**:
   - *Chỉ tiêu đề ra*: Xây dựng giao diện web kết nối liền mạch giữa mô hình gợi ý cá nhân hóa và khung chat trợ lý ảo.
   - *Kết quả thực tế đạt được*:
     - Hoàn thiện giao diện thương mại điện tử hiện đại trên Next.js 14, React, Tailwind CSS.
     - Triển khai hệ thống phân tán gồm 5 dịch vụ độc lập kết nối qua Docker Compose: Frontend, Gateway API, Recommender Service, RAG Service, PostgreSQL và Qdrant.
     - Luồng trải nghiệm người dùng hoạt động trơn tru: từ duyệt hàng, xem sản phẩm "Dành riêng cho bạn", quản lý giỏ hàng đến tương tác đa phương thức với Chatbot.
   - *Đánh giá*: **Hoàn thành tốt**.

---

### 1.2. Bảng đối chiếu yêu cầu tối thiểu và kết quả thực hiện

Bảng tổng hợp đối chiếu giữa yêu cầu tối thiểu của đồ án tốt nghiệp đại học ngành Công nghệ Thông tin và kết quả thực hiện thực tế của đề tài:

| STT | Nội dung / Yêu cầu | Mức yêu cầu tối thiểu | Kết quả thực hiện thực tế trong đề tài | Đánh giá |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **Chuẩn bị dữ liệu** | Có bộ dữ liệu phù hợp, làm sạch và chia train/test hợp lệ. | Bộ Amazon Reviews 2023 chuẩn hóa `balanced_u5_i2_v1` (21.690 users, 32.557 items, 198.200 interactions). Lọc K-core lặp và chia tách nhân quả theo dòng thời gian nghiêm ngặt, chống rò rỉ 100%. | **Đạt** |
| **2** | **Xây dựng mô hình** | Xây dựng được mô hình giải quyết bài toán đặt ra. | Xây dựng thành công mô hình Two-Stage: User Tower (SASRec kết hợp biểu diễn sản phẩm lai CLIP) và Residual Listwise Reranker. | **Đạt** |
| **3** | **Đánh giá mô hình** | Có các chỉ số đánh giá định lượng khoa học, rõ ràng. | Đánh giá Full-Ranking trên toàn bộ catalog không lấy mẫu: Candidate Recall, HitRate@K, NDCG@K ($K \in \{10, 50, 100, 1000\}$). | **Đạt** |
| **4** | **So sánh & Cải tiến** | Có phân tích, so sánh đối chứng hoặc thử nghiệm thành phần. | Thực hiện Ablation Study chuyên sâu: so sánh CF thuần vs CF+Image vs CF+Text vs Full Multimodal; so sánh 4 chiến lược ứng viên; chứng minh tính trực giao 76.8% so với Popularity. | **Vượt** |
| **5** | **Xây dựng hệ thống** | Có ứng dụng demo giao diện web hoặc hệ thống API minh họa. | Xây dựng hoàn chỉnh ứng dụng web Next.js 14 kết nối Backend FastAPI microservices, CSDL PostgreSQL và Qdrant Vector DB đóng gói Docker. | **Đạt** |
| **6** | **Chatbot RAG** | Chatbot truy xuất thông tin trước khi trả lời, giảm ảo giác. | Hệ thống Closed-domain RAG chuyên biệt cho TMĐT thời trang, hỗ trợ tìm kiếm bằng ảnh/chữ, tinh chỉnh tức thời trên candidate pool, 47/47 unit tests đạt, ảo giác giá bán bằng 0%. | **Đạt** |
| **7** | **Khả năng tái lập** | Có mã nguồn rõ ràng, lưu trữ cấu hình thực nghiệm. | Lưu trữ chi tiết tệp cấu hình YAML, seed cố định 20260813, checkpoint tối ưu và manifest tệp kèm mã băm SHA256/MD5 bất biến. | **Vượt** |

---

## 2. Hạn chế và phạm vi chưa hoàn thành

Mặc dù đạt được những kết quả rất tích cực, đề tài vẫn còn tồn tại một số hạn chế khách quan cần được nhìn nhận trung thực:

### 2.1. Hạn chế của mô hình gợi ý
1. **Giới hạn trần hiệu năng từ tầng Candidate Retrieval**:
   - Hiệu năng của mạng Reranker bị chặn trên bởi Candidate Recall của tầng truy hồi ($34,444\%$). Nếu sản phẩm người dùng thực sự muốn mua không lọt vào danh sách 2.000 ứng viên ban đầu, Reranker dù phức tạp đến đâu cũng không thể đề xuất sản phẩm đó vào Top-10.
2. **Dữ liệu đánh giá thưa thớt so với hành vi thực tế**:
   - Bộ dữ liệu Amazon Reviews chỉ ghi nhận hành vi khi người dùng đã mua và để lại đánh giá (Explicit Review). Trong môi trường vận hành thực tế, luồng dữ liệu clickstream (lượt xem, thời gian dừng trang, thêm vào yêu thích) có tần suất dày đặc hơn nhiều. Do đó, mô hình chưa được thử nghiệm với các luồng phản hồi ngầm thời gian thực.
3. **Chưa tiến hành thử nghiệm trực tuyến (Online A/B Testing)**:
   - Các kết quả trong đồ án đều dựa trên đánh giá ngoại tuyến (*Offline Evaluation*) trên dữ liệu lịch sử. Tỷ lệ nhấp chuột thực tế (*Click-Through Rate - CTR*) và tỷ lệ chuyển đổi đơn hàng (*Conversion Rate*) cần được kiểm chứng thông qua môi trường triển khai thực tế với người dùng thật.

### 2.2. Hạn chế của Chatbot RAG
1. **Tiến độ lập chỉ mục dữ liệu đánh giá (Review Indexing)**:
   - Do tài nguyên phần cứng máy trạm cá nhân bị giới hạn (GPU 4 GB VRAM), hệ thống mới chỉ hoàn thành lập chỉ mục vector cho **77.824 / 295.383** đánh giá tiêu biểu nhất. Đối với những sản phẩm nằm ở vùng cực đuôi (ít đánh giá), chatbot đôi khi phải thông báo "Chưa có đủ nhận xét" do chưa nạp dữ liệu.
2. **Độ trễ xử lý của mô hình nhúng trên CPU**:
   - Nhánh Text Encoder của Jina CLIP v2 sử dụng kiến trúc XLM-RoBERTa đa ngôn ngữ tương đối nặng. Khi chạy trên môi trường CPU thuần, thời gian trích xuất vector truy vấn mất khoảng 18–20 giây. Hệ thống bắt buộc phải được cấp phát tài nguyên GPU hoặc tối ưu hóa lượng tử hóa (*Quantization*) để đạt độ trễ lý tưởng trong môi trường sản xuất.
3. **Thử nghiệm với mô hình ngôn ngữ lớn (LLM)**:
   - Trong quá trình phát triển cục bộ, cơ chế kiểm duyệt và chống ảo giác được kiểm chứng tuyệt đối thông qua bộ kiểm thử tự động (Mock LLM) và bộ sinh mẫu (*Validated Template Generator*). Việc kết nối với các LLM mã nguồn mở cục bộ (như Qwen2.5-7B qua Ollama) đòi hỏi dung lượng RAM/VRAM lớn hơn cấu hình máy trạm hiện tại.

---

## 3. Khai báo chi tiết việc sử dụng AI trong đồ án

Tuân thủ quy định về liêm chính học thuật, bảng dưới đây kê khai chi tiết các hạng mục công việc có sử dụng sự hỗ trợ của các công cụ AI trong suốt quá trình thực hiện đề tài:

| STT | Nội dung có dùng AI hỗ trợ | Công cụ AI sử dụng | Mức độ hỗ trợ | Trách nhiệm và kiểm chứng của sinh viên | Minh chứng trong đồ án |
| :---: | :--- | :---: | :--- | :--- | :---: |
| **1** | **Tham khảo cú pháp và cấu trúc mã nguồn** | ChatGPT-4o, Antigravity IDE | Sinh các đoạn mã khung (Boilerplate) cho Pydantic schemas, FastAPI routers và cấu hình Docker Compose. | Sinh viên tự viết toàn bộ logic xử lý chính, thuật toán K-core, kiến trúc Two-Tower, Reranker và bộ quy tắc Intent. | Toàn bộ mã nguồn trong thư mục `hm/src/datn/` và `apps/` |
| **2** | **Xử lý xung đột thư viện trên Kaggle** | ChatGPT-4o | Gợi ý đoạn mã monkey-patch sửa lỗi buffer RoPE của thư viện `transformers 5.3.0` và lỗi xung đột kernel `torchvision`. | Sinh viên phân tích cơ chế bộ đệm trong PyTorch, kiểm chứng vector trích xuất không bị `NaN`/`Inf`, đảm bảo độ dài $L_2 = 1.0$. | Tài liệu [multimodal_embeddings_report.md](../../legacy/docs/multimodal_embeddings_report.md) mục 2.2 |
| **3** | **Tạo khung giao diện Web cơ bản** | Antigravity IDE | Hỗ trợ tạo cấu trúc các component React và class định kiểu Tailwind CSS cho khung chat và thẻ sản phẩm. | Sinh viên tái cấu trúc giao diện, kết nối API state management, hydrate dữ liệu từ backend và xử lý luồng sự kiện giỏ hàng. | Thư mục `hm/apps/web/` |
| **4** | **Rà soát ngôn ngữ và chuẩn hóa báo cáo** | Gemini / Antigravity | Gợi ý cách diễn đạt tiếng Việt học thuật, kiểm tra lỗi chính tả và định dạng bảng biểu Markdown. | Sinh viên tự tổng hợp số liệu thực tế từ manifest, tự chịu trách nhiệm về toàn bộ nội dung học thuật và lập luận khoa học. | Các tài liệu trong thư mục `docs/LuanVan/` |

> [!IMPORTANT]
> **Cam kết của sinh viên**: Toàn bộ các kết quả thực nghiệm, số liệu trong các bảng biểu và kết luận khoa học của đồ án đều xuất phát từ dữ liệu chạy thực tế, không có bất kỳ số liệu nào do công cụ AI tự sinh hoặc ngụy tạo.

---

## 4. So sánh nội dung thực tập và yêu cầu trong cẩm nang đồ án

Đối chiếu với các yêu cầu quy định trong Cẩm nang Hướng dẫn Đồ án Tốt nghiệp của Khoa Công nghệ Thông tin – Trường Đại học Mỏ - Địa chất:

1. **Về tính thực tiễn và quy mô đề tài**:
   - Đề tài giải quyết bài toán gợi ý sản phẩm và trợ lý mua sắm đàm thoại – một trong những hướng ứng dụng có giá trị kinh tế và tính thời sự cao nhất trong ngành công nghiệp thương mại điện tử hiện nay.
   - Dữ liệu thực nghiệm có quy mô lớn ($> 32.000$ sản phẩm, gần $200.000$ tương tác, mở rộng tìm kiếm trên $152.000$ vector), vượt xa các bài toán mẫu ở mức độ bài tập lớn thông thường.

2. **Về hàm lượng khoa học và kỹ thuật**:
   - Làm chủ và ứng dụng thành công các công nghệ học sâu tiên tiến nhất hiện nay: Gợi ý tuần tự với Self-Attention (SASRec), Học biểu diễn liên kết đa phương thức (CLIP), Cơ chế học phần dư trên hard-negatives (Residual Listwise Reranker), Tìm kiếm láng giềng gần xấp xỉ trên cơ sở dữ liệu vector chuyên dụng (Qdrant HNSW) và Kiến trúc RAG kiểm soát ảo giác.
   - Có phương pháp nghiên cứu rõ ràng, thực hiện đầy đủ các nghiên cứu loại trừ thành phần (Ablation Study) để bảo vệ các quyết định thiết kế kiến trúc.

3. **Về sản phẩm phần mềm bàn giao**:
   - Không dừng lại ở các đoạn mã thử nghiệm trên Jupyter Notebook, đề tài đã đóng gói mã nguồn thành một thư viện Python chuẩn (`hm/src/datn`), cung cấp giao diện dòng lệnh CLI chuyên nghiệp (`datn-balanced-data`, `datn-user-tower`, `datn-retrieval`).
   - Xây dựng hoàn chỉnh hệ thống ứng dụng web thương mại điện tử đa dịch vụ có khả năng triển khai thực tế bằng Docker Compose.

---

## 5. Hướng phát triển trong tương lai

Từ các kết quả đạt được và những hạn chế đã được nhận diện, đề tài đề xuất các hướng mở rộng và hoàn thiện trong tương lai:

1. **Nâng cấp tầng Reranker bằng mô hình Cross-Encoder đa phương thức**:
   - Nghiên cứu áp dụng kiến trúc Cross-Attention tương tác sâu giữa vector ngữ cảnh người dùng và từng sản phẩm ứng viên, kết hợp thêm các đặc trưng thuộc tính danh mục và thương hiệu để đẩy mạnh hơn nữa chỉ số HitRate@10 và NDCG@10.
2. **Tối ưu hóa tốc độ suy luận mô hình (Inference Optimization)**:
   - Áp dụng các kỹ thuật lượng tử hóa trọng số (Quantization FP16 / INT8) và tối ưu hóa đồ thị tính toán thông qua TensorRT hoặc ONNX Runtime cho mạng Jina CLIP v2, giúp giảm thời gian mã hóa trên CPU xuống dưới 200ms, loại bỏ nút cổ chai độ trễ khi chạy trên máy chủ không có GPU chuyên dụng.
3. **Hoàn thiện lập chỉ mục toàn bộ kho đánh giá khách hàng**:
   - Sử dụng hạ tầng điện toán đám mây mạnh hơn để hoàn tất lập chỉ mục cho toàn bộ 295.383 đánh giá còn lại, giúp Chatbot có khả năng giải thích và so sánh chi tiết trên 100% danh mục sản phẩm.
4. **Tích hợp tín hiệu tương tác thời gian thực (Real-time Clickstream)**:
   - Bổ sung cơ chế cập nhật trạng thái User Tower tức thời (*Online Inference*) ngay khi người dùng thực hiện hành vi click hoặc xem sản phẩm trong phiên duyệt web, giúp danh sách gợi ý phản ánh chuyển biến tâm lý khách hàng ngay trong tích tắc.
5. **Mở rộng sang các danh mục sản phẩm khác**:
   - Áp dụng quy trình kỹ thuật của đề tài sang các ngành hàng thương mại điện tử khác như Đồ điện tử (Electronics), Đồ gia dụng (Home & Kitchen) hoặc Sách (Books) để kiểm chứng tính tổng quát hóa của giải pháp.
