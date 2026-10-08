# MỞ ĐẦU VÀ THÔNG TIN ĐỒ ÁN TỐT NGHIỆP

> **Phạm vi bộ dữ liệu (cập nhật 05/10/2026):** bộ khung luận văn này viết theo bộ **Amazon**. Bộ dữ liệu chính mới là **H&M**; luận điểm được phép khẳng định, thực nghiệm còn thiếu và ánh xạ sửa từng chương nằm ở [`../hm/05_thesis_plan.md`](../../hm/docs/05_thesis_plan.md). Các số liệu Amazon bên dưới vẫn đúng cho Amazon.

---

## 1. Thông tin chung về đồ án

| Thuộc tính | Chi tiết |
| :--- | :--- |
| **Tên đề tài** | **Xây dựng hệ thống gợi ý sản phẩm đa phương thức ứng dụng RAG** |
| **Tên tiếng Anh** | *Developing a Multimodal Product Recommendation System using RAG* |
| **Sinh viên thực hiện** | Nguyễn Văn Tiến |
| **Mã sinh viên** | 2221050201 |
| **Lớp chuyên ngành** | DCCTKH67A |
| **Khoa / Trường** | Khoa Công nghệ Thông tin – Trường Đại học Mỏ - Địa chất (HUMG) |
| **Giảng viên hướng dẫn** | Nguyễn Duy Huy |
| **Hệ đào tạo** | Chính quy |
| **Niên khóa / Năm bảo vệ** | 2022 – 2026 / Năm 2026 |
| **Email liên hệ** | `2221050201@student.humg.edu.vn` |

---

## 2. Mục tiêu và phạm vi đề tài

### 2.1. Mục tiêu đề tài
Đề tài hướng tới việc giải quyết các thách thức cố hữu của hệ thống gợi ý truyền thống trong thương mại điện tử (dữ liệu thưa thớt, khởi đầu lạnh của sản phẩm mới và thiếu tính tương tác hai chiều) thông qua việc kết hợp học máy biểu diễn đa phương thức hiện đại và mô hình tạo sinh có tăng cường truy xuất tri thức (RAG). Các mục tiêu cụ thể, đo lường được gồm:

1. **Xây dựng mô hình gợi ý sản phẩm đa phương thức hai giai đoạn (Two-Stage Recommendation)**:
   - Giai đoạn 1 (*Candidate Retrieval*): Huấn luyện mạng **User Tower** dựa trên cơ chế tự chú ý (Self-Attention / SASRec) để nắm bắt chuỗi tương tác tuần tự theo thời gian của người dùng, tích hợp đồng thời vector nhúng đa phương thức (thị giác và văn bản trích xuất từ CLIP) nhằm bù đắp tín hiệu tương tác thưa. Kết hợp chiến lược truy hồi đa nguồn (Multi-source Retrieval) để mở rộng diện bao phủ, đạt mục tiêu **Candidate Recall $\ge 30\%$** trên không gian toàn bộ danh mục sản phẩm.
   - Giai đoạn 2 (*Reranking*): Xây dựng mạng **Residual Listwise Reranker** tinh chỉnh thứ hạng danh sách ứng viên Top-K với hàm mất mát listwise softmax trên các mẫu hard-negative. Đạt mục tiêu hiệu năng trên tập kiểm thử độc lập: **HitRate@10 $\ge 3.5\%$** và **NDCG@10 $\ge 2.4\%$**.

2. **Xây dựng hệ thống Chatbot RAG hỗ trợ hội thoại mua sắm thông minh**:
   - Xây dựng trợ lý ảo vận hành theo cơ chế đóng (*Closed-domain RAG*), chỉ sử dụng kho tri thức được lập chỉ mục từ metadata, thông số kỹ thuật và đánh giá thực tế của khách hàng từ tập dữ liệu đã chuẩn hóa.
   - Hỗ trợ đầy đủ các nhóm ý định người dùng bằng ngôn ngữ tự nhiên: Tìm kiếm ngữ nghĩa (*Search*), Gợi ý cá nhân hóa (*Recommend*), Tinh chỉnh bộ lọc (*Refine*), Giải thích lý do gợi ý (*Explain*) và So sánh sản phẩm (*Compare*).
   - Thiết lập cơ chế kiểm chứng tính có căn cứ dữ liệu (*Grounding Verification*): Đảm bảo 100% câu trả lời có trích dẫn mã bằng chứng xác thực (`[P#]`, `[R#.#]`), tuyệt đối loại bỏ hiện tượng bịa đặt thông tin (Hallucination) về giá cả, thuộc tính hoặc sản phẩm không có thật trong kho dữ liệu; thời gian phản hồi ở mức chấp nhận được cho môi trường thực nghiệm.

3. **Phát triển và tích hợp ứng dụng web thực nghiệm hoàn chỉnh**:
   - Xây dựng kiến trúc hệ thống phân tán hướng dịch vụ gồm: Frontend (Next.js/React), Backend Marketplace API (FastAPI, PostgreSQL), Vector Database (Qdrant), RAG Service và Recommender Service.
   - Mang lại luồng trải nghiệm người dùng liền mạch từ việc duyệt danh mục, nhận danh sách đề xuất cá nhân hóa, quản lý giỏ hàng đến tương tác trực tiếp với Chatbot trợ lý mua sắm.

### 2.2. Phạm vi đề tài
- **Phạm vi dữ liệu thực nghiệm**: Đề tài tập trung nghiên cứu trên nhánh ngành hàng **Clothing, Shoes and Jewelry** thuộc bộ dữ liệu mở tiêu chuẩn **Amazon Reviews 2023** (phát hành bởi nhóm nghiên cứu của Giáo sư Julian McAuley, Đại học California San Diego - UCSD). Phiên bản dữ liệu thực nghiệm chính thức được chốt là `balanced_u5_i2_v1`, bao gồm:
  - **21.690** người dùng có hành vi tương tác tích cực hợp lệ.
  - **32.557** sản phẩm thương mại điện tử có đầy đủ siêu dữ liệu văn bản và hình ảnh.
  - **198.200** lượt tương tác tích cực (rating $\ge 4$).
  - Không gian danh mục mở rộng được lập chỉ mục vector phục vụ tìm kiếm đạt **152.086** sản phẩm.
- **Giới hạn tri thức của Chatbot**: Chatbot đóng vai trò trợ lý mua sắm chuyên biệt trên danh mục sản phẩm thời trang. Tri thức của hệ thống được giới hạn nghiêm ngặt trong phạm vi thuộc tính sản phẩm và phản hồi khách hàng đã được nạp vào cơ sở dữ liệu vector Qdrant; hệ thống không tự ý suy diễn hoặc tìm kiếm dữ liệu ngoại lai từ Internet.

---

## 3. Lý do chọn đề tài và tính cấp thiết

### 3.1. Bối cảnh thực tiễn của thương mại điện tử
Sự bùng nổ của thương mại điện tử toàn cầu đã đem lại sự phong phú chưa từng có về hàng hóa, song đồng thời dẫn đến hiện tượng **"quá tải thông tin" (Information Overload)**. Khách hàng thường xuyên bị choáng ngợp trước hàng vạn sản phẩm tương đồng, khiến việc tìm kiếm món đồ phù hợp trở nên tốn thời gian và làm giảm tỷ lệ chuyển đổi mua sắm. 

Thách thức này đặc biệt khắt khe trong lĩnh vực **thời trang (Fashion E-commerce)** – một ngành hàng có tính chất thẩm mỹ và cảm xúc cao:
- Khách hàng không chỉ đưa ra quyết định dựa trên các thông số kỹ thuật khô khan hay tên gọi, mà bị chi phối mạnh mẽ bởi **diện mạo thị giác** (kiểu dáng, phom mẫu, hoa văn, màu sắc) và **mô tả ngữ nghĩa** (chất liệu vải, phong cách phối đồ, dịp sử dụng).
- Các hệ thống gợi ý truyền thống chủ yếu dựa trên **Lọc cộng tác (Collaborative Filtering - CF)** thông qua ma trận tương tác User-Item (lượt xem, đánh giá, mua hàng). Dù phát huy hiệu quả khi dữ liệu dày đặc, phương pháp này lập tức thất bại trước bài toán **dữ liệu thưa thớt (Data Sparsity)** và **khởi đầu lạnh (Cold-start)**: sản phẩm mới ra mắt hoặc sản phẩm ngách (Long-tail) chưa có lượt tương tác trong lịch sử sẽ không có tín hiệu để mô hình học biểu diễn, dẫn đến việc chúng bị loại trừ hoàn toàn khỏi danh sách gợi ý.

### 3.2. Tiềm năng của học sâu đa phương thức và gợi ý tuần tự
Để vượt qua giới hạn của lọc cộng tác thuần túy, việc khai thác đồng thời các nguồn dữ liệu đa phương thức (Multimodal Information) là hướng đi tất yếu:
- **Gợi ý tuần tự (Sequential Recommendation)** với kiến trúc **SASRec** (Kang & McAuley, 2018) đã chứng minh sức mạnh của cơ chế tự chú ý (*Self-Attention*) trong việc nắm bắt sự biến chuyển sở thích và động cơ tiêu dùng ngắn hạn của khách hàng theo chuỗi thời gian thực tế.
- **Mô hình thị giác - ngôn ngữ liên kết (CLIP)** (Radford et al., 2021) mang lại bước đột phá khi ánh xạ cả hình ảnh sản phẩm và văn bản mô tả vào một không gian vector ngữ nghĩa chung (*Shared Latent Space*). Nhờ đó, ngay cả khi một sản phẩm chưa từng có lượt mua nào trong lịch sử huấn luyện, hệ thống vẫn có thể trích xuất chính xác đặc trưng nội dung để đề xuất tới người dùng có gu thẩm mỹ tương thích.

### 3.3. Nhu cầu tương tác tự nhiên và vai trò của Retrieval-Augmented Generation (RAG)
Song song với việc nhận danh sách gợi ý thụ động, người tiêu dùng ngày nay có nhu cầu tương tác linh hoạt hai chiều bằng ngôn ngữ tự nhiên: hỏi đáp thông số, yêu cầu tư vấn theo ngân sách, hỏi lý do tại sao sản phẩm này phù hợp hoặc đối chiếu ưu nhược điểm giữa hai món đồ.

Mặc dù các Mô hình Ngôn ngữ Lớn (LLM) thể hiện năng lực giao tiếp vượt trội, chúng lại đối mặt với nguy cơ nghiêm trọng là **hiện tượng ảo giác thông tin (Hallucination)**: tự bịa đặt giá cả, thuộc tính kỹ thuật hoặc tình trạng hàng tồn kho. Để ứng dụng an toàn vào thương mại điện tử, kiến trúc **Retrieval-Augmented Generation (RAG)** (Lewis et al., 2020) là giải pháp hàng đầu hiện nay. Bằng cách cưỡng chế LLM chỉ tạo phản hồi dựa trên các đoạn văn bản (metadata, reviews) được truy xuất thực tế từ cơ sở dữ liệu vector, hệ thống đảm bảo tính minh bạch, chính xác tuyệt đối và có khả năng trích dẫn dẫn chứng xác thực cho mọi nhận định.

Xuất phát từ các cơ sở khoa học và thực tiễn trên, đề tài **"Xây dựng hệ thống gợi ý sản phẩm đa phương thức ứng dụng RAG"** mang tính cấp thiết cao, kết hợp hài hòa giữa mô hình gợi ý hai giai đoạn tối ưu hiệu năng xếp hạng và trợ lý đàm thoại RAG gia tăng niềm tin của người dùng.

---

## 4. Tóm tắt nội dung chính và kế hoạch thực hiện

Quá trình thực hiện đề tài được tổ chức thành các nội dung công việc cụ thể:

1. **Thu thập, xử lý và chuẩn hóa dữ liệu**:
   - Thu thập bộ dữ liệu Amazon Reviews 2023 nhánh *Clothing_Shoes_and_Jewelry*.
   - Xây dựng quy trình làm sạch, khử trùng lặp bản ghi tương tác theo mốc thời gian, lọc gán nhãn tương tác tích cực ($rating \ge 4$) và tương tác âm mạnh ($rating \le 2$).
   - Thiết kế giải thuật lọc lặp **Iterative K-core** ($K_u = 5, K_i = 2$) để ổn định mật độ tương tác mà vẫn bảo lưu đặc trưng đuôi dài của danh mục sản phẩm.
   - Thực hiện phân chia dữ liệu theo chiến lược **Chronological leave-last-out split**, kiểm soát nghiêm ngặt ngăn chặn rò rỉ dữ liệu (*Data Leakage*), đóng băng và bảo đảm tính tái lập bằng mã băm kiểm tra checksum.

2. **Khai phá đặc trưng đa phương thức (Multimodal Embeddings)**:
   - Sử dụng mô hình liên kết thị giác - ngôn ngữ **Jina CLIP v2** (kiến trúc EVA-02 ViT-L/14) để trích xuất vector đặc trưng $D = 1024$ chiều cho toàn bộ hình ảnh và văn bản của danh mục sản phẩm.
   - Chuẩn hóa vector đơn vị $L_2 = 1.0$, xử lý ngoại lệ cho các sản phẩm thiếu ảnh bằng cơ chế fallback có gắn cờ định danh.

3. **Thiết kế và huấn luyện mô hình gợi ý hai giai đoạn**:
   - **Tầng 1 (User Tower)**: Xây dựng mạng mã hóa chuỗi hành vi tuần tự tích hợp biểu diễn sản phẩm lai ($e_i = id\_residual(i) + content\_proj(clip(i))$); huấn luyện bằng sampled-softmax loss với phân phối âm popularity-based có hiệu chỉnh log-Q.
   - **Chiến lược sinh ứng viên đa nguồn**: Kết hợp 4 nhánh ứng viên độc lập (User Tower, Global Popularity, Content Centroid, Last-item Similarity) nhằm nâng cao độ bao phủ Candidate Recall.
   - **Tầng 2 (Residual Listwise Reranker)**: Xây dựng mạng xếp hạng lại tối ưu hóa danh sách Top-K dựa trên điểm truy hồi ban đầu và phần dư residual học được từ các mẫu hard-negative.

4. **Xây dựng kho tri thức và pipeline Chatbot RAG**:
   - Lập chỉ mục không gian vector trên hệ quản trị **Qdrant Vector Database** cho cả sản phẩm (`products`) và nhận xét người dùng (`reviews`).
   - Xây dựng module nhận dạng ý định (*Intent Classifier*) hỗ trợ 9 nhóm hành động và trích xuất tham số ngữ nghĩa (màu sắc, thương hiệu, khoảng giá).
   - Thiết lập bộ kiểm duyệt căn cứ (*Grounding & Hallucination Guardrails*): kiểm tra mã trích dẫn, kiểm tra khớp giá tiền hiển thị với dữ liệu thực tế.

5. **Thực nghiệm, đánh giá định lượng và đối chiếu**:
   - Đo lường hiệu năng xếp hạng theo giao thức full-ranking trên toàn bộ danh mục sản phẩm: Candidate Recall, HitRate@K, NDCG@K ($K \in \{10, 50, 100, 1000\}$).
   - Tiến hành nghiên cứu loại bỏ thành phần (*Ablation Study*) để khẳng định vai trò then chốt của các phương thức thị giác và văn bản so với lọc cộng tác thuần túy.
   - Kiểm thử tự động hệ thống Chatbot với bộ 47 kịch bản unit test chuyên sâu.

6. **Tích hợp hệ thống ứng dụng web**:
   - Xây dựng giao diện người dùng hiện đại trên Next.js 14, React, Tailwind CSS.
   - Xây dựng hệ thống Backend phân tán qua FastAPI, kết nối CSDL PostgreSQL lưu trữ nghiệp vụ và dịch vụ Qdrant lưu trữ vector.

---

## 5. Kết quả chính đạt được

Sau quá trình nghiên cứu và triển khai thực nghiệm nghiêm túc, đề tài đã hoàn thành xuất sắc các mục tiêu đề ra với những kết quả nổi bật:

1. **Bộ dữ liệu chuẩn hóa có khả năng tái lập hoàn toàn**:
   - Xây dựng thành công bộ dữ liệu `balanced_u5_i2_v1` gồm 21.690 người dùng, 32.557 sản phẩm và 198.200 tương tác tích cực. Toàn bộ quy trình từ dữ liệu thô đến dữ liệu huấn luyện đều được cố định bằng manifest, cấu hình YAML và mã băm SHA256/MD5.

2. **Mô hình gợi ý hai giai đoạn vượt trội về hiệu năng xếp hạng**:
   - Chiến lược sinh ứng viên đa nguồn kết hợp 4 nhánh đạt **Candidate Recall = 34.444%** trên tập kiểm thử (vượt xa mục tiêu 30%), giúp phục hồi được **11.72%** các sản phẩm hoàn toàn cold-start mà mô hình cộng tác đơn lẻ không thể truy xuất.
   - Mạng **Residual Listwise Reranker** cải thiện toàn diện thứ hạng Top-K: đạt **HitRate@10 = 3.518%** (vượt chỉ tiêu 3.5%) và **NDCG@10 = 2.489%** (vượt chỉ tiêu 2.4%), tăng đáng kể so với User Tower đơn lẻ (**HitRate@10 tăng từ 3.038% lên 3.518%**, **NDCG@10 tăng từ 2.106% lên 2.489%**).
   - Phân tích trực giao chứng minh: **76.8%** lượt đoán trúng trong Top-50 của mô hình gợi ý thuộc về các sản phẩm vùng thân và đuôi dài (Mid-tail, Long-tail) mà chiến lược đề xuất theo độ phổ biến (Popularity) hoàn toàn bỏ lỡ, giúp phá vỡ "bong bóng lọc" (Filter Bubble).

3. **Hệ thống Chatbot RAG hội thoại đáng tin cậy**:
   - Triển khai thành công kiến trúc Closed-domain RAG với cơ sở dữ liệu vector Qdrant.
   - Vượt qua **47/47 kịch bản kiểm thử tự động**, hỗ trợ mượt mà các tác vụ tìm kiếm đa phương thức (chữ và ảnh), gợi ý theo gu lịch sử, tinh chỉnh bộ lọc tức thì trên vùng đệm ứng viên, giải thích và so sánh sản phẩm kèm trích dẫn dẫn chứng xác thực.
   - Đảm bảo **100% câu trả lời có trích dẫn dữ liệu hợp lệ**, kiểm soát nghiêm ngặt không sinh ảo giác về thông tin giá bán hay thuộc tính.

4. **Ứng dụng web thử nghiệm trực quan, hoàn chỉnh**:
   - Hệ thống web hoạt động ổn định trên kiến trúc microservices với Docker Compose, giao diện trực quan, kết nối liền mạch giữa hiển thị sản phẩm, giỏ hàng thương mại điện tử và khung trò chuyện trợ lý ảo thông minh.

---

## 6. Bố cục của đồ án tốt nghiệp

Báo cáo đồ án tốt nghiệp được tổ chức thành 3 chương chính cùng phần Mở đầu, Kết luận và Phụ lục:

- **Mở đầu**: Giới thiệu thông tin đề tài, mục tiêu, phạm vi nghiên cứu, tính cấp thiết và tóm tắt các kết quả đạt được.
- **Chương 1: Tổng quan lý thuyết về lĩnh vực nghiên cứu**: Trình bày tổng quan bài toán gợi ý sản phẩm, cơ sở lý thuyết về dữ liệu, gợi ý tuần tự (SASRec), biểu diễn đa phương thức (CLIP), kiến trúc gợi ý hai giai đoạn, nguyên lý Retrieval-Augmented Generation (RAG), các phương pháp đánh giá định lượng và tổng quan công nghệ sử dụng.
- **Chương 2: Quy trình xây dựng hệ thống gợi ý sản phẩm đa phương thức và Chatbot RAG**: Mô tả chi tiết quy trình xử lý dữ liệu thực nghiệm Amazon Reviews 2023, kiến trúc kỹ thuật User Tower & Residual Reranker, cơ chế tìm kiếm lai và trích xuất bằng chứng của Chatbot RAG, thiết kế kiến trúc phần mềm, cơ sở dữ liệu, API và giao diện người dùng.
- **Chương 3: Thực nghiệm và đánh giá hệ thống**: Trình bày chi tiết môi trường phần cứng/phần mềm, thiết lập siêu tham số, bảng số liệu thực nghiệm định lượng của mô hình gợi ý, phân tích các nghiên cứu thành phần (Ablation Study), kết quả kiểm thử chức năng của Chatbot RAG, đánh giá hệ thống web và khai báo minh bạch việc sử dụng công cụ AI.
- **Kết luận và hướng phát triển**: Tổng kết các đóng góp chính của đồ án, phân tích trung thực các hạn chế hiện tại và đề xuất các hướng nâng cấp, hoàn thiện trong tương lai.
- **Tài liệu tham khảo & Phụ lục**: Danh mục các công trình khoa học đã trích dẫn và các tài liệu kỹ thuật liên quan.
