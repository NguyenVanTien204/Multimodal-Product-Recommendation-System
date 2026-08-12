# Tầm nhìn sản phẩm

## 1. Tuyên bố tầm nhìn

Xây dựng một hệ thống gợi ý sản phẩm cá nhân hóa có khả năng kết hợp lịch sử tương tác của người dùng, nội dung văn bản và hình ảnh sản phẩm để đưa ra danh sách Top-K phù hợp; đồng thời cho phép người dùng tìm kiếm, tinh chỉnh, so sánh và nhận giải thích bằng ngôn ngữ tự nhiên dựa trên dữ liệu sản phẩm và đánh giá thực tế.

Hệ thống hướng tới việc trả lời câu hỏi nghiên cứu trung tâm:

> Việc kết hợp thông tin hình ảnh và văn bản với lịch sử tương tác có cải thiện chất lượng gợi ý sản phẩm, đặc biệt đối với các sản phẩm có ít dữ liệu tương tác, so với collaborative recommendation truyền thống hay không?

## 2. Vấn đề cần giải quyết

Các hệ thống collaborative filtering phụ thuộc nhiều vào lịch sử tương tác. Chúng thường gặp khó khăn khi sản phẩm có ít tương tác và không khai thác trực tiếp những tín hiệu có giá trị như kiểu dáng, màu sắc, mô tả, thuộc tính hay hình ảnh sản phẩm. Mặt khác, danh sách gợi ý truyền thống thường thiếu khả năng hội thoại và khó giải thích cho người dùng.

Đề tài giải quyết hai lớp vấn đề:

1. **Lớp nghiên cứu:** đo lường đóng góp của interaction, image và text đối với chất lượng Top-K recommendation, đặc biệt trong điều kiện dữ liệu thưa.
2. **Lớp sản phẩm:** chuyển mô hình gợi ý thành trải nghiệm tương tác, trong đó người dùng có thể diễn đạt nhu cầu, tinh chỉnh kết quả, hỏi lý do và so sánh sản phẩm.

## 3. Đối tượng sử dụng

- **Người mua hàng:** nhận gợi ý cá nhân hóa, tìm sản phẩm bằng văn bản/hình ảnh, tinh chỉnh và so sánh kết quả.
- **Nhà nghiên cứu/sinh viên:** chạy thí nghiệm, ablation, cold-start evaluation và phân tích kết quả.
- **Quản trị viên hệ thống:** quản lý dữ liệu, cấu hình mô hình, index và theo dõi trạng thái dịch vụ.

## 4. Giá trị mang lại

### Giá trị nghiên cứu

- Cung cấp so sánh có kiểm soát giữa collaborative, content-only và multimodal recommendation.
- Định lượng đóng góp riêng của image và text thông qua ablation.
- Đánh giá khả năng hỗ trợ sản phẩm ít tương tác với các ngưỡng `<= 5` và `<= 10` tương tác huấn luyện.
- Cho phép kết luận có giá trị ngay cả khi multimodal không vượt mọi baseline.

### Giá trị sản phẩm

- Cá nhân hóa dựa trên lịch sử và nhu cầu hiện tại.
- Hỗ trợ truy vấn bằng văn bản, ảnh hoặc sản phẩm tham chiếu.
- Cho phép refinement theo phiên mà không cần huấn luyện lại mô hình.
- Giải thích và so sánh dựa trên metadata/review đã truy xuất, giảm nguy cơ LLM bịa thông tin.

## 5. Trải nghiệm đích

Người dùng có thể thực hiện các yêu cầu như:

- “Gợi ý cho tôi sneaker giống những sản phẩm tôi từng thích.”
- “Tôi muốn màu đen và ít thể thao hơn.”
- “Tại sao bạn gợi ý sản phẩm thứ 2?”
- “So sánh sản phẩm 2 và 4.”
- “Tìm cái giống sản phẩm này nhưng rẻ hơn.”

Luồng trải nghiệm tổng quát:

```text
Người dùng: văn bản / ảnh / lịch sử
                 ↓
        Lớp hội thoại và ý định
                 ↓
       Multimodal Recommender
                 ↓
            Top-K sản phẩm
              ↙       ↘
      Hiển thị kết quả   RAG theo ngữ cảnh
                              ↓
                    Explain / Compare / Refine
```

## 6. Mục tiêu

### Mục tiêu bắt buộc

- Tạo dataset cố định từ Amazon Reviews 2023 — `Clothing_Shoes_and_Jewelry`.
- Xây dựng pipeline Top-K recommendation và protocol đánh giá tái lập được.
- Hoàn thành baseline Popularity và collaborative recommendation.
- Trích xuất và cache image/text embeddings.
- Xây dựng và đánh giá các biến thể content-only và multimodal fusion.
- Thực hiện ablation và đánh giá sản phẩm ít tương tác.
- Xây dựng demo end-to-end tối thiểu bằng FastAPI và Streamlit.

### Mục tiêu mở rộng

- Nhận diện preference từ hội thoại và refinement theo phiên.
- RAG giải thích và so sánh dựa trên metadata/review.
- Lightweight tool-calling agent để điều phối các chức năng.
- Nếu còn thời gian, so sánh CLIP text encoder với SentenceTransformer hoặc nâng cấp frontend.

## 7. Ngoài phạm vi

- Multi-agent, autonomous planning, self-reflection loop hoặc web-browsing agent.
- Để LLM tự chọn sản phẩm thay cho recommender.
- Dùng RAG trong quá trình huấn luyện recommender.
- Strict cold-start với item hoàn toàn chưa xuất hiện trong train.
- Hạ tầng Spark, Hadoop, Kafka, Airflow, Kubernetes, microservices hoặc data warehouse.
- Tối ưu hệ thống ở quy mô production thương mại.

## 8. Câu hỏi nghiên cứu

- **RQ1 — Multimodal:** Image + Text có cải thiện recommendation so với Collaborative Filtering không?
- **RQ2 — Ablation:** Image và Text đóng góp bao nhiêu, modality nào quan trọng hơn trong từng nhóm dữ liệu?
- **RQ3 — Sparse items:** Thông tin đa phương thức có giúp recommendation cho sản phẩm ít interaction không?

Conversational recommendation, RAG, agent và explanation là các system extension, không phải câu hỏi nghiên cứu chính.

## 9. Tiêu chí thành công

Đồ án được coi là hoàn chỉnh khi:

- Có pipeline dữ liệu và đánh giá cố định, tái lập được.
- Có kết quả định lượng cho Interaction only, Content only và Multimodal.
- Có phân tích normal recommendation và sparse-item recommendation.
- Giải thích được khi nào multimodal tốt hơn hoặc kém hơn, cũng như đóng góp của image/text.
- Có demo cho thấy kết quả Top-K có thể được kế thừa bởi conversational refinement, RAG explanation/comparison và agent.

Thành công không được định nghĩa là multimodal bắt buộc phải thắng tất cả baseline.

