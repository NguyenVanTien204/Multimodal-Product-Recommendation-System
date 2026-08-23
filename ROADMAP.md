# Lộ trình Phát triển Hệ thống (ROADMAP.md)

Tài liệu này phác thảo lộ trình phát triển chi tiết cho dự án **"Xây dựng hệ thống gợi ý sản phẩm đa phương thức ứng dụng RAG"**. Lộ trình bao gồm các công việc từ xây dựng nền tảng dữ liệu, trích xuất đặc trưng (embeddings), huấn luyện mô hình gợi ý, cho đến đóng gói dịch vụ API và phát triển hệ thống giao diện bọc ngoài (Wrapper System).

---

## Tổng quan Lộ trình Triển khai (Roadmap Overview)

Hệ thống được phát triển qua 6 giai đoạn (Phases) kết nối chặt chẽ với nhau:

```mermaid
gantt
    title Lộ trình Phát triển Hệ thống Gợi ý Đa phương thức RAG
    dateFormat  YYYY-MM-DD
    section Dữ liệu & ETL
    Pha 1: Thu thập & ETL Dữ liệu nền tảng         :active, p1, 2026-08-01, 14d
    section Mô hình & Trích chọn
    Pha 2: Trích chọn Embeddings & Train CF Baseline :active, p2, after p1, 14d
    section Retrieval & Indexing
    Pha 3: Xây dựng Hybrid Retrieval & FAISS Index  : p3, after p2, 14d
    section RAG & Tương tác
    Pha 4: Bộ máy RAG & Hội thoại (Conversational)   : p4, after p3, 14d
    section Đóng gói & Giao diện
    Pha 5: API FastAPI & Streamlit Frontend Wrapper: p5, after p4, 14d
    section Đánh giá & Tối ưu
    Pha 6: Đánh giá Hệ thống & Viết Luận văn        : p6, after p5, 14d
```

---

## Chi tiết các Giai đoạn Triển khai (Detailed Phases)

### 🟢 Pha 1: Xây dựng nền tảng dữ liệu thô & ETL Pipeline (Data Ingestion & ETL)
*Trạng thái: Đã hoàn thành (DONE)*

Giai đoạn này giải quyết bài toán làm sạch dữ liệu lớn của Amazon Reviews 2023 (`Clothing_Shoes_and_Jewelry`) đảm bảo không bị tràn bộ nhớ (OOM) và tránh rò rỉ dữ liệu (data leakage).

*   **Các nội dung đã thực hiện:**
    *   [x] Thiết lập kịch bản tải dữ liệu tự động cho Amazon Reviews (interactions & metadata).
    *   [x] Xây dựng bộ lọc dữ liệu thông minh qua kỹ thuật **Iterative K-Core Lọc** để loại bỏ các user/item quá thưa thớt (sparse), giữ lại tập con chất lượng: $30k-50k$ users và $20k-40k$ items.
    *   [x] Thiết kế pipeline ETL streaming bằng **Polars** ghi file Parquet theo từng phân đoạn (parts) và gộp lại để tối ưu hóa RAM.
    *   [x] Áp dụng chiến lược chia tập dữ liệu theo thời gian **Chronological Leave-Last-Out split** (chia Train, Valid, Test theo thứ tự thời gian mua hàng của từng user).
    *   [x] Kiểm tra liên kết dữ liệu qua `parent_asin` và xuất báo cáo chất lượng dữ liệu bằng **DuckDB** để sinh tệp manifest (`dataset_manifest.json`).
*   **Kết quả đầu ra:** 
    *   [`data/train.parquet`](file:///d:/WorkSpace/Work/DATN/data/train.parquet) (Tập huấn luyện tương tác).
    *   [`data/valid.parquet`](file:///d:/WorkSpace/Work/DATN/data/valid.parquet) (Tập validation chọn hyperparameter).
    *   [`data/test.parquet`](file:///d:/WorkSpace/Work/DATN/data/test.parquet) (Tập test báo cáo cuối kỳ).
    *   [`data/items.parquet`](file:///d:/WorkSpace/Work/DATN/data/items.parquet) (Thông tin chi tiết và metadata sản phẩm).

---

### 🟡 Pha 2: Trích chọn Đặc trưng Đa phương thức & Huấn luyện Collaborative (Embeddings & CF Models)
*Trạng thái: Đang triển khai (IN PROGRESS)*

Trích xuất các đặc trưng ngữ nghĩa từ hình ảnh, văn bản sản phẩm và học vector biểu diễn hành vi người dùng (collaborative signal).

*   **Các nội dung đang thực hiện (Dựa trên Notebooks):**
    *   [/] Viết mã trích xuất đặc trưng hình ảnh bằng pretrained **CLIP-ViT** (`kaggle_image_embeddings.ipynb`).
    *   [/] Viết mã trích xuất đặc trưng văn bản sản phẩm (Title + Description + Features) bằng pretrained **CLIP-Text** hoặc **Sentence-Transformers** (`kaggle_text_embeddings.ipynb`).
    *   [/] Huấn luyện mô hình Collaborative Filtering (CF) như **BPR / Matrix Factorization** (hoặc mở rộng lên **LightGCN** qua PyTorch/RecBole) trên tập `train.parquet` để học hành vi tương tác ngầm của người dùng.
    *   [/] Áp dụng các phép biến đổi hình học (Projection Layers) để đưa vector biểu diễn Content (Ảnh, Text) và Collaborative (CF) về chung một không gian biểu diễn (Joint Latent Space).
*   **Kết quả kỳ vọng:**
    *   Các tệp lưu trữ vector: `image_embeddings.npy`, `text_embeddings.npy`, `collaborative_embeddings.npy`.
    *   Checkpoint mô hình CF được lưu trữ trong thư mục checkpoint dự án.

---

### ⚪ Pha 3: Xây dựng cơ chế Tìm kiếm lai & Indexing (FAISS Index & Hybrid Retrieval)
*Trạng thái: Chưa thực hiện (PLANNED)*

Xây dựng cơ sở dữ liệu vector và bộ máy tìm kiếm sản phẩm tương tự phục vụ quá trình sinh ứng viên gợi ý nhanh (Candidate Generation).

*   **Các nội dung cần thực hiện:**
    *   [ ] Lập chỉ mục Vector **FAISS** (`products.faiss`) cho các vector sản phẩm đa phương thức đã chuẩn hóa.
    *   [ ] Thiết lập thuật toán **Late Fusion** kết hợp điểm số tương đồng từ 3 nguồn:
        $$\text{Score} = \lambda \cdot \text{Similarity}_{\text{CF}} + (1-\lambda) \cdot (\alpha \cdot \text{Similarity}_{\text{Image}} + \beta \cdot \text{Similarity}_{\text{Text}})$$
    *   [ ] Tích hợp cơ chế **Hybrid Retrieval**: Cho phép người dùng kết hợp tìm kiếm ngữ nghĩa kèm lọc các thuộc tính cứng (Metadata Filtering) như khoảng giá, danh mục bằng DuckDB trên dữ liệu Parquet.
*   **Kết quả kỳ vọng:**
    *   Index file: `products.faiss` được tối ưu hóa cho tìm kiếm cosine similarity.
    *   Module truy xuất: `src/datn/retrieval/` hỗ trợ tìm kiếm kết hợp (hybrid search).

---

### ⚪ Pha 4: Bộ máy RAG & Giao tiếp Hội thoại (RAG Engine & Conversational Logic)
*Trạng thái: Chưa thực hiện (PLANNED)*

Xây dựng cấu phần cốt lõi của RAG: Truy xuất thông tin bổ trợ (nhận xét của người dùng, chi tiết sản phẩm) và điều phối tương tác qua LLM.

*   **Các nội dung cần thực hiện:**
    *   [ ] Cắt nhỏ nhận xét sản phẩm (reviews) thành các chunks và lập chỉ mục vector FAISS (`reviews.faiss`).
    *   [ ] Thiết kế prompt ngữ cảnh (Prompt Engineering): Tự động trích xuất các review hữu ích và metadata của Top-K sản phẩm được gợi ý để đưa vào LLM.
    *   [ ] Xây dựng bộ quản lý phiên hội thoại (`Session State Manager`): Lưu trữ cấu hình sở thích của người dùng và lịch sử chat.
    *   [ ] Xây dựng module phân tích ý định (`Intent Parser`): Chuyển câu lệnh chat tiếng Việt tự nhiên của người dùng thành các bộ lọc tương ứng (ví dụ: *"Rẻ hơn"* $\rightarrow$ giảm `max_price`, *"Màu đỏ"* $\rightarrow$ cập nhật bộ lọc màu sắc) để xếp hạng lại sản phẩm (Reranking).
*   **Kết quả kỳ vọng:**
    *   Index file: `reviews.faiss` hỗ trợ trích xuất nhanh review liên quan.
    *   Module RAG & Agent: `src/datn/rag/` và `src/datn/agent/`.

---

### ⚪ Pha 5: Đóng gói API FastAPI & Xây dựng Giao diện Streamlit (Wrapper System)
*Trạng thái: Chưa thực hiện (PLANNED)*

Đóng gói các mô hình toán học và logic RAG thành một **Hệ thống phần mềm** chạy thực tế, cho phép người dùng cuối tương tác.

*   **Các nội dung cần thực hiện:**
    *   [ ] Thiết kế RESTful API bằng **FastAPI** phục vụ các endpoints:
        *   `POST /recommend` (Gợi ý cá nhân hóa dựa trên ID người dùng).
        *   `POST /search` (Tìm kiếm bằng text hoặc ảnh tải lên).
        *   `POST /refine` (Cập nhật sở thích hội thoại).
        *   `POST /explain` (Giải thích tại sao gợi ý sản phẩm này dựa trên RAG).
        *   `POST /compare` (So sánh ưu nhược điểm của 2 sản phẩm dựa trên nhận xét của người mua).
        *   `POST /chat` (Endpoint chính cho Agent đối thoại).
    *   [ ] Xây dựng Frontend giao diện bằng **Streamlit** (hoặc React/Next.js nếu thời gian cho phép):
        *   Tải ảnh lên trực quan $\rightarrow$ gọi API tìm kiếm.
        *   Cửa sổ chat tương tác với Chatbot RAG.
        *   Giao diện hiển thị thẻ sản phẩm (Product Cards) kèm hình ảnh, giá, thương hiệu và nút "Giải thích/So sánh".
*   **Kết quả kỳ vọng:**
    *   Backend API chạy ngầm trên cổng mặc định (ví dụ: `http://localhost:8000`).
    *   Ứng dụng Streamlit chạy trực quan trên trình duyệt cho phép người dùng click và chat.

---

### ⚪ Pha 6: Thử nghiệm, Đánh giá Hệ thống & Viết Báo cáo (System Evaluation & Thesis Hardening)
*Trạng thái: Chưa thực hiện (PLANNED)*

Chạy thực nghiệm tổng thể để viết báo cáo khoa học phục vụ bảo vệ đồ án tốt nghiệp.

*   **Các nội dung cần thực hiện:**
    *   [ ] Đánh giá độ chính xác gợi ý (Recommendation Metrics) trên tập test: So sánh BPR baseline với các mô hình fusion đa phương thức để trả lời câu hỏi nghiên cứu về vai trò của Hình ảnh/Văn bản.
    *   [ ] Chạy thử nghiệm phân rã (Ablation Study) bằng cách lần lượt tắt các modality (CF, Text, Image) để đo mức độ sụt giảm metric.
    *   [ ] Đánh giá chất lượng RAG bằng **Ragas** (Faithfulness, Answer Relevance, Context Recall).
    *   [ ] Phân tích các ca lỗi (Error Analysis) và đo đạc độ trễ hệ thống (System Latency).
    *   [ ] Rerun toàn bộ pipeline để chốt số liệu, vẽ biểu đồ và hoàn thiện luận văn tốt nghiệp.
*   **Kết quả kỳ vọng:**
    *   Bảng số liệu so sánh chi tiết và các biểu đồ trực quan.
    *   Mã nguồn dự án hoàn chỉnh, dễ tái lập (Thesis-ready).

---

## Quản trị Rủi ro và Mốc kiểm tra (Checkpoints & Governance)

*   **Safety Lock:** Không thay đổi hay làm bẩn tập dữ liệu Parquet đã đóng băng ở Pha 1. Mọi thay đổi dữ liệu phải tạo phiên bản mới trong `dataset_manifest.json`.
*   **Quy trình Độc lập:** Hệ thống API và Gợi ý của Pha 5 phải có cơ chế fallback. Nếu dịch vụ LLM bên ngoài bị ngắt kết nối, API `/recommend` vẫn phải trả về sản phẩm bình thường kèm theo lời cảnh báo lỗi RAG trên UI.
*   **Definition of Done (DoD) cho mỗi giai đoạn:** Mỗi giai đoạn hoàn thành phải đi kèm với mã nguồn nằm trong thư mục `src/` và có kịch bản chạy mẫu (Smoke Test hoặc Notebook chạy thử) để kiểm chứng, không chỉ báo cáo lý thuyết.
