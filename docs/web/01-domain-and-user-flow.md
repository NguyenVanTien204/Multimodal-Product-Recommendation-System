# Phân tích nghiệp vụ web

## 1. Bài toán và phạm vi

ShopSense là storefront demo cho nhóm sản phẩm Clothing, Shoes & Jewelry của Amazon Reviews 2023. Mục tiêu trải nghiệm là rút ngắn hành trình khám phá sản phẩm bằng hai năng lực của đề tài:

1. Gợi ý tuần tự: dùng lịch sử tương tác, truy vấn và preference trong phiên để tạo/rerank Top-K.
2. Chatbot mua hàng agentic: nhận yêu cầu tiếng Việt, chọn tool được phép (`recommend`, `search`, `refine`, `explain`, `compare`) và trả lời có căn cứ.

“Agentic” nghĩa là điều phối tool có kiểm soát; agent không tự tạo giá, thuộc tính, review hoặc danh sách sản phẩm.

## 2. Tác nhân và mục tiêu

| Tác nhân | Mục tiêu | Quyền trong web |
|---|---|---|
| Người mua demo | Khám phá, lọc, nhận gợi ý, hỏi và so sánh | Tìm kiếm, chat, chọn/so sánh sản phẩm |
| Recommender service | Sinh và xếp hạng candidate | Quyết định ranking, fallback popularity |
| Agent/RAG service | Hiểu ý định và diễn đạt từ evidence | Điều phối tool, giải thích/so sánh có nguồn |
| Quản trị viên/nghiên cứu viên | Kiểm tra trạng thái artifact | Chỉ xem trạng thái/version qua API nội bộ |

## 3. Luồng nghiệp vụ chính

### A. Khám phá và gợi ý tuần tự

1. Người mua mở trang chủ, nhập `user_id` demo hoặc bắt đầu không có lịch sử.
2. Web gọi `/recommend`; backend dùng history + preference hiện hành để lấy Top-K.
3. Người mua mở sản phẩm, lưu hai sản phẩm để so sánh, hoặc gửi yêu cầu tiếp theo.
4. Với câu như “rẻ hơn, màu đen”, agent/parser tạo delta preference.
5. Web gọi `/refine`; backend merge state phiên và rerank, không retrain model.
6. Web hiển thị kết quả mới, chip filter đã hiểu và lý do/fallback nếu có.

### B. Hỏi đáp hỗ trợ mua hàng

1. Người mua chat bằng ngôn ngữ tự nhiên.
2. `/chat` nhận `session_id`, text và context kết quả hiện tại.
3. Agent chọn tool hợp lệ; tool trả dữ liệu có schema.
4. Nếu cần giải thích/so sánh, RAG chỉ nhận metadata/review của item liên quan.
5. Web hiển thị câu trả lời, card sản phẩm và evidence/source khi API cung cấp.

### C. Tìm kiếm trực tiếp

1. Người mua nhập mô tả, chọn category/giá hoặc chọn “tương tự sản phẩm này”.
2. Web gửi `/search`; backend parse hard constraints, retrieval và rerank.
3. Trường dữ liệu thiếu được hiển thị “Chưa có dữ liệu”, không coi là thỏa filter.

## 4. Quy tắc UX và nghiệp vụ web

| ID | Quy tắc |
|---|---|
| WBR-01 | Mỗi card phải có `item_id`, title, rank và chỉ hiển thị price/brand khi API cung cấp. |
| WBR-02 | Kết quả fallback phải được gắn nhãn rõ là “phổ biến”/“demo”, không gọi là cá nhân hóa. |
| WBR-03 | Chat không tự đổi kết quả đang xem; chỉ update khi response trả `products` hoặc `action_result`. |
| WBR-04 | Chỉ cho phép explain item thuộc kết quả hiện tại và compare từ 2 item hợp lệ. |
| WBR-05 | Hiển thị preference đã hiểu để người dùng phát hiện parser hiểu sai. |
| WBR-06 | Lỗi LLM/agent không làm mất chức năng tìm kiếm và gợi ý trực tiếp. |
| WBR-07 | Ảnh tải lên (giai đoạn sau) phải kiểm tra loại/kích thước ở web và backend; không lưu lâu dài mặc định. |

## 5. Trang và điều hướng MVP

| Trang/khu vực | Nội dung | Năng lực đề tài |
|---|---|---|
| Trang chủ | Hero, quick query, Top-K card, filter | Recommend/Search |
| Kết quả | Danh sách card, chip preference, refine | Sequential recommendation |
| Trợ lý mua sắm | Hội thoại, action card, evidence | Agentic chat/RAG |
| So sánh | Hai card và các trường chung | RAG comparison |
| Giới thiệu | Phạm vi demo, privacy, model status | Minh bạch nghiên cứu |

## 6. Chỉ số trải nghiệm demo

- Tỷ lệ request gợi ý/search/refine/chat thành công và latency p50/p95.
- Tỷ lệ fallback và nguyên nhân.
- Tỷ lệ chat có tool call, explain/compare có evidence.
- Không dùng click/CTR của người dùng demo làm metric thay thế Recall/NDCG/HitRate nghiên cứu.
