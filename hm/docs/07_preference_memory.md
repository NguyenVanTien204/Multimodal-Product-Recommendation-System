# Bộ nhớ sở thích tương tác (Interactive Preference Memory)

Mục tiêu: để chatbot không chỉ *tìm* mà còn *học gu* người dùng qua hội thoại, và gu đó đi vào đúng hai tầng của hệ gợi ý hai giai đoạn (retrieval + reranking). Không huấn luyện lại mô hình nào.

## Luồng dữ liệu

```
Người dùng: "không thích sản phẩm 2" / nút 👍👎 / xem, click
        │
        ▼
 RAG (orchestrator)  ──ghi──▶  PreferenceEvent(sku, kind, ts, event_id)   ← session (RAM, TTL 1h)
        │                                   │
        │  trả về events mới của lượt này   │
        ▼                                   ▼
 Backend (Postgres: preference_events)  ◀── lưu bền, idempotent theo (user_id, event_id)
        │
        └─ mỗi request /chat gửi lại events 90 ngày gần nhất ──▶ RAG dựng PreferenceProfile
```

RAG giữ nguyên tính không trạng thái giữa các phiên; backend là kho bền duy nhất. `event_id` do RAG sinh nên một sự kiện vừa nằm trong session vừa được backend gửi lại chỉ tính một lần.

## Cách hồ sơ được tính (`hm/src/datn/agent/preferences.py`)

| Loại | Trọng số | Ghi chú |
|---|---|---|
| purchase | +1.0 | reset: thay mọi tín hiệu cũ của item |
| cart | +0.8 | cộng dồn |
| like | +0.7 | reset |
| click | +0.3 | cộng dồn (hỏi/so sánh/tìm giống) |
| view | +0.1 | cộng dồn, một mình không đủ thành "positive" |
| dislike | −1.0 | reset |

Mỗi sự kiện suy giảm theo nửa đời 30 ngày. Điểm ròng ≥ 0.25 là *positive*, ≤ −0.25 là *negative*. "Đổi ý" luôn thắng quá khứ (like sau dislike → positive).

## Hồ sơ đi vào retrieval và reranking như thế nào

- **Positives** → lịch sử đầu vào của User Tower + Reranker (sau giỏ hàng/đơn hàng của shop, cũ trước mới sau). Đây là đường duy nhất mà "thích" ảnh hưởng đến mô hình.
- **Negatives** → (1) loại cứng khỏi mọi kết quả (`exclude_product_ids` trong Qdrant filter, và `filters.matches` ở nhánh recommend); (2) phạt mềm các sản phẩm gần nó: `score −= 0.6 / (60 + rank)` với `rank` là thứ hạng láng giềng gần nhất trong không gian CLIP (tối đa 4 negative gần nhất, 30 láng giềng mỗi cái, cùng thang RRF với điểm cộng cá nhân hoá).
- Dislike còn làm danh sách đang xem được làm mới ngay (refine trên pool đã có, không gọi lại encoder).

## Giao diện hội thoại

"thích cái 2", "không thích sản phẩm 3", "cái 4 xấu quá", "I don't like #3". Phải có tham chiếu tới kết quả đang hiển thị; "không thích màu đỏ" vẫn là sở thích danh mục, không phải phản hồi. Tham chiếu mơ hồ ("không thích 9" khi chỉ có 5 kết quả) bị hỏi lại, không đoán.

API cho giao diện: `POST /chat` với `action={"type":"feedback","kind":"like|dislike","product_id":..}`; `POST /me/preferences/events` (view/click/like/dislike), `GET /me/preferences` (cái bot đang nhớ), `DELETE /me/preferences[/{product_id}]` (quên).

## Thí nghiệm mô phỏng offline (`hm/scripts/sim_preference_feedback.py`)

Người dùng mô phỏng bắt đầu từ lịch sử mua thật; mỗi vòng được xem 12 sản phẩm. Sản phẩm nằm trong các lần mua tương lai (tập test) được *thích* với xác suất `p_like`; sản phẩm khác bị *không thích* với xác suất `p_dislike`. Ba nhánh dùng chung mọi thứ, chỉ khác cách dùng phản hồi: **none** / **like** (thêm vào lịch sử tower) / **like+dislike** (thêm phạt láng giềng). Mọi nhánh đều ẩn sản phẩm đã hiện, nên chênh lệch chỉ đến từ việc dùng phản hồi. Phản ứng giả lập tính theo (seed, user, item) nên các nhánh được ghép cặp; báo cáo chênh lệch recall tích luỹ theo vòng kèm CI bootstrap ghép cặp.

## Kết quả mô phỏng (06/10/2026, VM Azure, 2.799 người dùng test, 5 vòng × 12 sản phẩm, p_like 0.8, p_dislike 0.3, seed 0)

Recall tích luỹ của các lần mua tương lai được hiển thị (`hm/reports/feedback_sim_full.json`):

| Nhánh | vòng 1 | vòng 3 | vòng 5 |
|---|---|---|---|
| none | 0.0646 | 0.1135 | 0.1479 |
| like | 0.0646 | 0.1163 | 0.1513 |
| like+dislike | 0.0646 | 0.1140 | 0.1497 |

Chênh lệch ghép cặp ở vòng 5 (CI 95% bootstrap): **like − none = +0.0034 [+0.0015, +0.0055]** (có ý nghĩa nhưng nhỏ, ≈ +2.3% tương đối); **like+dislike − like = −0.0016 [−0.0034, −0.0001]** (phạt láng giềng làm *giảm* nhẹ recall); hit-rate cuối không đổi (0.294 / 0.294 / 0.293).

Đọc kết quả cho đúng:
- Phản hồi "thích" cải thiện nhẹ nhưng ổn định; mỗi người chỉ thích trung bình 0.33 món trong 5 vòng nên lịch sử của mô hình gần như không đổi. Đây là giới hạn của mô phỏng (tỉ lệ mua rất thưa), không phải của cơ chế.
- Trong mô phỏng, "không thích" được rút từ các món *không mua*, vốn không mang thông tin về gu thật; vì vậy phạt láng giềng không thể tăng recall và còn kéo xuống vài món đúng nằm gần món bị loại. Giá trị của tín hiệu âm nằm ở trải nghiệm (không lặp lại, đổi hướng ngay) hơn là ở recall, và cần khảo sát người dùng thật để đo.
- Chưa quét trọng số phạt (`NEGATIVE_WEIGHT`=0.6 là giá trị đặt tay, cùng thang với điểm cộng cá nhân hoá). Nếu quét, phải báo cáo như phân tích độ nhạy, không chọn giá trị tốt nhất rồi coi là kết quả.

## Giới hạn cần nói trung thực trong luận văn

- Ground truth là hành vi mua, không phải ý kiến: "không mua" không đồng nghĩa "không thích", nên `p_dislike` là giả định mô phỏng, cần quét độ nhạy.
- Mô phỏng không thay thế khảo sát người dùng thật; chỉ cho thấy cơ chế có hướng cải thiện đúng.
- Reranker huấn luyện trên lịch sử mua, còn click/like khác phân phối; vì vậy chỉ tiêm lúc suy luận, không retrain.
- Màn hình "for-you" của trang chủ shop chưa dùng bộ nhớ này (mới áp dụng trong chat).
- Đánh giá trên review là dữ liệu minh hoạ (H&M không có review thật).
