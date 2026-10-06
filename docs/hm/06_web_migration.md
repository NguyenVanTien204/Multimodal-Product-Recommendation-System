# H&M — Chuyển web, RAG và recommender sang bộ H&M

> Cập nhật 06/10/2026. Tài liệu này ghi lại **những gì đã làm, đo được gì và còn hạn chế gì** khi chuyển phần hệ thống (cửa hàng web, chatbot RAG, dịch vụ gợi ý) từ bộ Amazon sang bộ H&M. Quyết định D2/D3/D4 của [`05_thesis_plan.md`](05_thesis_plan.md) được chốt ở đây. Quy trình vận hành: [`deploy/azure/README.md`](../../deploy/azure/README.md), mục "Chạy bản H&M". Mọi con số bên dưới lấy từ `shop_catalog_manifest.json`, `hm_reviews_manifest.json` và các lần chạy trên máy chủ Azure ngày 06/10/2026.

## 1. Kiến trúc sau khi chuyển

```
Web (Next.js) → Backend (FastAPI + Postgres mini_market_hm) → RAG service ─┬ Qdrant hm_products (512-d, ảnh + văn bản)
                                                                           ├ Qdrant hm_reviews (đánh giá minh hoạ)
                                                                           └ Recommender service (DATN_ENGINE=hm)
                                                                               tower + luật phục vụ + 7 nguồn + LightGBM
```
Amazon vẫn còn nguyên (database `mini_market`, collection `products`/`reviews`, `DATN_ENGINE=amazon`) làm đối chứng và đường lùi; đổi qua lại bằng biến môi trường, không cần dựng lại dữ liệu.

## 2. Mô hình gợi ý (D4)

`src/datn/recommenders/hm/pipeline.py` (`HMRecommender`) là bản port của notebook 02/03: tower đa phương thức, luật phục vụ có nhận thức thời gian, 7 nguồn ứng viên, 24 đặc trưng, LightGBM LambdaRank, kênh cold. Kiểm chứng:
* `scripts/hm/parity_check.py`: với 40 khách test, cùng tập ứng viên, ma trận đặc trưng sai khác tối đa 0,0 và cùng top-12 so với chính code trong notebook.
* `scripts/hm/eval_port.py` (không dùng code notebook), 2.799 khách test: HitRate@12/50/100 = **0,1275 / 0,2701 / 0,3651** (chèn cold mỗi 10 vị trí: 0,1229 / 0,2558 / 0,3508; mỗi 5: 0,1186 / 0,2437 / 0,3358), trùng `reranker_metrics.json` đến 4 chữ số, kể cả Recall@K.
* Hiệu năng: ≈ 80 khách/giây trên CPU khi chấm hàng loạt; một yêu cầu qua HTTP trên VM 2 vCPU ≈ 56 ms; container dùng ≈ 550 MB RAM.

Cách phục vụ: **đồng hồ của cửa hàng cố định ở 16/09/2020** (cutoff của cửa sổ test), nên số liệu luận văn vẫn đúng cho cấu hình đang chạy; thống kê bán hàng không đổi khi có đơn mới trên web. Khách chưa có lịch sử nhận danh sách xếp theo "bán chạy 7 ngày" qua reranker. Dịch vụ lọc kết quả theo danh sách sản phẩm có trong shop (`shop_item_ids.txt`) và theo lịch sử đã mua. Mặc định không chèn vị trí cold (`hm_cold_every = 0`, tối ưu chỉ số tổng); đổi bằng `DATN_HM_COLD_EVERY` hoặc tham số `cold_every` của yêu cầu.

## 3. Catalog của cửa hàng

| | Giá trị |
|---|---|
| Toàn bộ catalog dữ liệu | 105.542 sản phẩm (`product_id` = dòng `items.parquet` + 1 = chỉ số trong tower) |
| **Bán trong shop** | **72.811** sản phẩm có ảnh (gói ảnh `active_only`); 32.731 sản phẩm còn lại vẫn nằm trong mô hình nhưng không hiển thị/không được gợi ý vì không có ảnh |
| Danh mục (10) | Áo 29.905 · Quần & Chân váy 13.670 · Đầm & Jumpsuit 9.251 · Phụ kiện 7.360 · Đồ lót 4.248 · Giày dép 3.082 · Đồ bơi 2.459 · Tất & Quần tất 1.508 · Đồ ngủ 1.170 · Khác 158 (gom từ `product_group_name`) |
| Nhóm khách (6) | Nữ 38.015 · Trẻ em 12.809 · Divided (teen) 11.810 · Nam 5.476 · Em bé 3.851 · Khác 850 |
| Còn bán trong 28 ngày trước mốc | 24.945 · chưa từng bán (cold) có ảnh: **144** |
| Tên | `prod_name – colour_group_name` (VD "Strap top – Black"), vì `prod_name` trùng giữa các màu |

**Những giá trị là quy ước, không phải dữ liệu thật** (phải nêu trong luận văn):
* **Giá.** Giá giao dịch H&M đã bị chuẩn hoá (trung vị 0,0229), không phải tiền. `giá_VND = giá_TB_trước_mốc × 15.308.495,6`, hệ số chọn để trung vị về 350.000₫ (trung vị các sản phẩm trong shop: 369.000₫), làm tròn nghìn, tối thiểu 20.000₫. Sản phẩm chưa có giao dịch (1.662) nhận trung vị của cùng `product_type`. Mọi sản phẩm gắn `price_estimated = true`; chatbot luôn nói "giá tham khảo".
* **Nhóm khách** suy ra từ `section_name` (tiền tố Womens/Ladies/Mama/Contemporary → nữ; Men/Mens → nam; Kids/Young/Girls/Boys → trẻ em; Baby → em bé; Divided → teen) vì `items.parquet` không có `index_group_name`.
* **Tồn kho** sinh ngẫu nhiên (15–120, seed cố định).

## 4. Vector và tìm kiếm

* `hm_products`: 72.811 điểm, hai vector có tên (`text`, `image`) 512 chiều cosine; payload gồm màu, loại, nhóm sản phẩm, hoạ tiết, khu vực, phòng ban, nhóm khách, `sold_28d`, danh mục, giá, ảnh. `hm_product_embeddings`: vector văn bản cho "sản phẩm tương tự".
* Encoder truy vấn (Jina CLIP v2) cắt vector về 512 chiều rồi chuẩn hoá lại, đúng với `truncate_dim` đã dùng khi embed catalog. Kích thước vector, tên collection và khoá "phổ biến" đọc từ biến môi trường (mặc định vẫn là Amazon).
* Bộ lọc cứng trong Qdrant: giá, nhóm khách, màu (nhóm màu của H&M, VD "đỏ" → Red, Dark Red), danh mục. Backend `/products` lọc theo `audience`, `colour`, `product_type`, sắp xếp `bestseller`/giá trên toàn bộ catalog.

## 5. Chatbot RAG

Giữ nguyên luồng cũ (intent → truy xuất lai → cá nhân hoá → bằng chứng → LLM có kiểm chứng/template). Thay đổi:
* Hiểu nhóm khách ("nam", "nữ", "bé gái", "trẻ em", "sơ sinh") và màu tiếng Việt; từ điển Việt → Anh bổ sung loại đồ H&M; "váy" dịch thành *dress skirt* vì tiếng Việt dùng chung cho cả hai.
* Không còn nói về "thương hiệu" khi dữ liệu không có.
* Mọi nhắc tới điểm đánh giá hoặc trích review đều ghi **"đánh giá minh hoạ"** (trong dữ liệu `is_mock`, thẻ trên giao diện, prompt LLM và câu trả lời mẫu).
* Kiểm chứng câu trả lời LLM chặt hơn: nhóm ngoặc có dạng trích dẫn nhưng kèm chữ thừa (VD `[P5 - chờ chút, P3 mới đúng]`, lỗi Gemini thật đã gặp) bị coi là vi phạm và bị thử lại / dùng câu trả lời mẫu.

## 6. Đánh giá minh hoạ (D2)

H&M không công bố văn bản đánh giá, nên review **mượn** từ Amazon (collection `reviews`, 295.383 review) và gắn vào sản phẩm H&M gần nhất. Không sinh văn bản mới.

Phương pháp (`src/datn/catalog/mock_reviews.py`, `mock_reviews_job.py`):
1. Với mỗi sản phẩm Amazon có review (151.598), tìm 20 sản phẩm H&M gần nhất theo cosine văn bản của Jina CLIP v2 (cả hai cắt 512 chiều).
2. Nếu tiêu đề Amazon nêu nhóm khách ("Women's", "Boys'"…), chỉ giữ ứng viên H&M có nhóm khách tương thích (áp dụng cho 123.881 / 151.598 sản phẩm).
3. Mỗi review đi đến **đúng một** sản phẩm (không lặp văn bản giữa các sản phẩm), chọn sản phẩm đang có ít review nhất trong các ứng viên đạt ngưỡng; tối đa 6 review/sản phẩm.
4. Loại: độ tương đồng < **0,65**; có liên kết hoặc từ khoá chợ (amazon, seller…); nhắc tên thương hiệu thật (889 thương hiệu có ≥ 25 sản phẩm; danh sách thô 28.849 chuỗi chủ yếu là tên người bán/cụm SEO nên bị loại, chi tiết bên dưới); nêu màu cơ bản không khớp màu sản phẩm; dưới 20 ký tự. Văn bản được làm sạch (`[[VIDEOID:…]]`, thực thể HTML).
5. Mỗi review lưu nguồn gốc (`source_item_id`, `match_score`), nên có thể gỡ hoặc thay bằng review thật bằng một lệnh.

Kết quả: **91.256 review (30,9% số review Amazon)** cho **32.813 / 72.811 sản phẩm (45,1%)**, trung bình 2,78 review mỗi sản phẩm có review. Điểm sao: 1★ 5.409 · 2★ 5.270 · 3★ 10.098 · 4★ 18.135 · 5★ 52.344. Độ tương đồng khi ghép: trung vị 0,670, phân vị 5% = 0,651, 95% = 0,728. Lý do loại: không đủ giống 129.189, trùng thương hiệu 20.536, liên kết/từ khoá chợ 16.111, màu mâu thuẫn 9.872, hết chỗ 28.302, quá ngắn 117.

**Bài học đáng ghi lại:** lần đầu bộ lọc thương hiệu loại 41–44% review vì cột `brand` của Amazon chứa nhiều cụm thông thường ("if you", "which is", "party"). Chuyển sang chỉ dùng thương hiệu có ≥ 25 sản phẩm, trừ một danh sách từ đời thường ("guess", "match"…), giảm xuống còn ≈ 7% review bị loại vì lý do này.

**Chất lượng ghép chưa được đo định lượng.** Mới chỉ xem tay 27 cặp ngẫu nhiên (9 cặp mỗi mức 0,62–0,65 / 0,65–0,70 / ≥ 0,70, trên bản dựng thử ngưỡng 0,62): ở 0,62–0,65 nhiều cặp lệch loại (áo choàng tắm ↔ váy), từ 0,65 trở lên phần lớn cùng loại (jogger ↔ jogger, váy ↔ váy, quần jean ↔ quần jean) nhưng vẫn có cặp lệch. Đó là lý do chọn ngưỡng 0,65 và không dùng review này làm đáp án để đánh giá truy xuất.

**Điều được phép nói trong luận văn:** review là dữ liệu mô phỏng nhằm giữ nguyên luồng RAG; chúng không phản ánh ý kiến khách mua sản phẩm H&M, không dùng để kết luận chất lượng sản phẩm hay chất lượng mô hình. Metric "review → sản phẩm" của phần Amazon (`docs/rag_chatbot_design.md` mục 9) **không** áp dụng cho H&M vì liên kết review–sản phẩm là nhân tạo.

## 7. Kiểm chứng đã làm

* 166 test tự động, 165 qua; 1 lỗi có sẵn từ trước và không liên quan (`test_coveo_pipeline`, lỗi DuckDB).
* Đầu cuối trên Azure: đăng nhập, "tương tự", gợi ý cá nhân hoá (bỏ hoodie/áo nỉ nam đen vào giỏ → được gợi ý đồ nam cùng dòng), tìm kiếm có lọc, 7 kịch bản chat tiếng Việt với Gemini thật (search → "rẻ hơn" → "tại sao sản phẩm 1?" → "so sánh 1 và 2", váy nữ, đồ bé gái, gợi ý), 2–4,4 giây mỗi lượt.
* Lỗi phát hiện nhờ chạy thật và đã sửa: chip màu lặp, "váy" chỉ ra chân váy, prompt còn nhắc thương hiệu, backend chưa nhận trường `reviews_mock`, Gemini chèn ghi chú vào mã trích dẫn.

## 8. Hạn chế đã biết

* Giá, tồn kho và nhóm khách là quy ước/suy diễn (mục 3); review là mô phỏng (mục 6).
* Thống kê bán hàng đóng băng ở 16/09/2020; đơn hàng mới trên web không làm đổi thứ tự "bán chạy".
* Chỉ **144** sản phẩm chưa từng bán có ảnh, nên kênh cold gần như không có gì để hiển thị khi demo.
* 32.731 sản phẩm không có ảnh không được bán, dù mô hình vẫn xếp hạng chúng.
* LLM nhỏ (Gemini flash-lite) thỉnh thoảng sinh lỗi trình bày; bộ kiểm chứng chỉ bắt được các lỗi có dạng quy tắc (trích dẫn, số tiền, liên kết).
* Chưa đo độ chính xác tìm kiếm văn bản H&M (cần một bộ truy vấn sinh từ thuộc tính sản phẩm; chưa làm).
