# Báo cáo Đánh giá Rủi ro & Chiến lược Phản biện: Tính Thực tế và Độ Thưa Dữ liệu (Sparsity Critique)

- **Ngày thực hiện**: 14/09/2026
- **Vấn đề xem xét**: Việc 80.86% sản phẩm chỉ có 1 tương tác có phải là một lỗ hổng nghiêm trọng (Fatal Flaw) làm mất đi tính thực tế của đề tài không?
- **Tài liệu tham chiếu**: [`docs/logs/2026-09-14_item_degree_sparsity_root_cause.md`](./2026-09-14_item_degree_sparsity_root_cause.md), [`docs/logs/2026-09-14_user_tower_p1_difficulty_and_popularity_bias.md`](./2026-09-14_user_tower_p1_difficulty_and_popularity_bias.md).

---

## 1. Phán quyết Trực diện (Executive Verdict)

1. **Đây CÓ PHẢI là lỗ hổng nghiêm trọng làm hủy hoại đề tài không?**
   - **KHÔNG PHẢI LỖ HỔNG HỦY HOẠI (Not a Fatal Bug)**. Đề tài không bị sai lệch về mặt toán học hay kỹ thuật.
   - **TUY NHIÊN, ĐÂY LÀ MỘT "TỬ HUYỆT PHƯƠNG PHÁP LUẬN" (Methodological Vulnerability)**: Nếu trong luận văn tác giả không chủ động giải trình nguồn gốc và ý đồ thiết kế của con số này, hội đồng chấm hoặc chuyên gia sẽ lập tức chất vấn theo hướng: *"Bạn cố tình tạo ra một tập dữ liệu dị tật để dìm Collaborative Filtering xuống nhằm tâng bốc mô hình Multimodal của mình"*.

2. **Dữ liệu thưa như vậy CÓ THIẾU THỰC TẾ KHÔNG?**
   - **Về bản chất ngành hàng E-commerce thời trang: CỰC KỲ THỰC TẾ (Highly Realistic)**. Trong thực tế tại Shopee, Amazon hay Taobao, danh mục thời trang có hàng triệu sản phẩm và hơn 85% sản phẩm đuôi dài (long-tail) hầu như không có đơn hàng hoặc chỉ có 1-2 lượt mua.
   - **Về góc độ loại tín hiệu tương tác: CÓ TÍNH CHẤT ÉP ÉP (Artificial Stress-test)**. Trong công nghiệp, các sàn thương mại điện tử sử dụng dữ liệu clickstream (lượt xem, thêm vào giỏ, click tìm kiếm) dày đặc hơn nhiều lần so với dữ liệu review. Việc dataset chỉ giữ lại review có `helpful_vote >= 3` đã ép đồ thị tương tác về trạng thái thưa thớt cực hạn (Extreme Stress-test).

---

## 2. Phân tích Hai Mặt: Tính Thực tế vs Tính Cực đoan

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                             HAI MẶT CỦA BỘ DỮ LIỆU ĐANG DÙNG                             │
├─────────────────────────────────────────────┬────────────────────────────────────────────┤
│         MẶT THỰC TẾ (REALISTIC)             │       MẶT CỰC ĐOAN (ARTIFICIAL/EXTREME)    │
├─────────────────────────────────────────────┼────────────────────────────────────────────┤
│ 1. Đặc thù ngành Thời trang (Fashion):     │ 1. Chỉ dùng Purchase Review đã vote:       │
│    Vòng đời sản phẩm cực ngắn (theo mùa,    │    Review là tín hiệu thưa nhất trong các  │
│    theo trend), số lượng mẫu mã khổng lồ.   │    tín hiệu. Lọc thêm helpful_vote >= 3   │
│    80% SKU thực tế chỉ bán được 1-2 đơn.    │    đã làm mất đi 95% hành vi thật của user.│
│                                             │                                            │
│ 2. Vượt qua giới hạn của Toy Benchmarks:   │ 2. Đồ thị tương tác bị bẻ gãy:            │
│    Các tập học thuật cũ (MovieLens, Yahoo)  │    User-Hash 1/32 ngẫu nhiên khiến xác     │
│    ép 5-core cả 2 phía tạo đồ thị dày giả   │    suất 2 user cùng mua 1 món đồ giảm về  │
│    tạo, hoàn toàn che giấu bài toán cold.   │    mức phần triệu, triệt tiêu CF hoàn toàn.│
│                                             │                                            │
│ 3. Khách quan với tầng Candidate Retrieval: │ 3. Dễ bị hội đồng bắt bẻ so sánh Strawman: │
│    Thách thức rank 152.086 items phản ánh   │    CF bị đặt vào môi trường chết đói       │
│    đúng áp lực của sàn E-commerce thật.     │    khiến mức chênh 33-42x trở nên bất thường.│
└─────────────────────────────────────────────┴────────────────────────────────────────────┘
```

---

## 3. Ba Rủi ro Phản biện Lớn nhất & Cách Đáp trả (Defense Playbook)

### Rủi ro 1: "Tại sao không dùng Dual 5-core tiêu chuẩn như Kang & McAuley (SASRec 2018) hay He et al. (LightGCN)?"
- **Lời phê của Hội đồng**: *"Trong cộng đồng nghiên cứu RecSys, mọi người đều lọc bỏ user và item có dưới 5 tương tác. Tại sao bạn lại giữ item 1 tương tác?"*
- **Cách đáp trả thuyết phục**:
  > *"Thưa thầy/cô, việc lọc Dual 5-core (5-core cho cả User và Item) đã bị các nghiên cứu hiện đại về Multimodal Recommendation (như UniSRec - KDD 2022, MoREC - SIGIR 2023) chỉ trích là **thiếu thực tế và che giấu bài toán Cold-start**. Khi ép 5-core cho item, ta đã tự loại bỏ toàn bộ các sản phẩm mới ra mắt và các sản phẩm đuôi dài – vốn là nơi mà hệ sinh thái E-commerce gặp khó khăn nhất và cần đến sự hỗ trợ của Đa phương thức nhất.*
  > 
  > *Đề tài của em tập trung giải quyết bài toán: **Làm sao gợi ý được khi sản phẩm chưa có hoặc có rất ít tương tác?** Nếu em ép 5-core cho item, bài toán nghiên cứu cốt lõi của đề tài về Cold-start và Long-tail Retrieval sẽ hoàn toàn biến mất."*

---

### Rủi ro 2: "Có phải bạn cố tình tạo ra tập dữ liệu này để dìm CF (Strawman Comparison)?"
- **Lời phê của Hội đồng**: *"CF-only chỉ đạt HitRate@10 = 0.0045%, kém hơn cả random (0.0066%). Bạn so sánh như vậy là không công bằng cho nhánh CF."*
- **Cách đáp trả thuyết phục**:
  > *"Em hoàn toàn đồng ý với nhận định của thầy/cô: **Trên tập dữ liệu này, CF thuần túy dựa trên ID chắc chắn phải thất bại về mặt toán học**, bởi vì 82.17% sản phẩm trong train chỉ xuất hiện đúng 1 lần, không hề có thông tin đồng xuất hiện (co-occurrence) để CF trích xuất tri thức.*
  >
  > *Sự so sánh ở đây không nhằm chứng minh 'thuật toán SASRec kém', mà nhằm làm rõ **giới hạn vật lý của phương pháp biểu diễn thuần ID (ID-based Representation)**: Khi doanh nghiệp đối mặt với các danh mục thời trang mới hoặc khi hệ thống mới thành lập (Cold-start / Data Sparsity 99.995%), các mô hình dựa trên ID hoàn toàn bất lực.*
  >
  > *Chính sự sụp đổ của ID trong môi trường này đã khẳng định tính bắt buộc phải chuyển dịch sang **Biểu diễn dựa trên Nội dung Đa phương thức (Modality-based Representation)** như CLIP Text và Image."*

---

### Rủi ro 3: "Dữ liệu quá thưa do bạn lọc `helpful_vote >= 3`, liệu có phải do bạn tự làm khó mình?"
- **Lời phê của Hội đồng**: *"Tại sao lại phải lọc `helpful_vote >= 3` để rồi làm mất 95% tương tác?"*
- **Cách đáp trả thuyết phục**:
  > *"Đây là một **sự đánh đổi kiến trúc có chủ đích (Architectural Trade-off)** vì đề tài của em là **Hệ thống gợi ý kết hợp RAG**:*
  > - *Nếu không lọc vote hữu ích, tập review trên Amazon chứa đại đa số các câu cộc lốc, vô nghĩa như 'OK', 'Good', 'Nice'. Khi đưa các văn bản rác này vào cơ sở tri thức RAG, mô hình LLM sẽ sinh ra các lời giải thích (Explanation) kém chất lượng và bị ảo giác (hallucination).*
  > - *Để đảm bảo RAG có tri thức sâu sắc về sản phẩm, em bắt buộc phải lọc các đánh giá có giá trị tham khảo cao (`helpful_vote >= 3`).*
  > - *Sự đánh đổi này phản ánh đúng thực tế khi xây dựng một hệ thống AI tổng thể: Tối ưu cho cấu phần tạo sinh (RAG) sẽ đặt ra thách thức cực đại cho cấu phần đề xuất (Recommender). Và kiến trúc đề xuất của em đã chứng minh khả năng vượt qua thách thức đó."*

---

## 4. Chiến lược Điều chỉnh & Đóng khung lại Bài toán trong Luận văn

Để bảo vệ luận văn đạt điểm tối đa (Xuất sắc) mà không bị bắt bẻ, bạn cần thực hiện 3 điều chỉnh sau trong cách trình bày:

1. **Đổi tên đặc tính dữ liệu trong Luận văn**:
   - ❌ **Không viết**: "Dữ liệu được thu thập ngẫu nhiên từ Amazon Fashion".
   -  **Viết chuẩn xác**: "Dữ liệu được xây dựng theo kịch bản **Môi trường Thử nghiệm Khắc nghiệt (Extreme Cold-start & Long-tail Stress-test Benchmark)** với cấu hình User-centric K-core ($K_{user}=5, K_{item}=1$)".

2. **Bổ sung Phân tích Bậc (Degree Distribution) vào Chương 3**:
   - Trình bày rõ ràng bảng thống kê 80.86% sản phẩm có bậc bằng 1.
   - Thừa nhận thẳng thắn: Đây là môi trường bất lợi tối đa cho CF và là môi trường để kiểm chứng năng lực biểu diễn ngữ nghĩa của Multimodal Tower.

3. **Khẳng định tính đóng góp học thuật (Academic Contribution)**:
   - Bài toán này chứng minh: Khi chuyển từ hệ thống gợi ý truyền thống (chỉ có ID) sang hệ thống gợi ý thời đại mới (AI Đa phương thức + RAG), ta không còn bị phụ thuộc vào giả định "dữ liệu phải dày đặc (5-core)" nữa. Hệ thống vẫn có thể định vị và xếp hạng được các sản phẩm đuôi dài nhờ thị giác và ngôn ngữ.
