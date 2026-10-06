# Kiểm toán baseline và ý nghĩa thống kê của pipeline `balanced_two_stage_v2` (2026-09-30)

- **Mã:** `scripts/eval_baselines.py`, `scripts/eval_diagnostics.py` (CPU/GPU nhẹ, khoảng 4 phút).
- **Kết quả thô:** `data/artifacts/baseline_eval/{test_baselines.json, diagnostics.json, test_ranks.npz}`.
- **Giao thức:** giống hệt repo (cùng loader, leave-last-out, xếp hạng toàn bộ 32.557 sản phẩm, loại sản phẩm đã xem), 21.690 user test.
  Harness tái hiện đúng số đã công bố (tower 3,038/6,556/9,142; tower+reranker 3,518/6,962/9,521).
- **Khoảng tin cậy:** paired bootstrap 2.000 lần theo user; p-value là McNemar (chính xác hoặc xấp xỉ chuẩn).
- **Tie:** baseline dùng thứ hạng trung bình khi bằng điểm, không thiên vị đáp án.

## 1. Kết quả trên test (HR = HitRate)

| Phương pháp | HR@10 | HR@50 | HR@100 | NDCG@10 | Coverage@10 |
|---|---:|---:|---:|---:|---:|
| Random | 0,03 | 0,18 | 0,35 | 0,01 | 99,9% |
| Popularity | 2,56 | 4,26 | 5,93 | 2,05 | 0,0% |
| ItemKNN co-occurrence, cosine thô | 0,22 | 0,69 | 1,23 | 0,11 | 85,7% |
| ItemKNN đã tinh chỉnh trên valid (shrink 1000) | 2,10 | 3,36 | 4,22 | — | — |
| Transition kNN (3 bước) | 0,90 | 3,12 | 4,96 | 0,55 | 83,0% |
| Content centroid (CLIP ảnh+text) | 1,32 | 3,20 | 4,69 | 0,69 | 60,9% |
| Content sản phẩm cuối | 1,71 | 3,44 | 4,74 | 0,96 | 85,2% |
| Cùng brand/category với sản phẩm cuối + popularity | 2,36 | 4,79 | 6,63 | 1,32 | 45,9% |
| RRF (kNN + content + popularity), không huấn luyện | 3,26 | 5,55 | 7,25 | 1,74 | 49,9% |
| User Tower thô | 3,04 | 6,56 | 9,14 | 2,11 | 8,0% |
| **Tower + Reranker v2** | **3,52** | **6,96** | **9,52** | **2,49** | 30,1% |

95% CI của pipeline cuối: HR@10 [3,28; 3,78], HR@50 [6,61; 7,31], HR@100 [9,10; 9,91].

## 2. So sánh cặp (chênh lệch, CI 95%)

| So sánh | @10 | @50 | @100 |
|---|---|---|---|
| Pipeline cuối − Popularity | +0,95 [+0,76; +1,16] | +2,71 [+2,39; +3,02] | +3,59 [+3,20; +3,95] |
| Reranker v2 − Tower thô | +0,48 [+0,36; +0,61] | +0,41 [+0,27; +0,53] | +0,38 [+0,24; +0,51] |
| Reranker v2 − RRF không huấn luyện | +0,25 [+0,06; +0,45] (p=0,012) | +1,42 [+1,10; +1,72] | +2,27 [+1,90; +2,63] |
| Tower − Tower với lịch sử hoán đổi giữa các user | +1,27 | +3,62 | +5,17 |

Tất cả đều có ý nghĩa thống kê (p < 0,05). Riêng chênh lệch với RRF ở @10 chỉ vừa đủ.

## 3. Tower có dùng lịch sử không?

| Đầu vào cho tower (test) | HR@10 | HR@50 | HR@100 |
|---|---:|---:|---:|
| Toàn bộ lịch sử | 3,04 | 6,56 | 9,14 |
| Chỉ sản phẩm cuối | 2,42 | 5,18 | 7,28 |
| Lịch sử của user khác | 1,77 | 2,93 | 3,97 |

Có dùng. Lịch sử đúng cho kết quả cao hơn lịch sử sai +3,6 điểm ở @50, và lịch sử đầy đủ cao hơn chỉ sản phẩm cuối +1,4 điểm.
Tuy vậy HR@50 gần như phẳng theo độ dài lịch sử (6,6 / 6,4 / 6,8 / 6,6 cho nhóm ≤4, 5–7, 8–11, ≥12 sản phẩm): thêm lịch sử ngoài vài sản phẩm gần nhất không mang lại lợi ích đo được.

## 4. Kết luận

1. Pipeline cuối **thật sự hơn** popularity (gấp 1,4 lần @10, 1,6 lần @50 và @100), hơn ItemKNN đã tinh chỉnh và hơn baseline lai không huấn luyện. Cải thiện có ý nghĩa thống kê nhưng nhỏ về độ lớn tuyệt đối.
2. Lọc cộng tác thuần (ItemKNN) **không thắng popularity** trên dữ liệu này: đồ thị quá thưa (khoảng 7 sản phẩm mỗi user, đa số sản phẩm chỉ có vài tương tác). Lợi thế của tower đến từ nội dung CLIP và mô hình chuỗi.
3. Reranker v2 thêm +0,4 đến +0,5 điểm so với tower thô. Ở @10 nó chỉ hơn RRF không huấn luyện +0,25 điểm.
4. Sản phẩm cold (5,3% target): tower và reranker đều 0 hit; chỉ content centroid đạt 1,5% @10 và 3,1% @50. Cold chiếm ít nên tác động tổng nhỏ.
5. Catalog: 94% sản phẩm có `category = AMAZON FASHION` (30.723/32.557) nên category gần như hằng số; khoảng 6% là hàng ngoài thời trang hoặc rỗng. `brand` có mặt ở 99,8% sản phẩm, dùng được hơn category.
6. Coverage@10 của tower thô chỉ 8,0% (thiên về sản phẩm phổ biến); reranker nâng lên 30,1%.

## 5. Hạn chế

- Baseline chưa gồm EASE, SASRec/LightGCN từ thư viện chuẩn; chỉ ItemKNN được tinh chỉnh (trên valid).
- Một seed, một bộ dữ liệu (Amazon Reviews, review làm tín hiệu thay giao dịch thật).
- Các số so sánh chỉ hợp lệ trên catalog 32.557 sản phẩm của `balanced_u5_i2_v1`; không so với các báo cáo trên catalog 152.086 sản phẩm.
- Baseline popularity ở bản này (4,26 @50) hơi khác con số nhanh 4,33 tính trước đó do cách xử lý tie.

## 6. Cold-start: vì sao tower "mù" và cách đo đúng (bổ sung cùng ngày)

**Mã:** `scripts/eval_cold_start.py`, `scripts/eval_cold_fix_probe.py`. **Kết quả thô:** `data/artifacts/baseline_eval/{cold_start,cold_fix_probe}.json`.
Cold = sản phẩm có 0 tương tác dương trong train.

### 6.1. Bộ đánh giá hiện tại gần như không đo cold-start
- Chỉ có **894 sản phẩm cold (2,7% của 32.557)**; chỉ **5,27% mục tiêu test** (1.143 user) là cold. Bộ lọc core 5-user/2-item đã loại phần lớn sản phẩm mới khỏi catalog.
- Recommender chỉ biết 32.557/152.086 sản phẩm của shop; 119.529 sản phẩm còn lại chưa có trong vocab.

### 6.2. Tower không "mù", nó bị lệch mạnh (warm bias)
| Đo trên test | Kết quả |
|---|---|
| Hạng trung vị của mục tiêu cold (tower) | 90,8% (gần cuối bảng); content centroid: 26,9% |
| Chênh lệch điểm trung bình cold − warm (tower) | **−1,86 độ lệch chuẩn** |
| Số user có ít nhất một sản phẩm cold trong top-100 | **0%** (0 sản phẩm cold nào từng lọt top-100) |
| HR@10 / @50 / @100 khi chỉ xếp hạng trong nhóm cold (894 sản phẩm) | tower 6,7 / 21,8 / 33,4%; content 7,9 / 18,3 / 27,6% (random @10: 1,1%) |

Trong nhóm cold, nội dung CLIP xếp hạng có ý nghĩa (gấp khoảng 6 lần random ở @10). Vấn đề là **hiệu chuẩn giữa cold và warm**, không phải thiếu thông tin.

### 6.3. Nguyên nhân (đã kiểm chứng trên checkpoint)
- Cấu hình huấn luyện có 50% mẫu âm lấy đều toàn catalog (`negative_uniform_ratio: 0.5`). Sản phẩm cold không bao giờ là mẫu dương nên chỉ nhận gradient âm.
- Docstring `model.py` giả định "id_residual của sản phẩm chưa có tương tác giữ nguyên bằng 0"; điều này **sai** với cấu hình trên: cả 894 hàng của sản phẩm cold đều khác 0, chuẩn L2 trung bình 0,60 so với 0,25 của sản phẩm warm.
- Phần nội dung (`content_proj`) của cold và warm gần bằng nhau (0,53 và 0,53), nên chính phần ID bị học lệch đã đẩy sản phẩm cold xuống.

### 6.4. Sửa ngay khi suy luận, không cần huấn luyện lại (đặt id_residual của sản phẩm cold = 0)
| Test | Chênh lệch điểm | HR@10/50/100 tổng | Mục tiêu cold | Mục tiêu warm | User có sản phẩm cold trong top-100 |
|---|---:|---|---|---|---:|
| Như đã huấn luyện | −1,86 sd | 3,04 / 6,56 / 9,14 | 0 / 0 / 0 | 3,21 / 6,92 / 9,65 | 0% |
| **Zero id_residual của cold** | +0,85 sd | **3,07 / 6,59 / 9,25** | **0,26 / 1,49 / 3,15** | 3,23 / 6,88 / 9,59 | 75,5% |
| Zero cho cả sản phẩm có ≤1 tương tác | +0,67 sd | 3,09 / 6,58 / 9,23 | 0,09 / 0,87 / 2,45 | 3,26 / 6,90 / 9,60 | 71,3% |

HR tổng không giảm (thậm chí nhích lên trong nhiễu). Kết quả trên valid tương tự (tổng 2,96 / 6,52 / 9,37; cold @100 = 3,82%).

### 6.5. Các cách khác đã thử (test)
- Cộng điểm thưởng cho sản phẩm cold: cần thưởng +3 sd mới có hit cold (HR@50 cold 3,06%), làm HR@50 tổng giảm 6,56 → 6,43.
- Dành chỗ cho sản phẩm cold trong top-10 (chọn theo nội dung): r=1 → cold 2,54%, tổng 3,07% (không mất); r=2 → cold 3,85%, tổng 3,02%; r=3 → cold 4,90%, tổng 2,97%.

### 6.6. Việc còn lại
1. Xây bộ đánh giá **strict item cold-start** (giữ lại 10–20% sản phẩm, gỡ toàn bộ tương tác của chúng khỏi train, dùng làm mục tiêu) thay cho định nghĩa cold hiện tại.
2. Sửa huấn luyện: loại sản phẩm chưa từng là mẫu dương khỏi mẫu âm đều, hoặc khoá `id_residual` của chúng bằng 0; huấn luyện lại reranker trên tower đã sửa với đặc trưng "độ phổ biến / cold flag".
3. Mở rộng vocab để tower chấm điểm cả 119.529 sản phẩm còn lại bằng vector nội dung CLIP (id_residual = 0).

## 7. Metric "thực dụng" (brand-hit, neighbor-hit) — định nghĩa trước khi xem kết quả

**Mã:** `scripts/eval_practical_metrics.py` → `data/artifacts/baseline_eval/practical_metrics.json`. Test, 21.690 user, cùng giao thức và cùng baseline.
- **brand-hit@K:** top-K có sản phẩm cùng thương hiệu với món mua tiếp theo (99,9% mục tiêu có brand).
- **neighbor-hit@K:** top-K có một trong 20 sản phẩm gần nhất về nội dung CLIP (ảnh+text) của món mua tiếp theo.

| Phương pháp | item@10 | brand@10 | neigh@10 | item@50 | brand@50 | neigh@50 |
|---|---:|---:|---:|---:|---:|---:|
| Popularity | 2,56 | 4,67 | 0,97 | 4,26 | 16,46 | 6,93 |
| Cùng brand với món cuối + popularity | 2,44 | 4,39 | 3,63 | 4,91 | 13,06 | 8,46 |
| Content centroid | 1,32 | 5,45 | 6,87 | 3,20 | 10,84 | 13,09 |
| RRF (content + popularity) | 3,21 | 7,88 | 6,05 | 5,62 | 18,03 | 12,96 |
| User Tower thô | 3,04 | 8,91 | 4,53 | 6,56 | 18,25 | 14,27 |
| **Tower + Reranker v2** | **3,52** | **9,70** | 5,49 | **6,96** | **18,71** | **14,92** |

Pipeline cuối so với popularity (paired bootstrap 95% CI): brand@10 +5,04 điểm [+4,68; +5,44] (gấp 2,1 lần); neighbor@10 +4,51 [+4,20; +4,84] (gấp 5,7 lần); brand@20 +1,16; brand@50 +2,25 [+1,78; +2,78]; neighbor@50 +7,99 [+7,49; +8,50].

**Đọc kết quả cho đúng:**
- Lợi thế lớn nhất ở K=10; ở K=20–50 brand-hit chỉ hơn popularity 1–2 điểm vì các thương hiệu phổ biến đã phủ nhiều mục tiêu.
- neighbor-hit được định nghĩa trong chính không gian CLIP nên **thiên vị các phương pháp dựa trên nội dung**: content centroid đạt 6,87% ở @10, cao hơn pipeline cuối (5,49%). Không dùng metric này để kết luận vượt phương pháp nội dung thuần.
- Số tuyệt đối vẫn khiêm tốn (brand@10 9,7%; neighbor@10 5,5%): không có cách trung thực làm cho các con số này "đẹp" hơn nhiều.

## 8. Giao thức lấy mẫu (1 đáp án + N sản phẩm ngẫu nhiên) — chỉ dùng như bảng phụ

**Mã:** `scripts/eval_sampled.py` → `data/artifacts/baseline_eval/sampled_protocol.json`. Test, 21.690 user, mọi phương pháp dùng **cùng** bộ mẫu âm; mẫu âm loại sản phẩm đã xem và đáp án; bằng điểm tính nửa. Reranker được chuyển thành xếp hạng đầy đủ (điểm reranker cho ứng viên, điểm tower cho phần còn lại).
Đối chiếu với `sampled_ranking_evaluate` của repo (tower thô): 1+99 HR@10 = 45,4% (repo) so với 45,1% (script này); 1+999: 16,1% và 16,1%.

| Giao thức | Random (tham chiếu) | Popularity | Content centroid | Tower thô | **Tower + Reranker v2** |
|---|---:|---:|---:|---:|---:|
| 1+99 mẫu âm ngẫu nhiên, HR@10 | 10,2 | 34,8 | 29,7 | 45,1 | **46,4** [45,7; 47,1] |
| 1+99, HR@5 / HR@1 | 5,2 / 1,1 | 24,5 / 10,1 | 20,1 / 7,7 | 33,2 / 14,4 | 33,8 / 14,6 |
| 1+99, NDCG@10 / MRR | 4,7 / 5,3 | 20,5 / 18,1 | 17,0 / 15,5 | 27,8 / 24,5 | 28,5 / 24,9 |
| 1+999 mẫu âm ngẫu nhiên, HR@10 | 1,1 | 11,1 | 8,4 | 16,1 | **16,4** [15,9; 16,9] |
| 1+99 mẫu âm theo độ phổ biến, HR@10 | 10,5 | 10,6 | **27,2** | 24,2 | 24,5 [23,9; 25,1] |

- Chênh lệch so với popularity (HR@10, paired bootstrap): 1+99 ngẫu nhiên +11,6 điểm [+11,0; +12,3]; 1+999 +5,3 [+4,8; +5,7]; 1+99 theo độ phổ biến +13,9 [+13,4; +14,5].

**Cảnh báo bắt buộc khi trình bày:**
1. Các con số này **không so sánh được** với HR toàn catalog (3,52%) hay với bài báo khác dùng xếp hạng toàn catalog; phải ghi rõ giao thức.
2. Con số phụ thuộc mạnh vào giao thức: đổi 99 thành 999 mẫu âm thì HR@10 từ 46,4% xuống 16,4%.
3. **Thứ tự các phương pháp đổi theo giao thức:** với mẫu âm theo độ phổ biến, content centroid (27,2%) **hơn** tower + reranker (24,5%) ở HR@10 và HR@20; với mẫu âm ngẫu nhiên thì ngược lại.
4. Với mẫu âm ngẫu nhiên, popularity đạt 34,8% chỉ vì đa số mẫu âm là sản phẩm ít người biết, không vì nó cá nhân hoá.
5. Mức hơn popularity tương đối (1,3 lần ở 1+99) tương đương mức đo toàn catalog (1,4 lần ở HR@10): giao thức lấy mẫu làm số tuyệt đối trông đẹp hơn chứ không thay đổi bản chất kết quả.

## 9. Siết k-core của Amazon có "làm mịn" được không? (chỉ đo baseline popularity, không huấn luyện)

Áp thêm bộ lọc core lặp lên các tương tác dương của `balanced_u5_i2_v1`, cùng giao thức (mục tiêu = sản phẩm cuối, loại sản phẩm đã xem). Dòng u5-i2 tái hiện đúng 2,56 / 4,26 / 5,93.

| Core (user, item) | User | Sản phẩm | Tương tác | Popularity HR@10 | @50 | @100 |
|---|---:|---:|---:|---:|---:|---:|
| u5-i2 (hiện tại) | 21.690 | 32.557 | 198.200 | 2,56 | 4,26 | 5,93 |
| u5-i3 | 18.022 | 20.374 | 160.048 | 3,02 | 4,98 | 6,85 |
| u5-i5 | 11.623 | 8.858 | 97.000 | 4,11 | 6,97 | 9,40 |
| u6-i5 | 6.597 | 6.130 | 62.019 | 4,61 | 7,87 | 11,11 |
| u6-i8, u8-i8 | rỗng | | | | | |

Kết luận: siết core làm baseline đơn giản nhất cũng tăng (HR@10 từ 2,56% lên 4,11% ở u5-i5) trong khi catalog co còn 27% và số user giảm một nửa; siết thêm nữa thì dữ liệu sụp đổ (rỗng). Đây là cách làm số đẹp hơn bằng cách làm bài toán dễ và ít thực tế hơn (catalog nhỏ, gần như không còn cold-start), và tăng cho **mọi** phương pháp nên không tạo thêm bằng chứng cho hệ thống.

## 10. HR 60–70% có đạt được không? Hai phân tích đã chốt trước (`scripts/eval_coarse.py`, `coarse_analyses.json`)

### 10.1. Đường cong giao thức lấy mẫu (User Tower thô, mẫu âm ngẫu nhiên; công thức giải tích từ hạng đầy đủ)
Kiểm chứng: N=99 cho HR@10 = 45,3% và HR@20 = 60,1%, khớp giá trị đo trực tiếp (45,1% và 60,1%); N=999 → 16,1% (đo: 16,1%).

| N (mẫu âm) | HR@1 | HR@5 | HR@10 | HR@20 | Random HR@10 |
|---:|---:|---:|---:|---:|---:|
| 19 | 30,0 | 64,0 | 80,9 | — | 50,0 |
| 29 | 25,0 | 55,0 | 71,3 | 88,1 | 33,3 |
| 49 | 19,8 | 44,7 | 59,7 | 75,9 | 20,0 |
| 99 | 14,4 | 33,2 | 45,3 | 60,1 | 10,0 |
| 199 | 10,3 | 24,3 | 33,6 | 45,6 | 5,0 |
| 999 | 4,8 | 11,4 | 16,1 | 22,3 | 1,0 |

Chất lượng mô hình không đổi; con số chỉ là hàm của N và K. Quy ước phổ biến trong tài liệu là 99 mẫu âm; không chọn N để đạt một con số định trước.

### 10.2. Hit ở mức loại sản phẩm (k-means trên vector CLIP, hai cỡ báo cáo cùng lúc)
| Phương pháp | 30 cụm top-10 | 30 cụm top-20 | 100 cụm top-10 | 100 cụm top-20 |
|---|---:|---:|---:|---:|
| Popularity | 30,2 | 40,5 | 13,8 | 20,3 |
| Content centroid | 20,3 | 24,4 | 13,2 | 16,5 |
| User Tower thô | 35,9 | 46,9 | 21,9 | 30,7 |
| **Tower + Reranker v2** | **36,5** | **47,3** | **22,6** | **31,1** |

(Luôn đoán cụm lớn nhất chỉ trúng 3,8% / 1,7% mục tiêu.) Ngay cả ở mức thô 30 cụm và top-20, mô hình cũng chưa tới 50%.

### 10.3. Kết luận
- Xếp hạng toàn catalog: cận trên của pipeline hiện tại bị chặn bởi recall tập ứng viên (36,5% ở 2.500 ứng viên); HR@10 60–70% không khả thi trên bài toán này nếu không rò rỉ dữ liệu.
- 60–70% chỉ xuất hiện khi thu hẹp bài toán (giao thức lấy mẫu với ít mẫu âm hoặc K lớn so với số ứng viên), và phải ghi rõ giao thức cùng mức random tương ứng.
