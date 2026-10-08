# H&M — Dữ liệu và giao thức đánh giá

> Cập nhật 05/10/2026. Số liệu mô tả dữ liệu lấy từ `hm/scripts/dataset_stats.py` (kết quả: [`results/dataset_stats.json`](results/dataset_stats.json)); định nghĩa metric khớp mã trong `hm/notebooks/02_hm_retrieval.ipynb` và `03_hm_reranking_evaluation.ipynb`.

## 1. Nguồn dữ liệu

| Thành phần | Nội dung |
|---|---|
| Nguồn | Kaggle *H&M Personalized Fashion Recommendations* (`transactions_train.csv`, `articles.csv`, `customers.csv`, thư mục `images/`) |
| Số giao dịch toàn bộ | 31.788.324 (đếm từ bảng `item_daily_counts.parquet`, notebook 00) |
| Catalog | 105.542 article (`articles.csv`); khoá `article_id` (chuỗi) → chỉ số nguyên `1..105.542` (0 = padding) |
| Metadata dùng | `prod_name`, `product_type_name`, `product_group_name`, `graphical_appearance_name`, `colour_group_name`, `department_name`, `section_name`, `detail_desc` (null chỉ 0,39% ở `detail_desc`) |
| Ảnh | 105.100/105.542 article có ảnh (độ phủ embedding ảnh 99,58%; 442 thiếu → vector 0) |
| Embedding | Jina CLIP v2 (`jinaai/jina-clip-v2`), cắt Matryoshka 512 chiều, chuẩn hoá L2, FP32: `text_embeddings.npy` và `image_embeddings.npy` (105.542 × 512). Văn bản embedding = `prod_name. product_type_name. product_group_name. colour_group_name. department_name. detail_desc`, `task=None` (checkpoint này chỉ nhận `retrieval.query` cho truy vấn, không hỗ trợ `retrieval.passage`) |

**Vì sao Jina CLIP v2**: một encoder đa phương thức dùng chung không gian cho ảnh và văn bản (căn chỉnh để ảnh/văn bản của cùng sản phẩm gần nhau), hỗ trợ cắt Matryoshka từ 64 đến 1024 chiều; dự án cố định 512 chiều để cân bằng bộ nhớ và chất lượng. Dự án **không** có phép đo riêng so sánh 512 chiều với chiều đầy đủ hay với CLIP nguyên bản, nên không khẳng định mức giữ chất lượng.

H&M **không có review văn bản**; dữ liệu tương tác chỉ gồm giao dịch mua (không có rating, không có tương tác âm tường minh). Đây là khác biệt lớn so với Amazon Reviews 2023 và ảnh hưởng đến lớp RAG (xem `05_thesis_plan.md`, mục 5).

## 2. Mẫu `hm_v1` (profile `quick`)

Toàn bộ 31,8 triệu giao dịch quá lớn để lặp ablation trên một phiên Kaggle, nên notebook 00 lấy mẫu **50.000 khách** theo cách tất định, chỉ nhìn dữ liệu **trước** cửa sổ chọn checkpoint:

1. Chỉ giữ khách có ≥ 3 giao dịch trước ngày `rerank_train_start` (`min_train_events = 3`).
2. Chia khách thành các nhóm theo `floor(log10(số giao dịch train))` (kẹp 0..4), phân bổ chỉ tiêu tỉ lệ thuận với kích thước nhóm.
3. Trong mỗi nhóm, lấy theo thứ tự hash có seed (`seed = 20260922`) → không chọn khách theo nhãn tương lai.
4. Mang toàn bộ dòng thời gian của khách đã chọn; chỉ cắt lịch sử **cũ** của cửa sổ train ở tối đa 100 giao dịch gần nhất (`max_events_per_customer = 100`); giữ nguyên ba cửa sổ tương lai.

Profile `full` (300.000 khách) có trong mã nhưng **chưa chạy**.

### Hệ quả cần nêu trong luận văn
* Mọi khách trong tập test đều có ≥ 3 giao dịch lịch sử (tỉ lệ khách test không có lịch sử = 0,0%). **Cold-start người dùng không được đo**; toàn bộ phần cold-start trong tài liệu này là *cold-start sản phẩm*.
* Mẫu 50k khách chỉ chứa khoảng 4% giao dịch. Thống kê cấp **sản phẩm** (độ phổ biến, bán chạy 7 ngày, ngày xuất hiện đầu tiên) tính từ mẫu rất nhiễu; vì vậy notebook 00 (phiên bản v2) xuất thêm bảng đếm theo (item, ngày) từ **toàn bộ** khách để các notebook sau dùng (xem mục 5).

## 3. Cửa sổ thời gian

Ba cửa sổ tương lai dài 7 ngày liên tiếp, lùi từ ngày cuối cùng của dữ liệu (22/09/2020):

| Cửa sổ | Tên file | Khoảng ngày | Sự kiện | Khách | Item khác nhau | Vai trò |
|---|---|---|---:|---:|---:|---|
| Train | `train.parquet` | 20/09/2018 – 01/09/2020 | 1.242.735 | 50.000 | 71.357 | Huấn luyện tower (cặp *lịch sử → món kế tiếp*) |
| Chọn checkpoint ("selection", file cũ tên `rerank_train`) | `rerank_train.parquet` | 02/09 – 08/09/2020 | 10.452 | 3.063 | 4.569 | Chọn epoch tower; **nhãn huấn luyện của reranker** |
| Validation | `valid.parquet` | 09/09 – 15/09/2020 | 10.690 | 3.030 | 4.505 | Dừng sớm reranker; quét 3 tham số luật phục vụ |
| Test | `test.parquet` | 16/09 – 22/09/2020 | 9.800 | 2.799 | 4.200 | Chỉ để báo cáo |

Ngữ cảnh (lịch sử) tại thời điểm dự đoán: cửa sổ selection dùng `train`; valid dùng `train + selection`; test dùng `train + selection + valid`. Nhãn của một cửa sổ chỉ gồm các item khách đó mua trong cửa sổ; khách không mua gì trong cửa sổ không được tính vào metric.

**Refit.** Sau khi chốt số epoch và tham số bằng cửa sổ selection/valid, tower và reranker được train lại trên dữ liệu gồm thêm selection + valid để chấm test (xem `02_methods.md`, mục 6). Số **valid** vì thế luôn do mô hình chưa thấy valid; số **test** do mô hình refit.

## 4. Đặc điểm dữ liệu quyết định thiết kế

| Hiện tượng | Số đo | Hệ quả |
|---|---|---|
| Hoạt động của khách | 24,85 giao dịch/khách (trung vị 14), 7,2 ngày mua khác nhau/khách; test có 3,5 item/khách (trung vị 2) | Cửa sổ 7 ngày rất ngắn: xếp đúng vài món trong 105.542 là bài toán rất khó |
| Mua lại | 2,36% đích test đã nằm trong lịch sử train của khách | Nhóm "repeat" nhỏ nhưng dễ (Hit@12 ≈ 0,65 với reranker) |
| Popularity toàn thời gian lỗi thời | Top-12 / 100 / 1000 popularity (train) chỉ trúng 0,72% / 4,52% / 12,55% sự kiện test | Popularity cổ điển là baseline yếu trong thời trang |
| Item đang bán rất ít so với catalog | Mỗi tuần chỉ ~4,2–5,1 nghìn / 105 nghìn item xuất hiện trong mẫu | Phần lớn catalog là hàng "không hoạt động" trong tuần đích |
| Tính thời vụ mạnh | Item đích test đã bán trong 7 / 14 / 28 / 56 ngày trước: **68,9% / 77,6% / 84,0% / 87,2%** (mẫu 50k; với thống kê toàn cục tỉ lệ còn cao hơn) | "Bán chạy tuần trước" là baseline rất mạnh; tower không biết "đang bán" sẽ lãng phí năng lực |
| Cold-start sản phẩm | Sự kiện test có item **chưa từng xuất hiện trong train**: 29,6% (valid 23,2%; selection 11,6%); sau khi tính cả selection + valid: **10,3%** (611 khách test có ≥ 1 đích cold theo tower refit). Theo thống kê **toàn cục** (mọi khách), nhóm *never-sold* thật sự còn **236 khách** | Hai loại cold khác bản chất: (a) *mới-bán-gần-đây* (có thống kê bán, tower train cũ chưa biết) và (b) *chưa-từng-bán* (không có bất kỳ tín hiệu tương tác nào) |
| Cold người dùng | 0% | Không đo được; ghi vào hạn chế |

## 5. Thống kê item theo thời điểm (không rò rỉ)

Cho mỗi cửa sổ đích với ngày bắt đầu `c` (cutoff): `c_selection` = 02/09, `c_valid` = 09/09, `c_test` = 16/09/2020.
* `pop_all(c)` = số giao dịch của item trước `c`; `pop_recent(c)` = số giao dịch trong 7 ngày `[c−7, c)`; `first_seen` = ngày đầu tiên item xuất hiện; `age(c)` = `c − first_seen` nếu `first_seen < c`, ngược lại −1 (item chưa bán); `price(c)` = giá trung bình các giao dịch trước `c`.
* Nguồn đếm: file `item_daily_counts.parquet` (đếm theo (`article_id`, `t_dat`) trên **toàn bộ** 31,8 triệu giao dịch, chỉ lấy ngày `< c`) nếu có; nếu không, mẫu 50k khách. Đây là tín hiệu cấp **item, không cá nhân hoá**, và production luôn có sẵn nên không phải rò rỉ nhãn. Tuy nhiên đó là một giả định mô hình hoá cần nêu rõ trong luận văn: các số báo cáo cuối cùng dùng thống kê **toàn cục**.
* Item *never-sold* tại `c` = `first_seen ≥ c` theo nguồn đếm đang dùng.

## 6. Định nghĩa metric

Tất cả metric là **full-ranking**: xếp hạng toàn bộ 105.542 item (cho phép mua lại, không loại item trong lịch sử), lấy top-K, so với tập item khách mua trong cửa sổ (`T` = tập đích).

| Metric | Công thức (mỗi khách, rồi lấy trung bình trên khách có ít nhất 1 đích) |
|---|---|
| HitRate@K | 1 nếu top-K chứa ≥ 1 phần tử của `T`, ngược lại 0 |
| Recall@K | `|top-K ∩ T| / |T|` |
| NDCG@K | `DCG@K / IDCG@K`, `IDCG` tính với `min(K, |T|)` đích, chiết khấu `1/log2(vị trí + 1)` |
| MAP@K | `Σ_{vị trí ≤ K} P@vị trí · rel(vị trí) / min(K, |T|)` |
| CandidateRecall / CandidateHitRate | Recall / HitRate của **tập ứng viên** (union) so với `T` — cận trên của reranker |
| CatalogCoverage@K | Số item khác nhau xuất hiện trong top-K của mọi khách / 105.542 |

K báo cáo: 12 (chuẩn cuộc thi H&M), 50, 100 (và 500, 1000 cho tower). **Số headline = full-ranking trên test.**

### Các lát (slice) báo cáo
Mỗi lát lọc `T` của từng khách (khách không còn đích nào bị loại khỏi lát đó, nên số khách thay đổi theo lát):
* `warm_items`: item có `degree > 0` trong tập fit của tower đang dùng; `strict_cold_items`: `degree = 0` (cold theo **tower**).
* `never_sold_items`: item chưa có giao dịch nào trước cutoff theo nguồn thống kê (cold theo **dữ liệu**; tập con thực sự khó).
* `repeat_items` / `explore_items`: item có / không có trong lịch sử của khách.

### Baseline luôn in kèm
`popularity` (đếm toàn thời gian trước cutoff), `recent_popularity` (đếm 7 ngày trước cutoff, cùng cho mọi khách), `random`.

### Giao thức phụ: sampled 1 + 99 (giống `sampled_ranking_evaluate` của bộ Amazon)
Với **mỗi cặp** (khách, item đích): xếp hạng item đích cùng 99 item âm tính được lấy ngẫu nhiên từ catalog, loại lịch sử và **mọi** item khách mua trong cửa sổ đích. Hai chiến lược lấy mẫu: `uniform` (đều) và `popularity` (theo số lần mua, khó hơn). Đồng hạng tính trung bình (`rank = #âm_cao_hơn + 0,5·#đồng_hạng`); `HitRate@10/20/50`, `NDCG@10`, `MRR`; khoảng tin cậy 95% bootstrap/ratio theo **cụm khách**. Kiểm tra giao thức khi khởi động notebook: oracle đạt HitRate@10 = 1,000; random đạt 0,101 (kỳ vọng 0,100).

Quy tắc đọc số sampled (đã rút ra từ kết quả, xem `03`, mục 9):
1. Chỉ là bảng **phụ**, không so với full-ranking, không so giữa hai giao thức/bộ dữ liệu.
2. Luôn đọc cạnh `random` và baseline. Với negative `uniform`, phần lớn negative là item "không hoạt động" nên chỉ cần biết "item này có đang bán không" là đạt HR@10 ≈ 0,88 (`recent_popularity`).
3. Một reranker chỉ được train trên tập ứng viên khi chấm trên negative ngẫu nhiên là chấm *ngoài phân phối* train; bản "union-gated" (item ngoài union xếp cuối) ≈ recall của union và chỉ để tham khảo.

### Kiểm định ý nghĩa thống kê
Mọi so sánh quan trọng dùng **bootstrap ghép cặp theo khách** (2.000 lần lấy mẫu lại) trên chênh lệch HitRate@12 / NDCG@12; chỉ coi là cải thiện khi khoảng tin cậy 95% loại trừ 0. Sai số chuẩn của HitRate@12 ở ~2.800 khách là ≈ 0,006, nên chênh lệch nhỏ hơn ~1 điểm phần trăm thường không phân biệt được với nhiễu.

## 7. So sánh với bộ Amazon

| | Amazon (`balanced_u5_i2_v1`) | H&M (`hm_v1`) |
|---|---|---|
| Bài toán | Mục kế tiếp của từng khách (leave-last-out), 32.557 item | Giỏ hàng 7 ngày kế tiếp, 105.542 item |
| Tín hiệu | Review (rating ≥ 4), rất thưa | Giao dịch mua, dày hơn (24,9/khách) |
| Chia | Theo từng khách | Cửa sổ toàn cục theo thời gian |
| Review văn bản | Có (RAG) | Không |
| Vai trò thời gian | Yếu | **Rất mạnh** (tính thời vụ, vòng đời sản phẩm) |

Các con số HR/NDCG của hai bộ **không so sánh trực tiếp** được (khác bài toán, catalog, cửa sổ, cách chia).
