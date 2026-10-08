# Nhật ký phiên 05/10/2026 — H&M: cold-start, đánh giá 1+99, luật phục vụ, reranker v2

> Loại: nhật ký diễn tiến (chronological). Kết quả đã được chuẩn hoá và đầy đủ ở [`docs/hm/`](../README.md) — đặc biệt [`03_experiments_and_results.md`](../03_experiments_and_results.md). Tài liệu này ghi **thứ tự quyết định và lý do**, kể cả những bước đi sai và đã sửa, để phục vụ viết phần "quy trình" và "bài học" của luận văn.

## 1. Tóm tắt điều hành
* Yêu cầu: xem xét dữ liệu/mô hình H&M, sửa notebook train để xử lý cold-start, thêm đánh giá 1 + 99 như bộ Amazon, rồi cải thiện để đạt kết quả tốt nhất.
* Kết quả: viết lại notebook 02 (tower) và 03 (reranker), thêm xuất thống kê item toàn cục vào 00; HitRate@12 test từ ≈ 0,05 (tower thuần v1) lên 0,128 (pipeline cuối), vượt baseline "bán chạy tuần trước" (0,076) +0,052 [+0,037, +0,066].
* Phát hiện chủ đạo: trong thời trang, **tính thời vụ chi phối**; cold-start gồm hai loại; loại chưa-từng-bán là giới hạn thông tin.
* Chạy: thăm dò + thử nghiệm rút gọn trên RTX 3050 (có xin phép, ≤ 15 phút GPU cho mỗi đợt), 3 lần chạy đầy đủ trên Kaggle do chủ đề tài thực hiện.

## 2. Dòng thời gian

| # | Bước | Bằng chứng / kết quả | Quyết định |
|---|---|---|---|
| T1 | Đọc 4 notebook v1 và thống kê `data/hm` | 29,6% sự kiện test cold so với train; ~4,5k/105k item bán mỗi tuần; top-12 popularity trúng 0,7%; hàng ID item cold giữ giá trị khởi tạo ngẫu nhiên | Mục tiêu: ID-dropout, refit, thêm baseline "bán chạy tuần trước", sampled 1+99 |
| T2 | Viết lại notebook 02 (ID-dropout, cold-fix quét ngưỡng, refit, sampled 1+99, baseline, CI cụm) | Smoke test 2.000 khách: oracle 1,000, random 0,101 ≈ kỳ vọng | Giữ cờ môi trường để tái lập v1 |
| T3 | Vectorize tính AP/NDCG (O(K²) → O(K)) | Lần smoke đầu mất 8,5 phút, chủ yếu do đánh giá | — |
| T4 | Viết lại notebook 03 (6 nguồn, 22 đặc trưng, LightGBM, sampled, bootstrap; về sau thêm kênh cold → 7 nguồn, 24 đặc trưng) | Smoke chạy thông 27 giây | Thêm bản "union-gated" cho sampled |
| T5 | Chấm sampled bằng reranker "union-gated" | HR@10 = HR@20 ≈ recall của union (negative hầu như không vào union) | Đổi: số chính chấm trực tiếp mọi item; gated chỉ tham khảo, có chú thích |
| T6 | Đo tốc độ và xin phép chạy cục bộ | 59 ms/bước, ≈ 1,9 phút/epoch (RTX 3050) | Chủ đề tài chọn "chạy rút gọn" (4 epoch, không refit) |
| T7 | Chạy v2, v1, reranker (cục bộ) | v1: cold HR@10 (1+99 uniform) 0,033 < random; v2: 0,146. Tắt ID: v1 sụp (−69%), v2 giảm 8%. Reranker +0,063 Hit@12 so với recent_popularity | ID-dropout là phần giúp cold; cold-fix lúc suy luận không được chọn (ngưỡng 0) |
| T8 | Bổ sung đường dùng `item_daily_counts.parquet` (thống kê toàn cục) vào 00 và 03; kiểm tra tương đương bằng file giả dựng từ mẫu | Kết quả đường mẫu và đường "global giả" giống hệt | Giữ fallback về mẫu 50k |
| T9 | Kaggle lần 1 (8 epoch + refit) | Tower test refit Hit@12 0,056; cold sau refit = 0; recent_popularity (0,076) vẫn hơn | Cần đưa "đang bán" vào tower |
| T10 | Thăm dò bộ lọc "item đang bán" (cục bộ, chỉ suy luận) | Hit@12 0,044 → 0,066 (pool 7 ngày); cold Hit@100 0,004 → 0,016 | Thêm "luật phục vụ" vào 02, chọn tham số bằng Recall@100 |
| T11 | Kaggle lần 2 (luật phục vụ v1 quét trên selection) | 0,056 → 0,062 Hit@12; `cold_bonus` luôn 0 | Phân tích: cửa sổ selection không có item cold đang bán nên không chọn được `cold_bonus` |
| T12 | Thăm dò trộn `w·log(1+bán 7 ngày)` + `cold_bonus` (cục bộ) | Hit@12 0,060 → 0,104; Hit@100 0,171 → 0,317; cold Hit@100 0,015 → 0,187 | Luật phục vụ v2: 3 tham số, **quét trên valid** (ghi chú về số valid không sạch) |
| T13 | Kaggle lần 3 (luật v2, thống kê toàn cục) | Tower test refit Hit@12 0,121, Hit@100 0,346; chọn `D=0, b=0,2, w=0,1` | Tower đã mạnh hơn baseline; cold sau refit vẫn 0,0196 |
| T14 | Thăm dò cold thật sự bằng CLIP trong pool item chưa bán | Hit@100 trong pool: random 0,005; CLIP-khách 0,038; prior 0,038; tổ hợp 0,092 | Thêm kênh cold + `launch_prior` + lát `never_sold` vào 03 |
| T15 | Đo kênh cold trong pipeline (cục bộ) | Recall union của never-sold 0,030 → 0,098, nhưng LightGBM xếp chúng cuối | Thêm chèn vị trí cold (`HM_COLD_EVERY`) |
| T16 | Kaggle lần 4 (03 đầy đủ) | Reranker test Hit@12 0,1275, Hit@100 0,365; never-sold Hit@100 0 → 0,021 (mỗi 10) → 0,051 (mỗi 5), giá −0,014/−0,029 Hit@100 tổng | Chốt báo cáo cả ba cấu hình |
| T17 | Thêm cell xuất model vào 03 (`hm_export/`, zip) | Kiểm thử cục bộ: tower gọn 58 MB, hai LightGBM, `model_meta.json` | Reranker một mình không đủ suy luận (cần pipeline đặc trưng) |
| T18 | Đồng bộ tài liệu, gỡ tài liệu H&M cũ, lưu kết quả vào `docs/hm/results/` | Xem `docs/hm/README.md` | H&M được ghi nhận là bộ dữ liệu chính mới |

## 3. Sai sót trong quá trình và cách điều chỉnh (trung thực)
1. **Tune `cold_bonus` trên cửa sổ không chứa hiện tượng cần chọn** (T11) — mất một lượt chạy Kaggle trước khi nhận ra; sửa bằng quét trên valid.
2. **Thăm dò với ngữ cảnh test thiếu lịch sử** (script nháp): `tctx` chỉ dựng cho khách có trong valid, làm số test lệch (Hit@12 0,058 thay vì 0,044); phát hiện vì số "không lọc" không khớp lần chạy trước, sửa rồi chạy lại. Script đưa vào repo (`scripts/hm/probe_serving_rule.py`) đã dùng ngữ cảnh đúng.
3. **Chấm gated cho sampled** (T5) cho kết quả vô nghĩa (≈ recall union); đã đổi cách báo cáo.
4. **Một số quyết định thiết kế được đưa ra sau khi xem số trên cả valid và test** (T10, T12, T14): số test vì vậy có thể hơi lạc quan; dữ liệu đã hết sau tuần test nên không có cửa sổ sạch để kiểm tra lại. Nêu rõ trong hạn chế.
5. **Kênh cold lần đầu không giúp ranker** (T15): hiệu ứng nằm ở chính sách vị trí chứ không phải ở việc thêm ứng viên.
6. **Epoch chọn bằng cấu hình khác cấu hình cuối** (T9–T13): chọn epoch áp ngưỡng cold-fix = 1, trong khi ngưỡng cuối là 0; ghi nhận như hạn chế nhỏ.

## 4. Việc còn lại
Xem `docs/hm/05_thesis_plan.md` (mục 4 và 6): ablation modality (ưu tiên P0), CI ghép cặp reranker vs tower + luật, nhiều seed, và các quyết định D1–D6.
