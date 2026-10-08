# H&M — Thực nghiệm và kết quả

> Cập nhật 05/10/2026. Đây là bản ghi **đầy đủ và có thứ tự** các thực nghiệm trong phiên làm việc 05/10/2026 trên bộ `hm_v1` (mẫu 50.000 khách, xem `01_dataset_and_protocol.md`; phương pháp ở `02_methods.md`). Mỗi mục ghi rõ: môi trường chạy, cấu hình, kết quả, nguồn số liệu và cách đọc. Số liệu "Kaggle" lấy từ JSON do notebook sinh ra (lưu ở [`results/`](results/)); số liệu "cục bộ" từ các lần chạy trên RTX 3050 4GB bằng `hm/scripts/` (lệnh tái lập kèm theo).

## 0. Tóm tắt kết quả

**Kết quả chính (test, full-ranking trên 105.542 item, 2.799 khách, tower + reranker refit, thống kê item toàn cục):**

| Phương pháp | HitRate@12 | HitRate@50 | HitRate@100 | Recall@100 | NDCG@12 | MAP@12 |
|---|---:|---:|---:|---:|---:|---:|
| Popularity toàn thời gian | 0,0186 | 0,0715 | 0,1165 | 0,0479 | 0,0046 | 0,0025 |
| Bán chạy 7 ngày trước (`recent_popularity`) | 0,0757 | 0,1708 | 0,2529 | 0,1180 | 0,0186 | 0,0107 |
| User Tower thuần (không luật phục vụ) | 0,0564 | 0,1061 | 0,1383 | 0,0683 | 0,0175 | 0,0112 |
| User Tower + luật phục vụ | 0,1208 | 0,2451 | 0,3455 | 0,1711 | 0,0351 | 0,0217 |
| Tower + reranker, 7 đặc trưng kiểu v1 | 0,0961 | 0,1836 | 0,2687 | 0,1311 | 0,0360 | 0,0252 |
| **Tower + reranker, 24 đặc trưng** | **0,1275** | **0,2701** | **0,3651** | **0,1923** | **0,0456** | **0,0310** |
| + chèn 1 item cold mỗi 10 vị trí | 0,1229 | 0,2558 | 0,3508 | 0,1823 | 0,0447 | 0,0307 |
| + chèn 1 item cold mỗi 5 vị trí | 0,1186 | 0,2437 | 0,3358 | 0,1738 | 0,0435 | 0,0301 |

Nguồn: [`results/retrieval_metrics.json`](results/retrieval_metrics.json) (`test`) và [`results/reranker_metrics.json`](results/reranker_metrics.json) (`test`).

**Các kết luận được hỗ trợ bằng số liệu:**
1. Trong thời trang, **tính thời vụ chi phối**: baseline "bán chạy 7 ngày trước" (Hit@12 0,076) vượt User Tower thuần (0,056) và gấp ~4 lần popularity toàn thời gian (0,019). Thêm tín hiệu "đang bán" vào tower tăng Hit@12 từ 0,056 lên 0,121 và Hit@100 từ 0,138 lên 0,346 (gấp 2,5 lần) mà không train lại.
2. Reranker 24 đặc trưng vượt baseline `recent_popularity` +0,052 HitRate@12 (CI 95% [+0,037, +0,066]) và vượt reranker 7 đặc trưng +0,031 [+0,020, +0,042].
3. **ID-dropout** đưa xếp hạng item cold từ *thấp hơn random* lên *cao hơn random* (mục E1), nhưng cold *thật sự* (item chưa từng bán) vẫn gần như không giải được bằng tương tác (mục E9–E11).
4. Cold thật sự có tín hiệu nội dung từ CLIP ≈ 5–21 lần random (trong pool, tuỳ K và tổ hợp tín hiệu), khai thác bằng kênh cold + chèn vị trí; đánh đổi định lượng được (mục E11).

**Các điều chưa được chứng minh** (phải nói trong luận văn): phần đóng góp *tăng thêm* của reranker so với tower + luật phục vụ ở Hit@12 (+0,0067, nhỏ hơn một sai số chuẩn ≈ 0,006; chưa có CI ghép cặp); đóng góp riêng của từng modality (id/text/image) trên H&M (chưa chạy ablation); độ ổn định theo seed (một seed); cold-start người dùng. Chi tiết ở mục 13.

---

## E0. Chẩn đoán dữ liệu và notebook gốc (v1)
* **Tái lập**: `python hm/scripts/dataset_stats.py` → [`results/dataset_stats.json`](results/dataset_stats.json).
* Kết quả: 29,6% sự kiện test là item chưa có trong train; chỉ ~4,5 nghìn / 105 nghìn item bán mỗi tuần; 68,9% đích test đã bán trong 7 ngày trước (mẫu); top-12 popularity trúng 0,72% sự kiện test (`01`, mục 4).
* Các khiếm khuyết của notebook 02/03 v1 đã xác định: (i) hàng ID của item cold giữ giá trị khởi tạo ngẫu nhiên (mục 2.4 của `02`); (ii) không có thông tin thời gian trong reranker (chỉ popularity toàn thời gian); (iii) không có sampled 1+99; (iv) không có baseline "bán chạy tuần trước"; (v) chọn checkpoint bằng cửa sổ không phản ánh tỉ lệ cold của test (11,6% so với 29,6%); (vi) tower không thấy 2 tuần cuối khi chấm test. Tài liệu chẩn đoán cũ (`hm_retrieval_diagnosis_and_business_metrics.md`) mô tả một phiên bản notebook trước đó và đã được thay thế (xem mục 14, bài học cuối).

## E1. Tower v1 so với v2 (cục bộ, 4 epoch, không refit)
Cùng dữ liệu/seed/4 epoch, chỉ khác: v1 = `ID_DROPOUT=0, COLD_FIX=0`; v2 = `ID_DROPOUT=0,3`. Test, context = train + selection + valid, tower base.

| | v1 | v2 |
|---|---:|---:|
| HitRate@100 toàn bộ | 0,1008 | 0,1140 |
| HitRate@100 item warm | 0,1168 | 0,1318 |
| HitRate@100 item cold | 0,0007 | 0,0037 |
| 1+99 uniform, tower, toàn bộ HR@10 | 0,4766 | 0,4970 |
| 1+99 uniform, **đích cold**, HR@10 (random ≈ 0,090) | **0,0326** | **0,1458** |
| 1+99 popularity-neg, đích cold, HR@10 (random ≈ 0,095) | 0,0173 | 0,0542 |

Quét cold-fix trên cửa sổ selection (Recall@100; `T` = ngưỡng degree để tắt ID):

| `T` | 0 (tắt) | 1 | 2 | 3 | 5 | 10 |
|---|---:|---:|---:|---:|---:|---:|
| v1 (không ID-dropout) | 0,0601 | **0,0188** | 0,0204 | 0,0224 | 0,0260 | 0,0293 |
| v2 (ID-dropout 0,3) | 0,0655 | 0,0601 | 0,0576 | 0,0587 | 0,0594 | 0,0595 |

Cách đọc: tắt ID trên mô hình *không* được train với ID-dropout làm sụp đổ chất lượng (−69%); ID-dropout làm mô hình chịu được việc bỏ ID (−8%). Cả hai trường hợp `T = 0` có Recall@100 cao nhất trên cửa sổ selection ⇒ cold-fix lúc suy luận **không** được chọn; lợi ích cold đến từ ID-dropout. (Lưu ý: cold HR@10 của v2 cao hơn random trong giao thức uniform không có nghĩa là cold được giải quyết; ở popularity-neg và full-ranking cold vẫn thấp.)

## E2. Kiểm tra giao thức 1+99
Mỗi lần chạy notebook 02 in: oracle HitRate@10 = 1,000; random 0,101 (kỳ vọng 0,100) trên 200 khách valid. Số cặp: valid 9.532, test 8.715.

## E3. Reranker v2 so với v1 (cục bộ, tower 4 epoch của E1, không refit, thống kê item từ mẫu 50k, 22 đặc trưng, chưa có kênh cold)
Test, full-ranking:

| | HitRate@12 | HitRate@50 | HitRate@100 | Recall@100 | NDCG@12 |
|---|---:|---:|---:|---:|---:|
| Reranker 22 đặc trưng | 0,1229 | 0,2354 | 0,3308 | 0,1676 | 0,0439 |
| Reranker 7 đặc trưng | 0,0832 | 0,1643 | 0,2540 | 0,1191 | 0,0297 |
| `recent_popularity` | 0,0597 | 0,1583 | 0,2333 | 0,1021 | 0,0147 |
| `popularity` | 0,0207 | 0,0682 | 0,1197 | 0,0494 | 0,0049 |

Bootstrap ghép cặp (HitRate@12): reranker − recent_popularity = **+0,0632 [+0,0493, +0,0768]**; 7 đặc trưng − recent_popularity = +0,0236 [+0,0104, +0,0361]; 22 − 7 đặc trưng = **+0,0397 [+0,0282, +0,0504]**. Union ứng viên: recall 0,321, hit 0,547 (cold: 0,263 / 0,362). Valid: 0,1294 / 0,2452 / 0,3231 (reranker), 0,0875 / 0,1845 / 0,2601 (7 đặc trưng), CI vs recent_popularity +0,0792 [+0,0663, +0,0927].
Cách đọc: đặc trưng thời gian, tuổi item, affinity và giá là phần tạo ra phần lớn khác biệt so với bộ 7 đặc trưng kiểu v1.

## E4. Lần chạy Kaggle thứ nhất (8 epoch + refit; chưa có luật phục vụ; thống kê mẫu 50k)
Tower, full-ranking:

| | Hit@12 | Hit@50 | Hit@100 | Recall@100 | cold Hit@100 |
|---|---:|---:|---:|---:|---:|
| Valid (base) | 0,0459 | 0,0977 | 0,1356 | 0,0656 | 0,0008 |
| Test base | 0,0497 | 0,0904 | 0,1236 | 0,0589 | 0,0015 |
| Test refit | 0,0564 | 0,1061 | 0,1383 | 0,0683 | 0,0000 (611 khách) |

Refit giảm tỉ lệ đích cold của test từ 29,2% xuống 10,2% và nâng Hit@12 từ 0,0497 lên 0,0564; nhưng chính nhóm cold còn lại (611 khách) có Hit@100 = 0. 1+99 uniform, đích cold, test refit: HR@10 = 0,0517 (random 0,0966) — thấp hơn random.
Đối chiếu: `recent_popularity` (mẫu 50k) 0,0757 Hit@12 / 0,2529 Hit@100 > tower thuần ở mọi K.

## E5. Thăm dò bộ lọc "item đang bán" (cục bộ, chỉ suy luận)
* **Tái lập**: `python hm/scripts/probe_serving_rule.py --ckpt <best_retrieval.pt>` (thống kê mẫu 50k, tower 4 epoch cục bộ).
* Chỉ xếp hạng trong pool item có giao dịch trong `D` ngày trước cutoff (test, Hit@12 / Hit@100 / cold Hit@100):

| `D` | pool | Hit@12 | Hit@100 | cold Hit@100 |
|---|---:|---:|---:|---:|
| không lọc | 105.542 | 0,0443 | 0,1140 | 0,0037 |
| 56 | 16.473 | 0,0475 | 0,1300 | 0,0066 |
| 28 | 10.702 | 0,0518 | 0,1490 | 0,0073 |
| 14 | 6.898 | 0,0604 | 0,1708 | 0,0147 |
| 7 | 4.505 | 0,0657 | 0,1819 | 0,0162 |

Cách đọc: tower lãng phí phần lớn năng lực xếp hạng item không hoạt động; lọc theo "đang bán" tăng Hit@12 +48% (pool 7 ngày).

## E6. Lần chạy Kaggle thứ hai (luật phục vụ v1, chọn tham số trên cửa sổ selection; thống kê toàn cục)
Luật chọn `b = 0` (cold_bonus), tức không có cộng điểm cold; (xem lý do ở `02`, mục 2.5). Test refit: Hit@12 0,0618, Hit@100 0,1576, cold Hit@100 0,0229; không luật 0,0564 / 0,1383 / 0. Thống kê toàn cục làm `recent_popularity` mạnh hơn hẳn: 1+99 uniform HR@10 = **0,885** (mẫu 50k: 0,68), Hit@100 0,2529; tower thuần vẫn thua nó.

## E7. Thăm dò trộn điểm tower với độ bán chạy và cộng điểm cold (cục bộ, chỉ suy luận)
Cùng lệnh E5 (bảng đầy đủ do script in). Test (thống kê mẫu 50k, tower 4 epoch):

| `D / b / w` | Hit@12 | Hit@50 | Hit@100 | cold Hit@100 |
|---|---:|---:|---:|---:|
| 14 / 0 / 0 | 0,0604 | 0,1218 | 0,1708 | 0,0147 |
| 14 / 0 / 0,05 | 0,0932 | 0,1936 | 0,2705 | 0,0411 |
| 14 / 0 / 0,1 | 0,0961 | 0,2133 | 0,2940 | 0,0668 |
| 14 / 0,1 / 0,05 | 0,1090 | 0,2126 | 0,2944 | 0,1564 |
| 14 / 0,1 / 0,1 | 0,1036 | 0,2308 | 0,3169 | 0,1865 |
| 0 / 0,2 / 0,1 | 0,0982 | 0,2047 | 0,2933 | 0,3142 |

Cách đọc: thêm `w·log(1 + c7)` là bước nhảy lớn nhất (Hit@12 0,060 → 0,096); thêm `b` tăng hit cold rất mạnh (cold Hit@100 0,067 → 0,187) mà không làm giảm Hit@100 tổng. Khi `w > 0`, bộ lọc `D` gần như không còn tác dụng (so `0 / 0,1 / 0,1` với `14 / 0,1 / 0,1`). Đây là cơ sở để chuyển việc quét tham số sang cửa sổ valid (selection không thấy item cold đang bán).

## E8. Lần chạy Kaggle thứ ba (luật phục vụ v2, quét trên valid; thống kê toàn cục) — kết quả tower cuối cùng
Cấu hình: `ID_DROPOUT 0,3`, `best_epoch 7`, `cold_min_degree 0`, luật phục vụ **`D = 0, b = 0,2, w = 0,1`**, thống kê toàn cục (31.788.324 giao dịch). Quét trên valid (Recall@100): không luật 0,0650; tốt nhất 0,1550 (`0d/b0,2/w0,1`, cũng `14d` và `28d` giống hệt); các lựa chọn kế tiếp `b0,2/w0,05` 0,1544, `b0,1/w0,05` 0,1503.

| | Hit@12 | Hit@50 | Hit@100 | Recall@100 | cold Hit@100 | (số khách cold) |
|---|---:|---:|---:|---:|---:|---:|
| Valid, có luật | 0,0960 | 0,2234 | 0,3109 | 0,1550 | 0,0587 | 1.209 |
| Valid, không luật | 0,0455 | 0,0983 | 0,1343 | 0,0650 | – | – |
| Test base, có luật | 0,1115 | 0,2422 | 0,3312 | 0,1638 | 0,1615 | 1.362 |
| Test base, không luật | 0,0497 | 0,0915 | 0,1222 | 0,0581 | – | – |
| **Test refit, có luật** | **0,1208** | **0,2451** | **0,3455** | **0,1711** | 0,0196 | 611 |
| Test refit, không luật | 0,0564 | 0,1061 | 0,1383 | 0,0683 | – | – |
| `recent_popularity` (test) | 0,0757 | 0,1708 | 0,2529 | 0,1180 | 0 | – |

Lưu ý quan trọng khi đọc: cold Hit@100 của "test base" (0,1615, tỉ lệ cold 29,2%) cao vì nhóm cold chủ yếu là item *mới-bán-gần-đây* (có thống kê bán toàn cục, `b` cộng điểm giúp chúng lên top); sau refit, các item này đã warm, và nhóm cold còn lại (611 khách, 10,2%) là item hầu như chưa bán ⇒ Hit@100 chỉ 0,0196. Hai con số mô tả hai hiện tượng cold khác nhau.

Đường cong huấn luyện (cửa sổ selection, `retrieval_history.json`; Recall@100 / HitRate@100, áp `min_degree = 1`):

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | **7** | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| loss | 4,679 | 4,278 | 4,218 | 4,165 | 4,112 | 4,062 | 4,009 | 3,956 |
| Recall@100 | 0,0494 | 0,0569 | 0,0584 | 0,0609 | 0,0613 | 0,0602 | **0,0680** | 0,0625 |
| HitRate@100 | 0,1002 | 0,1133 | 0,1159 | 0,1237 | 0,1254 | 0,1244 | 0,1312 | 0,1247 |

Epoch 7 là một đỉnh đột ngột (+0,0078 so với epoch 6, −0,0055 ở epoch 8) — gợi ý nhiễu chọn epoch cỡ ±0,003 Recall@100; hiệu ứng lên kết quả cuối được hạn chế vì refit dùng 7 epoch cố định.

## E9. Thăm dò cold thật sự bằng CLIP (cục bộ, chỉ numpy)
* **Tái lập**: `python hm/scripts/probe_cold_clip.py`.
* Pool = 33.013 item chưa bán trước cutoff test (mẫu); 611 khách có đích trong pool (890 sự kiện, 512 item khác nhau). Hit@K *trong pool*:

| Cách xếp hạng trong pool | Hit@100 | Hit@500 |
|---|---:|---:|
| Random (kỳ vọng) | ≈ 0,005 | ≈ 0,025 |
| Chỉ độ giống CLIP với hồ sơ khách | 0,0376 | 0,1195 |
| Chỉ prior "giống hàng mới ra 56 ngày" | 0,0376 | 0,1538 |
| `z(khách) + z(prior)` (56 ngày) | 0,0867 | 0,1866 |
| `0,5·z(khách) + z(prior)` (56 ngày) | **0,0917** | **0,2095** |

Cách đọc: thông tin cold có thật nhưng thưa — thứ duy nhất dự báo item chưa bán sắp được mua là *nó giống hàng mới ra gần đây* (prior) kết hợp sở thích khách. Đây là cơ sở của kênh cold (`02`, mục 5). Chú ý: không so trực tiếp 0,092 (xếp hạng riêng trong pool 33 nghìn) với Hit@100 toàn catalog.

## E10. Kênh cold trong pipeline: tập ứng viên tăng nhưng ranker không đẩy lên (cục bộ, tower 4 epoch)
So sánh `HM_COLD_K=0` và `100` (test, cùng ranker, thống kê mẫu):

| | Union recall tổng | Union recall nhóm never-sold | Reranker Hit@12 | Hit@100 | Hit@100 nhóm never-sold (611 khách) |
|---|---:|---:|---:|---:|---:|
| Không kênh cold | 0,321 | 0,030 | 0,1279 | 0,3380 | 0,0033 |
| Có kênh cold (K = 100) | 0,327 | **0,098** | 0,1136 | 0,3390 | 0,0000 |

Kênh cold nhân ba recall ứng viên của nhóm never-sold nhưng LightGBM vẫn xếp chúng cuối vì nhãn cold hiếm trong dữ liệu train của ranker và mọi đặc trưng tương tác của chúng bằng 0. Hit@12 lệch −0,014 giữa hai lần chạy (một seed; trong khoảng nhiễu của lần huấn luyện lại).

## E11. Chèn vị trí cold (cục bộ rồi Kaggle)
**Cục bộ** (E10, `HM_COLD_EVERY`): nhóm never-sold Hit@100 0 → **0,0213** (mỗi 10) → **0,0344** (mỗi 5); Hit@100 tổng 0,339 → 0,328 → 0,316.
**Kaggle (kết quả cuối, test, 236 khách never-sold theo thống kê toàn cục; union recall của nhóm này 0,189):**

| | Hit@12 | Hit@100 tổng | Hit@50 never-sold | Hit@100 never-sold | Recall@100 never-sold |
|---|---:|---:|---:|---:|---:|
| Reranker | 0,1275 | 0,3651 | 0,0000 | 0,0000 | 0,0000 |
| + mỗi 10 vị trí | 0,1229 | 0,3508 | 0,0085 | **0,0212** | 0,0191 |
| + mỗi 5 vị trí | 0,1186 | 0,3358 | 0,0212 | **0,0508** | 0,0466 |

Chi phí độ chính xác tổng: Hit@100 −0,0143 (mỗi 10), −0,0293 (mỗi 5); Hit@12 −0,0046 / −0,0089. Lợi ích: nhóm never-sold từ 0 lên 2,1% / 5,1% Hit@100. Hit@100 tổng giảm khi chèn, nghĩa là mỗi vị trí dành cho item cold mang lại ít hit hơn vị trí warm mà nó thay thế; chèn cold vì vậy **không** tăng chỉ số tổng — đây là lựa chọn sản phẩm (tăng độ phủ/tiếp xúc item mới), không phải tối ưu độ chính xác.

## E12. Kết quả cuối cùng của pipeline (Kaggle, test)
**(a) Phân tích theo lát** (HitRate@12 / HitRate@100; số khách trong ngoặc):

| Lát | Reranker 24 đặc trưng | Reranker 7 đặc trưng | `recent_popularity` | Tower + luật |
|---|---:|---:|---:|---:|
| Toàn bộ (2.799) | 0,1275 / 0,3651 | 0,0961 / 0,2687 | 0,0757 / 0,2529 | 0,1208 / 0,3455 |
| Warm (2.669) | 0,1315 / 0,3792 | 0,1000 / 0,2776 | 0,0794 / 0,2653 | 0,1263 / 0,3608 |
| Strict-cold theo tower (611) | 0,0131 / 0,0442 | 0,0033 / 0,0278 | 0 / 0 | 0,0016 / 0,0196 |
| Never-sold (236) | 0 / 0 | 0 / 0,0085 | 0 / 0 | – |
| Mua lại (287) | 0,6516 / 0,8885 | – | – | 0,2578 / 0,5122 |
| Khám phá (2.709) | 0,0716 / 0,3119 | – | – | 0,1000 / 0,3219 |

**(b) Tập ứng viên** (union ≈ 997 item): recall 0,434, hit 0,657 (warm 0,464 / 0,676; strict-cold 0,153 / 0,178; never-sold 0,189 / 0,208; repeat 1,000; explore 0,404 / 0,628).

**(c) Bootstrap ghép cặp (test, 2.799 khách, 2.000 lần lấy mẫu lại):**

| So sánh | HitRate@12 | NDCG@12 |
|---|---|---|
| Reranker − `recent_popularity` | **+0,0518 [+0,0368, +0,0661]** | +0,0269 [+0,0206, +0,0330] |
| Reranker 7 đặc trưng − `recent_popularity` | +0,0204 [+0,0064, +0,0347] | +0,0173 [+0,0110, +0,0235] |
| Reranker 24 − Reranker 7 đặc trưng | **+0,0314 [+0,0204, +0,0418]** | +0,0096 [+0,0060, +0,0132] |

(Valid: reranker − recent_popularity +0,0789 [+0,0657, +0,0924]; 24 − 7 đặc trưng +0,0383 [+0,0281, +0,0492].)

**(d) Chú ý đọc đúng**: Reranker − (Tower + luật phục vụ) = +0,0067 Hit@12, +0,0196 Hit@100, +0,0105 NDCG@12, +0,0093 MAP@12. Chưa có bootstrap ghép cặp cho so sánh này (JSON lưu số trung bình, không lưu theo khách). Với sai số chuẩn Hit@12 ≈ 0,006, chênh lệch +0,0067 **chưa chứng minh được** reranker cải thiện Hit@12 so với tower + luật phục vụ; ở K = 50/100 và NDCG/MAP chênh lệch lớn hơn nhưng vẫn cần CI.

**(e) Đối chiếu độ khó**: MAP@12 test = 0,0310 (reranker). Không so trực tiếp với bảng xếp hạng cuộc thi Kaggle (mẫu 50k khách, chỉ tính khách có mua trong tuần, thống kê toàn cục dùng theo cách khác, không tái hiện submission).

## E13. Sampled 1 + 99 (kết quả cuối, test, 8.715 cặp; HR@10 ± CI95)

| Bộ chấm | negative `uniform` | negative `popularity` |
|---|---:|---:|
| random | 0,100 ± 0,006 | 0,094 ± 0,006 |
| popularity | 0,354 ± 0,012 | 0,084 ± 0,006 |
| tower thuần (điểm thô, không áp luật phục vụ) | 0,606 ± 0,013 | 0,329 ± 0,013 |
| `recent_popularity` | 0,882 ± 0,009 | 0,574 ± 0,014 |
| **Reranker 24 đặc trưng** | **0,906 ± 0,008** | **0,653 ± 0,013** |
| Reranker 7 đặc trưng | 0,378 ± 0,012 | 0,236 ± 0,010 |
| Reranker, item ngoài union xếp cuối ("union-gated") | 0,419 ± 0,013 | 0,417 ± 0,013 |

Theo lát (uniform, reranker / tower thuần / random): warm 0,939 / 0,669 / 0,096; **cold theo tower 0,615 / 0,054 / 0,085** (890 cặp). Bảng này chỉ để đối chiếu giao thức với Amazon (Amazon, 1+99 uniform: pipeline 46,4%, popularity 34,8%, random 10,2%) — hai bộ dữ liệu không so sánh được với nhau.
Cách đọc: (1) `recent_popularity` đạt 0,88 chỉ vì phần lớn negative uniform là hàng không hoạt động; (2) reranker 7 đặc trưng thấp (0,378) trong khi vượt `recent_popularity` ở full-ranking — vì khi chấm negative ngẫu nhiên ngoài tập ứng viên, mô hình thiếu đặc trưng thời gian chấm sai (ngoài phân phối); (3) bản union-gated chỉ phản ánh recall của union.

## 12. Thời gian và tài nguyên
* Tower: ≈ 75 giây/epoch (Kaggle), ≈ 99 giây/epoch (RTX 3050 4GB, 985.376 cặp, batch 512, ≈ 59 ms/bước); 8 epoch + refit 7 epoch ≈ 20 phút Kaggle (chưa tính đánh giá).
* Notebook 03 (dựng 3 cửa sổ ứng viên + 24 đặc trưng + LightGBM + đánh giá): ≈ 3 phút cục bộ ở kích thước đầy đủ.
* Quét 3 tham số luật phục vụ (84 cấu hình trên valid): vài phút.

## 13. Các hạn chế và việc chưa làm (phải nêu rõ trong luận văn)
1. **Ablation modality chưa chạy** (`HM_MODALITY = id | text | image | multimodal`): chưa có bằng chứng riêng về đóng góp của text/ảnh trên H&M — câu hỏi nghiên cứu cốt lõi của đề tài. Bằng chứng gián tiếp duy nhất: CLIP cho tín hiệu cold ≈ 5–21× random trong pool (E9).
2. **Một seed**; chưa có độ lệch giữa các lần chạy. Nhiễu chọn epoch cỡ ±0,003 Recall@100.
3. **Không có CI ghép cặp** cho Reranker − (Tower + luật phục vụ) (E12d).
4. **Cold người dùng không được đo** (mẫu chỉ có khách có lịch sử).
5. **Thống kê item toàn cục** là giả định hệ thống; số liệu trên mẫu 50k khách cho kết quả khác (E3 so với E8).
6. **Tham số luật phục vụ chọn trên valid** (số valid của tower không sạch); một số quyết định thiết kế (trộn độ bán chạy, kênh cold, chèn vị trí) được đưa ra sau khi xem chẩn đoán trên cả valid và test; dữ liệu đã hết sau tuần test nên không có cửa sổ chưa dùng để kiểm tra lại.
7. Reranker train trên ~1,7–2,0 nghìn nhóm (một cửa sổ 7 ngày) và dừng sớm ở 84 vòng: dữ liệu train của ranker là nút thắt tiềm năng; thêm cửa sổ cần thêm tower theo mốc thời gian để tránh rò rỉ.
8. Profile `full` (300.000 khách) chưa chạy.
9. Chưa thử kiến trúc user encoder khác (Transformer/attention), chưa có đặc trưng khách (tuổi, hạng thành viên) vì `customers.csv` không được xuất trong `hm_v1`.

## 14. Bài học kỹ thuật
* **Cold có hai loại.** "Mới-bán-gần-đây" giải được bằng thống kê bán hàng (đếm 7 ngày + cộng điểm cold + refit); "chưa-từng-bán" chỉ còn tín hiệu nội dung ⇒ là giới hạn thông tin.
* **Cửa sổ chọn tham số phải chứa hiện tượng cần chọn**: selection không có item cold đang bán nên không thể quét `cold_bonus`; chọn ở đó luôn cho 0 (E6).
* **Metric sampled phải đọc cạnh baseline**: với negative uniform trong catalog 105 nghìn item, "item có đang bán không" quyết định gần hết kết quả (E13).
* **Chấm điểm ngoài phân phối train**: chỉ chấm negative ngẫu nhiên bằng reranker huấn luyện trên tập ứng viên cho kết quả sai; bản gated chỉ phản ánh recall của union.
* **Chi phí chạy**: tính AP O(K²) của bản v1 làm đánh giá chậm nhiều lần; v2 vectorize bằng cộng dồn.
* **Thay thế tài liệu cũ**: `hm_retrieval_diagnosis_and_business_metrics.md` chứa khoảng giá trị kỳ vọng (ví dụ cold HitRate 8–15%, lift 150–250×) chưa từng được đo và đã bị số liệu mới bác bỏ (cold Hit@100 sau refit = 2,0% cho tower và 4,4% cho reranker); đã gỡ khỏi repo (còn trong lịch sử git; xem `README.md` của thư mục này, mục "Tài liệu đã gỡ").

