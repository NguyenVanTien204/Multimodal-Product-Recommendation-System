# H&M — Kế hoạch chuyển luận văn sang bộ dữ liệu H&M

> Cập nhật 05/10/2026. Tài liệu này là **cầu nối giữa thực nghiệm H&M (`01`–`04`) và việc viết lại luận văn** (`docs/LuanVan/`, hiện dựng trên bộ Amazon). Nó trả lời: đã có gì để viết, điều gì được phép khẳng định, còn thiếu thực nghiệm nào, mỗi chương hiện tại phải sửa gì, và những quyết định nào cần chủ đề tài chốt. Không chứa kết luận nào chưa có số liệu; mọi số dẫn về `03_experiments_and_results.md`.

## 1. Hiện trạng bộ khung luận văn (Amazon)

Các file trong `docs/LuanVan/` được viết quanh bộ `balanced_u5_i2_v1` (21.690 khách, 32.557 sản phẩm, 198.200 tương tác, HR@10 test 3,518%, Candidate Recall 34,444%, kho tri thức Qdrant 152.086 sản phẩm / 295.383 review). Sự khác biệt giữa hai bộ không chỉ ở dữ liệu mà còn ở **thiết kế mô hình**:

| | Amazon (luận văn hiện tại) | H&M (tài liệu này) |
|---|---|---|
| Bài toán | Mục kế tiếp, leave-last-out | Giỏ hàng 7 ngày kế tiếp, cửa sổ toàn cục |
| User encoder | Transformer 2 lớp, 4 head, `L_max = 20`, right-padding | Trung bình có trọng số theo độ mới, `L = 30` |
| Item | ID khởi tạo bằng 0 + `W·CLIP` | ID (cổng, N(0,1)) + `W_t·text` + cổng·`W_m·ảnh`, **ID-dropout** |
| Mất mát | Sampled softmax, 512 âm (50% uniform), **log-Q** | Sampled softmax, 128 âm popularity^0,75, **không log-Q** |
| Ứng viên | 4 nguồn, 2.000 | 7 nguồn, ≈ 1.000, **kênh cold CLIP**, thống kê thời gian |
| Reranker | Residual listwise NN | **LightGBM LambdaRank**, 24 đặc trưng |
| Tín hiệu then chốt | Chuỗi hành vi + nội dung | **Tính thời vụ / đang bán** |
| Giao thức | Full-ranking HR/NDCG@10; sampled 1+99 phụ | Full-ranking @12/50/100 + slice; sampled 1+99 phụ; bootstrap ghép cặp |

**Hệ quả**: nếu luận văn dùng cả hai bộ, phải giải thích vì sao hai mô hình khác nhau (hoặc chuyển cùng một kiến trúc sang cả hai bộ — chưa làm). Đây là quyết định D1 ở mục 6.

## 2. Câu hỏi nghiên cứu viết lại cho H&M

Câu hỏi gốc của đề tài (`docs/01-vision.md`): *kết hợp hình ảnh và văn bản với lịch sử tương tác có cải thiện chất lượng gợi ý, đặc biệt với sản phẩm ít tương tác, so với collaborative thuần hay không?* Với H&M, dữ liệu cho thấy câu hỏi cần được tách thành các câu nhỏ hơn có thể trả lời bằng thực nghiệm:

| RQ | Câu hỏi | Trạng thái |
|---|---|---|
| RQ1 | Trong thời trang, tín hiệu **đang bán / thời vụ** đóng góp bao nhiêu so với hành vi cá nhân và nội dung? | **Có bằng chứng** (E5, E7, E8, E12) |
| RQ2 | **Đóng góp của từng modality** (ID / text / ảnh / đa phương thức) trên H&M, tổng thể và trên sản phẩm ít/không có tương tác? | **Chưa chạy** (mục 4, P0) |
| RQ3 | Cold-start sản phẩm: (a) mới-bán-gần-đây và (b) chưa-từng-bán — kỹ thuật nào giải quyết được loại nào? | **Có bằng chứng một phần** (E1, E8, E9–E11); thiếu so sánh theo modality |
| RQ4 | Giao thức nào phản ánh đúng chất lượng: full-ranking hay sampled 1+99? | **Có bằng chứng** (E13) |
| RQ5 | Reranker với đặc trưng đa dạng có cải thiện so với retrieval + luật phục vụ? | **Chưa kết luận** ở Hit@12 (E12d); cần CI ghép cặp |

## 3. Bảng luận điểm: được khẳng định, cần thực nghiệm, không nên khẳng định

### 3.1. Có thể khẳng định (có số liệu và CI/baseline)
1. "Bán chạy 7 ngày trước" mạnh hơn popularity toàn thời gian gần 4 lần (Hit@12 0,0757 vs 0,0186) và mạnh hơn tower thuần (0,0564); baseline này phải có mặt trong mọi so sánh H&M. (E12)
2. Thêm tín hiệu đang bán (và cộng điểm item cold) vào điểm tower — **không train lại** — nâng Hit@12 từ 0,056 lên 0,121 và Hit@100 từ 0,138 lên 0,346 (test, refit). (E8)
3. Reranker 24 đặc trưng vượt `recent_popularity` +0,052 HitRate@12, CI 95% [+0,037, +0,066], và vượt reranker 7 đặc trưng +0,031 [+0,020, +0,042]. (E12c)
4. ID-dropout đưa xếp hạng item cold từ thấp hơn random lên cao hơn random trong 1+99 uniform (0,033 → 0,146) và làm mô hình chịu được việc tắt ID (Recall@100 −8% thay vì −69%). (E1)
5. Cold-start gồm hai loại khác nhau về bản chất; loại chưa-từng-bán chỉ còn tín hiệu nội dung (CLIP + prior hàng mới ra: Hit@100 trong pool 0,092 so với ≈ 0,005 random). (E9)
6. Chèn vị trí cho kênh cold nâng Hit@100 nhóm never-sold từ 0 lên 2,1–5,1% với cái giá −1,4 đến −2,9 điểm Hit@100 tổng. (E11)
7. Sampled 1+99 với negative uniform bị chi phối bởi "item có đang bán không" (`recent_popularity` 0,88); phải đọc cạnh baseline. (E13)

### 3.2. Chỉ khẳng định sau khi chạy thêm (mục 4)
* Đóng góp của modality text/ảnh (RQ2), kể cả "đa phương thức tốt hơn từng modality".
* Reranker cải thiện Hit@12 so với retrieval + luật phục vụ (chênh +0,0067 hiện nằm trong nhiễu).
* Từng thành phần của luật phục vụ / từng nhóm đặc trưng (thời gian, affinity, giá, cold) đóng góp bao nhiêu **với thống kê toàn cục** (các bảng E5/E7 dùng thống kê mẫu 50k).
* Độ ổn định theo seed (hiện một seed).

### 3.3. Không nên khẳng định (không có bằng chứng, hoặc đã bị bác bỏ)
* "Multimodal giải quyết cold-start": tower không giải được nhóm chưa-từng-bán (Hit@100 0,0196 sau refit); chỉ kênh CLIP + chèn vị trí đạt 2–5% và có cái giá.
* Các con số "kỳ vọng" trong tài liệu cũ (cold HitRate 8–15%, lift 150–250× so với random, HitRate@100 28–42%) — chưa từng được đo; một số đã bị đo bác bỏ (cold). Đã gỡ.
* So sánh trực tiếp MAP@12 với bảng xếp hạng cuộc thi, hoặc HR/NDCG của H&M với Amazon.
* Cold-start **người dùng** (mẫu chỉ có khách có lịch sử).
* "Reranker 7 đặc trưng tốt hơn/kém hơn..." theo sampled 1+99: kết quả sampled của mô hình thiếu đặc trưng thời gian bị méo do chấm ngoài phân phối (E13).

## 4. Thực nghiệm còn thiếu (theo ưu tiên)

| Ưu tiên | Thực nghiệm | Cách làm | Chi phí ước tính |
|---|---|---|---|
| **P0** | Ablation modality trên H&M (RQ2) | Chạy 02 + 03 với `HM_MODALITY=id/text/image/multimodal` (4 lần), cùng `item_daily_counts.parquet`; báo cáo tổng thể + `strict_cold` + `never_sold`; bootstrap ghép cặp so với đa phương thức | 4 × (≈ 20 phút 02 + vài phút 03) trên Kaggle |
| **P0** | CI ghép cặp Reranker − (Tower + luật phục vụ) (RQ5) | Sửa 03: thêm xếp hạng "tower + luật phục vụ" vào `full_report` và `paired_diff` (hiện chỉ so với `recent_popularity`) | Sửa mã nhỏ + chạy lại 03 |
| **P0** | Nhiều seed | Chạy 02/03 với ≥ 3 seed (đổi `seed` trong `run_config.json` và `torch.manual_seed`); báo cáo trung bình ± độ lệch chuẩn | 3 × toàn bộ pipeline |
| **P1** | Ablation luật phục vụ với thống kê toàn cục | `HM_ACTIVE_GRID/HM_BONUS_GRID/HM_POP_GRID` cố định từng thành phần = 0; bootstrap trên test | Chạy lại phần đánh giá của 02 (nên bổ sung chế độ "chỉ đánh giá từ checkpoint") |
| **P1** | Ablation nhóm đặc trưng của reranker | Train ranker bỏ lần lượt: thời gian (`log_pop_recent, recent_share, age_days, seen_before`), affinity, giá, cold (`is_cold, from_cold, launch_prior`) | Rẻ (LightGBM) |
| **P1** | ID-dropout × luật phục vụ | Tower v1 (ID-dropout 0) + luật phục vụ; kiểm tra lợi ích ID-dropout còn tồn tại khi đã có luật | 1–2 lần 02 |
| **P2** | Cold-start theo modality | Từ P0 lấy lát `never_sold`/`strict_cold` cho 4 modality | Đã gồm trong P0 |
| **P2** | `HM_PROFILE=full` (300k khách) | Chạy 00 `full`, 02, 03; xem mức tăng khi tăng dữ liệu | Nặng; làm nếu còn thời gian |
| **P2** | Encoder người dùng Transformer cho H&M | Chuyển User Tower Amazon sang H&M để thống nhất kiến trúc giữa hai bộ (quyết định D1) | Công sức lớn |
| **P2** | Đặc trưng khách (tuổi, hạng thành viên) | Xuất thêm từ `customers.csv` ở notebook 00 | Nhỏ; hiệu quả chưa rõ |

## 5. Bảng và hình đề xuất cho chương thực nghiệm (nguồn dữ liệu)

| Mục | Nội dung | Nguồn |
|---|---|---|
| Bảng 1 | Mô tả dữ liệu `hm_v1` và cửa sổ thời gian | `01` mục 2–4, `results/dataset_stats.json` |
| Bảng 2 | Kết quả chính (thang: popularity → recent_popularity → tower → +luật → +reranker → +cold slot) | `03` mục 0, `results/*.json` |
| Bảng 3 | Phân tích theo lát (warm / strict-cold / never-sold / repeat / explore) | `03` E12a |
| Bảng 4 | Ablation: ID-dropout, cold-fix (E1), luật phục vụ (E5/E7), đặc trưng (E3, E12c) | `03` |
| Bảng 5 | Ablation modality (sau P0) | chưa có |
| Bảng 6 | Sampled 1+99 (bảng phụ, kèm baseline) | `03` E13 |
| Hình 1 | Sơ đồ pipeline | `02` mục 1 (Mermaid) |
| Hình 2 | Tính thời vụ: % đích test đã bán trong 7/14/28/56 ngày trước; số item bán mỗi tuần | `01` mục 4 |
| Hình 3 | Đường cong Hit@K (K = 12…100) cho các phương pháp | `results/reranker_metrics.json` (cần vẽ) |
| Hình 4 | Đánh đổi chèn vị trí cold: Hit@100 tổng và never-sold theo N | `03` E11 |
| Hình 5 | Hit@100 theo `pop_weight` / `cold_bonus` (lưới luật phục vụ) | `results/retrieval_metrics.json` (`serving_grid_valid_window`) |
| Hình 6 | Đường cong huấn luyện tower | `results/retrieval_history.json` |

Khi vẽ, dùng quy ước màu nhất quán và ghi rõ baseline (kỹ năng `dataviz` của công cụ có thể dùng).

## 6. Quyết định cần chủ đề tài chốt

| # | Quyết định | Lựa chọn | Ghi chú |
|---|---|---|---|
| **D1** | Vai trò bộ dữ liệu | (A) H&M là bộ chính, Amazon là bộ đối chứng/thứ cấp; (B) chỉ H&M, đưa Amazon vào phụ lục; (C) hai bộ ngang hàng | A giữ được toàn bộ công sức Amazon (nhất là RAG, vì Amazon có review) nhưng phải giải thích hai thiết kế mô hình khác nhau; B gọn nhất nhưng mất bằng chứng Amazon; C đòi hỏi chuyển cùng một kiến trúc sang cả hai |
| **D2** | RAG trên H&M | (a) giữ RAG trên Amazon (có review) như thành phần độc lập; (b) RAG trên H&M chỉ với metadata/mô tả (`detail_desc`, thuộc tính) — không có review nên "giải thích/so sánh" không thể trích dẫn review; (c) khác | **H&M không có review văn bản.** Hiện kho Qdrant (`products` 152.086, `reviews` 295.383) và `rag_chatbot_design.md` đều là Amazon |
| **D3** | Web/backend demo | Giữ dữ liệu Amazon hay nạp lại danh mục H&M (105.542 sản phẩm, embedding Jina 512 chiều đã có; ảnh ở `data/hm/Image`) | Cần `datn-retrieval index-products` cho H&M và đường suy luận trong `src/` |
| **D4** | Đưa suy luận H&M vào `src/datn/` | Cần port: thống kê item theo cutoff, tower + luật phục vụ, 7 nguồn ứng viên, `featurize`, LightGBM, kênh cold, chèn vị trí | Hiện logic chỉ nằm trong notebook (`AGENT.md` yêu cầu mã nằm trong `src/`) |
| **D5** | Thống kê toàn cục như một phần của phương pháp | Chấp nhận như giả định hệ thống (có sẵn khi triển khai) hay chỉ báo cáo bản mẫu 50k | Nên báo cả hai (đã có hai bộ số: E3, E8) |
| **D6** | Mục tiêu chính của kênh cold | Tối ưu chỉ số tổng (`HM_COLD_EVERY=0`) hay độ phủ cold (10 hoặc 5) | Là lựa chọn sản phẩm; báo cáo cả ba |

## 7. Ánh xạ sang các file luận văn hiện tại (`docs/LuanVan/`)

Phần này dựa trên mục lục và nội dung các file; các số Amazon trong đó vẫn đúng cho Amazon và phải được giữ nếu chọn D1 = A hoặc C.

| File / mục | Hiện tại (Amazon) | Cần làm khi H&M là bộ chính |
|---|---|---|
| `00_Mo_dau.md` §2.1, §2.2, §4, §5 | Mục tiêu, phạm vi dữ liệu `balanced_u5_i2_v1`, kết quả chính | Viết lại phạm vi dữ liệu và kết quả; cập nhật mục tiêu theo RQ1–RQ5 |
| `01_Chuong_1…` §1.2.2 | Tiền xử lý K-core, leave-last-out | Thêm mẫu 50k khách + cửa sổ 7 ngày + thống kê item theo cutoff |
| §1.3.1 | User Tower SASRec/Transformer | Nêu rõ tower H&M dùng trung bình có trọng số + ID-dropout (hoặc chuyển Transformer sang H&M — D1/P2) |
| §1.3.3 | Two-Stage | Bổ sung tính thời vụ, thống kê item toàn cục, kênh cold |
| §1.5.1 | HR/NDCG@K | Thêm HitRate/Recall/MAP/NDCG@12, các lát, sampled 1+99, bootstrap ghép cặp (`01` mục 6) |
| `02_Chuong_2…` §2.2 | Nguồn Amazon Reviews 2023 | Thêm H&M và nêu rõ không có review (D2) |
| §2.4 | K-core, leave-last-out | Thêm quy trình lấy mẫu/cửa sổ/rò rỉ của H&M (`01`) |
| §2.5.1 | User Tower Transformer, log-Q | Thêm mục H&M (`02` mục 2–3) hoặc thay |
| §2.5.2 | 4 nguồn, 2.000 ứng viên | 7 nguồn, ≈ 1.000, kênh cold (`02` mục 4–5) |
| §2.5.3 | Residual Listwise Reranker | LightGBM LambdaRank + 24 đặc trưng (`02` mục 6–7) |
| §2.6–§2.8 | RAG, kiến trúc web, mã nguồn | Phụ thuộc D2–D4 |
| `03_Chuong_3…` §3.2–§3.3 | Thiết lập và kết quả Amazon | Thay bằng `03` của tài liệu này (+ P0) |
| §3.4–§3.5 | Kiểm thử chatbot, web | Giữ nếu RAG/web giữ trên Amazon (D2/D3) |
| `04_Ket_luan.md` | Đối chiếu mục tiêu, hạn chế | Viết lại; hạn chế: `03` mục 13 |
| `Bao_cao.md` | Bản gộp toàn bộ | Sinh lại sau khi các chương cập nhật (đang trùng nội dung các chương) |

## 8. Câu hỏi phản biện thường gặp và câu trả lời có căn cứ

| Câu hỏi | Trả lời dựa trên số liệu |
|---|---|
| Vì sao HitRate@12 chỉ 12,8%? | Cửa sổ 7 ngày, trung vị 2 món/khách, 105.542 item; random Hit@12 ≈ 0,011% × số đích; mốc so sánh là baseline mạnh (7,6%) chứ không phải 100% |
| Tower thua "bán chạy tuần trước" — tower có giá trị gì? | Thua khi thuần (0,056 vs 0,076), nhưng tower + tín hiệu đang bán đạt 0,121; tower cung cấp cá nhân hoá, reranker cộng thêm; ablation modality (P0) sẽ định lượng phần nội dung |
| Dùng thống kê toàn bộ khách có phải rò rỉ không? | Chỉ dùng ngày < cutoff, cấp item, không cá nhân hoá, và production có sẵn; báo cả bản mẫu 50k để thấy độ nhạy (E3 vs E8) |
| Cold-start đã giải quyết chưa? | Loại mới-bán-gần-đây: có (thống kê + refit + cộng điểm cold). Loại chưa-từng-bán: chỉ một phần, 2–5% Hit@100 với chèn vị trí, và có cái giá; giới hạn bản chất là thiếu thông tin |
| Vì sao không tối ưu thêm mô hình? | Chênh lệch nhỏ hơn ~1 điểm phần trăm không phân biệt được với nhiễu (SE ≈ 0,006); chỉ nhận cải thiện khi CI ghép cặp loại trừ 0 |
| Sampled 1+99 cho 0,906, có đáng tin không? | Đó là bảng phụ; với negative uniform, "đang bán" quyết định gần hết (baseline 0,88); số có ý nghĩa là chênh lệch với baseline, không phải con số tuyệt đối |
| Vì sao không so MAP@12 với cuộc thi Kaggle? | Mẫu 50k khách, chỉ tính khách có mua, cách dùng thống kê khác; không tái hiện submission |
| Tham số chọn trên valid, test có sạch không? | Ba tham số luật phục vụ chọn trên valid; test không dùng để chọn tham số. Một số quyết định thiết kế được đưa ra khi đã xem chẩn đoán trên cả valid/test (E5–E7); thừa nhận trong hạn chế |

## 9. Quy tắc trình bày số liệu (để luận văn nhất quán)
1. Headline = full-ranking trên **test**, mô hình refit; luôn kèm `popularity` và `recent_popularity`.
2. Mọi cải thiện được khẳng định phải có CI ghép cặp loại trừ 0; chênh lệch nhỏ gọi là "tương đương".
3. Tách rõ hai loại cold; không dùng một con số "cold" duy nhất.
4. Số valid của tower không dùng làm kết quả độc lập (tham số luật phục vụ chọn trên đó).
5. Sampled chỉ là bảng phụ, kèm `random` và baseline.
6. Ghi nguồn số liệu (file JSON, mục E) cho mọi bảng.

## 10. Việc kỹ thuật kế tiếp (không phải thực nghiệm)
* Port suy luận vào `src/datn/` (D4) và thêm kiểm thử đơn vị cho `featurize`, luật phục vụ, chèn vị trí.
* Thêm chế độ "chỉ đánh giá từ checkpoint" cho 02 (tách quét tham số khỏi huấn luyện).
* Nạp danh mục H&M vào Qdrant nếu chọn demo trên H&M (D3).
* Đổi tên `best_retrieval (1).pt` → `best_retrieval.pt` trong `checkpoints/hm/` (tên hiện tại do tải về trùng tên).
* **Cần xác nhận mã sinh viên**: `docs/LuanVan/Bao_cao.md` ghi `2121050201`, còn `docs/LuanVan/00_Mo_dau.md` và `README.md` ghi `2221050201`. Hai file không nhất quán; chỉ chủ đề tài mới biết giá trị đúng nên chưa sửa.
