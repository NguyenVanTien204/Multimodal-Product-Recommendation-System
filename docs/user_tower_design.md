# Thiết kế User Tower (Attention/Transformer trên chuỗi hành vi)

Tài liệu này mô tả kiến trúc và các quyết định thiết kế của module
[`src/datn/recommenders/user_tower/`](../src/datn/recommenders/user_tower/), đồng thời
**sửa một lỗi logic trong công thức Late Fusion ở [`02-overview.md`](./02-overview.md)
mục 6-7** đã được phát hiện trong review.

## 1. Vai trò trong kiến trúc Two-Tower

- **User Tower** (module này): một bộ mã hoá tự-attention kiểu SASRec, nhận chuỗi
  hành vi (các item positive, sắp theo thời gian) của một user, trả về vector truy
  vấn `h_user`.
- **Item Tower**: bảng vector `e_i` cho từng item trong catalog, được định nghĩa
  **ngay trong cùng module này** (không phải một encoder tách biệt), vì đây là nơi
  duy nhất cần biết "item i trông như thế nào" — cả để đưa vào self-attention và để
  làm mục tiêu chấm điểm.
- **Điểm tương thích duy nhất giữa hai tháp**: `score(u, i) = h_user(u) · e_i(i)`
  (tích vô hướng). Không có phép cộng vector user + vector item ở đâu trong hệ thống.

## 2. Vì sao Item Embedding không còn "thuần ID"

Bản thiết kế đầu tiên dùng `nn.Embedding` ngẫu nhiên cho ID sản phẩm, tách biệt hoàn
toàn với vector CLIP (ảnh/chữ) đã trích xuất ở Pha 2. Đây là một lỗ hổng thật:

- SASRec thuần ID **không học được ngữ nghĩa** ảnh/chữ trong quá trình self-attention
  — nó chỉ học từ đồng-xuất-hiện (co-occurrence), nên self-attention không thể "biết"
  hai sản phẩm giống nhau về hình ảnh dù chưa từng xuất hiện cùng nhau.
- **Cold-start item** (chưa có tương tác nào, chiếm ~61-62% ở Valid/Test theo
  [`multimodal_embeddings_report.md`](./multimodal_embeddings_report.md)) không có
  gradient nào chạm tới hàng embedding của nó → vector ngẫu nhiên vô nghĩa mãi mãi.
  Late fusion ở "phút chót" không cứu được điều này vì nhánh CF vẫn luôn trả về rác
  cho các item đó.

### Fix: `e_i = id_residual(i) + content_proj(clip(i))`

Xem [`model.py`](../src/datn/recommenders/user_tower/model.py) (`UserTower.embed_items`):

- `id_residual`: `nn.Embedding` **khởi tạo bằng 0** (không phải `N(0, 0.02)` như bản
  cũ). Chỉ item nào thực sự có tương tác trong train mới nhận gradient và "lệch" ra
  khỏi 0.
- `content_proj`: một `nn.Linear` (có thể train) chiếu vector CLIP (ảnh, và text khi
  sẵn sàng) về chiều `d_model`, cộng vào `id_residual`.
- Hệ quả: một item **chưa từng được huấn luyện** có `id_residual(i) == 0` chính xác,
  nên `e_i` của nó **chính là** hình chiếu nội dung CLIP — không phải nhiễu ngẫu
  nhiên. Đây là chỗ cold-start được giải quyết, không phải ở bước late-fusion.
- `e_i` này được dùng ở **cả hai chỗ**: làm input embedding cho từng vị trí trong
  chuỗi self-attention, và làm vector mục tiêu khi tính loss / khi chấm điểm toàn bộ
  catalog lúc evaluate (`UserTower.item_vectors`). Một nguồn sự thật duy nhất, không
  có hai bảng embedding lệch nhau.

Nguồn nội dung được nạp qua [`content.py`](../src/datn/recommenders/user_tower/content.py)
(`load_content_matrix`), khớp theo `item_id` với `data/embedding/image_embedding_metadata
(1).parquet` hiện có; text sẽ được thêm khi `text_embeddings.npy` hoàn tất (còn
"Pending" theo `multimodal_embeddings_report.md`).

## 3. Sửa lỗi công thức Late Fusion (mục 6-7, `02-overview.md`)

Công thức cũ:

```text
E_content = α E_image + β E_text
E_final   = λ E_cf + (1 - λ) E_content
```

Lỗi: `E_cf` (dự kiến là `h_last`, tức **vector user**) bị cộng thẳng với
`E_image`/`E_text` (vector **item**). Hai không gian này không thể cộng — chỉ có thể
nhân vô hướng (dot product) để so khớp user với item.

**Công thức đúng, ở mức điểm số (score-level), không ở mức vector:**

```text
score_cf(u, i)      = h_user(u) · e_i^{id}(i)          # id_residual only
score_content(u, i) = h_user(u) · e_i^{content}(i)     # content_proj(clip(i)) only
score_final(u, i)   = λ · score_cf(u, i) + (1 - λ) · score_content(u, i)
```

Vì `e_i = e_i^{id} + e_i^{content}`, ta có
`h_user · e_i = score_cf + score_content` — tách được đúng hai "nhánh" CF và content
**sau khi** đã tính dot product, chứ không tách/cộng ở mức vector. Đây là cách hybrid
scoring chuẩn trong recommender học lai (weighted hybrid), và vẫn giữ được đúng thí
nghiệm quét `λ ∈ {0, 0.25, 0.5, 0.75, 1}` mà roadmap Tuần 6 yêu cầu.

### Map vào bảng thí nghiệm (mục 8, `02-overview.md`)

Cấu hình `content.enabled` (xem `configs/user_tower.yaml`) chọn trực tiếp biến thể
mô hình, không cần huấn luyện lại kiến trúc khác nhau cho mỗi dòng:

| Dòng trong bảng thí nghiệm | `content.enabled` |
|---|---|
| Collaborative (Interaction only) | `[]` |
| CF + Image | `[image]` |
| CF + Text | `[text]` (khi có text embedding) |
| Full Multimodal | `[image, text]` |

`λ` vẫn là một tham số **suy luận** độc lập với việc huấn luyện: có thể huấn luyện
một model duy nhất với `content.enabled: [image, text]` rồi quét `λ` bằng cách tính
lại `score_cf` và `score_content` riêng từ `e_i^{id}` và `e_i^{content}` đã lưu, không
cần train lại cho mỗi giá trị `λ`.

## 4. Negative sampling: popularity-based, không còn uniform

Uniform trên 152.086 item khiến hầu hết negative là "easy negative" — mô hình phân
biệt được ngay không cần học kỹ. `train.build_negative_sampler` giờ lấy mẫu theo
`popularity^0.75` (kiểu word2vec/BPR): item phổ biến bị lấy làm negative thường xuyên
hơn, buộc mô hình học ranking tốt hơn giữa các item "hợp lý" thay vì giữa các item
ngẫu nhiên hoàn toàn khác biệt.

Giới hạn còn lại: đây vẫn không phải hard-negative mining thật (theo độ tương đồng
CLIP hoặc in-batch). Nếu NDCG@10 vẫn thấp sau khi tối ưu kiến trúc, hướng mở rộng tiếp
theo là lấy negative trong cùng batch (in-batch negatives) hoặc theo độ tương đồng nội
dung với item positive.

## 5. Vì sao right-padding, không left-padding

`dataset.pad_right` cố ý xếp token thật ở đầu, PAD ở cuối. Kết hợp causal mask +
key-padding mask: nếu dùng left-padding, các vị trí PAD ở đầu chuỗi sẽ bị chặn tất cả
key hợp lệ (causal chỉ cho nhìn về trước, mà "trước" toàn PAD), softmax ra `-inf` ở
mọi vị trí → NaN, và giá trị NaN đó lan sang các vị trí thật ở layer sau qua
attention (trọng số ~0 nhưng `0 * NaN = NaN`). Right-padding đảm bảo vị trí 0 luôn là
token thật (mọi chuỗi có độ dài ≥ 1), nên không bao giờ có dòng softmax bị mask toàn
bộ. Đã kiểm chứng bằng smoke test (2 epoch trên dữ liệu thật, không NaN).

## 6. Giao thức đánh giá

- Target hợp lệ cho Valid/Test **chỉ khi** tương tác leave-last-out đó `is_positive
  == 1` (rating ≥ 4) — rating thấp không phải tín hiệu "thích" nên không dùng làm
  ground-truth "đề xuất đúng" (đúng theo mục 5, `02-overview.md`).
- Đánh giá bằng **full-ranking** trên toàn bộ catalog (152.086 item, batch theo user),
  không sample negative lúc test — tránh bias của sampled-negative evaluation.
- `HitRate@10 == Recall@10` về mặt số học ở đây vì mỗi user chỉ có đúng 1 item liên
  quan (leave-one-out); cả hai vẫn được báo cáo riêng cho khớp với bảng kết quả đã
  thiết kế ở mục 8.

## 7. Trạng thái các nguồn đặc trưng (Content Sources)

- `image_embeddings.npy` (1024-d) & `text_embeddings.npy` (1024-d): Đã hoàn thành trích xuất đầy đủ cho 152.086 sản phẩm trong `data/embedding/`.
- Cấu hình mặc định kích hoạt đồng thời cả 2 phương thức (`content.enabled: [image, text]`), tạo thành vector nội dung 2048-d đưa vào `content_proj`.

## 8. Các cải tiến tối ưu hóa & Hướng dẫn Ablation Study

Nhằm hạn chế overfitting sau epoch 10 và nâng cao chất lượng xếp hạng trên 152k items:
1. **Kiến trúc mô hình**: Nâng `d_model: 128`, `n_heads: 4`, `d_ff: 512` giúp bảo toàn tốt hơn ngữ nghĩa từ 2048 chiều CLIP.
2. **Học tập tương phản (Negative Sampling)**: Tăng `num_negatives: 4` (lấy mẫu theo phân phối `popularity^0.75`), tạo lực đẩy tương phản mạnh hơn gấp 4 lần.
3. **Tối ưu hóa & Điều chuẩn (Regularization)**: Chuyển sang `AdamW` với `weight_decay: 0.0001`, bổ sung `CosineAnnealingLR` scheduler (giảm dần từ `5e-4` về `1e-5`), giúp gradient hội tụ ổn định và bền vững.
4. **Mở rộng đánh giá ứng viên**: Hỗ trợ đánh giá đồng thời `ks: [10, 20, 50]`.
5. **Hỗ trợ CLI Ablation nhanh**:
   - CF thuần ID: `python -m datn.recommenders.user_tower.cli train --content --artifacts-dir data/artifacts/user_tower_cf`
   - CF + Image: `python -m datn.recommenders.user_tower.cli train --content image --artifacts-dir data/artifacts/user_tower_image`
   - CF + Text: `python -m datn.recommenders.user_tower.cli train --content text --artifacts-dir data/artifacts/user_tower_text`
   - Full Multimodal: `python -m datn.recommenders.user_tower.cli train --content image text --artifacts-dir data/artifacts/user_tower`

> [!TIP]
> Kết quả thực nghiệm chi tiết và bảng so sánh các biến thể trên tập Test (22.414 users, 152.086 items) được ghi nhận đầy đủ tại [`docs/user_tower_experiments.md`](./user_tower_experiments.md). Mô hình **Full Multimodal** vượt trội hoàn toàn so với CF thuần ID (gấp 31.7x về HitRate@10 và 41.7x về NDCG@10), chứng minh hiệu quả giải quyết bài toán Cold-start của hệ thống.
