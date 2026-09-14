# Báo cáo Thực nghiệm & Nghiên cứu Thành phần (Ablation Study) - User Tower

Tài liệu này ghi nhận kết quả thực nghiệm hoàn chỉnh của các biến thể mô hình **User Tower (SASRec Two-Tower Retrieval)** trong đề tài: *"Xây dựng hệ thống gợi ý sản phẩm đa phương thức ứng dụng RAG"*.

Toàn bộ các thực nghiệm được đánh giá theo giao thức **Full-Ranking trên toàn bộ danh mục 152.086 sản phẩm** (Leave-one-out), không sử dụng phương pháp lấy mẫu negative khi đánh giá để đảm bảo tính khách quan và trung thực học thuật.

---

## 1. Bảng Tổng hợp Kết quả Thực nghiệm (Experimental Results)

### A. Kết quả trên tập Kiểm thử chính thức (Test Split - 22.414 Users)

| STT | Mô hình (Variant) | Phương thức sử dụng (Modalities) | HitRate@10 | Recall@10 | NDCG@10 | HitRate@20 | NDCG@20 | HitRate@50 | NDCG@50 |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | **Random Baseline** | Không (Ngẫu nhiên tuyệt đối) | 0.0066% | 0.0066% | 0.0030% | 0.0131% | 0.0046% | 0.0329% | 0.0078% |
| 1 | **Collaborative Filtering (CF)** | Chỉ ID tương tác (`id_residual`) | 0.0045% | 0.0045% | 0.0017% | 0.0134% | 0.0041% | 0.0178% | 0.0050% |
| 2 | **CF + Text** | Tương tác + Text CLIP (1024-d) | 0.1026% | 0.1026% | 0.0576% | 0.1562% | 0.0710% | 0.3257% | 0.1043% |
| 3 | **CF + Image** | Tương tác + Image CLIP (1024-d) | 0.1383% | 0.1383% | 0.0608% | 0.2409% | 0.0864% | 0.5532% | 0.1472% |
| 4 | **Full Multimodal (Đề xuất)** | **Tương tác + Image + Text (2048-d)** | **0.1428%** | **0.1428%** | **0.0721%** | **0.2811%** | **0.1074%** | **0.5711%** | **0.1649%** |

*Ghi chú: $100\% = 1.0$. Giá trị trong bảng hiển thị theo đơn vị phần trăm (%) để thuận tiện so sánh trực quan; các giá trị NDCG nhân với 100 để hiển thị cùng thang đo.*

---

### B. Kết quả trên tập Thẩm định (Validation Split - 22.268 Users, Best Checkpoint)

| STT | Mô hình (Variant) | Best Epoch | HitRate@10 | NDCG@10 | HitRate@20 | NDCG@20 | HitRate@50 | NDCG@50 |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Collaborative Filtering (CF)** | Epoch 5 | 0.0180% | 0.0071% | 0.0225% | 0.0082% | 0.0449% | 0.0126% |
| 2 | **CF + Text** | Epoch 20 | 0.1751% | 0.0842% | 0.2470% | 0.1018% | 0.5075% | 0.1529% |
| 3 | **CF + Image** | Epoch 15 | 0.1841% | 0.0936% | 0.3009% | 0.1227% | 0.6107% | 0.1829% |
| 4 | **Full Multimodal (Đề xuất)** | **Epoch 10** | **0.2470%** | **0.1355%** | **0.4176%** | **0.1777%** | **0.8038%** | **0.2551%** |

---

## 2. Phân tích Chuyên sâu & Ý nghĩa Luận văn (Key Findings)

### 2.1. Hiện tượng sụp đổ của Collaborative Filtering thuần trước Cold-Start
- Trên tập Test, **Collaborative Filtering (ID thuần) chỉ đạt HitRate@10 = 0.0045%**, thậm chí thấp hơn cả mức lựa chọn ngẫu nhiên (`0.0066%`).
- **Nguyên nhân cốt lõi:** Theo thống kê tại [`dataset_analysis.md`](./dataset_analysis.md), tập Test có tới **62.57% sản phẩm là Cold-Start** (hoàn toàn chưa từng xuất hiện trong tập Train). Nhánh CF chỉ dựa vào ma trận tương tác nên vector của các sản phẩm này giữ nguyên giá trị khởi tạo bằng 0 (`id_residual = 0`), khiến mô hình hoàn toàn "bị mù" và không thể xếp hạng các sản phẩm mới vào danh sách gợi ý.

### 2.2. Sự vượt trội của Đa phương thức (Multimodal Gain)
- Khi tích hợp **Đặc trưng Văn bản (CF + Text)**:
  - `HitRate@10` tăng từ $0.0045\% \to 0.1026\%$ (**gấp ~23 lần** so với CF thuần).
  - `HitRate@50` tăng từ $0.0178\% \to 0.3257\%$ (**gấp ~18.3 lần** so với CF thuần).
- Khi tích hợp **Đặc trưng Hình ảnh (CF + Image)**:
  - `HitRate@10` đạt $0.1383\%$ (**gấp ~30.7 lần** so với CF thuần).
  - `HitRate@50` đạt $0.5532\%$ (**gấp ~31.1 lần** so với CF thuần).
- Khi tích hợp **Toàn diện Đa phương thức (Full Multimodal - Image + Text)**:
  - Mô hình đạt đỉnh hiệu năng trên mọi chỉ số: `HitRate@10 = 0.1428%` (**gấp ~31.7 lần** CF), `HitRate@50 = 0.5711%` (**gấp ~32.1 lần** CF), và `NDCG@10 = 0.0721%` (**gấp ~41.7 lần** CF).
  - So với Random Baseline, Full Multimodal vượt trội **~21.7 lần ở Top-10** và **~17.4 lần ở Top-50**.

### 2.3. So sánh vai trò giữa Thị giác (Vision) và Văn bản (Text)
- Trong miền dữ liệu thời trang thương mại điện tử (Amazon Fashion), **Đặc trưng Hình ảnh (Image) có đóng góp lớn hơn rõ rệt so với Văn bản (Text)**:
  - `CF + Image` vượt trội hơn `CF + Text` khoảng **+34.8%** ở `HitRate@10` ($0.1383\%$ vs $0.1026\%$) và **+69.9%** ở `HitRate@50` ($0.5532\%$ vs $0.3257\%$).
  - Tuy nhiên, việc kết hợp cả hai (`Full Multimodal`) vẫn mang lại hiệu năng cao nhất ($0.5711\%$ ở `HitRate@50` và $0.0721\%$ ở `NDCG@10`), chứng minh rằng văn bản (thương hiệu, chất liệu, tính năng) và hình ảnh (kiểu dáng, màu sắc) có tính chất **bổ trợ lẫn nhau (complementary)** chứ không triệt tiêu.

### 2.4. Khẳng định vai trò tầng Candidate Retrieval trong Kiến trúc Two-Stage
- Trong bài toán gợi ý quy mô lớn với 152.086 sản phẩm:
  - User Tower hoàn thành nhiệm vụ **Candidate Retrieval**: thu hẹp không gian tìm kiếm từ **152.086 sản phẩm xuống Top 50 ứng viên tiềm năng nhất** (với `HitRate@50` đạt gần 0.6% trên Test và 0.8% trên Valid).
  - 50 ứng viên này tiếp tục được chuyển sang tầng Reranker / RAG / LLM Agent để xếp hạng chi tiết và giải thích cá nhân hóa theo thời gian thực.

---

## 3. Quy trình Tái lập Thực nghiệm (Reproducibility)

Tất cả các mô hình có thể được huấn luyện lại và kiểm định độc lập thông qua giao diện dòng lệnh:

```powershell
# 1. Collaborative Filtering thuần (ID-only)
python -m datn.recommenders.user_tower.cli train --content --artifacts-dir data/artifacts/user_tower_cf
python -m datn.recommenders.user_tower.cli evaluate --split test --content --artifacts-dir data/artifacts/user_tower_cf

# 2. CF + Text
python -m datn.recommenders.user_tower.cli train --content text --artifacts-dir data/artifacts/user_tower_text
python -m datn.recommenders.user_tower.cli evaluate --split test --content text --artifacts-dir data/artifacts/user_tower_text

# 3. CF + Image
python -m datn.recommenders.user_tower.cli train --content image --artifacts-dir data/artifacts/user_tower_image
python -m datn.recommenders.user_tower.cli evaluate --split test --content image --artifacts-dir data/artifacts/user_tower_image

# 4. Full Multimodal (Image + Text)
python -m datn.recommenders.user_tower.cli train --content image text --artifacts-dir data/artifacts/user_tower
python -m datn.recommenders.user_tower.cli evaluate --split test --content image text --artifacts-dir data/artifacts/user_tower
```
