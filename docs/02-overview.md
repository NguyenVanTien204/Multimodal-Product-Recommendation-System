# Tổng quan đề tài và hệ thống

## 1. Thông tin chung

| Thuộc tính | Nội dung |
|---|---|
| Tên tiếng Việt | Xây dựng hệ thống gợi ý sản phẩm đa phương thức ứng dụng RAG |
| Tên tiếng Anh | Developing a Multimodal Product Recommendation System using RAG |
| Lĩnh vực | Recommendation Systems, Multimodal Learning, Information Retrieval, RAG |
| Dataset | Amazon Reviews 2023 — Clothing_Shoes_and_Jewelry |
| Bài toán chính | Personalized Top-K Product Recommendation |
| Phạm vi nghiên cứu | Collaborative, image, text, fusion, ablation, sparse-item evaluation |
| Phần mở rộng | Conversational refinement, RAG explanation/comparison, lightweight agent |

## 2. Dataset và phạm vi dữ liệu

Dataset được liên kết bằng `parent_asin`, dùng làm `item_id` thống nhất giữa interaction và product metadata. Dữ liệu cuối cùng chỉ là một subset đủ lớn để thực nghiệm, không dùng toàn bộ category.

| Thành phần | Khoảng mục tiêu |
|---|---:|
| Users | 30.000–50.000 |
| Products | 20.000–40.000 |
| Interactions | 200.000–500.000 |
| Image | 1 ảnh chính/item |
| Product text | title + features + description |
| Reviews dùng cho RAG | Một số review hữu ích/item |

Các khoảng trên là mục tiêu định hướng; số lượng cuối cùng được ghi trong `dataset_manifest.json` sau preprocessing.

## 3. Mô hình dữ liệu đích

### `interactions.parquet`

| Trường | Ý nghĩa |
|---|---|
| user_id | Định danh người dùng đã ẩn danh |
| item_id | `parent_asin` của sản phẩm |
| rating | Điểm đánh giá |
| timestamp | Thời điểm tương tác |
| verified_purchase | Trạng thái mua hàng xác thực |

### `items.parquet`

| Trường | Ý nghĩa |
|---|---|
| item_id | Định danh sản phẩm |
| title | Tên sản phẩm |
| description | Mô tả |
| features | Thuộc tính/đặc trưng |
| category | Danh mục |
| brand | Thương hiệu |
| price | Giá |
| image_url | URL ảnh chính |

### `reviews.parquet`

| Trường | Ý nghĩa |
|---|---|
| item_id, user_id | Liên kết sản phẩm và người dùng |
| rating | Điểm review |
| review_title | Tiêu đề review |
| review_text | Nội dung review |
| helpful_vote | Số lượt hữu ích |
| timestamp | Thời điểm review |

### Artifact sinh ra

```text
embeddings/image_embeddings.npy
embeddings/text_embeddings.npy
embeddings/collaborative_embeddings.npy
indexes/products.faiss
indexes/reviews.faiss
```

## 4. Pipeline dữ liệu

```text
Amazon Reviews 2023 / Raw JSONL
                ↓
         Schema validation
                ↓
Interaction ↔ Metadata bằng parent_asin
                ↓
      Loại item thiếu metadata thiết yếu
                ↓
 Positive interaction: rating >= 4
                ↓
          Iterative K-core
                ↓
          Subset sampling
                ↓
          Chạy lại K-core
                ↓
 Chronological leave-last-out split
          ↙        ↓        ↘
       Train     Valid      Test
                ↓
          Processed Parquet
```

Rating thấp có thể được giữ trong dữ liệu raw phục vụ phân tích nhưng không được coi là tín hiệu “thích” trong bài toán implicit Top-K.

## 5. Chiến lược chia dữ liệu

Với mỗi người dùng có chuỗi tương tác theo thời gian, `N-2` tương tác đầu thuộc train, tương tác `N-1` thuộc validation và tương tác cuối thuộc test. Không dùng random 80/20 vì có thể làm rò rỉ tín hiệu tương lai vào quá trình huấn luyện.

Việc lọc, sampling và split chỉ được thực hiện trên dữ liệu cần thiết; các tham số, seed, số lượng bản ghi và checksum cần được lưu trong manifest để tái lập.

## 6. Kiến trúc logic

```text
                         USER
                 Text / Image / History
                          │
                          ▼
               Lightweight Recommendation Agent
                          │
                  Preference / Intent
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
     User History     Text Query       Image Query
          │               │               │
          ▼               ▼               ▼
      CF Encoder      Text Encoder     Image Encoder
          └───────────────┼───────────────┘
                          ▼
                 Multimodal Fusion
                          │
                          ▼
                        FAISS
                          │
                     Candidates
                          │
                        Ranking
                          │
                        Top-K
                    ┌─────┴─────┐
                    ▼           ▼
                Sản phẩm       RAG
                                │
                    Metadata + Reviews
                                │
                               LLM
                         Explain / Compare
```

Agent là lớp điều phối. Khi agent không khả dụng, recommender, retrieval và các API lõi vẫn phải hoạt động độc lập.

## 7. Recommendation core

### Baseline

- Popularity.
- BPR hoặc Matrix Factorization.
- LightGCN nếu nguồn lực cho phép sau baseline tối thiểu.

### Encoder

- Image: pretrained CLIP-family image encoder.
- Text giai đoạn đầu: CLIP text encoder với `title + features + description`.
- Thử nghiệm phụ: SentenceTransformer nếu core đã hoàn thành.
- Collaborative: embedding từ mô hình CF được huấn luyện trên train interactions.

### Late fusion

`E_cf` là vector **user** (đầu ra User Tower, xem
[`user_tower_design.md`](./user_tower_design.md)); `E_image`/`E_text` là vector
**item** (CLIP). Hai không gian này không được cộng trực tiếp — chỉ so khớp
user-item qua tích vô hướng. Fusion diễn ra ở **mức điểm số**, sau dot product,
không ở mức vector:

```text
score_cf(u, i)      = E_cf(u)      · E_id(i)
score_content(u, i) = E_cf(u)      · (α E_image(i) + β E_text(i))
score_final(u, i)   = λ · score_cf(u, i) + (1 - λ) · score_content(u, i)
```

trong đó `E_id(i)` và `α E_image(i) + β E_text(i)` cộng lại thành vector item đầy đủ
`e_i` mà User Tower dùng để tự-attention và để làm mục tiêu huấn luyện (chi tiết ở
`user_tower_design.md` mục 2-3). Các vector dùng cosine similarity cần được chuẩn hóa
trước khi lập FAISS index. Embedding phải được cache và version theo encoder/config/dataset manifest.

## 8. Thiết kế thí nghiệm & Kết quả thực nghiệm

### 8.1. Ma trận phân loại mô hình

| Model | Interaction | Text | Image |
|---|:---:|:---:|:---:|
| Popularity | ✓ |  |  |
| Collaborative (ID-only) | ✓ |  |  |
| Text-only |  | ✓ |  |
| Image-only |  |  | ✓ |
| Image + Text |  | ✓ | ✓ |
| CF + Text | ✓ | ✓ |  |
| CF + Image | ✓ |  | ✓ |
| Full Multimodal | ✓ | ✓ | ✓ |

### 8.2. Kết quả Thực nghiệm Ablation Study (Full-Ranking trên 152.086 items)

Chi tiết báo cáo và quy trình thực nghiệm xem tại [`docs/user_tower_experiments.md`](./user_tower_experiments.md).

| Biến thể (Variant) | Phương thức | Test HitRate@10 | Test NDCG@10 | Test HitRate@50 | Test NDCG@50 | Ghi chú & Đánh giá |
|---|---|:---:|:---:|:---:|:---:|---|
| **Random Baseline** | Không | 0.0066% | 0.0030% | 0.0329% | 0.0078% | Ngẫu nhiên tuyệt đối |
| **Collaborative (CF)** | Chỉ ID tương tác | 0.0045% | 0.0017% | 0.0178% | 0.0050% | Thất bại do 62.5% Cold-Start |
| **CF + Text** | Tương tác + Text CLIP | 0.1026% | 0.0576% | 0.3257% | 0.1043% | Tăng gấp 23x so với CF |
| **CF + Image** | Tương tác + Image CLIP | 0.1383% | 0.0608% | 0.5532% | 0.1472% | Tăng gấp 31x so với CF (Ảnh > Chữ) |
| **Full Multimodal** | **Tương tác + Ảnh + Chữ** | **0.1428%** | **0.0721%** | **0.5711%** | **0.1649%** | **Đạt đỉnh toàn diện (gấp 41.7x NDCG của CF)** |

Metric chính: `Recall@K`, `NDCG@K`, `HitRate@K` ($K \in \{10, 20, 50\}$). Kiểm thử trên 22.414 người dùng với 152.086 sản phẩm.

## 9. Conversational recommendation

LLM chỉ chuyển ngôn ngữ tự nhiên thành preference có cấu trúc, ví dụ:

```json
{
  "category": "shoes",
  "color": "black",
  "style": "casual",
  "max_price": 100,
  "sporty": "negative"
}
```

Recommender và reranker quyết định kết quả. Trạng thái preference được lưu trong phiên; câu tiếp theo chỉ cập nhật phần thay đổi và rerank danh sách ứng viên, không retrain model.

## 10. RAG và agent

RAG chạy sau Top-K. Với sản phẩm được chọn, hệ thống truy xuất description, features, metadata, review liên quan và preference của người dùng, sau đó yêu cầu LLM tạo câu trả lời có căn cứ.

Toolset dự kiến cho một agent duy nhất:

```text
get_user_profile()
recommend()
search_similar_products()
get_product_context()
compare_products()
refine_preferences()
```

Agent chỉ chọn công cụ và tổng hợp câu trả lời; không tự tạo thông tin giá, thuộc tính, review hoặc sản phẩm không có trong dữ liệu truy xuất.

## 11. API và giao diện

| Endpoint | Mục đích |
|---|---|
| `POST /recommend` | Gợi ý Top-K từ user/history/preference |
| `POST /search` | Tìm kiếm tương tự bằng text/image/item |
| `POST /refine` | Cập nhật preference và rerank |
| `POST /explain` | Giải thích một sản phẩm đã gợi ý |
| `POST /compare` | So sánh các sản phẩm bằng evidence |
| `POST /chat` | Giao diện hội thoại điều phối các chức năng |

Backend dùng FastAPI; frontend chốt dùng Next.js (App Router, TypeScript) và Tailwind CSS. Frontend chỉ là presentation layer, không truy cập trực tiếp artifact model/index.

## 12. Công nghệ

| Layer | Công nghệ |
|---|---|
| Language | Python |
| Raw/processed storage | JSONL / Parquet |
| ETL | Polars |
| EDA và kiểm tra | DuckDB + Jupyter |
| ML | PyTorch |
| Recommendation | RecBole + custom fusion |
| Image/Text | CLIP / Transformers |
| Optional text encoder | Sentence Transformers |
| Vector search | FAISS |
| RAG | Custom lightweight pipeline |
| Agent | Single tool-calling agent |
| API / Demo | FastAPI / Next.js + Tailwind CSS |
| Results | CSV/JSON + plots |
| Version control | Git |

## 13. Cấu trúc project đề xuất

```text
project/
├── data/{raw,interim,processed,images}/
├── embeddings/{image,text,collaborative}/
├── indexes/{products.faiss,reviews.faiss}
├── src/{data,features,recommenders,evaluation,retrieval,rag,agent,api}/
├── experiments/
├── notebooks/
├── frontend/
├── tests/
└── configs/
```
