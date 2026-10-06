# Chatbot RAG gợi ý & tìm kiếm sản phẩm đa phương thức

> **Phạm vi bộ dữ liệu (cập nhật 06/10/2026):** tài liệu này mô tả hệ thống được xây và đo trên dữ liệu **Amazon** (152.086 sản phẩm, 1024 chiều, review thật). Từ 06/10/2026 web, RAG và recommender chạy trên **H&M** (quyết định D2/D3 đã chốt trong [`./hm/05_thesis_plan.md`](./hm/05_thesis_plan.md)): xem [`./hm/06_web_migration.md`](./hm/06_web_migration.md) cho những gì đã đổi (catalog, vector 512 chiều, bộ lọc nhóm khách/màu, review minh hoạ). Các con số đo ở các mục 7–10 bên dưới (độ trễ, review → sản phẩm, tiếng Việt) **chỉ đúng cho Amazon**. Mã nguồn vẫn hỗ trợ cả hai catalog, chọn bằng biến môi trường `DATN_PRODUCTS_COLLECTION`, `DATN_REVIEWS_COLLECTION`, `DATN_VECTOR_SIZE`.

Tài liệu mô tả hệ thống chatbot đã xây dựng (Pha 3–5 trong ROADMAP): kiến trúc, quyết định thiết kế, cách chạy, và các giới hạn đã biết.

## 1. Kiến trúc tổng quan

```
Web (Next.js)  ──►  Backend marketplace (FastAPI, :8000)  ──►  RAG service (FastAPI, :8200)
 chat UI, upload ảnh   POST /chat  (proxy + hydrate sản phẩm      ├─ Agent: intent · session · refine
                        từ Postgres, gắn lịch sử giỏ/đơn hàng)    ├─ Retrieval: Jina CLIP v2 → Qdrant `products`
                                                                  │     (vector text + vector ảnh + payload filter)
                                                                  ├─ RAG: Qdrant `reviews` → context → LLM/template
                                                                  └─ Personalization ──► Recommender service (:8100)
                                                                        User Tower + Reranker checkpoint (Amazon)
```

| Thành phần | Mã nguồn | Vai trò |
|---|---|---|
| Encoder truy vấn | `src/datn/retrieval/encoder.py` | Jina CLIP v2 (cùng model đã embed catalog). Text query dùng `task="retrieval.query"`; ảnh người dùng dùng `encode_image`. |
| Chỉ mục | `src/datn/retrieval/indexer.py`, CLI `datn-retrieval` | Dựng collection `products` (152.086 điểm) và `reviews`. |
| Tìm kiếm lai | `src/datn/retrieval/search.py`, `filters.py` | Vector search trên 2 named vector (text, image) + filter payload trong Qdrant, hợp nhất bằng weighted RRF. |
| Bằng chứng (RAG) | `src/datn/rag/evidence.py` | Lấy review **giới hạn theo đúng sản phẩm đang bàn**, xếp theo độ liên quan tới câu hỏi + lượt hữu ích. |
| Context + kiểm chứng | `src/datn/rag/context.py` | Đánh mã trích dẫn `[P#]`/`[R#.#]`, kiểm tra câu trả lời không bịa giá/không trích dẫn ảo. |
| Sinh câu trả lời | `src/datn/rag/answer.py`, `llm.py` | LLM (OpenAI-compatible, tuỳ chọn) có kiểm chứng; hỏng/không cấu hình → template dựng thẳng từ dữ liệu. |
| Agent | `src/datn/agent/*` | Phân tích ý định VI/EN, quản lý session, refine không cần train lại, điều phối các bước. |
| Service | `apps/rag/` | FastAPI: `/chat /search /recommend /refine /explain /compare /health`. |
| Gateway | `apps/backend/app/chat/` | `POST /chat` cho web; có fallback tìm từ khoá khi RAG service hỏng. |

## 2. Dữ liệu và chỉ mục

**`products`** (Qdrant, id = `products.id` của Postgres = dòng `items.parquet` + 1)
- Named vectors `image` và `text` (1024-d, cosine) lấy từ `data/embedding/*.npy` — không tính lại.
- Payload: `item_id` (ASIN, khoá của model checkpoint), `title`, `brand`, `category_slug`, `price` (VND như shop hiển thị), `price_estimated`, `image_url`, `description`/`features` (rút gọn), `review_count`, `avg_rating`, cờ `is_image_fallback`…
- Payload index: keyword (brand, category, item_id), float (price, avg_rating), integer, bool, và full-text trên `title` (dùng cho chế độ dự phòng khi không có encoder).

**`reviews`** (Qdrant): vector Jina CLIP v2 của `tiêu đề + nội dung` review; payload `product_id`, `rating`, `helpful_vote`, `text`.
Nguồn: hợp nhất `train/valid/test/candidate_interactions.parquet` (551.217 cặp user–item duy nhất; sau khi lọc ≥ 20 ký tự). **Chỉ embed tối đa 4 review/sản phẩm** (ưu tiên nhiều lượt hữu ích) vì embed toàn bộ ~515k review mất ~2 giờ trên GPU laptop 4 GB; `review_count`/`avg_rating` của sản phẩm vẫn được tính trên **toàn bộ** review.

Kiểm chứng encoder: mã hoá lại 5 sản phẩm bằng đúng template của notebook và so với vector đã lưu → cosine 0,9999 (chênh do fp16), nên truy vấn nằm đúng trong không gian của catalog.

## 3. Luồng xử lý một lượt chat

1. **Intent** (`intent.py`, rule-based, có unit test): tách giá (`dưới 500k`, `từ 300k đến 600k`, `under $100`), màu, thương hiệu (từ vocabulary trong Qdrant), số sao, tham chiếu thứ tự (`sản phẩm 2`, `#2`, `cái đầu tiên`), rồi chọn hành động: `search | refine | recommend | similar | explain | compare | reset | greet | help`. Câu tiếng Việt được bổ sung từ khoá tiếng Anh theo glossary (catalog là dữ liệu Amazon tiếng Anh) nhưng vẫn giữ nguyên câu gốc vì Jina CLIP v2 đa ngôn ngữ. LLM chỉ được hỏi khi rule trả về `unknown`.
2. **Retrieval**: encode text và/hoặc ảnh → mỗi query tìm trên cả vector `text` và `image` (text→ảnh và ảnh→text đều dùng được vì CLIP căn chỉnh hai nhánh) → weighted RRF (điểm cosine giữa các vector không so sánh được trực tiếp nên hợp nhất theo hạng; điểm gốc vẫn được giữ để giải thích). Filter giá/thương hiệu/rating thực thi **bên trong Qdrant**.
3. **Cá nhân hoá**: nếu có lịch sử (giỏ hàng/đơn hàng từ shop + sản phẩm người dùng đã hỏi trong phiên), gọi recommender (User Tower → Reranker) và cộng điểm hạng của model vào điểm hợp nhất (`PERSONAL_WEIGHT=0.6`). Không có lịch sử/service lỗi → bỏ qua, không ảnh hưởng tìm kiếm.
4. **Refine** (`rẻ hơn`, `màu đỏ`, `thương hiệu Nike`, `cái khác`, `bỏ lọc`): cập nhật `Preferences` của session rồi chạy lại. Nếu chỉ siết chặt bộ lọc và truy vấn ngữ nghĩa không đổi, dùng lại **candidate pool 60 sản phẩm** đã có (không query vector mới) — đúng yêu cầu "lọc lại trên tập ứng viên có sẵn, không retrain".
5. **Evidence + trả lời**: lấy review của đúng các sản phẩm kết quả → dựng context có mã trích dẫn → LLM (nếu cấu hình) hoặc template. Câu trả lời LLM bị kiểm tra (mã trích dẫn tồn tại, mọi số tiền phải trùng giá trong dữ liệu / chênh lệch giữa hai giá / ngưỡng do người dùng nêu); vi phạm → thử lại một lần với danh sách lỗi, vẫn sai → dùng template.
6. **Giải thích/So sánh**: `explain` liệt kê lý do có cấu trúc (khớp mô tả với độ tương đồng bao nhiêu, nằm trong ngân sách, hạng gợi ý cá nhân hoá, điểm đánh giá) cộng review dẫn chứng; `compare` thêm cả review điểm thấp để nêu nhược điểm, và **không kết luận khi thiếu dữ liệu** (mỗi sản phẩm cần ≥ 3 đánh giá để nói "tốt hơn").

## 4. Suy giảm có kiểm soát (độc lập với LLM)

| Sự cố | Hành vi |
|---|---|
| Không cấu hình / lỗi LLM | Trả lời template dựng từ dữ liệu; API và giao diện vẫn đầy đủ sản phẩm + lý do + review |
| Recommender lỗi hoặc user chưa có lịch sử | Bỏ boost cá nhân hoá; `recommend` rơi về tương đồng nội dung rồi popularity |
| Encoder không nạp được | Tìm từ khoá trên `title` (full-text index của Qdrant) |
| RAG service sập | Backend trả kết quả tìm theo từ khoá tên sản phẩm, gắn cảnh báo `rag_unavailable` |

## 5. Chạy hệ thống

```powershell
docker compose up -d postgres qdrant
pip install -e ".[rag]"
datn-retrieval index-products                 # ~3,5 phút, dùng vector đã có
datn-retrieval index-reviews                  # GPU khuyến nghị, có thể chạy lại để tiếp tục
copy %USERPROFILE%\.cache\huggingface data\hf_cache   # (tuỳ chọn) tránh tải lại Jina CLIP v2 trong container
docker compose up -d --build recommender rag backend
```

LLM là tuỳ chọn, đặt trong `.env` cạnh `docker-compose.yml` (mọi endpoint tương thích OpenAI đều dùng được):

```
RAG_LLM_BASE_URL=http://host.docker.internal:11434/v1
RAG_LLM_MODEL=qwen2.5:7b-instruct
```

## 6. Giới hạn đã biết (trung thực)

- **Giá trong shop không hoàn toàn là giá thật**: `price = USD × 25.000`; ~46,7% sản phẩm không có giá gốc nên nhận giá mặc định theo nhóm. Chatbot luôn gắn nhãn *giá tham khảo* cho các sản phẩm này, và lọc giá vẫn tính trên giá hiển thị.
- **Danh mục của shop được gán bằng regex** (ví dụ tất *GOLDTOE* rơi vào "Đồng hồ & Trang sức" vì chứa "gold"), nên chatbot **không** dùng danh mục làm bộ lọc cứng, chỉ dùng ngữ nghĩa.
- **Recommender chỉ biết 32.557/152.086 sản phẩm** (tập `balanced_u5_i2_v1`); boost cá nhân hoá chỉ tác động lên phần giao nhau. HR@10 của checkpoint chỉ ≈ 3,5% (xem `docs/logs/2026-09-22_...`), nên vai trò của nó là *tín hiệu phụ* cho tìm kiếm ngữ nghĩa chứ không phải bộ xếp hạng chính.
- Review chỉ được embed tối đa 4/sản phẩm; nhiều sản phẩm chỉ có 0–2 review nên câu "chưa có nhận xét" là hành vi đúng, không phải lỗi.
- Nhận xét của người mua là tiếng Anh; câu trả lời tiếng Việt trích dẫn nguyên văn tiếng Anh (template) hoặc diễn đạt lại (LLM).
- Intent là rule-based: các câu quá tự do có thể bị phân loại sai (đã có fallback hỏi lại / LLM khi `unknown`). Session lưu trong RAM của tiến trình (mất khi restart; đổi sang Redis chỉ cần thay `SessionStore`).
- **Chưa chạy với LLM thật** trong lần kiểm thử đầu; đường LLM được kiểm bằng LLM giả (unit test) — cần thử với mô hình thật trước khi demo.

## 7. Trạng thái kiểm chứng (cập nhật 30/09/2026)

| Hạng mục | Trạng thái |
|---|---|
| Test tự động (`tests/`: thư viện RAG, service `apps/rag`, gateway `/chat` của backend) | 78/78 đạt |
| Index `products` (152.086 điểm), `reviews` (295.383 điểm, nạp từ Kaggle) | Xong; sha256 file khớp `manifest.json` |
| Tìm text (có lọc giá), refine, explain, compare, tìm bằng ảnh, gọi recommender checkpoint | Đã chạy thật trên dữ liệu (host) |
| Đo độ trễ và đánh giá truy xuất | Xong trên Kaggle (T4), xem mục 9 |
| LLM Gemini | Client + retry + ngắt mạch đã kiểm bằng test (transport giả); **chưa gọi Gemini thật** (cần khoá) — chạy `python scripts/check_llm.py` |
| Docker `rag` + gateway backend, giao diện web trên trình duyệt, lọc giá/thương hiệu bằng `datn-eval-retrieval` | Chưa chạy (code có, đã qua `tsc`/`py_compile`) |

## 8. Phần nặng chạy trên Kaggle

Embed review, đo độ trễ GPU/CPU và đánh giá truy xuất nằm trong `notebooks/kaggle_rag_reviews_and_eval.ipynb` (sinh bởi `scripts/build_rag_notebooks.py`).

**Notebook độc lập, một nguồn dữ liệu.** Chỉ cần một Kaggle Dataset chứa 9 file (`items`, `train`, `valid`, `test`, `candidate_interactions` `.parquet`; `image_embeddings.npy`, `text_embeddings.npy` và 2 file metadata của chúng). Không cần dataset chứa mã nguồn: mọi hàm dùng trong notebook được sao chép nguyên văn từ `src/datn/` lúc sinh notebook bằng `ast` (các cell gắn thẻ `embedded`), nên không thể lệch với code của dịch vụ. Sau khi sửa `src/datn/`, chạy lại `python scripts/build_rag_notebooks.py --check` để sinh lại và kiểm tra.

1. Tạo Dataset từ 9 file (`python scripts/pack_kaggle_rag_inputs.py` liệt kê đường dẫn và dung lượng, ~1,3 GB), Add Input, bật GPU + Internet, Run all.
2. Notebook ghim `transformers==4.46.3` và **dừng ngay** nếu vector tính lại của 16 sản phẩm không khớp vector catalog (cosine < 0,99), để không tạo 300k vector sai.
3. Tải `review_embeddings.npy`, `reviews_meta.parquet`, `manifest.json`, `results.json` về, rồi nạp vào Qdrant (không cần GPU; lệnh xoá và tạo lại collection `reviews`, nên phần index dở dang trước đó bị thay thế):
   ```
   datn-retrieval import-reviews --embeddings review_embeddings.npy --meta reviews_meta.parquet
   ```
4. `results.json` gồm: độ trễ encode (GPU; CPU fp32/bf16, số luồng, GFLOPS matmul CPU để chẩn đoán vụ ~20 giây), HR/MRR review→sản phẩm (text, ảnh, hợp nhất; trọng số ảnh của RRF chọn trên nửa dev, báo cáo trên nửa test), căn chỉnh chéo ảnh↔text, precision@10 cho 24 truy vấn tiếng Việt (câu thô so với câu mở rộng glossary).

Kết quả là tìm kiếm **exact** (không sai số ANN) nên là cận trên so với Qdrant; lọc giá/thương hiệu và kiểm thử end-to-end (Docker `rag` + backend + web) vẫn chạy ở local bằng `datn-eval-retrieval` sau khi có collection `reviews`.

## 9. Kết quả đo trên Kaggle (`data/embedding/results.json`)

Tìm kiếm exact trên GPU T4, seed 20260930; trọng số ảnh của RRF chọn trên nửa dev (500 truy vấn), báo cáo trên nửa test (500 truy vấn).

**Độ trễ encode một truy vấn**

| Cấu hình | Trung bình |
|---|---:|
| GPU T4, text | 102 ms (p95 107 ms) |
| GPU T4, ảnh | 152 ms |
| Tìm kiếm exact + RRF trên GPU | 6 ms |
| CPU Xeon 4 vCPU, fp32 | 2,4 s (1 luồng 4,7 s; 4 luồng 2,5 s) |
| CPU, bf16 | 1,8 s |

Trên laptop đã đo ~20 s/truy vấn, chậm hơn ~10 lần dù matmul CPU của Kaggle chỉ 214 GFLOPS. Nguyên nhân chưa xác định (nghi ngờ tranh chấp tài nguyên với job GPU lúc đó hoặc cấu hình torch/Windows); cần đo lại trên máy rảnh hoặc trong container Linux. Với CPU, mỗi truy vấn vẫn ~2 s; GPU/bf16 giảm đáng kể.

**Review → sản phẩm** (dùng nội dung một review làm truy vấn; là proxy, không phải đánh giá liên quan của người)

| Phương án | HR@1 | HR@10 | HR@50 | MRR@50 |
|---|---:|---:|---:|---:|
| Chỉ text | 1,8% | 7,6% | 13,2% | 0,037 |
| Chỉ ảnh (chéo) | 0,8% | 3,8% | 8,8% | 0,019 |
| Hợp nhất, trọng số ảnh 0,8 (mặc định) | 2,4% | 6,6% | 14,2% | 0,038 |
| Hợp nhất, trọng số ảnh 0,4 | – | 8,2% | – | 0,040 |

Với 500 truy vấn, chênh lệch giữa các phương án hợp nhất nằm trong nhiễu (≈ ±1–2 điểm %), nên **chưa có bằng chứng** rằng nhánh ảnh cải thiện tìm kiếm bằng văn bản; dev chọn trọng số 0, test lại nhỉnh nhất ở 0,4. Giữ mặc định 0,8 hiện chưa được kết quả này ủng hộ hay bác bỏ. Nhánh ảnh vẫn cần cho truy vấn bằng ảnh.

**Căn chỉnh chéo ảnh ↔ text** (rank của chính sản phẩm trong 152.086): ảnh→text HR@1 13,0%, HR@10 34,1%, HR@50 54,8%, hạng trung vị 36,5; text→ảnh HR@10 32,6%. Hai nhánh gần nhau nhưng không khớp tuyệt đối.

**Truy vấn tiếng Việt** (24 truy vấn, precision@10 theo loại sản phẩm): câu thô 0,938; mở rộng glossary 0,958. Glossary chỉ tăng ~2 điểm %, không đáng kể ở cỡ mẫu này (có truy vấn giảm: "váy đầm dự tiệc" 1,0 → 0,6, có truy vấn tăng: "quần jean nữ" 0,7 → 0,9). Jina CLIP v2 đã xử lý tiếng Việt tốt; glossary là tuỳ chọn, không phải thành phần thiết yếu.

Lưu ý: kết quả review→sản phẩm thấp về tuyệt đối vì review thường không nhắc tên hàng; nó chỉ nên dùng để so sánh tương đối các cấu hình.

## 10. Đo hiệu năng dịch vụ (RTX 3050 4 GB, 30/09/2026)

`scripts/benchmark_rag.py` gọi HTTP vào dịch vụ thật (recommender ở Docker CPU, Qdrant ở Docker, không có LLM), 10 lượt/kịch bản sau 2 lượt khởi động; số chi tiết theo từng giai đoạn nằm trong `data/artifacts/rag_eval/benchmark_gpu.json` (`meta.timings_ms`). VRAM dùng ~2,4 GB, RAM tiến trình ~1,5 GB.

| Kịch bản | Trước tối ưu p50 | Sau tối ưu p50 | Sau tối ưu p95 |
|---|---:|---:|---:|
| Tìm text | 1.344 ms | 331 ms* | 497 ms |
| Tìm text có lọc giá | 463 ms | 200 ms | 316 ms |
| Tìm bằng ảnh | 550 ms | 323 ms | 349 ms |
| Refine giá / màu | 406 / 437 ms | 132 / 169 ms | 216 / 221 ms |
| Giải thích | 411 ms | 123 ms | 144 ms |
| So sánh | 435 ms | 149 ms | 168 ms |
| Tương tự | 86 ms | 65 ms | 140 ms |
| Gợi ý cá nhân hoá (có lịch sử) | 704 ms | 449 ms | 489 ms |
| Tìm text + lịch sử | 889 ms | 561 ms | 699 ms |
| 4 yêu cầu song song | 2,2 req/s | 7,5 req/s | p95 556 ms |

\* lần chạy này Qdrant vừa bị đẩy khỏi cache nên `retrieval_ms` trung bình 164 ms; khi đã nóng chỉ ~30–40 ms.

**Ba nguyên nhân chậm đã tìm ra**
1. **Khớp thương hiệu bằng ~3.000 biểu thức chính quy cho mỗi tin nhắn** tốn ~230 ms/lượt (`intent_ms`); parse_intent lúc test đơn lẻ chỉ 0,4 ms nên không lộ ra. Đã đổi sang tra cứu n-gram (test `test_brand_lookup_is_fast...` giữ < 5 ms).
2. **Đọc đĩa lạnh của Qdrant** (vector lưu `on_disk`): truy vấn đầu 450–860 ms so với 30–65 ms khi đã nóng. Chưa xử lý; hướng: lượng tử hoá scalar int8 `always_ram` (giảm ~4 lần RAM) hoặc bỏ `on_disk` nếu đủ RAM.
3. **LoRA của Jina CLIP v2 bị nhân lại ở mỗi lần gọi** (kể cả bảng embedding 250.002×1024). `JinaClipEncoder` giờ gộp adapter mặc định một lần khi nạp (`bake_lora=True`); embedding giống hệt (cosine 1,0000) nhưng CPU nhanh gấp ~2 lần, GPU giảm ~30 ms.

**CPU (cấu hình Docker mặc định)**: trên Windows của máy này encode một truy vấn vẫn **~6 giây** sau khi gộp LoRA (trước đó 20–24 giây), dù matmul CPU đạt 385 GFLOPS và RAM đủ; không phải do bf16 hay số luồng. Trên CPU Xeon 4 vCPU của Kaggle chỉ 2,4 giây khi chưa gộp LoRA. Chưa đo trong container Linux thật. Nếu cần chạy CPU, hãy dùng container Linux hoặc GPU.

**Điểm nghẽn còn lại**: cá nhân hoá (~400 ms, recommender chạy CPU trong Docker) và encode (~100–140 ms trên GPU). Chưa có LLM nên `answer_ms` = 0; khi bật LLM, thời gian sinh văn bản sẽ là thành phần lớn nhất.

## 11. LLM (Gemini)

Dùng endpoint tương thích OpenAI của Google, nên không cần SDK riêng:

```
RAG_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
RAG_LLM_MODEL=gemini-3.5-flash-lite      # nhanh/rẻ nhất; đổi sang gemini-3.5-flash nếu cần chất lượng cao hơn
RAG_LLM_API_KEY=<khoá từ https://aistudio.google.com/apikey>
RAG_LLM_REASONING_EFFORT=low             # giữ "thinking" ngắn để giảm độ trễ
RAG_LLM_MAX_TOKENS=1024
```

Đặt trong `.env` (mẫu: `.env.example`; file `.env` đã bị `.gitignore`). Khoá không bao giờ được ghi vào log, `/health` hay `meta` của phản hồi (`llm_status` chỉ có trạng thái, model, số lần lỗi).

**Độ bền của client** (`src/datn/rag/llm.py`)
- Lỗi tạm thời (429, 5xx, timeout) được thử lại tối đa 2 lần với backoff, tôn trọng `Retry-After`; lỗi 4xx khác (sai khoá, sai tên model) báo ngay, không thử lại.
- Ngắt mạch: sau 3 lần lỗi liên tiếp, bỏ qua LLM 30 giây; hết quota chỉ tốn một lần lỗi nhanh thay vì mỗi tin nhắn chờ timeout. Trong lúc đó chatbot vẫn trả lời bằng bản dựng từ dữ liệu.
- `max_tokens` không bao giờ dưới 1024 vì các model suy luận dùng một phần ngân sách cho "thinking"; nếu câu trả lời rỗng do `finish_reason=length`, lỗi nêu rõ cách chỉnh.
- Chế độ JSON: nếu nhà cung cấp trả 400 vì `response_format`, tự gửi lại không có tham số đó (kết quả vẫn được trích JSON từ văn bản).

**Chống bịa và chống prompt-injection** (`src/datn/rag/context.py`, `answer.py`)
- Prompt tuyên bố nội dung review chỉ là dữ liệu, không phải mệnh lệnh.
- Câu trả lời của LLM bị kiểm tra: mã trích dẫn `[P#]/[R#.#]` phải tồn tại; mọi số tiền (kể cả "1 đồng") phải trùng giá trong dữ liệu, chênh lệch giữa hai giá, hoặc ngưỡng người dùng nêu; mọi đường dẫn/tên miền phải có trong dữ liệu. Vi phạm → thử lại một lần kèm danh sách lỗi → vẫn sai thì dùng bản template.
- Giới hạn: kiểm tra này bắt được payload phổ biến (giá giả, link lạ, trích dẫn ảo) nhưng không chứng minh được LLM không bị dẫn dắt về mặt ngữ nghĩa (ví dụ khen quá mức theo lời một review). `scripts/check_llm.py` có một ca tấn công mẫu; cần thêm bộ ca thử nếu đưa vào sản phẩm thật.

**Tự kiểm với Gemini thật**: `python scripts/check_llm.py` (kết nối, câu trả lời search/explain/compare có căn cứ, ca prompt-injection, JSON intent). Trả về mã 0 chỉ khi mọi kiểm tra đạt. Script này đã được kiểm bằng một server giả tương thích OpenAI; **chưa** chạy với Gemini thật.
