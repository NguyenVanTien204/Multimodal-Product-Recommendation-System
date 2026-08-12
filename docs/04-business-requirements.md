# Đặc tả nghiệp vụ và yêu cầu hệ thống

## 1. Mục đích

Tài liệu mô tả nghiệp vụ của hệ thống gợi ý sản phẩm đa phương thức, ranh giới trách nhiệm giữa recommender, RAG và agent, các use case chính, quy tắc nghiệp vụ, yêu cầu chức năng/phi chức năng và tiêu chí nghiệm thu.

## 2. Phạm vi nghiệp vụ

Hệ thống hỗ trợ ba nhóm hoạt động:

1. **Gợi ý và tìm kiếm:** tạo Top-K cá nhân hóa từ lịch sử, văn bản, ảnh hoặc sản phẩm tham chiếu.
2. **Tương tác theo phiên:** hiểu preference, cập nhật yêu cầu và rerank kết quả.
3. **Hỗ trợ quyết định:** giải thích lý do gợi ý và so sánh sản phẩm bằng metadata/review được truy xuất.

Huấn luyện, đánh giá và quản trị artifact thuộc nghiệp vụ nghiên cứu/vận hành nội bộ, không phải thao tác trực tiếp của người mua.

## 3. Tác nhân

| Tác nhân | Vai trò |
|---|---|
| Người dùng | Yêu cầu gợi ý, tìm kiếm, refinement, giải thích, so sánh |
| Nhà nghiên cứu | Chuẩn bị dữ liệu, huấn luyện, chạy thí nghiệm và phân tích |
| Quản trị viên | Cấu hình model/index, kiểm tra artifact và trạng thái dịch vụ |
| Recommender | Sinh candidate và xếp hạng sản phẩm |
| RAG/LLM | Truy xuất bằng chứng và diễn đạt giải thích/so sánh |
| Lightweight Agent | Chọn và gọi công cụ phù hợp với ý định |

## 4. Danh sách use case

| ID | Use case | Tác nhân chính | Mức ưu tiên |
|---|---|---|---|
| UC-01 | Nhận gợi ý cá nhân hóa từ lịch sử | Người dùng | P0 |
| UC-02 | Tìm sản phẩm bằng văn bản | Người dùng | P1 |
| UC-03 | Tìm sản phẩm bằng ảnh/sản phẩm tham chiếu | Người dùng | P1 |
| UC-04 | Lọc và tinh chỉnh preference | Người dùng | P1 |
| UC-05 | Hỏi lý do gợi ý | Người dùng | P2 |
| UC-06 | So sánh sản phẩm | Người dùng | P2 |
| UC-07 | Hội thoại điều phối nhiều ý định | Người dùng | P3 |
| UC-08 | Chuẩn bị và đóng băng dataset | Nhà nghiên cứu | P0 |
| UC-09 | Huấn luyện và đánh giá model | Nhà nghiên cứu | P0 |
| UC-10 | Quản lý embedding và index | Nhà nghiên cứu/Quản trị viên | P1 |

## 5. Đặc tả use case chính

### UC-01 — Gợi ý cá nhân hóa

**Tiền điều kiện:** model, item catalog và FAISS index hợp lệ; user có history hoặc hệ thống có fallback.

**Luồng chính:**

1. Người dùng yêu cầu gợi ý.
2. Hệ thống lấy lịch sử positive interactions và preference hiện tại.
3. Recommender tạo user/query representation.
4. FAISS truy xuất candidate, ranking tính điểm và áp dụng filter.
5. Hệ thống trả Top-K kèm thông tin hiển thị và mã tham chiếu trong phiên.

**Ngoại lệ:** user mới hoặc history không đủ → dùng preference hiện tại; nếu cũng không có, dùng Popularity theo category/toàn cục và thông báo đây là gợi ý fallback.

**Hậu điều kiện:** Top-K và context cần thiết được lưu trong phiên để phục vụ refine/explain/compare.

### UC-02 — Tìm bằng văn bản

1. Người dùng nhập mô tả nhu cầu.
2. Hệ thống tách constraint có cấu trúc và phần semantic query.
3. Text encoder tạo embedding; FAISS lấy candidate.
4. Hệ thống lọc hard constraint như category/max price, sau đó rerank theo preference/history.
5. Trả Top-K và constraint đã hiểu để người dùng kiểm tra.

Nếu giá/thuộc tính bị thiếu, sản phẩm không được khẳng định là thỏa constraint tương ứng; policy include/exclude phải cấu hình và hiển thị nhất quán.

### UC-03 — Tìm bằng ảnh hoặc sản phẩm tham chiếu

1. Người dùng tải ảnh hoặc chọn một sản phẩm trong kết quả.
2. Hệ thống kiểm tra định dạng/kích thước ảnh hoặc item ID.
3. Image encoder lấy representation.
4. FAISS truy xuất item tương tự; loại item nguồn nếu cần.
5. Recommender rerank theo user preference và constraint.

Ảnh lỗi/không hỗ trợ phải trả thông báo rõ ràng, không chuyển âm thầm sang kết quả Popularity.

### UC-04 — Tinh chỉnh preference

1. Người dùng nói “rẻ hơn”, “màu đen”, “ít thể thao hơn” hoặc yêu cầu tương đương.
2. Hệ thống trích xuất phần preference thay đổi.
3. Người dùng có thể xem constraint đã hiểu.
4. Session state được merge theo quy tắc ghi đè rõ ràng.
5. Candidate hiện tại hoặc candidate mới được rerank; không retrain model.
6. Trả Top-K mới.

### UC-05 — Giải thích gợi ý

1. Người dùng chọn một sản phẩm thuộc kết quả hiện tại.
2. Hệ thống lấy reason signals từ ranking nếu có.
3. RAG truy xuất metadata, features, description và review phù hợp của đúng item.
4. LLM tạo giải thích liên hệ với preference/history nhưng chỉ dùng evidence đã cung cấp.
5. Hệ thống trả giải thích cùng bằng chứng/nguồn dữ liệu.

Nếu không đủ bằng chứng, hệ thống phải nói rõ giới hạn thay vì suy diễn.

### UC-06 — So sánh sản phẩm

1. Người dùng chọn từ hai sản phẩm hợp lệ trong catalog/kết quả.
2. Hệ thống truy xuất cùng một nhóm thuộc tính cho từng sản phẩm.
3. Chuẩn hóa giá/thuộc tính có thể so sánh.
4. LLM tổng hợp điểm giống, khác và độ phù hợp với preference.
5. Trả bảng/đoạn so sánh; trường thiếu được đánh dấu “không có dữ liệu”.

### UC-09 — Huấn luyện và đánh giá

1. Nhà nghiên cứu chọn dataset manifest và experiment config.
2. Pipeline huấn luyện chỉ sử dụng train.
3. Chọn model/hyperparameter trên validation.
4. Chạy test một cách nhất quán cho model được chốt.
5. Xuất Recall@10, NDCG@10, HitRate@10 cho toàn bộ và sparse-item segments.
6. Lưu config, seed, dataset version, artifact và kết quả.

## 6. Quy tắc nghiệp vụ

| ID | Quy tắc |
|---|---|
| BR-01 | `item_id` chuẩn là `parent_asin` khi liên kết interaction và metadata. |
| BR-02 | `rating >= 4` được coi là positive interaction cho implicit Top-K. |
| BR-03 | Rating thấp có thể phục vụ EDA/RAG nhưng không được chuyển thành positive interaction. |
| BR-04 | Dataset phải qua iterative K-core, sampling và K-core lần hai trước khi split. |
| BR-05 | Split theo thời gian: N-2 train, N-1 validation, N test cho mỗi user đủ điều kiện. |
| BR-06 | Không dùng test để chọn model, threshold hoặc fusion weight. |
| BR-07 | Sparse item được xác định bằng số interaction trong train, với báo cáo cho `<= 5` và `<= 10`. |
| BR-08 | Embedding phải được cache và gắn với dataset/model/config version. |
| BR-09 | LLM chỉ extract preference/điều phối/diễn đạt; recommender quyết định ranking. |
| BR-10 | Refinement chỉ cập nhật session và rerank; không kích hoạt retraining. |
| BR-11 | RAG chỉ chạy sau khi có candidate/Top-K hoặc item cụ thể. |
| BR-12 | Explanation và comparison chỉ được dùng context của đúng item liên quan. |
| BR-13 | Không khẳng định giá, thuộc tính hoặc nhận xét khi evidence không có. |
| BR-14 | Mỗi experiment phải có dataset version, config, seed, code version và kết quả lưu trữ. |
| BR-15 | Nếu model/agent mở rộng lỗi, baseline recommender/API lõi phải có đường fallback. |

## 7. Yêu cầu chức năng

### Recommendation và retrieval

- **FR-01:** Hệ thống phải sinh Top-K từ `user_id` hợp lệ.
- **FR-02:** Hệ thống phải hỗ trợ text query, image query và item-reference query.
- **FR-03:** Hệ thống phải hỗ trợ filter category, color/style khi có dữ liệu và max price.
- **FR-04:** Hệ thống phải loại item không hợp lệ và xử lý item đã tương tác theo evaluation/product policy.
- **FR-05:** Hệ thống phải có fallback cho user mới hoặc thiếu history.
- **FR-06:** Mỗi kết quả phải có item ID, title, score/rank và metadata hiển thị có sẵn.

### Hội thoại

- **FR-07:** Hệ thống phải chuyển câu người dùng thành preference schema có validation.
- **FR-08:** Hệ thống phải lưu preference và danh sách kết quả theo session.
- **FR-09:** Hệ thống phải merge, ghi đè hoặc loại bỏ constraint theo yêu cầu tiếp theo.
- **FR-10:** Hệ thống phải hỗ trợ tham chiếu “sản phẩm #N” trong session hiện tại.

### RAG và agent

- **FR-11:** Hệ thống phải truy xuất product metadata/review theo item ID trước khi explain/compare.
- **FR-12:** Câu trả lời phải phân biệt dữ liệu có bằng chứng và dữ liệu bị thiếu.
- **FR-13:** Hệ thống phải so sánh ít nhất hai sản phẩm bằng cùng một schema thuộc tính.
- **FR-14:** Agent phải chọn trong toolset cho phép và validate input/output của tool.
- **FR-15:** Người dùng vẫn có thể gọi các API chức năng khi agent không hoạt động.

### Dữ liệu và thực nghiệm

- **FR-16:** Pipeline phải tạo train/valid/test, items, reviews và dataset manifest.
- **FR-17:** Evaluation phải xuất Recall@10, NDCG@10 và HitRate@10.
- **FR-18:** Pipeline phải đánh giá đầy đủ model matrix và sparse-item segments.
- **FR-19:** Kết quả experiment phải xuất ra CSV/JSON và hỗ trợ tạo biểu đồ.
- **FR-20:** Hệ thống phải duy trì mapping nhất quán giữa item ID và embedding/index row.

## 8. Yêu cầu phi chức năng

| ID | Yêu cầu |
|---|---|
| NFR-01 | **Tái lập:** cùng manifest/config/seed phải tạo kết quả tương đương trong sai số cho phép. |
| NFR-02 | **Không rò rỉ:** feature/model/evaluation không được dùng interaction tương lai ngoài split quy định. |
| NFR-03 | **Truy vết:** explanation/compare phải truy được item và context nguồn. |
| NFR-04 | **Hiệu năng:** Top-K retrieval/ranking cho demo phải đủ tương tác; mục tiêu p95 nội bộ nên được đo và công bố thay vì cam kết production. |
| NFR-05 | **Tin cậy:** ảnh hỏng, item thiếu và service phụ lỗi phải có fallback/thông báo xác định. |
| NFR-06 | **Bảo mật:** không lưu thêm dữ liệu nhận dạng cá nhân; validate file upload và giới hạn loại/kích thước. |
| NFR-07 | **Khả năng bảo trì:** data, model, retrieval, RAG, agent và API phải tách module/config. |
| NFR-08 | **Quan sát:** ghi log request ID, model/index version, latency và lỗi mà không ghi nội dung nhạy cảm không cần thiết. |
| NFR-09 | **Tính nhất quán:** mọi model trong so sánh dùng cùng dataset, candidate/evaluation protocol và metric implementation. |

## 9. Mô hình trạng thái phiên

Một session tối thiểu gồm:

```json
{
  "session_id": "...",
  "user_id": "...",
  "preferences": {
    "category": "shoes",
    "color": "black",
    "style": "casual",
    "max_price": 100,
    "sporty": "negative"
  },
  "reference_item_ids": [],
  "last_result_item_ids": [],
  "updated_at": "..."
}
```

Preference mới ghi đè giá trị cũ cùng trường; câu phủ định/xóa constraint phải được biểu diễn tường minh. Session hết hạn theo cấu hình và không được coi là long-term user profile.

## 10. Tiêu chí nghiệm thu nghiệp vụ

| ID | Kịch bản nghiệm thu | Kết quả mong đợi |
|---|---|---|
| AC-01 | User có history gọi `/recommend` | Trả đúng K item hợp lệ và có rank/metadata |
| AC-02 | User mới yêu cầu gợi ý | Trả fallback có thông báo, không lỗi hệ thống |
| AC-03 | Query “giày đen casual dưới $100” | Preference được parse; hard constraint được áp dụng theo missing policy |
| AC-04 | Tiếp tục “ít thể thao hơn” | Session cập nhật và kết quả được rerank, không retrain |
| AC-05 | “Giống sản phẩm #2 nhưng rẻ hơn” | Resolve đúng item #2 và áp dụng constraint giá tương đối |
| AC-06 | Hỏi lý do cho item trong Top-K | Trả giải thích dựa trên context của đúng item, nêu thiếu dữ liệu nếu có |
| AC-07 | So sánh item 2 và 4 | Trả các thuộc tính đồng nhất, không bịa trường thiếu |
| AC-08 | Chạy evaluation hai lần cùng config/seed | Kết quả tái lập trong sai số đã quy định |
| AC-09 | Báo cáo sparse items | Có metric riêng cho `<= 5`, `<= 10` theo train count |
| AC-10 | Agent không khả dụng | `/recommend`, `/search`, `/refine`, `/explain`, `/compare` vẫn gọi trực tiếp được |

## 11. Rủi ro và biện pháp

| Rủi ro | Tác động | Giảm thiểu |
|---|---|---|
| Dataset quá lớn/ảnh tải lỗi | Trễ ETL, thiếu modality | Subset có kiểm soát, cache, retry, missing policy |
| Data leakage | Kết quả nghiên cứu sai | Chronological split, test kiểm tra timestamp và overlap |
| Fusion không vượt CF | Kỳ vọng kết quả không đạt | Xem là kết quả nghiên cứu; phân tích theo segment và ablation |
| Text/image embedding lệch domain | Retrieval kém | Chuẩn hóa input, error analysis, thử encoder phụ khi còn thời gian |
| LLM hallucination | Giải thích sai | RAG bị giới hạn theo item, provenance, abstain khi thiếu evidence |
| Scope phình vì UI/agent | Không xong thesis core | Khóa tuần 6 cho research core; cắt agent trước |
| Metric/candidate protocol không đồng nhất | So sánh thiếu công bằng | Một evaluation module/config dùng chung cho mọi model |

## 12. Ma trận truy vết mục tiêu

| Mục tiêu/RQ | Thành phần | Bằng chứng nghiệm thu |
|---|---|---|
| RQ1 Multimodal | Model matrix + common evaluation | Recall/NDCG/HR của CF, content, multimodal |
| RQ2 Ablation | `-CF`, `-Image`, `-Text`, fusion sweep | Bảng ablation và phân tích modality |
| RQ3 Sparse items | Train-count segmentation | Metric cho normal, `<=10`, `<=5` |
| Conversational extension | Preference extraction + session rerank | AC-03 đến AC-05 |
| RAG extension | Product/review retrieval + LLM | AC-06, AC-07 và provenance |
| Agent extension | Tool-calling orchestration | UC-07 và fallback AC-10 |

