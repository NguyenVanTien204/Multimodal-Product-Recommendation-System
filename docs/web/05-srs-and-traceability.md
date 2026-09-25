# Đặc tả yêu cầu phần mềm (SRS) và ma trận truy vết

Tài liệu này là lớp SRS dành riêng cho web demo. Nó cụ thể hóa `FR-01`–`FR-15` và `NFR-01`–`NFR-08` trong [đặc tả nghiệp vụ tổng thể](../04-business-requirements.md), để dùng trực tiếp ở chương Phân tích hệ thống và chương Kiểm thử của luận văn.

## 1. Mục tiêu, phạm vi và stakeholder

| Nhóm | Nhu cầu chính |
|---|---|
| Người mua demo | Khám phá sản phẩm, lọc, nhận Top-K, refinement, hỏi đáp và so sánh rõ ràng. |
| Sinh viên/nghiên cứu viên | Minh họa đúng hành vi hệ thống và provenance, không làm sai kết quả thực nghiệm. |
| Giảng viên phản biện | Kiểm tra được hai chức năng trung tâm: gợi ý tuần tự và chatbot hỗ trợ mua hàng. |
| Quản trị viên demo | Biết model/index nào đang được phục vụ và lỗi ở tầng nào. |

Phạm vi gồm storefront, session ngắn hạn, API integration, hiển thị evidence và mini marketplace (JWT, catalog, giỏ hàng, tồn kho, tạo đơn hàng). Không gồm cổng thanh toán thật, giao vận, hoàn tiền hoặc lưu PII ngoài dữ liệu tối thiểu người dùng tự nhập trong môi trường demo.

## 2. Yêu cầu chức năng web

| ID | Yêu cầu | Ưu tiên | Điều kiện kiểm thử |
|---|---|---:|---|
| WFR-01 | Hiển thị catalog/Top-K dạng card có rank, item ID, title và metadata sẵn có. | Must | Product thiếu giá/ảnh vẫn render với trạng thái “Chưa có dữ liệu”. |
| WFR-02 | Tìm kiếm bằng text, category, màu/phong cách khi dữ liệu có và giá tối đa. | Must | Request `/search` chứa query/filter; chip thể hiện constraint đã hiểu. |
| WFR-03 | Yêu cầu gợi ý từ history hoặc fallback cho user mới. | Must | `/recommend` trả source `model` hoặc `popularity_fallback` được gắn nhãn. |
| WFR-04 | Lưu `session_id`, preference và `last_result_item_ids` trong vòng đời phiên. | Must | Refresh thao tác trong phiên không làm mất context trừ khi hết TTL/reset. |
| WFR-05 | Tinh chỉnh bằng form hoặc chat, rồi cập nhật Top-K mà không retrain. | Must | “rẻ hơn” gọi `/refine` hoặc `/chat`; response hiển thị preference mới. |
| WFR-06 | Cho phép chọn đúng hai sản phẩm trong kết quả hiện tại để so sánh. | Should | UI chặn chọn quá hai item; `/compare` nhận đúng hai `item_ids`. |
| WFR-07 | Giải thích một item thuộc kết quả hiện tại và hiển thị evidence/provenance khi có. | Should | `/explain` không được gọi với item ngoài session result. |
| WFR-08 | Cung cấp cửa sổ chat, hiển thị user/assistant message, action result và product card trả về. | Must | `/chat` có `session_id`; message không bị gửi lại tự động khi timeout. |
| WFR-09 | Hiển thị trạng thái nguồn dữ liệu: model, popularity fallback hoặc demo fixture. | Must | Không có text nào gọi demo fixture là kết quả model. |
| WFR-10 | Cung cấp reset session và trạng thái lỗi có thể hành động. | Must | Reset tạo session mới; network/422/503 có thông báo và nút thử lại phù hợp. |
| WFR-11 | Có trang/khối giới thiệu phạm vi nghiên cứu, giới hạn dữ liệu và chính sách demo. | Should | Người xem phân biệt được demo khỏi sàn thương mại điện tử thật. |
| WFR-12 | Hỗ trợ image query sau khi image encoder/API được cung cấp. | Could | Validate MIME/size client rồi gọi endpoint theo contract mở rộng. |
| WFR-13 | Người dùng đăng ký, đăng nhập và xem hồ sơ bằng JWT. | Must | Password được hash; endpoint cần auth từ chối token thiếu/hết hạn. |
| WFR-14 | Người dùng thêm/xóa sản phẩm trong giỏ và không vượt tồn kho. | Must | Quantity không hợp lệ hoặc vượt stock trả lỗi xác định. |
| WFR-15 | Checkout tạo đơn hàng `PENDING`, snapshot tên/giá và trừ tồn kho trong một transaction. | Must | Không oversell khi hai checkout cùng sản phẩm; cart được làm rỗng khi thành công. |
| WFR-16 | Người dùng xem đơn hàng của chính mình; admin tạo category/product và index vector. | Must | Ownership/RBAC được kiểm thử bằng token user/admin khác nhau. |
| WFR-17 | Tìm sản phẩm tương tự qua Qdrant và gọi recommender tuần tự qua gateway riêng. | Should | Qdrant không khả dụng hoặc gateway chưa cấu hình trả 503 có thông báo rõ. |

## 3. Yêu cầu phi chức năng có thể đo

Các ngưỡng dưới đây là **mục tiêu nghiệm thu cho môi trường demo cục bộ**, không phải tuyên bố hiệu năng production và không thay thế metric recommender.

| ID | Thuộc tính | Yêu cầu/đo đạc |
|---|---|---|
| WNFR-01 | Hiệu năng | Đo p50/p95 riêng cho `/recommend`, `/search`, `/refine`, `/chat`; UI có loading state. Mục tiêu nội bộ: p95 retrieval API ≤ 3 giây khi index đã nạp; chat có thông báo đang xử lý nếu vượt 3 giây. |
| WNFR-02 | Tính sẵn sàng/fallback | Lỗi agent/LLM không chặn recommend/search/refine; 503 phải có thông báo xác định. |
| WNFR-03 | Bảo mật | Secret chỉ server-side; validate schema ở FastAPI; sanitize output hiển thị; không log token/PII/raw upload trái policy. |
| WNFR-04 | Privacy | Không yêu cầu đăng nhập và không lưu PII; `user_id` Amazon chỉ là định danh ẩn danh; session có TTL cấu hình. |
| WNFR-05 | Khả dụng | Desktop từ 1280px và mobile từ 360px không vỡ layout; thao tác chính bằng bàn phím; image có alt text; màu/lỗi không là tín hiệu duy nhất. |
| WNFR-06 | Tương thích | Chrome/Edge bản hiện hành; Node.js LTS; FastAPI/Python theo `pyproject.toml`. |
| WNFR-07 | Bảo trì | Next.js TypeScript strict, ESLint; FastAPI Pydantic/OpenAPI; không duplicate schema thủ công nếu có thể sinh type từ OpenAPI. |
| WNFR-08 | Quan sát/truy vết | Mỗi response có `request_id`, source và version artifact khi có; log latency/status không log nội dung chat không cần thiết. |
| WNFR-09 | Độ tin cậy thông tin | Explanation/compare hiển thị field thiếu và evidence; không tạo price, thuộc tính hoặc review ngoài context. |

## 4. Dữ liệu giao diện và ownership

| Dữ liệu | Chủ sở hữu | Lưu ở đâu | Vòng đời |
|---|---|---|---|
| `session_id`, preference, last results | FastAPI session manager | Backend store; frontend chỉ giữ token/context UI | Theo TTL cấu hình |
| Product metadata/rank/score | Recommender API | Catalog/index/backend | Theo artifact version |
| Evidence/review citation | RAG service | Backend response | Chỉ trong response/UI session |
| Theme, selected comparison | Frontend | Browser state | Một phiên browser |
| API/LLM secret | Backend/Next server | Environment | Không bao giờ gửi client |

## 5. Ma trận truy vết

| Mục tiêu luận văn / yêu cầu tổng | Yêu cầu web | API | Bằng chứng nghiệm thu |
|---|---|---|---|
| FR-01, FR-05; UC-01 | WFR-01, WFR-03 | `/recommend` | AC-01, AC-02; source badge |
| FR-02, FR-03; UC-02/03 | WFR-02, WFR-12 | `/search` | AC-03; test filter/upload |
| FR-07–10; UC-04 | WFR-04, WFR-05 | `/refine`, `/chat` | AC-04, AC-05; session test |
| FR-11–12; UC-05 | WFR-07 | `/explain` | AC-06; evidence/missing-state test |
| FR-13; UC-06 | WFR-06 | `/compare` | AC-07; same-schema test |
| FR-14–15; UC-07 | WFR-08–10 | `/chat`, direct APIs | AC-10; agent failure test |
| NFR-03, NFR-05–08 | WNFR-01–09 | `/health` + all APIs | latency log, security/accessibility checklist |

## 6. Kế hoạch kiểm thử tối thiểu

1. Unit: render card thiếu metadata, reducer preference, giới hạn hai item compare, API error mapping.
2. Component: search form tạo payload đúng; loading/empty/error states; keyboard navigation.
3. API contract: FastAPI OpenAPI khớp request/response TypeScript; test 422/404/503.
4. E2E: năm kịch bản trong `04-delivery-plan.md`, gồm fallback và agent unavailable.
5. Manual: Chrome/Edge, viewport 360px/1280px, keyboard-only và contrast cơ bản.
