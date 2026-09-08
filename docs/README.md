# Bộ tài liệu đồ án

## Tên đề tài

**Xây dựng hệ thống gợi ý sản phẩm đa phương thức ứng dụng RAG**  
**Developing a Multimodal Product Recommendation System using RAG**

## Danh mục tài liệu

1. [Tầm nhìn sản phẩm](./01-vision.md)
2. [Tổng quan đề tài và hệ thống](./02-overview.md)
3. [Roadmap triển khai 12 tuần](./03-roadmap.md)
4. [Đặc tả nghiệp vụ và yêu cầu hệ thống](./04-business-requirements.md)
5. [Báo cáo phân tích tập dữ liệu](./dataset_analysis.md)
6. [Báo cáo trích xuất đặc trưng đa phương thức (Embeddings)](./multimodal_embeddings_report.md)

## Nguyên tắc phạm vi

- Phần nghiên cứu cốt lõi là đánh giá **Interaction only vs Content only vs Multimodal** trên gợi ý thông thường và sản phẩm ít tương tác.
- Conversational recommendation, RAG, explanation và lightweight agent là phần mở rộng hệ thống.
- RAG không huấn luyện và không thay thế recommender; RAG chỉ tạo câu trả lời dựa trên bằng chứng sau khi hệ thống đã có Top-K.
- Agent chỉ điều phối các công cụ có sẵn, không tự quyết định sản phẩm ngoài kết quả của recommender.
- Nếu chậm tiến độ, ưu tiên theo thứ tự: dữ liệu và đánh giá → multimodal và cold-start → hội thoại → RAG → agent.

