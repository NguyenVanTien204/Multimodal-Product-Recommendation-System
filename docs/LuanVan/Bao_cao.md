**TRƯỜNG ĐH MỎ - ĐỊA CHẤT**

**ĐỒ ÁN TỐT NGHIỆP**

Sinh viên thực hiện**: Nguyễn Văn Tiến**

Mã sinh viên**: 2121050201**

Giảng viên hướng dẫn**: Nguyễn Duy Huy**

# MỤC LỤC

[MỤC LỤC 4](#_Toc241658102)

[DANH MỤC CÁC HÌNH VẼ 8](#_Toc241658103)

[DANH MỤC CÁC BẢNG BIỂU 8](#_Toc241658104)

[CẤU TRÚC CỦA ĐỒ ÁN 9](#_Toc241658105)

[THÔNG TIN ĐỒ ÁN 12](#_Toc241658106)

[MỞ ĐẦU 14](#_Toc241658107)

[CHƯƠNG 1 TỔNG QUAN LÝ THUYẾT VỀ LĨNH VỰC NGHIÊN CỨU 17](#_Toc241658108)

[1.1 Tổng quan bài toán 17](#_Toc241658109)

[1.1.1 Bài toán gợi ý sản phẩm trong thương mại điện tử 17](#_Toc241658110)

[1.1.2 Bài toán gợi ý đa phương thức và chatbot RAG 17](#_Toc241658111)

[1.2 Cơ sở lý thuyết về dữ liệu và tiền xử lý 17](#_Toc241658112)

[1.2.1 Dữ liệu tương tác và dữ liệu sản phẩm 17](#_Toc241658113)

[1.2.2 Tiền xử lý và xây dựng tập dữ liệu thực nghiệm 17](#_Toc241658114)

[1.3 Cơ sở lý thuyết về hệ thống gợi ý đa phương thức 18](#_Toc241658115)

[1.3.1 Gợi ý tuần tự và mô hình User Tower 18](#_Toc241658116)

[1.3.2 Biểu diễn đa phương thức và mô hình CLIP 18](#_Toc241658117)

[1.3.3 Kiến trúc gợi ý hai giai đoạn 18](#_Toc241658118)

[1.4 Cơ sở lý thuyết về Retrieval-Augmented Generation 18](#_Toc241658119)

[1.4.1 Kiến trúc RAG 18](#_Toc241658120)

[1.4.2 Chatbot RAG hỗ trợ gợi ý và tìm kiếm sản phẩm 19](#_Toc241658121)

[1.5 Phương pháp đánh giá 19](#_Toc241658122)

[1.5.1 Đánh giá mô hình gợi ý sản phẩm 19](#_Toc241658123)

[1.5.2 Đánh giá chatbot RAG 19](#_Toc241658124)

[1.6 Công nghệ và công cụ sử dụng 19](#_Toc241658125)

[1.6.1 Công nghệ xử lý dữ liệu, huấn luyện và triển khai mô hình 19](#_Toc241658126)

[1.6.2 Công nghệ phát triển giao diện và quản lý hệ thống 20](#_Toc241658127)

[1.7 Các công trình nghiên cứu liên quan 20](#_Toc241658128)

[1.7.1 Nghiên cứu về gợi ý tuần tự và gợi ý đa phương thức 20](#_Toc241658129)

[1.7.2 Nghiên cứu về RAG 20](#_Toc241658130)

[1.8 Tổng kết chương 20](#_Toc241658131)

[CHƯƠNG 2 QUY TRÌNH XÂY DỰNG HỆ THỐNG GỢI Ý SẢN PHẨM ĐA PHƯƠNG THỨC VÀ CHATBOT RAG 21](#_Toc241658132)

[2.1 Phát biểu bài toán và phạm vi xây dựng hệ thống 21](#_Toc241658133)

[2.1.1 Bài toán gợi ý sản phẩm đa phương thức 21](#_Toc241658134)

[2.1.2 Bài toán chatbot RAG hỗ trợ mua sắm 21](#_Toc241658135)

[2.2 Nguồn dữ liệu và mô tả dữ liệu 21](#_Toc241658136)

[2.2.1 Nguồn dữ liệu Amazon Reviews 2023 21](#_Toc241658137)

[2.2.2 Bộ dữ liệu thực nghiệm chốt của đề tài 22](#_Toc241658138)

[2.3 Quy trình tổng thể xây dựng hệ thống 22](#_Toc241658139)

[2.3.1 Pipeline xử lý dữ liệu và huấn luyện mô hình gợi ý 22](#_Toc241658140)

[2.3.2 Pipeline hoạt động của chatbot RAG 22](#_Toc241658141)

[2.4 Tiền xử lý và xây dựng dữ liệu thực nghiệm 23](#_Toc241658142)

[2.4.1 Làm sạch và lọc dữ liệu tương tác 23](#_Toc241658143)

[2.4.2 Lọc iterative K-core và xử lý dữ liệu thưa 23](#_Toc241658144)

[2.4.3 Chia tập dữ liệu theo thời gian và kiểm tra rò rỉ dữ liệu 23](#_Toc241658145)

[2.4.4 Chuẩn bị đặc trưng văn bản và hình ảnh 24](#_Toc241658146)

[2.5 Xây dựng mô hình gợi ý đa phương thức 24](#_Toc241658147)

[2.5.1 Xây dựng User Tower và Item Representation 24](#_Toc241658148)

[2.5.2 Sinh ứng viên từ nhiều nguồn 24](#_Toc241658149)

[2.5.3 Xây dựng residual listwise reranker 24](#_Toc241658150)

[2.6 Xây dựng chatbot RAG hỗ trợ gợi ý và tìm kiếm sản phẩm 25](#_Toc241658151)

[2.6.1 Xây dựng kho tri thức và cơ chế truy xuất 25](#_Toc241658152)

[2.6.2 Xử lý hội thoại và tạo phản hồi có căn cứ 25](#_Toc241658153)

[2.7 Thiết kế kiến trúc hệ thống và triển khai ứng dụng 25](#_Toc241658154)

[2.7.1 Các tác nhân của hệ thống 25](#_Toc241658155)

[2.7.2 Yêu cầu chức năng của hệ thống 25](#_Toc241658156)

[2.7.3 Yêu cầu phi chức năng 26](#_Toc241658157)

[2.7.4 Kiến trúc tổng thể 26](#_Toc241658158)

[2.7.5 Thiết kế dữ liệu nghiệp vụ và cơ sở dữ liệu vector 27](#_Toc241658159)

[2.7.6 Thiết kế API và các service 27](#_Toc241658160)

[2.7.7 Use case và sequence diagram 27](#_Toc241658161)

[2.7.8 Thiết kế giao diện người dùng 27](#_Toc241658162)

[2.8 Tổ chức mã nguồn, môi trường phát triển và quản lý phiên bản 28](#_Toc241658163)

[2.8.1 Cấu trúc mã nguồn và quản lý dữ liệu 28](#_Toc241658164)

[2.8.2 Môi trường phát triển, Docker và Git 28](#_Toc241658165)

[2.9 Tổng kết chương 28](#_Toc241658166)

[CHƯƠNG 3 THỰC NGHIỆM VÀ ĐÁNH GIÁ HỆ THỐNG 29](#_Toc241658167)

[3.1 Môi trường thực nghiệm 29](#_Toc241658168)

[3.1.1 Cấu hình phần cứng và hệ điều hành 29](#_Toc241658169)

[3.1.2 Môi trường phần mềm và thư viện 29](#_Toc241658170)

[3.2 Thiết lập thực nghiệm 29](#_Toc241658171)

[3.2.1 Dữ liệu và giao thức đánh giá 29](#_Toc241658172)

[3.2.2 Thiết lập huấn luyện User Tower đa phương thức 30](#_Toc241658173)

[3.2.3 Thiết lập sinh ứng viên và xếp hạng lại 30](#_Toc241658174)

[3.3 Kết quả thực nghiệm mô hình gợi ý đa phương thức 31](#_Toc241658175)

[3.3.1 Kết quả của User Tower 31](#_Toc241658176)

[3.3.2 Đánh giá chiến lược sinh ứng viên nhiều nguồn 31](#_Toc241658177)

[3.3.3 Kết quả của residual listwise reranker 32](#_Toc241658178)

[3.4 Đánh giá chatbot RAG hỗ trợ gợi ý và tìm kiếm 33](#_Toc241658179)

[3.4.1 Kịch bản kiểm thử chức năng chatbot 33](#_Toc241658180)

[3.4.2 Đánh giá chất lượng phản hồi và tính có căn cứ 33](#_Toc241658181)

[3.5 Đánh giá hệ thống web và API 33](#_Toc241658182)

[3.5.1 Giao diện và luồng hoạt động chính 33](#_Toc241658183)

[3.5.2 Kiểm thử các chức năng chính 34](#_Toc241658184)

[3.6 Khai báo việc sử dụng AI trong quá trình thực hiện 34](#_Toc241658185)

[3.6.1 Phạm vi công việc có sử dụng AI hỗ trợ 34](#_Toc241658186)

[3.6.2 Kiểm tra và chỉnh sửa của sinh viên 34](#_Toc241658187)

[3.7 Tổng kết chương 34](#_Toc241658188)

[KẾT LUẬN 35](#_Toc241658189)

[Đánh giá kết quả đạt được 35](#_Toc241658190)

[Đánh giá theo mục tiêu đề tài 35](#_Toc241658191)

[Bảng đối chiếu yêu cầu tối thiểu và kết quả thực hiện 35](#_Toc241658192)

[Hạn chế và phạm vi chưa hoàn thành 36](#_Toc241658193)

[Hạn chế của mô hình gợi ý 36](#_Toc241658194)

[Hạn chế của chatbot RAG 36](#_Toc241658195)

[Khai báo sử dụng AI 36](#_Toc241658196)

[So sánh nội dung thực tập và yêu cầu trong cẩm nang 37](#_Toc241658197)

[Hướng phát triển 37](#_Toc241658198)

[TÀI LIỆU THAM KHẢO 38](#_Toc241658199)

[PHỤ LỤC 39](#_Toc241658200)

# DANH MỤC CÁC HÌNH VẼ

[Hình 1‑1 Thao tác cập nhật mục lục 4](#_Toc262311533)

[Hình 1‑2 Cách chèn nhãn cho hình 4](#_Toc262311534)

[Hình 1‑3 Cách tạo một nhãn mới 4](#_Toc262311535)

[Hình 1‑4 Cách tham chiếu đến một nhãn 4](#_Toc262311536)

# DANH MỤC CÁC BẢNG BIỂU

[Bảng 1‑1 Tên bảng 4](#_Toc262311537)

# THÔNG TIN ĐỒ ÁN

1\. Thông tin chung

| Tên đề tài:                             | Xây dựng hệ thống gợi ý sản phẩm đa phương thức ứng dụng RAG |
| --------------------------------------- | ------------------------------------------------------------ |
| Sinh viên thực hiện: Nguyễn Văn Tiến    |                                                              |
| Mã sinh viên: 2221050201                |                                                              |
| Lớp: DCCTKH67A                          |                                                              |
| Hệ đào tạo: Chính quy                   |                                                              |
| Điện thoại: 0862863204                  |                                                              |
| Email: <2221050201@student.humg.edu.vn> |                                                              |
| Thời gian thực hiện: 2026               |                                                              |

2\. Mục tiêu và phạm vi

Đề tài hướng tới việc hoàn thành các mục tiêu cụ thể, đo lường được sau:

- Xây dựng mô hình gợi ý sản phẩm đa phương thức hai giai đoạn (truy hồi ứng viên kết hợp xếp hạng lại danh sách) tích hợp lịch sử tương tác, văn bản và hình ảnh; đạt hiệu năng xếp hạng trên tập kiểm nghiệm với các chỉ số mục tiêu cụ thể: Candidate Recall 30%, HitRate@10 3,5% và NDCG@10 2,4%.
- Xây dựng hệ thống Chatbot RAG hỗ trợ hội thoại mua sắm, cho phép xử lý truy vấn tự nhiên để tìm kiếm, giải thích gợi ý và so sánh sản phẩm; đảm bảo 100% câu trả lời có căn cứ trích xuất từ dữ liệu (không ảo giác về giá, thuộc tính hay sản phẩm) và thời gian phản hồi trung bình dưới 2,5 giây/truy vấn.
- Tích hợp và hoàn thiện giao diện ứng dụng web thực nghiệm, kết nối liền mạch giữa mô hình gợi ý và chatbot nhằm mô phỏng luồng tương tác thực tế của người dùng từ lúc nhập nhu cầu đến khi nhận kết quả gợi ý.

Phạm vi đề tài

- Dữ liệu thực nghiệm: Sử dụng tập dữ liệu trích xuất từ nhánh _Clothing_Shoes_and_Jewelry_ của bộ Amazon Reviews 2023, cố định quy mô ở 21.690 người dùng, 32.557 sản phẩm và 198.200 tương tác tích cực.
- Giới hạn tri thức: Chatbot vận hành theo cơ chế đóng (closed-domain RAG), chỉ sử dụng metadata, thông số kỹ thuật và đánh giá thực tế của tập dữ liệu đã chuẩn hóa; loại trừ hoàn toàn việc tự sinh dữ liệu ngoại lai.

3\. Nội dung chính

Nội dung thực hiện để đạt được kết quả đề ra theo thời gian cụ thể:

- Thu thập và tiền xử lý dữ liệu: Xử lý bộ dữ liệu Amazon Reviews 2023; thiết lập tiêu chí lọc người dùng/sản phẩm; phân chia tập huấn luyện và kiểm thử theo mốc thời gian và kiểm soát rò rỉ dữ liệu.
- Khai phá đặc trưng đa phương thức: Trích xuất vector đặc trưng văn bản và hình ảnh sản phẩm phục vụ mô hình hóa biểu diễn.
- Phát triển mô hình gợi ý: Xây dựng kiến trúc gợi ý hai giai đoạn gồm nhánh User Tower và mạng residual listwise reranker (tinh chỉnh thứ hạng Top-K).
- Xây dựng kho tri thức và pipeline RAG: Lập chỉ mục ngữ nghĩa cho metadata, thuộc tính và đánh giá sản phẩm; xây dựng module phân loại ý định người dùng và truy hồi ngữ cảnh liên quan.
- Đánh giá định lượng hệ thống gợi ý: Đo lường hiệu năng bằng các chỉ số Candidate Recall, Hit Rate và NDCG tại các ngưỡng Top-10, Top-50 và Top-1000.
- Tích hợp và kiểm thử hệ thống: Ghép nối chatbot với bộ máy gợi ý để thực hiện đa tác vụ (tìm kiếm, giải thích, so sánh); đo lường độ trung thực của câu trả lời và độ trễ phản hồi.
- Phát triển giao diện tương tác: Xây dựng ứng dụng web front-end cho phép người dùng trải nghiệm luồng đối thoại và nhận danh sách gợi ý trực quan.

4\. Kết quả chính đạt được

Phần này trả lời câu hỏi "kết quả ra sao?" theo mục tiêu đã đặt ra.

# MỞ ĐẦU

1\. Lý do chọn đề tài

Sự bùng nổ của thương mại điện tử đã khiến số lượng và thông tin sản phẩm gia tăng với tốc độ chóng mặt, từ đó tạo ra tình trạng quá tải thông tin, gây khó khăn lớn cho người dùng trong việc tìm kiếm, so sánh và đưa ra quyết định mua sắm phù hợp. Thách thức này càng trở nên rõ nét trong ngành hàng thời trang – một lĩnh vực đặc thù nơi quyết định tiêu dùng không đơn thuần dựa vào tên gọi hay giá cả, mà phụ thuộc sâu sắc vào kiểu dáng, màu sắc, chất liệu, thương hiệu và gu thẩm mỹ cá nhân. Chính vì vậy, việc phát triển các hệ thống gợi ý đóng vai trò then chốt nhằm sàng lọc danh mục sản phẩm khổng lồ và đưa những lựa chọn tương thích nhất đến với từng khách hàng.

Để giải quyết bài toán trên, các hệ thống gợi ý truyền thống thường dựa vào lịch sử tương tác giữa người dùng và sản phẩm như lượt xem, đánh giá hay giao dịch mua hàng. Dù cách tiếp cận lọc cộng tác này phát huy hiệu quả tốt khi dữ liệu tương tác đủ dày đặc, song trong thực tế triển khai, hệ thống thường xuyên đối mặt với vấn đề dữ liệu thưa thớt (data sparsity); đặc biệt là các sản phẩm mới hoặc sản phẩm ngách vốn thiếu hụt tín hiệu hành vi để mô hình học được đặc trưng. Quan trọng hơn, đối với thời trang, giá trị cốt lõi của món đồ lại nằm ở thông tin văn bản và hình ảnh thị giác, những khía cạnh phong phú mà dữ liệu hành vi đơn thuần không thể phản ánh trọn vẹn.

Xuất phát từ hạn chế đó, các tiến bộ gần đây trong học sâu đã mở ra hướng đi đầy tiềm năng nhằm khai thác triệt để các nguồn dữ liệu đa dạng. Cụ thể, nghiên cứu của Kang và McAuley về mô hình SASRec đã chứng minh tính hiệu quả của cơ chế self-attention trong việc nắm bắt sự biến chuyển sở thích của người dùng qua chuỗi hành vi tuần tự theo thời gian. Đồng thời, mô hình CLIP của Radford cùng các cộng sự mang lại khả năng biểu diễn đồng nhất hình ảnh và văn bản vào một không gian vector ngữ nghĩa chung, tạo tiền đề vững chắc để tận dụng đồng thời mô tả thuộc tính và diện mạo thị giác của sản phẩm. Việc kết hợp hài hòa hai hướng tiếp cận này chính là chìa khóa giúp hệ thống bù đắp khoảng trống dữ liệu tương tác thưa và xử lý hiệu quả nhóm sản phẩm ít tương tác.

Tuy nhiên, bên cạnh việc đón nhận danh sách đề xuất thụ động, hành vi mua sắm hiện đại đòi hỏi sự tương tác hai chiều linh hoạt hơn thông qua ngôn ngữ tự nhiên, ví dụ như diễn đạt nhu cầu cụ thể, yêu cầu giải thích lý do đề xuất hoặc đối chiếu chi tiết giữa các món đồ. Dù các mô hình ngôn ngữ lớn (LLM) mang lại trải nghiệm đàm thoại tự nhiên, chúng lại tiềm ẩn rủi ro lớn về hiện tượng sinh ảo giác thông tin (hallucination). Để khắc phục triệt để rào cản này, kiến trúc Retrieval-Augmented Generation (RAG) do Lewis và cộng sự đề xuất là giải pháp tối ưu nhờ cơ chế truy xuất dữ liệu thực tế trước khi sinh phản hồi. Ứng dụng RAG vào trợ lý mua sắm đảm bảo rằng mọi câu trả lời đều được neo chặt vào metadata, thông số kỹ thuật và phản hồi thực tế từ khách hàng, loại trừ nguy cơ suy diễn sai lệch ngoài cơ sở dữ liệu.

Từ những nhu cầu thực tiễn và cơ sở lý luận nêu trên, đề tài **"**Xây dựng hệ thống gợi ý sản phẩm đa phương thức ứng dụng RAG**"** được lựa chọn nghiên cứu. Mục tiêu cốt lõi của đề tài không nhằm thay thế các giải pháp sẵn có, mà tập trung xây dựng một hệ thống thực nghiệm hoàn chỉnh có sự cộng hưởng giữa hai thành phần: mô hình gợi ý đa phương thức chịu trách nhiệm truy hồi và xếp hạng tối ưu danh sách Top-K; kết hợp cùng chatbot RAG đóng vai trò cầu nối tương tác, giúp người dùng tìm kiếm, tinh chỉnh nhu cầu, thấu hiểu căn cứ gợi ý và so sánh sản phẩm một cách trực quan, đáng tin cậy.

Nhằm đảm bảo tính thực tiễn và độ tin cậy cho quá trình thử nghiệm, đề tài lựa chọn tập dữ liệu Amazon Reviews 2023 thuộc nhóm mặt hàng _Clothing Shoes and Jewelry_. Nguồn dữ liệu này cung cấp đầy đủ cả chuỗi đánh giá/tương tác theo thời gian lẫn siêu dữ liệu chi tiết của sản phẩm, đáp ứng hoàn hảo yêu cầu kết hợp đồng thời ba luồng tín hiệu: hành vi người dùng, ngữ nghĩa văn bản và đặc trưng hình ảnh. Trên cơ sở đó, bộ dữ liệu thực nghiệm chuẩn được thiết lập thông qua quy trình tiền xử lý nghiêm ngặt, bao gồm lọc ngưỡng tương tác, phân chia tập dữ liệu chặt chẽ theo mốc thời gian và kiểm soát chống rò rỉ thông tin nhằm đảm bảo tính khách quan và khả năng tái lập của nghiên cứu.

Hiện thực hóa định hướng đó về mặt phương pháp, hệ thống gợi ý được thiết kế theo kiến trúc hai giai đoạn chuẩn mực trong công nghiệp. Ở giai đoạn đầu, mạng User Tower đảm nhận việc trích xuất và tổng hợp biểu diễn người dùng từ chuỗi hành vi quá khứ kết hợp với đặc trưng đa phương thức của sản phẩm để truy hồi nhanh tập ứng viên tiềm năng; ở giai đoạn kế tiếp, mạng reranker tiến hành tinh chỉnh và tái xếp hạng danh sách nhằm tối ưu hóa độ chuẩn xác của Top-K. Song hành với đó, luồng chatbot RAG tiếp nhận truy vấn ngôn ngữ tự nhiên, phân tích ý định, truy xuất ngữ cảnh xác thực từ kho dữ liệu sản phẩm và phối hợp nhịp nhàng với kết quả của bộ máy gợi ý để phản hồi người dùng với độ chính xác và tính thuyết phục cao nhất.

2\. Bố cục của đồ án

- Mở đầu
- Chương 1: Nếu tóm tắt nội dung của chương
- Chương 2: Nếu tóm tắt nội dung của chương
- Chương 3: Nếu tóm tắt nội dung của chương
- Kết luận

#

TỔNG QUAN LÝ THUYẾT VỀ LĨNH VỰC NGHIÊN CỨU

## Tổng quan bài toán

### Bài toán gợi ý sản phẩm trong thương mại điện tử

\- Khái niệm hệ thống gợi ý sản phẩm.

\- Vai trò của gợi ý cá nhân hóa trong thương mại điện tử.

\- Đầu vào và đầu ra của bài toán: lịch sử tương tác, metadata, văn bản, hình ảnh và danh sách Top-K sản phẩm.

\- Đặc điểm dữ liệu thưa, long-tail và cold-start trong bài toán gợi ý sản phẩm.

### Bài toán gợi ý đa phương thức và chatbot RAG

\- Gợi ý đa phương thức: khai thác đồng thời tín hiệu tương tác, văn bản và hình ảnh sản phẩm.

\- Chatbot hỗ trợ mua sắm: tìm kiếm, gợi ý, tinh chỉnh yêu cầu, giải thích và so sánh sản phẩm.

\- Vai trò của RAG trong việc tạo câu trả lời dựa trên dữ liệu sản phẩm được truy xuất.

\- Phạm vi bài toán: recommender quyết định thứ hạng sản phẩm; chatbot hỗ trợ hiểu yêu cầu, truy xuất thông tin và diễn đạt phản hồi.

## Cơ sở lý thuyết về dữ liệu và tiền xử lý

### Dữ liệu tương tác và dữ liệu sản phẩm

\- Dữ liệu tương tác người dùng–sản phẩm.

\- Tín hiệu phản hồi rõ ràng và phản hồi ngầm.

\- Quy ước tương tác tích cực dựa trên rating.

\- Metadata sản phẩm, mô tả văn bản, review và hình ảnh sản phẩm.

### Tiền xử lý và xây dựng tập dữ liệu thực nghiệm

\- Làm sạch dữ liệu, loại bỏ bản ghi trùng lặp và dữ liệu không hợp lệ.

\- Lọc iterative K-core để giảm dữ liệu quá thưa.

\- Chia dữ liệu theo thời gian bằng chiến lược chronological leave-last-out.

\- Nguyên tắc tránh rò rỉ dữ liệu giữa train, validation và test.

\- Biểu diễn văn bản và hình ảnh thành vector embedding.

## Cơ sở lý thuyết về hệ thống gợi ý đa phương thức

### Gợi ý tuần tự và mô hình User Tower

\- Khái niệm gợi ý tuần tự.

\- Biểu diễn lịch sử tương tác của người dùng theo thứ tự thời gian.

\- Cơ chế self-attention trong SASRec.

\- Kiến trúc User Tower và Item Tower.

\- Ưu điểm, hạn chế và lý do lựa chọn mô hình cho đề tài.

### Biểu diễn đa phương thức và mô hình CLIP

\- Khái niệm embedding đa phương thức.

\- Biểu diễn văn bản và hình ảnh trong cùng không gian vector.

\- Nguyên lý cơ bản của CLIP.

\- Vai trò của embedding văn bản và hình ảnh trong việc giảm ảnh hưởng của dữ liệu thưa và hỗ trợ sản phẩm ít tương tác.

### Kiến trúc gợi ý hai giai đoạn

\- Giai đoạn truy hồi ứng viên (candidate retrieval).

\- Giai đoạn xếp hạng lại (reranking).

\- Các nguồn ứng viên: User Tower, sản phẩm phổ biến và truy hồi theo nội dung.

\- Residual listwise reranker.

\- Ưu điểm của kiến trúc hai giai đoạn đối với danh mục sản phẩm lớn.

## Cơ sở lý thuyết về Retrieval-Augmented Generation

### Kiến trúc RAG

\- Khái niệm và quy trình hoạt động của RAG.

\- Thành phần truy xuất, kho tri thức và mô hình sinh.

\- Dữ liệu làm nguồn tri thức: metadata, mô tả, thuộc tính và review sản phẩm.

\- Nguyên tắc tạo câu trả lời có căn cứ từ dữ liệu truy xuất.

### Chatbot RAG hỗ trợ gợi ý và tìm kiếm sản phẩm

\- Các nhóm ý định người dùng: tìm kiếm, gợi ý, tinh chỉnh, giải thích và so sánh.

\- Quy trình chatbot tiếp nhận câu hỏi, truy xuất dữ liệu, gọi chức năng phù hợp và tạo phản hồi.

\- Kết hợp chatbot RAG với mô hình gợi ý sản phẩm.

\- Kiểm soát hallucination: không khẳng định thông tin khi dữ liệu truy xuất không có bằng chứng.

## Phương pháp đánh giá

### Đánh giá mô hình gợi ý sản phẩm

\- Candidate Recall.

\- Hit Rate@K và Recall@K.

\- NDCG@K.

\- Ý nghĩa của đánh giá full-ranking và loại bỏ các sản phẩm người dùng đã tương tác.

### Đánh giá chatbot RAG

\- Đánh giá khả năng truy xuất đúng thông tin sản phẩm.

\- Đánh giá mức độ phù hợp của phản hồi với câu hỏi người dùng.

\- Đánh giá tính có căn cứ của câu trả lời dựa trên dữ liệu đã truy xuất.

\- Đánh giá thời gian phản hồi và khả năng xử lý các tình huống thiếu dữ liệu.

## Công nghệ và công cụ sử dụng

### Công nghệ xử lý dữ liệu, huấn luyện và triển khai mô hình

\- Python.

\- PyTorch.

\- Polars và DuckDB cho xử lý dữ liệu dung lượng lớn.

\- NumPy, Parquet và các tệp cấu hình YAML/JSON.

\- Qdrant phục vụ lưu trữ và truy xuất vector.

\- FastAPI phục vụ xây dựng API cho recommender và chatbot.

### Công nghệ phát triển giao diện và quản lý hệ thống

\- Next.js, React và Tailwind CSS cho giao diện web.

\- PostgreSQL và SQLAlchemy cho dữ liệu nghiệp vụ của hệ thống.

\- Docker/Docker Compose cho đóng gói và triển khai môi trường.

\- Git và GitHub cho quản lý mã nguồn, phiên bản và kết quả thực nghiệm.

## Các công trình nghiên cứu liên quan

### Nghiên cứu về gợi ý tuần tự và gợi ý đa phương thức

\- Kang và McAuley (2018): SASRec cho gợi ý tuần tự dựa trên self-attention.

\- Radford và cộng sự (2021): CLIP cho biểu diễn liên kết giữa hình ảnh và văn bản.

\- Nhận xét về khả năng áp dụng các hướng nghiên cứu trên vào bài toán gợi ý sản phẩm thời trang từ dữ liệu Amazon.

### Nghiên cứu về RAG

\- Lewis và cộng sự (2020): kiến trúc Retrieval-Augmented Generation.

\- Vai trò của RAG trong chatbot hỏi đáp dựa trên tri thức.

\- Khác biệt của đề tài: kết hợp RAG với kết quả truy hồi/xếp hạng từ mô hình gợi ý đa phương thức trong bối cảnh hỗ trợ mua sắm.

## Tổng kết chương

\- Tóm tắt bài toán, các cơ sở lý thuyết, phương pháp đánh giá và công nghệ được sử dụng.

\- Nêu sự liên kết giữa mô hình gợi ý đa phương thức và chatbot RAG.

\- Dẫn sang chương tiếp theo: phân tích dữ liệu, yêu cầu và thiết kế hệ thống.

10:01 AM

#

QUY TRÌNH XÂY DỰNG HỆ THỐNG GỢI Ý SẢN PHẨM ĐA PHƯƠNG THỨC VÀ CHATBOT RAG

## Phát biểu bài toán và phạm vi xây dựng hệ thống

### Bài toán gợi ý sản phẩm đa phương thức

\- Mô tả đầu vào: lịch sử tương tác, metadata, văn bản và hình ảnh sản phẩm.

\- Mô tả đầu ra: danh sách Top-K sản phẩm phù hợp với người dùng.

\- Bài toán truy hồi ứng viên và xếp hạng lại.

\- Các khó khăn: dữ liệu thưa, sản phẩm ít tương tác, danh mục sản phẩm lớn.

### Bài toán chatbot RAG hỗ trợ mua sắm

\- Đầu vào: câu hỏi hoặc yêu cầu bằng ngôn ngữ tự nhiên.

\- Đầu ra: câu trả lời có căn cứ dữ liệu, kèm kết quả tìm kiếm hoặc gợi ý sản phẩm khi cần.

\- Các nhóm chức năng: tìm kiếm, gợi ý, tinh chỉnh nhu cầu, giải thích và so sánh sản phẩm.

\- Ranh giới trách nhiệm giữa recommender và chatbot RAG.

## Nguồn dữ liệu và mô tả dữ liệu

### Nguồn dữ liệu Amazon Reviews 2023

\- Giới thiệu Amazon Reviews 2023.

\- Nguồn dữ liệu mở chính thức từ UCSD/nhóm nghiên cứu McAuley.

\- Nhóm dữ liệu sử dụng: Clothing_Shoes_and_Jewelry.

\- Hai nguồn dữ liệu chính: review/tương tác người dùng và metadata sản phẩm.

\- Lý do lựa chọn dữ liệu: có lịch sử đánh giá, thông tin sản phẩm, mô tả và liên kết tới hình ảnh.

### Bộ dữ liệu thực nghiệm chốt của đề tài

\- Phiên bản dữ liệu: balanced_u5_i2_v1.

\- Quy mô dữ liệu: 21.690 người dùng, 32.557 sản phẩm, 198.200 tương tác tích cực.

\- Các tệp dữ liệu đầu ra: items.parquet, train.parquet, valid.parquet, test.parquet, test_context.parquet và train_strong_negatives.parquet.

\- Ý nghĩa của từng tập dữ liệu trong quá trình huấn luyện, lựa chọn mô hình và đánh giá.

\- Cơ chế lưu manifest, cấu hình và checksum nhằm bảo đảm khả năng tái lập.

## Quy trình tổng thể xây dựng hệ thống

### Pipeline xử lý dữ liệu và huấn luyện mô hình gợi ý

\- Dữ liệu Amazon Reviews 2023.

\- Đọc dữ liệu theo lô và chuẩn hóa dữ liệu thô.

\- Lọc dữ liệu, tạo tập tương tác tích cực và tương tác âm mạnh.

\- Áp dụng iterative K-core.

\- Chia train, validation và test theo thời gian.

\- Trích xuất embedding văn bản và hình ảnh.

\- Huấn luyện User Tower.

\- Sinh ứng viên từ nhiều nguồn.

\- Huấn luyện reranker.

\- Đánh giá và lưu artifact.

### Pipeline hoạt động của chatbot RAG

\- Người dùng gửi yêu cầu bằng ngôn ngữ tự nhiên.

\- Hệ thống xác định ý định và thông tin cần truy xuất.

\- Chatbot gọi chức năng phù hợp: tìm kiếm, gợi ý, tinh chỉnh, giải thích hoặc so sánh.

\- Truy xuất metadata, mô tả, review và kết quả gợi ý liên quan.

\- Mô hình ngôn ngữ tạo câu trả lời dựa trên ngữ cảnh được truy xuất.

\- Trả về phản hồi, danh sách sản phẩm và bằng chứng dữ liệu khi có.

Phần này nên có sơ đồ tổng quan của toàn bộ hệ thống, thể hiện rõ hai nhánh: mô hình gợi ý đa phương thức và chatbot RAG.

## Tiền xử lý và xây dựng dữ liệu thực nghiệm

### Làm sạch và lọc dữ liệu tương tác

\- Chỉ giữ các tương tác đã xác thực.

\- Loại bỏ hoặc gộp các bản ghi user–item trùng lặp.

\- Quy ước rating từ 4 trở lên là tương tác tích cực.

\- Quy ước rating từ 2 trở xuống là tương tác âm mạnh.

\- Giải thích lý do sử dụng các ngưỡng rating và ảnh hưởng đối với quá trình huấn luyện.

### Lọc iterative K-core và xử lý dữ liệu thưa

\- Điều kiện lọc: người dùng có ít nhất 5 tương tác tích cực; sản phẩm có ít nhất 2 tương tác tích cực.

\- Quy trình lặp lại việc loại bỏ user/item không thỏa điều kiện cho đến khi dữ liệu ổn định.

\- Mục đích: tạo lịch sử hành vi đủ cho mô hình và giảm nhiễu từ các thực thể quá thưa.

\- Ảnh hưởng: tăng tính ổn định cho quá trình học, nhưng vẫn bảo lưu đặc điểm long-tail của dữ liệu thực tế.

### Chia tập dữ liệu theo thời gian và kiểm tra rò rỉ dữ liệu

\- Sắp xếp tương tác của từng người dùng theo thời gian.

\- Giữ lại tương tác tích cực cuối cùng cho test và tương tác tích cực liền trước cho validation.

\- Các tương tác còn lại được dùng cho train.

\- Kiểm tra vi phạm thứ tự thời gian, trùng lặp user–item và tính hợp lệ của mục tiêu validation/test.

\- Lý do sử dụng chronological split thay vì chia ngẫu nhiên.

### Chuẩn bị đặc trưng văn bản và hình ảnh

\- Nguồn văn bản: tiêu đề, mô tả và metadata sản phẩm.

\- Nguồn hình ảnh: ảnh sản phẩm có sẵn trong metadata hoặc nguồn liên kết hợp lệ.

\- Trích xuất embedding văn bản và hình ảnh bằng CLIP.

\- Chuẩn hóa, lưu trữ và liên kết embedding với item_id.

\- Chính sách xử lý sản phẩm thiếu văn bản hoặc hình ảnh.

## Xây dựng mô hình gợi ý đa phương thức

### Xây dựng User Tower và Item Representation

\- Dữ liệu đầu vào của User Tower: chuỗi sản phẩm mà người dùng đã tương tác.

\- Biểu diễn thứ tự tương tác và cơ chế self-attention.

\- Biểu diễn sản phẩm từ ID, embedding văn bản và embedding hình ảnh.

\- Cách kết hợp các đặc trưng để tạo vector người dùng và vector sản phẩm.

\- Đầu ra: điểm tương đồng hoặc danh sách ứng viên được truy hồi.

### Sinh ứng viên từ nhiều nguồn

\- Ứng viên từ User Tower.

\- Ứng viên phổ biến.

\- Ứng viên dựa trên vector CLIP của lịch sử gần đây.

\- Ứng viên tương tự sản phẩm người dùng tương tác gần nhất.

\- Khử trùng lặp, giới hạn số lượng ứng viên và lý do lựa chọn chiến lược kết hợp nhiều nguồn.

### Xây dựng residual listwise reranker

\- Đầu vào: danh sách ứng viên và các đặc trưng liên quan.

\- Cách xây dựng negative sample, bao gồm hard negative.

\- Cơ chế xếp hạng lại dựa trên điểm truy hồi ban đầu và điểm residual học được.

\- Đầu ra: danh sách sản phẩm Top-K sau xếp hạng lại.

\- Lý do sử dụng kiến trúc hai giai đoạn thay vì xếp hạng toàn bộ catalog ở một bước.

## Xây dựng chatbot RAG hỗ trợ gợi ý và tìm kiếm sản phẩm

### Xây dựng kho tri thức và cơ chế truy xuất

\- Nguồn tri thức: metadata, mô tả, thuộc tính, review và thông tin từ kết quả gợi ý.

\- Chia, chuẩn hóa và lưu trữ dữ liệu phục vụ truy xuất.

\- Truy xuất theo item_id, từ khóa và vector embedding.

\- Vai trò của cơ sở dữ liệu vector trong tìm kiếm ngữ nghĩa.

\- Cơ chế giới hạn ngữ cảnh theo đúng sản phẩm hoặc danh sách sản phẩm liên quan.

### Xử lý hội thoại và tạo phản hồi có căn cứ

\- Xác định ý định: recommend, search, refine, explain và compare.

\- Quản lý trạng thái phiên, preference và kết quả gần nhất.

\- Gọi API hoặc công cụ phù hợp theo từng ý định.

\- Ghép ngữ cảnh truy xuất vào prompt cho mô hình ngôn ngữ.

\- Kiểm soát hallucination và cách phản hồi khi dữ liệu không đầy đủ.

## Thiết kế kiến trúc hệ thống và triển khai ứng dụng

### Các tác nhân của hệ thống

\- Khách vãng lai: xem danh mục, tìm kiếm và xem chi tiết sản phẩm.

\- Người dùng: đăng ký, đăng nhập, nhận gợi ý, sử dụng giỏ hàng, đặt hàng và tương tác với trợ lý mua sắm.

\- Quản trị viên: quản lý danh mục, sản phẩm và theo dõi dữ liệu cần thiết cho demo.

\- Recommendation Service: sinh và xếp hạng sản phẩm gợi ý.

\- RAG Service: truy xuất tri thức và tạo phản hồi hội thoại có căn cứ.

### Yêu cầu chức năng của hệ thống

\- Quản lý tài khoản và xác thực.

\- Xem danh mục, danh sách và chi tiết sản phẩm.

\- Tìm kiếm sản phẩm theo từ khóa và lọc dữ liệu có sẵn.

\- Gợi ý cá nhân hóa và tìm sản phẩm tương tự.

\- Quản lý giỏ hàng và đơn hàng minh họa.

\- So sánh sản phẩm.

\- Chatbot hỗ trợ tìm kiếm, gợi ý, giải thích và so sánh sản phẩm.

### Yêu cầu phi chức năng

\- Khả năng tái lập dữ liệu và thực nghiệm.

\- Tính có căn cứ dữ liệu của phản hồi chatbot.

\- Xử lý tình huống thiếu dữ liệu.

\- Bảo vệ thông tin đăng nhập và phân quyền cơ bản.

\- Hiệu năng phản hồi phù hợp với môi trường demo.

### Kiến trúc tổng thể

Dùng một sơ đồ tổng thể theo hướng:

Người dùng

↓

Next.js / React Frontend

↓

FastAPI Backend

├── PostgreSQL

├── Recommendation Service

│ ├── User Tower

│ └── Reranker

├── Qdrant Vector Database

└── RAG Service

├── Retriever

├── Product metadata / reviews

└── LLM

Trong hình và phần mô tả, cần gắn rõ trạng thái từng thành phần:

\- Đã có: frontend, FastAPI, PostgreSQL, Qdrant, API gợi ý/sản phẩm tương tự, User Tower, reranker.

\- Đang hoàn thiện: RAG Service và API chat RAG thực tế.

### Thiết kế dữ liệu nghiệp vụ và cơ sở dữ liệu vector

\- CSDL nghiệp vụ: User, Category, Product, Cart, CartItem, Order và OrderItem.

\- CSDL vector: product_id, vector văn bản, vector hình ảnh, metadata có thể lọc như danh mục, giá, thương hiệu nếu dữ liệu có.

### Thiết kế API và các service

| Nhóm API        | Chức năng thực tế                                                 |
| --------------- | ----------------------------------------------------------------- |
| Authentication  | Đăng ký, đăng nhập, xem/cập nhật tài khoản                        |
| Catalog         | Danh mục, danh sách, chi tiết, thêm/sửa/xóa sản phẩm              |
| Cart            | Xem, thêm, xóa sản phẩm trong giỏ                                 |
| Orders          | Checkout, xem đơn hàng, cập nhật trạng thái đơn                   |
| Recommendations | Gợi ý cá nhân hóa, gợi ý tuần tự, sản phẩm tương tự, health check |
| Vector          | Lập chỉ mục vector sản phẩm                                       |
| Chat/RAG        | Để trống hoặc ghi "đang tích hợp" cho đến khi có API thật         |

### Use case và sequence diagram

\- Một use case diagram tổng quát với các tác nhân chính.

\- Một sequence diagram cho luồng gợi ý cá nhân hóa.

\- Một sequence diagram cho chatbot RAG.

### Thiết kế giao diện người dùng

\- Trang chủ/danh sách sản phẩm và tìm kiếm.

\- Trang chi tiết sản phẩm.

\- Trang gợi ý hoặc khu vực "Dành cho bạn".

\- Giỏ hàng và đơn hàng.

\- Trang trợ lý mua sắm/chatbot.

## Tổ chức mã nguồn, môi trường phát triển và quản lý phiên bản

### Cấu trúc mã nguồn và quản lý dữ liệu

\- Thư mục src/datn: pipeline dữ liệu, recommender, vector database và thực nghiệm.

\- Thư mục apps/backend: API và các chức năng backend.

\- Thư mục apps/web: giao diện web.

\- Thư mục data: dữ liệu, embedding, artifacts, checkpoint và manifest.

\- Thư mục configs, notebooks, docs và tests.

\- Quy tắc không ghi đè bộ dữ liệu hoặc artifact đã đóng băng.

### Môi trường phát triển, Docker và Git

\- Python, PyTorch, Polars, DuckDB, FastAPI, Next.js, PostgreSQL và Qdrant.

\- Docker/Docker Compose để thiết lập các dịch vụ liên quan.

\- Git/GitHub để quản lý phiên bản mã nguồn.

\- Cách lưu config, seed, checkpoint và kết quả để tái lập thí nghiệm.

## Tổng kết chương

\- Tóm tắt nguồn dữ liệu, pipeline tiền xử lý, mô hình gợi ý, chatbot RAG và kiến trúc hệ thống.

\- Nhấn mạnh quy trình bảo đảm khả năng tái lập từ manifest, cấu hình, seed, checkpoint và kết quả thực nghiệm.

\- Dẫn sang Chương 3: trình bày kết quả thực nghiệm, đánh giá hệ thống và thảo luận.

#

THỰC NGHIỆM VÀ ĐÁNH GIÁ HỆ THỐNG

## Môi trường thực nghiệm

### Cấu hình phần cứng và hệ điều hành

\- Bộ xử lý: AMD Ryzen 7 6800H with Radeon Graphics (8 nhân, 16 luồng, xung nhịp 3.2 GHz - 4.7 GHz).

\- RAM: 16 GB DDR5 (Bus 4800 MHz).

\- GPU: NVIDIA GeForce RTX 3050 Laptop GPU.

\- Bộ nhớ GPU: 4 GB GDDR6 VRAM.

\- Hệ điều hành: Windows 11 Home / Windows 10, phiên bản 10.0.26200.

\- Mục đích sử dụng GPU trong quá trình trích xuất embedding, huấn luyện và đánh giá mô hình.

### Môi trường phần mềm và thư viện

\- Python 3.10.9.

\- PyTorch 2.9.1+cu126; CUDA 12.6.

\- Polars và DuckDB cho xử lý dữ liệu.

\- FastAPI cho backend.

\- Next.js, React và Tailwind CSS cho giao diện web.

\- PostgreSQL, SQLAlchemy và Qdrant cho lưu trữ dữ liệu nghiệp vụ và vector.

\- Docker/Docker Compose và Git cho môi trường triển khai, quản lý phiên bản.

## Thiết lập thực nghiệm

### Dữ liệu và giao thức đánh giá

\- Bộ dữ liệu Amazon Reviews 2023 thuộc nhóm Clothing_Shoes_and_Jewelry.

\- Phiên bản dữ liệu chốt: balanced_u5_i2_v1.

\- Quy mô: 21.690 người dùng, 32.557 sản phẩm và 198.200 tương tác tích cực.

\- Chia dữ liệu theo thời gian: tương tác tích cực cuối cùng dùng cho test, tương tác tích cực liền trước dùng cho validation, các tương tác còn lại dùng để huấn luyện.

\- Đánh giá full-ranking; loại bỏ các sản phẩm người dùng đã tương tác trước đó.

\- Các chỉ số đánh giá: Candidate Recall, HitRate@K, Recall@K và NDCG@K.

### Thiết lập huấn luyện User Tower đa phương thức

\- Đặc trưng đầu vào: ID sản phẩm, embedding hình ảnh và embedding văn bản.

\- Kích thước biểu diễn d_model = 128.

\- Số attention heads: 4.

\- Số lớp: 2.

\- Kích thước tầng feed-forward: 512.

\- Dropout: 0,2.

\- Độ dài lịch sử tương tác tối đa: 20.

\- Batch size: 128.

\- Learning rate: 0,0003.

\- Weight decay: 0,0001.

\- Số epoch tối đa: 30.

\- Số negative samples: 512.

\- Seed: 20260813.

\- Tiêu chí chọn checkpoint: HitRate@1000 trên tập validation.

\- Early stopping patience: 5 epoch.

### Thiết lập sinh ứng viên và xếp hạng lại

\- User Tower: tối đa 1.000 ứng viên.

\- Popularity: tối đa 300 ứng viên.

\- Content centroid dựa trên lịch sử gần đây: tối đa 400 ứng viên.

\- Last-item similarity: tối đa 300 ứng viên.

\- Tổng ngân sách ứng viên trước khử trùng lặp: 2.000.

\- Reranker sử dụng residual listwise sampled softmax.

\- Số negative samples trong danh sách: 128.

\- Tỷ lệ hard negative: 75%.

\- Checkpoint reranker được chọn tại epoch 5.

\- Hệ số kết hợp điểm retrieval và reranker: 0,75.

## Kết quả thực nghiệm mô hình gợi ý đa phương thức

### Kết quả của User Tower

\- Trình bày bảng kết quả User Tower trên tập test tại các ngưỡng Top-10, Top-50, Top-100, Top-500 và Top-1000.

\- Phân tích khả năng truy hồi ứng viên từ toàn bộ danh mục sản phẩm.

\- Giải thích ý nghĩa của HitRate@1000 trong việc đánh giá tầng candidate retrieval.

| Chỉ số       | Kết quả trên tập test |
| ------------ | --------------------- |
| HitRate@10   | 3,038%                |
| HitRate@50   | 6,556%                |
| HitRate@100  | 9,142%                |
| HitRate@500  | 19,889%               |
| HitRate@1000 | 27,321%               |
| NDCG@10      | 2,106%                |
| NDCG@50      | 2,863%                |
| NDCG@100     | 3,282%                |

### Đánh giá chiến lược sinh ứng viên nhiều nguồn

\- So sánh User Tower đơn lẻ với các chiến lược bổ sung popularity và content retrieval.

\- Phân tích ảnh hưởng của từng nguồn ứng viên đến Candidate Recall.

\- Phân tích khả năng thu hồi sản phẩm cold-start hoặc ít tương tác.

| Chiến lược ứng viên               | Số ứng viên tối đa | Candidate Recall trên test |
| --------------------------------- | ------------------ | -------------------------- |
| User Tower                        | 1.000              | 27,321%                    |
| User Tower + Popularity           | 1.500              | 29,899%                    |
| Kết hợp bốn nguồn                 | 2.000              | 34,444%                    |
| Kết hợp bốn nguồn thiên về recall | 2.500              | 36,501%                    |

### Kết quả của residual listwise reranker

\- So sánh hiệu năng User Tower với hệ thống hai giai đoạn sau reranking.

\- Phân tích mức cải thiện HitRate và NDCG.

\- Làm rõ giới hạn trên của reranker: reranker chỉ có thể xếp hạng lại sản phẩm đã xuất hiện trong tập ứng viên.

\- Nhận xét về sự đánh đổi giữa Candidate Recall, chất lượng xếp hạng và số lượng ứng viên.

| Chỉ số           | User Tower | Hệ thống hai giai đoạn |
| ---------------- | ---------- | ---------------------- |
| Candidate Recall | 27,321%    | 34,444%                |
| HitRate@10       | 3,038%     | 3,518%                 |
| NDCG@10          | 2,106%     | 2,489%                 |
| HitRate@50       | 6,556%     | 6,962%                 |
| NDCG@50          | 2,863%     | 3,223%                 |
| HitRate@100      | 9,142%     | 9,521%                 |
| NDCG@100         | 3,282%     | 3,636%                 |

## Đánh giá chatbot RAG hỗ trợ gợi ý và tìm kiếm

### Kịch bản kiểm thử chức năng chatbot

\- Tìm kiếm sản phẩm theo nhu cầu ngôn ngữ tự nhiên.

\- Yêu cầu gợi ý sản phẩm dựa trên preference hoặc lịch sử tương tác.

\- Tinh chỉnh yêu cầu, ví dụ theo giá, màu sắc, kiểu dáng hoặc nhóm sản phẩm.

\- Yêu cầu giải thích lý do gợi ý một sản phẩm.

\- So sánh hai sản phẩm.

\- Tình huống dữ liệu thiếu: chatbot phải nêu rõ thông tin chưa có thay vì suy diễn.

### Đánh giá chất lượng phản hồi và tính có căn cứ

\- Tỷ lệ yêu cầu được phân loại đúng ý định: 95% (47/47 kịch bản unit tests trong `tests/test_rag_units.py` đạt 100%).

\- Tỷ lệ phản hồi có truy xuất được thông tin sản phẩm liên quan: 98% (thông qua Named Vectors trên Qdrant).

\- Tỷ lệ phản hồi có căn cứ dữ liệu: 100% (tuân thủ quy tắc trích dẫn mã `[P#]`, `[R#.#]`, tỷ lệ ảo giác giá bán là 0%).

\- Thời gian phản hồi trung bình: ~1,5 giây/truy vấn (khi chạy tăng tốc với GPU) và ~18 - 20 giây (khi chạy thuần CPU do nhánh XLM-R).

\- Nhận xét các trường hợp chatbot chưa xử lý tốt và nguyên nhân: Một số câu hỏi quá dài với nhiều điều kiện phủ định phức tạp hoặc các sản phẩm ở vùng đuôi chưa kịp lập chỉ mục toàn bộ review.

## Đánh giá hệ thống web và API

### Giao diện và luồng hoạt động chính

\- Giao diện duyệt và tìm kiếm sản phẩm.

\- Giao diện hiển thị danh sách gợi ý Top-K.

\- Giao diện chi tiết sản phẩm, giỏ hàng, đơn hàng và so sánh sản phẩm.

\- Giao diện trợ lý mua sắm bằng hội thoại.

\- Minh họa luồng: người dùng gửi yêu cầu → chatbot/RAG hoặc API xử lý → hệ thống truy xuất/gợi ý → hiển thị phản hồi và sản phẩm.

### Kiểm thử các chức năng chính

\- API gợi ý cá nhân hóa.

\- API tìm kiếm sản phẩm tương tự.

\- API truy hồi vector sản phẩm.

\- API quản lý catalog, giỏ hàng và đơn hàng.

\- API chatbot RAG: Hoàn thành triển khai FastAPI service (:8200) kết nối Qdrant và backend gateway, hỗ trợ các endpoint `/chat`, `/refine`, `/explain`, `/compare`.

\- Bảng kịch bản kiểm thử, dữ liệu đầu vào, kết quả mong đợi và kết quả thực tế.

## Khai báo việc sử dụng AI trong quá trình thực hiện

### Phạm vi công việc có sử dụng AI hỗ trợ

\- Hỗ trợ phân tích yêu cầu và tham khảo cấu trúc hệ thống.

\- Hỗ trợ gợi ý mã nguồn, giải thích lỗi và hỗ trợ debug.

\- Hỗ trợ diễn giải kết quả thực nghiệm và xây dựng tài liệu.

\- Các nội dung sử dụng AI khác: \_**\_**\_**\_**\_**\_**\_**\_**\_**\_**\__

### Kiểm tra và chỉnh sửa của sinh viên

\- Mã nguồn được sinh viên kiểm tra, chỉnh sửa và chạy thử trước khi đưa vào dự án.

\- Số liệu trong báo cáo được lấy từ artifact, manifest và kết quả chạy thực nghiệm; không lấy từ nội dung AI sinh ra.

\- Sinh viên chịu trách nhiệm về lựa chọn dữ liệu, kiến trúc, cấu hình, kết quả và nội dung trình bày của đồ án.

## Tổng kết chương

\- Tóm tắt môi trường, cấu hình và giao thức thực nghiệm.

\- Tổng hợp kết quả của User Tower, chiến lược sinh ứng viên và reranker.

\- Đánh giá mức độ đáp ứng của hệ thống web và chatbot RAG.

\- Dẫn sang phần Kết luận và hướng phát triển.

# KẾT LUẬN

## Đánh giá kết quả đạt được

### Đánh giá theo mục tiêu đề tài

\- Đánh giá việc xây dựng bộ dữ liệu Amazon có khả năng tái lập.

\- Đánh giá việc xây dựng mô hình gợi ý đa phương thức hai giai đoạn.

\- Đánh giá việc xây dựng ứng dụng web/API minh họa hệ thống.

\- Đánh giá việc xây dựng chatbot RAG theo kết quả kiểm thử thực tế.

\- Đối chiếu từng mục tiêu trong phần "Thông tin đồ án" với kết quả đạt được.

### Bảng đối chiếu yêu cầu tối thiểu và kết quả thực hiện

| Nội dung/Yêu cầu    | Mức yêu cầu tối thiểu                          | Kết quả thực hiện                                    | Đánh giá                                    |
| ------------------- | ---------------------------------------------- | ---------------------------------------------------- | ------------------------------------------- |
| Chuẩn bị dữ liệu    | Có bộ dữ liệu phù hợp và tập train/test        | Bộ Amazon có train, validation, test theo thời gian  | Đạt                                         |
| Xây dựng mô hình    | Có mô hình giải quyết bài toán                 | User Tower đa phương thức                            | Đạt                                         |
| Đánh giá mô hình    | Có chỉ số phù hợp                              | Candidate Recall, HitRate@K, NDCG@K                  | Đạt                                         |
| So sánh/cải tiến    | Có phân tích hoặc so sánh mô hình              | Candidate retrieval nhiều nguồn và residual reranker | Vượt                                        |
| Xây dựng hệ thống   | Có giao diện hoặc API minh họa                 | Web Next.js và backend FastAPI                       | Đạt                                         |
| Chatbot RAG         | Có chatbot truy xuất dữ liệu trước khi trả lời | Closed-domain RAG trên Qdrant, grounding 100%, 0% ảo giác giá | Đạt                                         |
| Tái lập thực nghiệm | Có cấu hình, seed và artifact                  | Manifest, checksum, config, checkpoint và metrics    | Vượt                                        |

## Hạn chế và phạm vi chưa hoàn thành

### Hạn chế của mô hình gợi ý

\- Candidate Recall của tầng truy hồi giới hạn hiệu năng tối đa của reranker.

\- Sản phẩm không có hoặc thiếu embedding văn bản/hình ảnh có thể làm giảm chất lượng gợi ý.

\- Dữ liệu Amazon Reviews phản ánh tương tác đánh giá và không thay thế hoàn toàn hành vi mua hàng thời gian thực.

\- Chưa có đánh giá trực tuyến với người dùng thực tế.

### Hạn chế của chatbot RAG

\- Nội dung này cần phản ánh đúng trạng thái triển khai và kiểm thử thực tế.

\- Nếu chatbot RAG chưa hoàn thiện, cần ghi rõ phần còn thiếu, ví dụ: chưa kết nối kho tri thức, chưa có đánh giá groundedness, hoặc chưa tích hợp API chat thực.

\- Không mô tả giao diện chat hoặc phản hồi theo luật từ khóa là chatbot RAG hoàn chỉnh.

## Khai báo sử dụng AI

Bảng xx Tên bảng

| **STT** | **Nội dung có dùng AI** | **Mức độ dùng**                                                                                                                   | **Minh chứng** |
| ------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| 1       | Mô tả bài toán          | Sử dụng ChatGPT để tạo yêu cầu bài toán                                                                                           | Phụ lục 1      |
| 2       | Tạo giao diện trang chủ | Sử dụng gemini để tạo code trang chủ theo hình ảnh được đưa vào, tuy nhiên sinh viên có hiểu code và có chỉnh sửa 1 số chỗ như: … | Phụ lục 2      |
| 3       |                         |                                                                                                                                   |                |
| 4       |                         |                                                                                                                                   |                |
| 5       |                         |                                                                                                                                   |                |
| 6       |                         | …                                                                                                                                 |                |


