# CHƯƠNG 1: TỔNG QUAN LÝ THUYẾT VỀ LĨNH VỰC NGHIÊN CỨU

> **Phạm vi chương:** H&M là bộ dữ liệu chính. Chương này trình bày cơ sở lý thuyết; cấu hình, kết quả thực nghiệm và chi tiết triển khai được dành cho các chương sau.

---

## 1.1. Tổng quan bài toán

### 1.1.1. Bài toán gợi ý sản phẩm trong thương mại điện tử
Hệ thống gợi ý sản phẩm là thành phần hỗ trợ người dùng khám phá những mặt hàng có khả năng phù hợp với nhu cầu của họ trong một danh mục lớn. Thay vì trình bày cùng một danh sách cho mọi khách hàng, hệ thống ước lượng mức độ phù hợp giữa từng người dùng và sản phẩm, sau đó sắp xếp các sản phẩm theo điểm dự đoán. Đây là cách nhìn phổ biến của bài toán gợi ý: khai thác thông tin về người dùng, sản phẩm và các tương tác đã quan sát để dự đoán những lựa chọn có ích tiếp theo ([Adomavicius và Tuzhilin, 2005](https://ids.csom.umn.edu/faculty/gedas/papers/recommender-systems-survey-2005.pdf)).

Trong thương mại điện tử, cá nhân hóa giúp giảm công sức duyệt danh mục và đưa các sản phẩm có liên quan đến sở thích của từng khách hàng lên những vị trí dễ nhìn thấy hơn. Chẳng hạn, hai khách hàng cùng tìm một loại áo có thể nhận các gợi ý khác nhau dựa trên lịch sử mua sắm, kiểu dáng hoặc màu sắc họ quan tâm. Hệ thống gợi ý cũng tạo cơ hội để người dùng khám phá những sản phẩm phù hợp mà họ chưa chủ động tìm kiếm; một ứng dụng thực tế của hướng tiếp cận này được mô tả trong hệ thống gợi ý của Amazon ([Linden và cộng sự, 2003](https://ieeexplore.ieee.org/document/1167344)). Những tác động kinh doanh như tăng tỷ lệ mua hàng hoặc giá trị đơn hàng cần được kiểm chứng bằng dữ liệu vận hành, vì vậy không được xem là kết quả mặc định của mọi hệ thống gợi ý.

Có thể mô hình hóa bài toán tại thời điểm $t$ như sau. Gọi $\mathcal{U}$ là tập người dùng, $\mathcal{I}_t$ là tập sản phẩm có thể được đề xuất, và $H_u^{<t}=\{(i_j,a_j,t_j)\mid t_j<t\}$ là lịch sử của người dùng $u$, trong đó $i_j$ là sản phẩm, $a_j$ là loại tương tác (ví dụ: xem, thêm vào giỏ hoặc mua) và $t_j$ là thời điểm xảy ra. Mỗi sản phẩm $i$ có thể đi kèm siêu dữ liệu $m_i$ như danh mục, màu sắc, giá, cùng văn bản mô tả $x_i^{\mathrm{text}}$ và hình ảnh $x_i^{\mathrm{image}}$. Từ các đầu vào đó, mô hình tính điểm phù hợp $s(u,i,t)$ và trả về danh sách có thứ tự $R_K(u,t)=[i_{(1)},i_{(2)},\ldots,i_{(K)}]$ gồm $K$ sản phẩm có điểm cao trong $\mathcal{I}_t$, sau khi áp dụng các điều kiện hợp lệ của hệ thống. Việc có loại sản phẩm từng mua khỏi danh sách hay không phụ thuộc vào mục tiêu sử dụng; với những mặt hàng có thể mua lặp lại, lịch sử mua không đồng nghĩa với việc sản phẩm phải bị loại.

Ba đặc điểm dữ liệu khiến việc xây dựng danh sách này trở nên khó khăn. **Thứ nhất, dữ liệu tương tác thưa:** mỗi khách hàng thường chỉ tương tác với một phần nhỏ danh mục, nên phần lớn cặp người dùng–sản phẩm không có bản ghi. Nếu $\mathcal{E}$ là tập cặp đã quan sát, mật độ của ma trận tương tác là $|\mathcal{E}|/(|\mathcal{U}|\,|\mathcal{I}|)$; giá trị thấp làm giảm tín hiệu để học sở thích và độ tương đồng. Đặc biệt, đối với phản hồi ngầm như lượt xem hoặc giao dịch mua, việc không quan sát thấy tương tác không đủ để kết luận người dùng không thích sản phẩm ([Hu và cộng sự, 2008](https://yifanhu.net/PUB/cf.pdf)).

**Thứ hai, phân phối đuôi dài:** một số sản phẩm phổ biến nhận nhiều tương tác, trong khi nhiều sản phẩm khác chỉ có ít dữ liệu. Mô hình phụ thuộc mạnh vào lịch sử tương tác vì thế có thể tiếp tục ưu tiên nhóm phổ biến và ít đưa sản phẩm ở phần đuôi vào danh sách gợi ý. Hiện tượng này được gọi là thiên lệch phổ biến và có thể làm giảm mức độ bao phủ của danh mục ([Abdollahpouri và cộng sự, 2019](https://cdn.aaai.org/ocs/18199/18199-78818-1-PB.pdf)).

**Thứ ba, khởi đầu lạnh:** người dùng mới có ít hoặc chưa có lịch sử để suy ra sở thích; sản phẩm mới có ít hoặc chưa có tương tác để ước lượng mức độ phù hợp từ hành vi cộng tác. Đây là hai trường hợp khác nhau và cần được phân biệt khi thiết kế, đánh giá hệ thống. Với sản phẩm mới, siêu dữ liệu, văn bản và hình ảnh có thể cung cấp tín hiệu nội dung ban đầu, nhưng hiệu quả gợi ý vẫn phải được kiểm chứng trên nhóm sản phẩm đó ([Adomavicius và Tuzhilin, 2005](https://ids.csom.umn.edu/faculty/gedas/papers/recommender-systems-survey-2005.pdf)). Các thách thức trên là cơ sở để xem xét việc kết hợp tín hiệu hành vi với thông tin đa phương thức ở những mục tiếp theo.

### 1.1.2. Bài toán gợi ý đa phương thức và Chatbot RAG
Mục 1.1.1 cho thấy lịch sử tương tác chỉ phản ánh một phần sở thích của người dùng, nhất là khi sản phẩm có ít giao dịch. Trong gợi ý đa phương thức (*multimodal recommendation*), mô hình khai thác thêm nội dung sản phẩm bên cạnh tín hiệu hành vi. Lịch sử mua cho biết người dùng đã chọn những sản phẩm nào; văn bản như tên, loại hàng và mô tả cung cấp thông tin về thuộc tính; hình ảnh bổ sung tín hiệu về màu sắc, kiểu dáng và vẻ ngoài. Các nguồn thông tin này có thể được mã hóa thành đặc trưng để hỗ trợ ước lượng điểm phù hợp giữa người dùng và sản phẩm. Nghiên cứu về gợi ý có nhận biết hình ảnh đã cho thấy đặc trưng thị giác có thể bổ sung cho tín hiệu phản hồi ngầm trong bài toán xếp hạng cá nhân hóa ([He và McAuley, 2016](https://ojs.aaai.org/index.php/AAAI/article/view/9973)). Tuy nhiên, việc kết hợp nhiều nguồn dữ liệu không mặc nhiên cải thiện kết quả; đóng góp của từng nguồn cần được xác định bằng thực nghiệm trên cùng một giao thức đánh giá.

Bên cạnh danh sách gợi ý, người mua có thể diễn đạt nhu cầu dưới dạng hội thoại và điều chỉnh nhu cầu đó sau khi xem kết quả. Chatbot hỗ trợ các tác vụ tìm kiếm, gợi ý theo sở thích, tinh chỉnh điều kiện, giải thích lý do gợi ý và so sánh sản phẩm. Ví dụ, sau khi yêu cầu “tìm áo sơ mi màu xanh”, người dùng có thể nói “rẻ hơn” hoặc “tôi không thích sản phẩm thứ hai”. Những lượt trao đổi này cung cấp thông tin về nhu cầu hiện tại và phản hồi trực tiếp mà lịch sử mua sắm chưa thể hiện. Khả năng thu nhận và cập nhật sở thích qua nhiều lượt là một đặc điểm quan trọng của hệ gợi ý hội thoại ([Gao và cộng sự, 2021](https://arxiv.org/abs/2101.09459)).

Trong hệ gợi ý hội thoại, phản hồi qua các lượt trao đổi có thể được tổ chức thành **bộ nhớ sở thích** của người dùng. Bộ nhớ này bổ sung cho lịch sử tương tác: thông tin về những sản phẩm được quan tâm có thể định hướng truy hồi ứng viên, còn các sở thích hoặc điều kiện mới giúp điều chỉnh thứ hạng kết quả. Nhờ đó, hệ thống có thể thích ứng với nhu cầu vừa được người dùng diễn đạt, thay vì chỉ dựa vào các giao dịch trong quá khứ. Đề tài xem việc đưa bộ nhớ sở thích vào cả truy hồi và xếp hạng là một phần của bài toán gợi ý, bên cạnh việc tạo câu trả lời trong hội thoại.

RAG (*Retrieval-Augmented Generation*) kết hợp truy xuất thông tin liên quan với quá trình sinh ngôn ngữ ([Lewis và cộng sự, 2020](https://papers.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf)). Trong bối cảnh dữ liệu thời trang của đề tài, thông tin được truy xuất cho câu trả lời chủ yếu là tên, thuộc tính và mô tả sản phẩm. Các nguồn này tạo ngữ cảnh để chatbot giải thích hoặc so sánh những đặc điểm có thể kiểm chứng; chúng không cung cấp ý kiến của người mua sau khi sử dụng sản phẩm. Về phạm vi trách nhiệm, mô hình gợi ý tính mức phù hợp và xếp hạng sản phẩm, còn lớp hội thoại hiểu yêu cầu, duy trì sở thích, chuyển chúng thành tín hiệu cho truy hồi–xếp hạng và diễn đạt kết quả dựa trên thông tin truy xuất.

---

## 1.2. Cơ sở lý thuyết về dữ liệu và tiền xử lý

### 1.2.1. Dữ liệu tương tác và dữ liệu sản phẩm
Dữ liệu đầu vào của bài toán gợi ý có thể chia thành **lịch sử hành vi của người dùng** và **thông tin về sản phẩm**. Với bộ dữ liệu [H&M Personalized Fashion Recommendations](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/overview) được sử dụng trong đề tài, lịch sử hành vi gồm các giao dịch mua gắn với mã khách hàng, mã sản phẩm và thời điểm. Một giao dịch có thể biểu diễn là $(u,i,t)$, trong đó $u$ là khách hàng, $i$ là sản phẩm được mua và $t$ là thời điểm mua. Trật tự thời gian cho phép mô hình sử dụng những giao dịch đã xảy ra để dự đoán nhu cầu ở thời điểm tiếp theo. Một khách hàng cũng có thể mua lại cùng sản phẩm, vì vậy các lần mua cần được xem là những sự kiện theo thời gian thay vì mặc nhiên gộp thành một cặp người dùng–sản phẩm duy nhất.

Giao dịch mua là một dạng **phản hồi ngầm** (*implicit feedback*): nó cho thấy khách hàng đã chọn sản phẩm, nhưng không trực tiếp cho biết mức độ hài lòng hay lý do mua. Dữ liệu H&M không có điểm đánh giá sao hoặc phản hồi “không thích” gắn với các giao dịch. Do đó, các giao dịch đã quan sát có thể được dùng làm tín hiệu mục tiêu cho bài toán dự đoán mua, còn việc không thấy một khách hàng mua sản phẩm nào đó không chứng minh họ không thích sản phẩm ấy. Phân biệt giữa “không quan sát thấy” và “phản hồi tiêu cực” là yêu cầu quan trọng khi học từ dữ liệu ngầm ([Hu và cộng sự, 2008](https://yifanhu.net/PUB/cf.pdf)). Các phản hồi được người dùng cung cấp trực tiếp trong hội thoại, nếu có, là một nguồn sở thích bổ sung và cần được hiểu khác với lịch sử giao dịch của bộ dữ liệu.

Thông tin sản phẩm trong H&M gồm các **thuộc tính có cấu trúc** như loại sản phẩm, nhóm hàng, màu sắc và bộ phận kinh doanh; **văn bản** như tên và mô tả chi tiết; cùng **hình ảnh** sản phẩm. Thuộc tính và văn bản giúp nhận biết đặc điểm được mô tả bằng ngôn ngữ, còn hình ảnh thể hiện những yếu tố trực quan như kiểu dáng và màu sắc. Khi một sản phẩm có ít giao dịch, các nguồn nội dung này vẫn cung cấp thông tin để xây dựng biểu diễn sản phẩm, dù mức độ hữu ích của từng nguồn cần được đánh giá thực nghiệm ([He và McAuley, 2016](https://ojs.aaai.org/index.php/AAAI/article/view/9973)). Bộ dữ liệu H&M không cung cấp review của khách hàng; vì vậy, phân tích sở thích và tạo nội dung giải thích dựa trên nguồn này phải xuất phát từ lịch sử mua và thông tin sản phẩm sẵn có, không suy diễn ra nhận xét sau mua.

### 1.2.2. Tiền xử lý và xây dựng tập dữ liệu thực nghiệm
Tiền xử lý nhằm chuyển dữ liệu giao dịch và dữ liệu sản phẩm thành những quan sát nhất quán, có thể dùng để học và đánh giá hệ gợi ý. Trước hết cần kiểm tra định danh người dùng, định danh sản phẩm, thời điểm giao dịch và quan hệ giữa bản ghi giao dịch với danh mục sản phẩm. Những bản ghi sai định dạng, thiếu trường thiết yếu hoặc không thể liên kết với sản phẩm cần được xử lý theo quy tắc công bố trước. Đối với ảnh hay mô tả bị thiếu, hệ thống cần ghi nhận sự thiếu vắng thông tin để không nhầm một sản phẩm thiếu dữ liệu với một sản phẩm thực sự có nội dung rỗng.

**Khử trùng lặp** phải dựa trên ý nghĩa của bản ghi. Hai dòng ghi lại cùng một giao dịch do lỗi nhập liệu có thể được xem là trùng kỹ thuật; ngược lại, hai lần mua cùng sản phẩm ở những thời điểm khác nhau là hai sự kiện hành vi có giá trị. Việc gộp mọi cặp người dùng–sản phẩm thành một dòng sẽ làm mất tín hiệu mua lặp lại và thay đổi lịch sử theo thời gian. Tương tự, lọc bỏ toàn bộ người dùng hoặc sản phẩm ít giao dịch có thể làm tập dữ liệu dễ học hơn nhưng cũng thay đổi phân phối ban đầu, đặc biệt làm giảm khả năng quan sát các trường hợp dữ liệu thưa và sản phẩm mới. Vì vậy, tiêu chí giữ hoặc loại bản ghi cần gắn với mục tiêu đánh giá và được báo cáo rõ ([Cañamares và cộng sự, 2020](https://doi.org/10.1007/s10791-020-09371-3)).

Khi dữ liệu lớn cần **chọn mẫu**, quyết định chọn người dùng hoặc sản phẩm chỉ nên dựa trên thông tin đã tồn tại trước thời điểm dự đoán. Nếu dùng chính hành vi trong giai đoạn kiểm thử để chọn mẫu, tập đánh giá sẽ bị chi phối bởi thông tin tương lai. Mẫu cũng cần được mô tả bằng các đặc điểm như mức độ hoạt động của khách hàng, tần suất giao dịch và số sản phẩm có tương tác, bởi việc chỉ giữ khách hàng có lịch sử dài sẽ làm kết quả không còn đại diện cho người dùng mới. Đây là vấn đề về phạm vi suy luận của thí nghiệm, không chỉ là bước giảm kích thước dữ liệu.

Với bài toán dự đoán giao dịch tương lai, cách chia dữ liệu phù hợp là dùng **các mốc thời gian chung** để tạo những giai đoạn liên tiếp: dữ liệu quá khứ cho huấn luyện, giai đoạn tiếp theo cho lựa chọn mô hình hoặc tham số, giai đoạn sau đó cho thẩm định và giai đoạn cuối cho kiểm thử. Tại mỗi mốc dự đoán, lịch sử người dùng và các thống kê sản phẩm như độ phổ biến chỉ được tính từ sự kiện trước mốc đó; giao dịch trong giai đoạn đích không được dùng để xây dựng đầu vào cho chính giai đoạn này. Khi một giai đoạn đã kết thúc, giao dịch của nó có thể trở thành lịch sử cho giai đoạn kế tiếp theo giao thức đã định. Cách chia theo từng người dùng nhưng không tôn trọng trục thời gian chung có thể đưa tương tác xảy ra trong tương lai của người này vào quá trình học trước khi dự đoán cho người khác, tạo ra rò rỉ dữ liệu ([Ji và cộng sự, 2020](https://arxiv.org/abs/2010.11060)). Tập kiểm thử cần được giữ riêng để báo cáo kết quả cuối cùng sau khi các lựa chọn phương pháp đã được chốt, nhờ đó phép đánh giá phản ánh rõ hơn tình huống dùng quá khứ để dự đoán tương lai.

---

## 1.3. Cơ sở lý thuyết về gợi ý đa phương thức có xét đến thời gian

### 1.3.1. Mô hình hóa sở thích từ lịch sử tương tác

Sở thích của người dùng vừa có thành phần tương đối ổn định, vừa thay đổi theo nhu cầu gần đây. Vì vậy, lịch sử mua sắm có thể được xem là một chuỗi có thứ tự $H_u^{<t}=((i_1,t_1),\ldots,(i_n,t_n))$, với $t_j<t$ là thời điểm dự đoán. Một mô hình gợi ý tạo biểu diễn người dùng $q_u(t)=f(H_u^{<t})$ để tóm tắt thông tin cần thiết cho việc so sánh với các sản phẩm. Hàm $f$ có thể là phép tổng hợp có trọng số, mạng hồi quy hoặc cơ chế tự chú ý. Chẳng hạn, tổng hợp có trọng số gán mức ảnh hưởng khác nhau cho các giao dịch trong lịch sử: $q_u(t)=\big(\sum_j w_jv_{i_j}\big)/\big(\sum_j w_j\big)$, trong đó $w_j\geq0$ và $v_{i_j}$ là biểu diễn của sản phẩm đã mua. Trọng số theo độ mới là một giả thuyết mô hình hóa; không phải mọi hành vi gần đây đều quan trọng hơn hành vi cũ đối với mọi người dùng.

Các mô hình tuần tự như SASRec dùng cơ chế tự chú ý để học quan hệ giữa những vị trí trong chuỗi và lựa chọn sản phẩm tiếp theo ([Kang và McAuley, 2018](https://arxiv.org/abs/1808.09781)). Một hướng khác là kiến trúc hai tháp: tháp người dùng tạo $q_u(t)$, tháp sản phẩm tạo $v_i$, sau đó tính điểm tương thích $s(u,i,t)=q_u(t)^\top v_i$. Nếu cả hai vector được chuẩn hóa theo chuẩn $L_2$, tích vô hướng bằng độ tương đồng cosine. Khi huấn luyện từ giao dịch mua, mô hình thường so sánh sản phẩm đã mua với các sản phẩm được lấy mẫu nhưng chưa quan sát thấy giao dịch; các sản phẩm lấy mẫu này là đối chứng cho mục tiêu học, không phải bằng chứng người dùng không thích chúng ([Hu và cộng sự, 2008](https://yifanhu.net/PUB/cf.pdf)).

### 1.3.2. Biểu diễn sản phẩm từ định danh, văn bản và hình ảnh

Biểu diễn dựa trên mã sản phẩm có thể học được tín hiệu cộng tác từ những mặt hàng đã xuất hiện nhiều lần trong lịch sử. Tuy nhiên, khi một sản phẩm mới có ít hoặc chưa có giao dịch, riêng mã định danh không cung cấp đủ thông tin để suy ra sự phù hợp. Tên, mô tả, thuộc tính và ảnh sản phẩm là các nguồn dữ liệu bổ sung. Văn bản có thể thể hiện loại hàng, chất liệu hoặc màu sắc được mô tả; hình ảnh thể hiện những đặc điểm trực quan khó diễn đạt đầy đủ bằng từ ngữ. Công trình VBPR cho thấy đặc trưng ảnh có thể được đưa vào mô hình xếp hạng cá nhân hóa cùng với phản hồi ngầm ([He và McAuley, 2016](https://ojs.aaai.org/index.php/AAAI/article/view/9973)).

Các mô hình thị giác–ngôn ngữ như CLIP học biểu diễn ảnh và văn bản trong một không gian có thể so sánh, dựa trên các cặp ảnh–văn bản khi tiền huấn luyện ([Radford và cộng sự, 2021](https://proceedings.mlr.press/v139/radford21a.html)). Biểu diễn nội dung có thể kết hợp với biểu diễn mã sản phẩm bằng phép cộng, ghép nối hoặc cơ chế học trọng số. Những cách kết hợp này tạo điều kiện sử dụng cả tín hiệu cộng tác lẫn thông tin nội dung, nhưng vẫn cần xử lý trường hợp thiếu ảnh hoặc mô tả. Việc giảm sự phụ thuộc của mô hình vào mã sản phẩm là một hướng ứng phó với dữ liệu thưa; nó không bảo đảm rằng sản phẩm chưa từng được mua sẽ được xếp hạng tốt. Hiệu quả đối với sản phẩm ít tương tác và sản phẩm chưa từng có giao dịch cần được đo riêng.

### 1.3.3. Truy hồi ứng viên, xếp hạng và tín hiệu thời gian

Khi danh mục lớn, hệ gợi ý thường chia quá trình dự đoán thành hai bước. **Truy hồi ứng viên** chọn một tập $C_u(t)$ nhỏ hơn danh mục từ các nguồn như độ tương đồng với sở thích người dùng, mặt hàng phổ biến gần đây hoặc sản phẩm có nội dung gần với lịch sử mua. **Xếp hạng lại** sử dụng nhiều đặc trưng hơn để sắp xếp các ứng viên và lấy danh sách Top-$K$. Cấu trúc truy hồi rồi xếp hạng đã được mô tả trong hệ thống gợi ý quy mô lớn của YouTube ([Covington và cộng sự, 2016](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/)). Khả năng của bước xếp hạng bị giới hạn bởi tập ứng viên: sản phẩm mục tiêu không có trong $C_u(t)$ sẽ không thể xuất hiện ở kết quả cuối.

Với hàng thời trang, mức độ quan tâm và sự hiện diện của sản phẩm trong giao dịch có thể thay đổi theo thời gian. Do đó, độ phổ biến trong một khoảng gần đây, thời điểm sản phẩm bắt đầu xuất hiện và lịch sử mua lại có thể là tín hiệu bổ sung cho điểm phù hợp cá nhân. Các thống kê này phải được tính từ dữ liệu có trước thời điểm cần dự đoán; dùng giao dịch của giai đoạn đích sẽ gây rò rỉ thông tin. Ở bước xếp hạng lại, mô hình học xếp hạng có thể kết hợp điểm truy hồi, tín hiệu thời gian, độ tương đồng nội dung và mức khớp thuộc tính. LambdaRank là một hướng học xếp hạng sử dụng tín hiệu theo cặp có xét đến thay đổi của thước đo theo vị trí, chẳng hạn NDCG; LambdaMART hiện thực hướng đó bằng cây quyết định tăng cường ([Burges, 2010](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/)).

Chỉ ưu tiên các mặt hàng đã bán nhiều có thể làm sản phẩm mới ít được hiển thị. Vì thế, ngoài độ chính xác, hệ thống có thể quan tâm đến độ bao phủ danh mục và cơ hội xuất hiện của sản phẩm mới. Những chính sách tăng hiển thị cho nhóm ít dữ liệu tạo ra sự đánh đổi giữa khám phá và độ chính xác của danh sách, cần được báo cáo bằng các thước đo riêng thay vì mặc nhiên xem là cải thiện toàn diện.

---

## 1.4. Cơ sở lý thuyết về RAG và gợi ý hội thoại

### 1.4.1. Truy xuất thông tin để tạo câu trả lời có căn cứ

RAG (*Retrieval-Augmented Generation*) kết hợp mô hình sinh ngôn ngữ với một nguồn tri thức bên ngoài được truy xuất theo câu hỏi. Trong mô hình gốc, các đoạn thông tin liên quan được truy hồi trước khi tạo câu trả lời, nhờ đó mô hình có thể sử dụng dữ liệu ngoài các tham số đã học ([Lewis và cộng sự, 2020](https://papers.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf)). Đối với trợ lý mua sắm, nguồn tri thức có thể là danh mục sản phẩm, thuộc tính, mô tả và thông tin có thể kiểm tra được về những sản phẩm đang được hỏi. Truy hồi sản phẩm liên quan và truy hồi bằng chứng để trả lời là hai nhu cầu gắn bó nhưng không đồng nhất: một sản phẩm có thể phù hợp để gợi ý, trong khi một khẳng định cụ thể về sản phẩm vẫn cần dữ liệu hỗ trợ.

[Bộ dữ liệu H&M Personalized Fashion Recommendations](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/overview) cung cấp giao dịch, thuộc tính, mô tả và ảnh sản phẩm nhưng không cung cấp đánh giá văn bản thực của người mua. Vì vậy, câu trả lời dựa trên nguồn này có thể giải thích sự phù hợp về loại hàng, màu sắc hoặc những thuộc tính được ghi nhận, nhưng không thể kết luận về trải nghiệm sau mua nếu không có nguồn độc lập. Tài liệu hoặc nhận xét dùng để minh họa giao diện phải được phân biệt với bằng chứng về chính sản phẩm đang xét.

### 1.4.2. Bộ nhớ sở thích và giới hạn của phản hồi sinh

Hệ gợi ý hội thoại khai thác những gì người dùng diễn đạt qua nhiều lượt để bổ sung hoặc điều chỉnh hồ sơ sở thích ([Gao và cộng sự, 2021](https://arxiv.org/abs/2101.09459)). Bộ nhớ sở thích có thể lưu các ràng buộc đang có hiệu lực, sản phẩm người dùng quan tâm và phản hồi thích hoặc không thích. Khi được đưa trở lại truy hồi và xếp hạng, thông tin này có thể làm thay đổi danh sách đề xuất. Đây là thành phần mở rộng của kiến trúc trợ lý gợi ý; RAG theo định nghĩa gốc không tự tạo ra một bộ nhớ sở thích bền vững. Cần phân biệt phản hồi không thích do người dùng chủ động cung cấp với việc không thấy một giao dịch mua trong dữ liệu lịch sử.

Việc cung cấp ngữ cảnh truy xuất cho mô hình sinh không bảo đảm mọi câu trả lời đều đúng. Một câu trả lời **có căn cứ** cần có các khẳng định được hỗ trợ bởi dữ liệu liên quan và không suy diễn vượt quá nội dung được cung cấp. Ví dụ, màu sắc có thể đối chiếu với thuộc tính hoặc ảnh sản phẩm, còn nhận định “chất lượng tốt hơn” cần loại bằng chứng khác và không thể rút ra chỉ từ ảnh. Hệ thống cần nhận biết thông tin thiếu, nêu giới hạn khi so sánh và có cách phản hồi phù hợp khi không đủ bằng chứng. Đánh giá RAG vì vậy phải xem xét riêng chất lượng truy xuất và độ trung thực của câu trả lời so với ngữ cảnh ([Es và cộng sự, 2024](https://aclanthology.org/2024.eacl-demo.16/)).

---

## 1.5. Phương pháp đánh giá

### 1.5.1. Đánh giá danh sách gợi ý theo thời gian

Đánh giá ngoại tuyến sử dụng các giao dịch trong tương lai làm mục tiêu để kiểm tra danh sách được tạo từ dữ liệu quá khứ. Với một mốc thời gian $t$, gọi $T_u(t)$ là tập sản phẩm người dùng $u$ mua trong giai đoạn đích, $R_K(u,t)$ là danh sách $K$ sản phẩm được gợi ý và $U^+$ là tập người dùng có ít nhất một sản phẩm mục tiêu. Cần xác định rõ cách chọn $U^+$, độ dài giai đoạn đích, danh mục ứng viên và việc có cho phép mua lặp lại hay không, vì các lựa chọn này ảnh hưởng trực tiếp đến ý nghĩa của điểm số. Chia theo mốc thời gian chung giúp hạn chế việc mô hình học từ giao dịch xảy ra sau thời điểm dự đoán ([Ji và cộng sự, 2020](https://arxiv.org/abs/2010.11060)).

Với nhiều sản phẩm mục tiêu cho mỗi người dùng, các chỉ số Top-$K$ thường dùng gồm:

$$\mathrm{HitRate@}K=\frac{1}{|U^+|}\sum_{u\in U^+}\mathbf{1}\big[R_K(u,t)\cap T_u(t)\ne\varnothing\big],$$

$$\mathrm{Recall@}K=\frac{1}{|U^+|}\sum_{u\in U^+}\frac{|R_K(u,t)\cap T_u(t)|}{|T_u(t)|}.$$

HitRate đo tỷ lệ người dùng có ít nhất một sản phẩm mua nằm trong danh sách; Recall đo phần mục tiêu được tìm thấy. Để xét vị trí, đặt $\mathrm{rel}_{u,r}=1$ nếu sản phẩm ở hạng $r$ thuộc $T_u(t)$, ngược lại bằng $0$. Khi đó:

$$\mathrm{DCG@}K(u)=\sum_{r=1}^{K}\frac{\mathrm{rel}_{u,r}}{\log_2(r+1)},\qquad
\mathrm{NDCG@}K=\frac{1}{|U^+|}\sum_{u\in U^+}\frac{\mathrm{DCG@}K(u)}{\mathrm{IDCG@}K(u)}.$$

$\mathrm{IDCG@}K(u)$ là điểm DCG khi tối đa $\min(K,|T_u(t)|)$ sản phẩm mục tiêu đứng ở các vị trí đầu. Một chỉ số khác là $\mathrm{MAP@}K$, trung bình của $\mathrm{AP@}K(u)=\sum_{r=1}^{K}\mathrm{P@}r(u)\,\mathrm{rel}_{u,r}/\min(K,|T_u(t)|)$, trong đó $\mathrm{P@}r(u)$ là tỷ lệ sản phẩm đúng trong $r$ vị trí đầu. NDCG và MAP đều quan tâm đến thứ tự, nhưng sử dụng cách chiết khấu vị trí khác nhau.

Ở hệ gợi ý nhiều giai đoạn, **Candidate Recall** đo tỷ lệ mục tiêu xuất hiện trong tập ứng viên trước khi xếp hạng và là giới hạn trên của Recall cuối cùng nếu các bước sau chỉ sắp xếp tập đó. **Độ bao phủ danh mục** đo tỷ lệ sản phẩm khác nhau được hiển thị qua nhiều người dùng; nên báo thêm kết quả theo nhóm sản phẩm đã có tương tác, sản phẩm chưa thấy trong tập huấn luyện và sản phẩm chưa từng có giao dịch trước mốc dự đoán. Các nhóm “cold” này có ý nghĩa khác nhau và không nên gộp thành một con số.

Kết quả chính nên được đo trên danh mục ứng viên được xác định nhất quán tại thời điểm dự đoán, đồng thời so sánh với các đường cơ sở như phổ biến toàn thời gian và phổ biến gần đây. Phép đánh giá dùng một mục tiêu cùng một số ít sản phẩm lấy mẫu có thể phục vụ phân tích phụ, nhưng điểm số của nó phụ thuộc mạnh vào cách lấy mẫu và không so sánh trực tiếp với xếp hạng toàn danh mục. Khi so sánh hai phương pháp trên cùng người dùng, có thể lấy mẫu lại theo người dùng để ước lượng khoảng tin cậy cho **chênh lệch chỉ số**; khoảng tin cậy và độ lớn cải thiện cần được đọc cùng nhau ([Cañamares và cộng sự, 2020](https://doi.org/10.1007/s10791-020-09371-3)).

### 1.5.2. Đánh giá trợ lý hội thoại và RAG

Chất lượng trợ lý cần được xem xét ở nhiều bước: hiểu đúng ý định và ràng buộc của người dùng; truy xuất đúng sản phẩm, thuộc tính và nguồn thông tin liên quan; duy trì nhất quán sở thích qua các lượt; và tạo câu trả lời đáp ứng yêu cầu mà không đưa ra khẳng định thiếu căn cứ. Độ chính xác truy xuất có thể đo bằng bộ câu hỏi có sản phẩm hoặc thuộc tính liên quan đã được gán nhãn. Độ trung thực của câu trả lời cần đối chiếu từng khẳng định với thông tin truy xuất, nhất là giá, màu sắc và các so sánh giữa sản phẩm. Cũng cần đánh giá khả năng từ chối kết luận khi thiếu dữ liệu, thời gian phản hồi và hành vi dự phòng khi một thành phần không sẵn sàng ([Es và cộng sự, 2024](https://aclanthology.org/2024.eacl-demo.16/)).

Đối với bộ nhớ sở thích, một phép đánh giá riêng có thể kiểm tra liệu phản hồi người dùng có làm thay đổi kết quả theo đúng hướng và liệu hệ thống còn ghi nhớ ràng buộc ở các lượt sau. Chỉ số gợi ý tính từ giao dịch lịch sử không đủ để chứng minh người dùng hài lòng với đối thoại; đánh giá mô phỏng và kiểm thử chức năng cần được phân biệt với phản hồi từ người dùng thật.

---

## 1.6. Nền tảng công nghệ liên quan

### 1.6.1. Xử lý dữ liệu, biểu diễn và học xếp hạng

Quy trình nghiên cứu cần công cụ xử lý dữ liệu dạng bảng và định dạng lưu trữ có thể tái sử dụng để xây dựng tập giao dịch theo thời gian. Các thư viện tính toán tensor hỗ trợ học biểu diễn người dùng và sản phẩm; mô hình thị giác–ngôn ngữ tạo đặc trưng cho văn bản và ảnh; mô hình học xếp hạng xử lý tập ứng viên với nhiều đặc trưng. Trong đề tài, Polars, DuckDB và Parquet phục vụ dữ liệu; PyTorch phục vụ mô hình biểu diễn; Jina CLIP v2 cung cấp đặc trưng đa phương thức; LightGBM phục vụ xếp hạng lại. Jina CLIP v2 hỗ trợ biểu diễn ảnh và văn bản trong cùng không gian và rút gọn chiều biểu diễn theo nguyên lý Matryoshka ([Koukounas và cộng sự, 2024](https://arxiv.org/abs/2412.08802); [Kusupati và cộng sự, 2022](https://papers.nips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html)). Việc chọn chiều vector, tham số và phiên bản cụ thể thuộc phần phương pháp triển khai.

### 1.6.2. Truy xuất và cung cấp dịch vụ

Cơ sở dữ liệu vector hỗ trợ truy hồi sản phẩm theo độ gần của biểu diễn văn bản hoặc hình ảnh và có thể kết hợp với bộ lọc thuộc tính. Cơ sở dữ liệu quan hệ lưu các thực thể nghiệp vụ và lịch sử tương tác; dịch vụ API nối giao diện với bộ gợi ý và trợ lý hội thoại. Qdrant, PostgreSQL, FastAPI và Next.js là các công cụ được dùng cho những vai trò tương ứng trong hệ thống. Việc liệt kê chúng ở đây nhằm xác định chức năng của từng nhóm công nghệ; cấu hình, phiên bản và kiến trúc triển khai sẽ được trình bày ở Chương 2.

---

## 1.7. Các công trình nghiên cứu liên quan

### 1.7.1. Mô hình hóa hành vi và nội dung sản phẩm

[Kang và McAuley (2018)](https://arxiv.org/abs/1808.09781) đề xuất SASRec để dự đoán sản phẩm tiếp theo từ chuỗi tương tác bằng cơ chế tự chú ý. Công trình cho thấy lịch sử có thứ tự chứa thông tin mà một hồ sơ người dùng tĩnh có thể bỏ qua; việc dùng kiến trúc chú ý hay phép tổng hợp nhẹ hơn vẫn là lựa chọn cần đánh giá theo dữ liệu và điều kiện tính toán. [He và McAuley (2016)](https://ojs.aaai.org/index.php/AAAI/article/view/9973) đưa đặc trưng ảnh vào xếp hạng cá nhân hóa từ phản hồi ngầm, đặt cơ sở cho việc khai thác vẻ ngoài của sản phẩm thời trang. [Radford và cộng sự (2021)](https://proceedings.mlr.press/v139/radford21a.html) trình bày cách học biểu diễn ảnh–văn bản bằng học tương phản; các mô hình kế thừa như [Jina CLIP v2](https://arxiv.org/abs/2412.08802) mở rộng khả năng truy hồi đa phương thức. Những công trình này hỗ trợ lựa chọn nguồn đặc trưng, nhưng không tự chứng minh việc kết hợp chúng cải thiện kết quả trên H&M.

### 1.7.2. Truy hồi ứng viên và học xếp hạng

[Covington và cộng sự (2016)](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/) mô tả hệ gợi ý có bước tạo ứng viên và bước xếp hạng riêng, một cách tổ chức hữu ích khi danh mục lớn. [Burges (2010)](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/) tổng hợp LambdaRank và LambdaMART, giải thích cách học xếp hạng có xét đến thước đo theo vị trí. [Ke và cộng sự (2017)](https://papers.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) giới thiệu LightGBM như một phương pháp cây quyết định tăng cường hiệu quả trên dữ liệu dạng bảng. Trong bài toán thời trang theo thời gian, các hướng này gợi ý việc kết hợp điểm cá nhân hóa với đặc trưng sản phẩm và thống kê gần thời điểm dự đoán; giá trị thực tế của từng nguồn phải được kiểm tra bằng đối chứng phù hợp.

### 1.7.3. Gợi ý hội thoại và sinh câu trả lời có căn cứ

[Gao và cộng sự (2021)](https://arxiv.org/abs/2101.09459) hệ thống hóa các bài toán của gợi ý hội thoại, gồm thu nhận sở thích, chiến lược đối thoại nhiều lượt và đánh giá trải nghiệm. [Lewis và cộng sự (2020)](https://papers.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf) đặt nền tảng cho RAG bằng cách kết hợp truy hồi nguồn tri thức ngoài với sinh văn bản. [Es và cộng sự (2024)](https://aclanthology.org/2024.eacl-demo.16/) nhấn mạnh cần đánh giá riêng độ liên quan của ngữ cảnh và độ trung thực của câu trả lời. Đề tài kết nối các hướng này qua bộ nhớ sở thích dùng cho gợi ý và dữ liệu sản phẩm dùng cho giải thích, đồng thời giới hạn những khẳng định không có bằng chứng trong dữ liệu H&M.

---

## 1.8. Tổng kết chương

Chương 1 đã trình bày bài toán gợi ý sản phẩm từ lịch sử giao dịch và nội dung đa phương thức, các nguyên tắc xử lý dữ liệu theo thời gian, kiến trúc truy hồi–xếp hạng, cùng vai trò của bộ nhớ sở thích và RAG trong trợ lý mua sắm. Chương cũng xác định cách đánh giá danh sách Top-$K$ và câu trả lời có căn cứ, đặc biệt là yêu cầu phân biệt tín hiệu mua với phản hồi tiêu cực và tránh dùng thông tin tương lai khi dự đoán. Các nguyên lý này làm nền tảng cho Chương 2, nơi phương pháp và thiết kế cụ thể của hệ thống được trình bày.
