1. Kết luận nhanh

Nếu phải chấm riêng phần thực nghiệm hiện tại:

Khía cạnh	Đánh giá
Full-ranking 152k item	🟢 Rất tốt
Có Random + Popularity baseline	🟢 Tốt
Có CF → Image → Text → Multimodal	🟢 Rất tốt
Có validation/test riêng	🟢
Có phân tích overlap với Popularity	🟢 Có giá trị
Chứng minh multimodal tốt hơn từng modality	🟡 Có, nhưng chưa hoàn toàn
Chứng minh Image + Text bổ trợ lẫn nhau	🟡 Chưa đủ
Chứng minh khả năng personalization	🟡 Có dấu hiệu, chưa đủ
Chứng minh retrieval đủ tốt cho production	🔴 Chưa
Chứng minh không leakage	🔴 Cần kiểm tra
Statistical significance	🔴 Thiếu
Ý nghĩa của gain “x lần”	🟡 Dễ gây hiểu nhầm

Điểm đáng chú ý nhất:
Full Multimodal không thực sự thắng Text trên mọi metric. Nó thắng rất rõ về NDCG và Top-50, nhưng Text vẫn thắng HR@10.

Và một vấn đề lớn hơn: HR@50 = 3.10% nghĩa là mô hình vẫn bỏ lỡ khoảng 96.9% positive test cases. Vì vậy chưa thể viết rằng User Tower đã “hoàn thành tốt nhiệm vụ Candidate Retrieval” nếu Top-50 là candidate set cuối cùng.

2. Điều thực sự đáng mừng: ablation đang kể một câu chuyện khá đẹp

Nhìn riêng trên Test:

CF             HR@50 = 1.4719%
CF + Image     HR@50 = 2.7244%
CF + Text      HR@50 = 3.0536%
Full           HR@50 = 3.1028%

Đây là pattern rất hợp lý:

                 HR@50

CF                 1.47
│
├── + Image        2.72
│
├── + Text         3.05
│
└── + Image+Text   3.10

Tức là:

ID-only đã học được collaborative signal.
Image bổ sung rất nhiều.
Text bổ sung nhiều hơn Image.
Image + Text tiếp tục cải thiện, nhưng phần gain cuối cùng khá nhỏ.

Đây là một kết quả có giá trị nghiên cứu, vì nó cho phép bạn nói:

Nội dung sản phẩm mang lại tín hiệu bổ sung đáng kể so với collaborative signal thuần túy.

Nhưng tôi sẽ không viết:

“Multimodal chứng minh vượt trội rõ rệt so với mọi single modality.”

Vì điều đó không hoàn toàn đúng.

3. Một phát hiện rất thú vị: Text có vẻ quan trọng hơn Image

Đây có thể là một trong những insight hay nhất của experiment.

So với CF:

Image

HR@50:

$$ 2.7244-1.4719=1.2525 $$
Text
$$ 3.0536-1.4719=1.5817 $$
Full
$$ 3.1028-1.4719=1.6309 $$

Tức là incremental gain:

Image       +1.2525 percentage points
Text        +1.5817 percentage points
Image+Text  +1.6309 percentage points

Text đóng góp lớn hơn Image.

Nhưng cái thú vị nằm ở phần Image được thêm vào sau Text:

$$ 3.1028-3.0536=0.0492\% $$

Trong khi Text thêm vào CF:

$$ 3.0536-1.4719=1.5817\% $$

Đây là chênh lệch cực lớn.

Nó đặt ra một câu hỏi nghiên cứu rất quan trọng:

Image thực sự bổ sung thông tin mới cho User Tower, hay phần lớn thông tin mà Image mang lại đã được Text capture?

Đây là chỗ tôi sẽ đào sâu tiếp.

4. Và đây là điểm tôi nghi ngờ mạnh nhất: “multimodal synergy” có thể đang bị nói quá

Báo cáo hiện tại viết:

Image + Text tạo nên biểu diễn người dùng toàn diện và giảm thiểu nhầm lẫn ngữ nghĩa.

Số liệu hiện tại chưa đủ để kết luận điều này.

Tại sao?

Full:

HR@50 = 3.1028%

Text:

HR@50 = 3.0536%

Gain:

+0.0492 percentage point

Tương đối:

$$ \frac{3.1028-3.0536}{3.0536}\approx1.61\% $$

Tức là sau khi đã có Text, Image chỉ giúp khoảng 1.6% tương đối ở HR@50.

Ở NDCG@50:

Full      0.9421
Text      0.9191

Gain ≈ 2.5% relative.

Đây là cải thiện có thể có ý nghĩa, nhưng chưa thể tự động gọi là “synergy mạnh”.

Tôi sẽ gọi kết quả hiện tại là:

Complementary contribution / incremental benefit

thay vì:

Strong multimodal synergy

cho đến khi có thêm experiment.

5. Full Multimodal thắng Text theo cách rất đáng chú ý

Có một pattern còn thú vị hơn:

Model	HR@10	NDCG@10
Text	0.9081	0.4583
Full	0.8779	0.4683

Text:

bắt được nhiều positive hơn ở Top-10.

Full:

không bắt được nhiều bằng Text, nhưng khi bắt được thì xếp đúng vị trí hơn.

Đây là sự khác biệt giữa:

retrieval coverage

và

ranking quality.

Full có:

$$ NDCG@10 / HR@10 $$

cao hơn Text.

Tính ra:

Text:
0.4583 / 0.9081 ≈ 0.505

Full:
0.4683 / 0.8779 ≈ 0.533

Điều này gợi ý:

Image có thể không giúp tăng nhiều khả năng “đưa đúng item vào Top-10”, nhưng có thể giúp điều chỉnh thứ tự các candidate theo hướng tốt hơn.

Đây mới là một hypothesis rất đáng kiểm tra.

6. Popularity baseline đang là đối thủ rất đáng gờm

Đây là phần tôi nghĩ báo cáo hiện tại cần trung thực hơn.

Ở Top-50:

Popularity       4.0525%
Full User Tower  3.1028%

Tức là:

$$ 3.1028 / 4.0525 \approx 76.6\% $$

User Tower mới đạt khoảng 76.6% HR của popularity.

Hay nói cách khác:

Popularity: 1071 hits
User Tower: 820 hits

Vì N = 26,428.

Cho nên nếu hội đồng hỏi:

“Mô hình của em có đánh bại popularity không?”

thì câu trả lời hiện tại là:

Không.

Và điều này hoàn toàn không làm experiment thất bại.

Ngược lại, đây là một kết quả thú vị:

Một mô hình personalization phức tạp chưa vượt được global popularity về raw hit rate.

Đó là điều rất bình thường trong recommender system, đặc biệt khi dataset có head-product bias.

7. Nhưng phân tích Popularity/User Tower của bạn có một vấn đề logic

Bạn có:

Popularity hits       1070
User Tower hits        820
Overlap                190

Only UT                630
Only Popularity        880

Phần này rất đáng giá.

Nhưng câu:

“630 users là sản phẩm nằm ở Mid-Tail và Long-Tail”

không được suy ra trực tiếp từ overlap.

Bạn mới chứng minh được:

630 test cases được User Tower hit nhưng Popularity không hit.

Bạn chưa chứng minh 630 đó là Mid-Tail/Long-Tail.

Muốn tuyên bố như vậy, phải có phân tích:

item popularity rank
        ↓
Head / Mid / Tail / Long-tail
        ↓
UT-only hit distribution

Ví dụ:

Segment	Catalog %	UT-only hits	Popularity hits
Head	?	?	?
Mid	?	?	?
Tail	?	?	?
Long-tail	?	?	?

Nếu 630 thật sự tập trung mạnh ở tail → đây sẽ là một finding rất đẹp.

8. Câu “95% danh mục không bao giờ được tiếp cận” cũng cần sửa

Nếu Popularity baseline của bạn là một danh sách Top-K cố định, thì đúng là nó chỉ expose một tập cực nhỏ của catalog.

Nhưng:

“95% danh mục sẽ không bao giờ được tiếp cận người dùng”

là một claim rộng hơn.

Bạn cần phân biệt:

Popularity recommendation
Top K global products

với

Product exposure trong toàn hệ thống

Một sản phẩm không nằm trong Top-50 popularity không đồng nghĩa nó “không bao giờ được tiếp cận” nếu còn các tầng khác.

Nên đổi thành:

“Popularity-only retrieval chỉ khai thác một tập sản phẩm rất nhỏ ở vùng head của catalog, dẫn tới độ bao phủ catalog thấp.”

Câu này chắc chắn hơn về mặt học thuật.

9. Vấn đề lớn nhất: HR@50 = 3.1% có đủ làm Candidate Retrieval không?

Đây là điểm tôi muốn bạn đặc biệt lưu ý.

Bạn đang viết:

User Tower thu hẹp 152,086 → 50 candidate.

Nhưng:

HR@50 = 3.1028%

nghĩa là:

chỉ khoảng 820 / 26,428 positive users có ground-truth item nằm trong Top-50.

Tức khoảng:

$$ 96.9\% $$

positive vẫn không xuất hiện trong candidate set.

Nếu downstream Reranker/RAG chỉ nhìn 50 candidates này thì:

Reranker không thể cứu được 96.9% những case mà User Tower đã loại bỏ.

Đây là bản chất của two-stage retrieval.

Reranker có thể sửa:

candidate ordering

nhưng không thể sửa:

candidate recall = 0
10. Vì vậy, câu chuyện kiến trúc nên được sửa

Không nên viết:

User Tower hoàn thành nhiệm vụ Candidate Retrieval.

Tôi sẽ viết:

User Tower đóng vai trò tầng candidate retrieval nhằm giảm đáng kể không gian tìm kiếm từ 152,086 sản phẩm xuống tập ứng viên nhỏ hơn. Tuy nhiên, kết quả HR@50 ≈ 3.1% cho thấy khả năng recall hiện tại còn hạn chế và cần được cải thiện trước khi sử dụng Top-50 làm candidate set cuối cùng cho tầng reranking.

Điều này mạnh hơn về mặt học thuật, chứ không yếu hơn.

Bạn đang cho thấy mình hiểu giới hạn của chính model.

11. Tôi nghi ngờ mạnh về một vấn đề khác: định nghĩa positive

Bạn ghi:

Leave-one-out

và:

26,428 Positive Users.

Nếu mỗi user chỉ có một ground-truth positive, thì:

$$ Recall@K = HitRate@K $$

Điều này giải thích tại sao:

HR@10 = Recall@10
HR@20 = ...

Điều đó không sai.

Nhưng luận văn cần nói rõ:

Recall@K trong experiment này tương đương HitRate@K do mỗi user có đúng một positive item được đánh giá.

Nếu không, hội đồng có thể hỏi:

“Tại sao Recall@10 và HitRate@10 giống hệt nhau?”

12. Cần kiểm tra leakage cực kỳ kỹ ở Dataset mới

Đây là thứ tôi sẽ kiểm tra trước tất cả mọi thứ khác.

Bạn viết:

“tái thu thập và làm giàu dữ liệu tương tác từ toàn bộ tập đánh giá gốc”

Cụm này làm tôi bật cảnh báo.

Cần xác minh pipeline:

Raw interactions
       ↓
chronological split?
       ↓
train / validation / test
       ↓
enrichment
       ↓
training

hay vô tình:

Raw evaluation dataset
       ↓
enrichment
       ↓
training
       ↓
test

Nếu interaction dùng để enrich user representation có chứa thông tin sau thời điểm test → temporal leakage.

Đặc biệt cần kiểm tra:

User history

Có interaction sau test timestamp đi vào user tower không?

Item features

CLIP embedding có được train/fine-tune trên dataset chứa test information không?

Popularity

Popularity được tính:

train only

hay:

train + validation + test?

Popularity baseline bắt buộc phải chỉ dùng train.

ID residual

id_residual được học từ toàn bộ item/user mapping hay mapping có leakage?

Mapping itself thường không phải leakage, nhưng feature construction cần xem.

13. “Dataset mới tốt hơn 120 lần” không nên dùng làm headline

Ví dụ:

CF:
0.0045% → 0.5411%

Bạn nói:

+120.2 lần

Về toán học đúng.

Nhưng về diễn giải học thuật, rất dễ gây ấn tượng sai.

Bởi vì baseline cũ gần zero.

Tốt hơn nên viết:

HR@10 tăng từ 0.0045% lên 0.5411%, tương đương tăng 0.5366 percentage point hoặc khoảng 120× theo tỷ lệ tương đối.

Và quan trọng hơn:

Không được diễn giải 120× là “model tốt hơn 120 lần”.

Nó chỉ là ratio của hai metric rất nhỏ.

14. Dataset enrichment thực sự là một finding lớn

Tôi nghĩ đây mới là một trong những phần đáng nghiên cứu nhất.

Bạn có:

80% singleton
        ↓
34,187 users ≥ 5 interactions
        ↓
90% warm products

và:

CF HR@10
0.0045%
    ↓
0.5411%

Đây không đơn thuần là “train model lâu hơn”.

Nó cho thấy:

data regime đang quyết định mạnh đến khả năng học collaborative signal.

Điều này rất phù hợp với recommender system.

Có thể giải thích:

Sparse regime
User:
[A]
[B]
[C]

→ không đủ sequence signal

Dense regime
User:
[A → C → F → G → K → M]

→ SASRec bắt đầu học transition / preference pattern

Đây là finding đáng đưa vào luận văn.

15. Nhưng tôi muốn kiểm tra một giả thuyết khác

Bạn đang gọi:

“CF đã học được quan hệ cộng tác thực sự”

Tôi sẽ chưa cho phép kết luận mạnh như vậy.

Vì CF tăng mạnh sau enrichment có thể đến từ:

nhiều user history hơn;
nhiều item xuất hiện trong training hơn;
item frequency tăng;
sequence length tăng;
warm-start tăng;
user embedding được học tốt hơn;
giảm cold-start;
distribution train/test thay đổi.

Tất cả đều có thể xảy ra.

Muốn nói:

“SASRec học được collaborative relationship”

thì cần dissect:

performance by user history length
performance by item popularity
performance by warm/cold item
performance by sequence length
16. Đây là các experiment tôi muốn bạn chạy tiếp

Nếu bạn đang chuẩn bị thesis, tôi sẽ không train thêm model ngay.

Hãy phân tích dữ liệu/code trước.

🔴 Priority 1 — Leakage audit

Kiểm tra:

split timestamp;
user history cutoff;
item feature construction;
CLIP embedding source;
popularity calculation;
negative/exclusion construction;
train/valid/test contamination.

Đây là blocker.

🔴 Priority 2 — Performance theo user history

Chia:

5–9 interactions
10–19
20–49
50–99
100+

Sau đó:

HR@10
HR@50
NDCG@10

cho:

CF
Image
Text
Full

Tôi đặc biệt muốn biết:

Full multimodal có thực sự tốt hơn ở user sparse không?

🔴 Priority 3 — Performance theo item popularity

Chia item:

Head
Mid
Tail
Long-tail

rồi đo HR/NDCG.

Đây là experiment có khả năng biến claim:

“User Tower khám phá long-tail”

từ suy đoán → bằng chứng.

🟠 Priority 4 — Head vs Tail của từng modality

Đây có thể rất thú vị:

             Head    Mid    Tail
CF
Image
Text
Full

Tôi dự đoán một khả năng:

CF       → Head
Text     → Mid/Tail
Image    → Mid/Tail
Full     → Tail?

Nếu dữ liệu thực sự cho pattern này thì luận văn sẽ có câu chuyện rất đẹp.

17. Priority 5 — User Tower Top-K curve

Đừng chỉ báo:

@10
@20
@50

Hãy lấy:

@1
@5
@10
@20
@50
@100
@200
@500
@1000

Sau đó xem:

Recall@K

curve.

Đây là experiment cực kỳ quan trọng đối với candidate retrieval.

Ví dụ nếu:

K       HR
10      0.88%
20      1.56%
50      3.10%
100     5.2%
200     8.9%
500     17%
1000    28%

thì câu chuyện hoàn toàn khác so với:

50  → 3.1%
100 → 3.3%
200 → 3.4%

Case đầu: model có signal nhưng cần candidate set lớn hơn.

Case sau: representation/retrieval đang có vấn đề.

18. Priority 6 — Statistical significance

Chênh lệch:

Text     3.0536
Full     3.1028

chỉ là:

+0.0492 percentage point

Có thể là improvement thật.

Nhưng cũng có thể nằm trong variance.

Với 26,428 users, bạn hoàn toàn có thể bootstrap user-level:

1000 bootstrap samples
        ↓
HR@50 Full - HR@50 Text
        ↓
95% CI

Nếu CI:

[+0.01%, +0.09%]

→ khá thuyết phục.

Nếu:

[-0.03%, +0.13%]

→ không thể khẳng định Full tốt hơn Text về HR@50.

Tôi đặc biệt muốn làm điều này cho:

Full vs Text
Full vs Image
Full vs CF
19. Một điểm nữa: validation rất đáng chú ý

Full:

Validation HR@50 = 3.1524%
Test       HR@50 = 3.1028%

rất gần nhau.

Tương tự NDCG:

0.9460
0.9421

Đây là tín hiệu khá tốt về generalization.

Khoảng cách cực nhỏ:

HR@50:
3.1524 → 3.1028

chỉ khoảng 1.6% relative.

Đặc biệt checkpoint Full là Epoch 5 trong khi các model kia Epoch 10.

Điều này khiến tôi muốn xem training curve.

Có khả năng:

Full
Epoch 1  ───
Epoch 3  ──────
Epoch 5  ─────────
Epoch 10 ───────
Epoch 20 ─────

Nếu validation bắt đầu giảm sau Epoch 5 thì đây là bằng chứng rất tốt cho early stopping.

20. Một câu chuyện khác tôi thấy từ số liệu

Có thể mô hình đang có hiện tượng:

Multimodal giúp ranking hơn là discovery.

Bằng chứng sơ bộ:

Text → HR@10 cao nhất
Full → NDCG@10 cao nhất

và:

Text → HR@50 3.0536
Full → HR@50 3.1028

Full không tăng coverage quá nhiều so với Text, nhưng NDCG tăng khá nhất quán:

NDCG@10
Text  0.4583
Full  0.4683

NDCG@20
Text  0.6326
Full  0.6406

NDCG@50
Text  0.9191
Full  0.9421

Tức:

Image có thể đang giúp disambiguate / reorder những item mà Text đã đưa vào vùng candidate.

Đây là hypothesis, chưa phải kết luận.

Nhưng tôi sẽ rất muốn kiểm tra nó.

21. Và cuối cùng: Random baseline của bạn thực sự rất tốt để chứng minh evaluation

Random:

HR@10 ≈ 0.0066%
HR@50 ≈ 0.0329%

Về lý thuyết:

$$ 10/152086=0.006575\% $$ $$ 50/152086=0.032876\% $$

Khớp gần như hoàn hảo.

Đây là một dấu hiệu tốt rằng full-ranking evaluation của bạn có vẻ nhất quán.

Full:

HR@10 = 0.8779%

so với random:

0.0066%

→ khoảng 133×.

Nhưng tôi sẽ dùng nó như:

“model có signal rõ ràng so với random”

chứ không dùng:

“model tốt gấp 133 lần”.
