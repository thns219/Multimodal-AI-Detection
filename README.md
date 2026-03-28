I - Abstract
Sự bùng nổ của các mô hình sinh nội dung dựa trên trí tuệ nhân tạo đã làm gia tăng đáng kể số lượng dữ liệu giả mạo dưới dạng văn bản và hình ảnh. Điều này đặt ra yêu cầu cấp thiết về các phương pháp phát hiện nội dung do AI tạo ra với độ chính xác và khả năng mở rộng cao. Trong nghiên cứu này, chúng tôi đề xuất một hệ thống phát hiện đa phương thức (multimodal) nhằm phân loại nội dung AI và nội dung thực (human-generated). Hệ thống kết hợp các mô hình học sâu tiên tiến cho cả hai miền dữ liệu, bao gồm Transformer cho văn bản và Vision Transformer cho hình ảnh. Kết quả thực nghiệm cho thấy cách tiếp cận đa phương thức giúp cải thiện hiệu năng so với các phương pháp đơn lẻ, đồng thời tăng khả năng tổng quát hóa trên dữ liệu quy mô lớn.

II - Kiến trúc hệ thống (System Architecture)
Hệ thống được thiết kế theo kiến trúc đa mô-đun, bao gồm hai nhánh xử lý chính:

1. Nhánh xử lý văn bản (Text Branch)
Sử dụng các mô hình Transformer như:
BERT - RoBERTa
Thực hiện:
Tokenization
Embedding
Fine-tuning cho bài toán phân loại (AI vs Human)

2. Nhánh xử lý hình ảnh (Image Branch)
Sử dụng mô hình:
Vision Transformer (ViT)
Thực hiện:
Tiền xử lý ảnh (resize, normalize)
Trích xuất đặc trưng
Phân loại ảnh (Fake vs Real)

3. Cơ chế kết hợp (Fusion Strategy)
Áp dụng late fusion:
Kết hợp kết quả dự đoán từ hai nhánh
Tăng độ tin cậy của hệ thống

Có thể mở rộng sang:
Feature-level fusion
Attention-based fusion

III - Phương pháp (Methodology)

Quy trình xử lý bao gồm các bước chính:
Thu thập dữ liệu
Văn bản: final_data.csv
Hình ảnh: diffusion_data

Tiền xử lý
Văn bản: làm sạch, tokenize
Hình ảnh: resize, normalize
Huấn luyện mô hình
Fine-tune mô hình Transformer cho NLP
Train ViT cho image classification

Đánh giá
Sử dụng các metrics:
Accuracy - Precision - Recall - F1-score

Confusion Matrix

IV - Thực nghiệm và kết quả (Experiments & Results)
Hệ thống được đánh giá trên tập dữ liệu gồm cả văn bản và hình ảnh với hai nhãn chính:

AI / Fake
Human / Real

Kết quả cho thấy:

Các mô hình Transformer (BERT, RoBERTa) đạt hiệu quả cao trong phân loại văn bản
Vision Transformer (ViT) hoạt động tốt trong phát hiện ảnh giả
Phương pháp đa phương thức cải thiện độ chính xác tổng thể so với từng mô hình riêng lẻ
Hệ thống có khả năng mở rộng tốt trên dữ liệu lớn và đa dạng

V - Kết luận (Conclusion)
Nghiên cứu này trình bày một hệ thống phát hiện nội dung AI dựa trên cách tiếp cận đa phương thức, kết hợp giữa xử lý ngôn ngữ tự nhiên và thị giác máy tính. Kết quả cho thấy việc tích hợp nhiều modality giúp cải thiện đáng kể hiệu năng và độ tin cậy của hệ thống.

Trong tương lai, hệ thống có thể được mở rộng theo các hướng:
Kết hợp thêm dữ liệu video hoặc audio
Áp dụng các kỹ thuật fusion nâng cao
Tối ưu hóa cho triển khai thực tế (real-time detection)

VI - Ứng dụng (Applications)
Phát hiện tin giả (Fake News Detection)
Kiểm chứng nội dung trên mạng xã hội
Hỗ trợ kiểm duyệt nội dung
Ứng dụng trong an ninh mạng và điều tra số

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/38b23739-80c2-4ae6-9b25-77669624467d" />

Hệ thống Multimodal AI Detection phát hiện nội dung do AI tạo ra bằng cách kết hợp phân tích văn bản (BERT, RoBERTa) và hình ảnh (ViT). Các đặc trưng từ hai nhánh được hợp nhất thông qua cơ chế fusion, giúp cải thiện độ chính xác và khả năng tổng quát hóa so với các phương pháp đơn phương thức.
