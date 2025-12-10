W# Web Attack Detection using Deep Learning

Dự án này tập trung vào việc xây dựng và đánh giá các mô hình **Học sâu (Deep Learning)** để phát hiện các cuộc tấn công Web. Hệ thống sử dụng các kiến trúc mạng nơ-ron tiên tiến như **CNN, LSTM, GRU, MLP** và mô hình lai **CNN-LSTM** để phân loại các request độc hại dựa trên bộ dữ liệu WEBIDS23.

## 📂 Cấu trúc Dự án

Cấu trúc thư mục được tổ chức module hóa để dễ quản lý:

```text
web-attack-detection/
├── models/                           # Chứa các file mô hình đã huấn luyện (.h5)
│   ├── cnn_webids23_model.h5
│   ├── lstm_webids23_model.h5
│   ├── gru_webids23_model.h5
│   ├── mlp_webids23_model.h5
│   └── cnn_lstm_webids23_model.h5
│
├── results/                          # Kết quả đánh giá (Confusion Matrix, History)
│   ├── *_confusion_matrix.png
│   └── *_train_history.png
│
├── src/
│   ├── preprocessing-src/            # Mã nguồn tiền xử lý dữ liệu
│   │   ├── preprocessed_data.ipynb
│   │   └── preprocessed_data_2.ipynb
│   │
│   └── train-src/                    # Mã nguồn huấn luyện mô hình
│       ├── train_cnn_balance.ipynb
│       ├── train_lstm_balance.ipynb
│       ├── train_gru_balance.ipynb
│       ├── train_mlp_balance.ipynb
│       ├── train_cnnlstm_balance.ipynb
│       └── ...
│
├── requirements.txt                  # Các thư viện cần thiết
└── README.md                         # Tài liệu hướng dẫn
```
🧠 Mô hình & Giải thuật
Dự án triển khai và so sánh hiệu năng của 5 kiến trúc mạng nơ-ron khác nhau. Dưới đây là chi tiết giải thuật và lý do sử dụng:

1. Multi-Layer Perceptron (MLP)
Kiến trúc: Mạng nơ-ron truyền thẳng (Feed-forward) cơ bản với các lớp Dense.

Vai trò: Dùng làm baseline để so sánh hiệu năng với các mô hình phức tạp hơn. Phù hợp với dữ liệu dạng bảng nhưng hạn chế trong việc bắt các đặc trưng chuỗi hoặc không gian.

2. Convolutional Neural Networks (CNN)
Kiến trúc: Sử dụng các lớp Conv1D để trích xuất đặc trưng cục bộ (local features) từ các chuỗi dữ liệu (ví dụ: các mẫu ký tự trong URL hoặc Payload).

Ưu điểm: Hiệu quả trong việc phát hiện các mẫu (patterns) cố định của các loại tấn công như SQL Injection hay XSS.

3. Long Short-Term Memory (LSTM)
Kiến trúc: Mạng nơ-ron hồi quy (RNN) có khả năng ghi nhớ dài hạn.

Ưu điểm: Xử lý tốt dữ liệu dạng chuỗi thời gian hoặc chuỗi ký tự, giúp mô hình hiểu ngữ cảnh của request trước và sau, khắc phục vấn đề vanishing gradient của RNN thường.

4. Gated Recurrent Unit (GRU)
Kiến trúc: Một biến thể đơn giản hóa của LSTM với ít tham số hơn.

Ưu điểm: Tốc độ huấn luyện nhanh hơn LSTM nhưng vẫn giữ được khả năng nắm bắt thông tin chuỗi tốt.

5. Hybrid CNN-LSTM
Kiến trúc: Kết hợp Conv1D (để trích xuất đặc trưng) và LSTM (để học sự phụ thuộc chuỗi).

Cơ chế: Dữ liệu đi qua CNN để lọc nhiễu và trích xuất đặc trưng quan trọng, sau đó output được đưa vào LSTM để phân tích ngữ nghĩa theo thời gian. Đây thường là mô hình cho độ chính xác cao nhất.

🛠 Yêu cầu cài đặt
Đảm bảo bạn đã cài đặt Python (khuyên dùng 3.9 - 3.11).

Clone dự án:

Bash

git clone https://github.com/Pbui3983/Web-attack-detection-using-deep-learning
cd web-attack-detection
Cài đặt thư viện: Sử dụng file requirements.txt đi kèm:

Bash

pip install -r requirements.txt
🚀 Hướng dẫn Sử dụng & Chi tiết Code
Quy trình thực hiện dự án đi qua 3 bước chính, tương ứng với các thư mục trong src/:

Bước 1: Tiền xử lý dữ liệu (src/preprocessing-src)
Sử dụng preprocessed_data.ipynb.

Load Data: Đọc dữ liệu thô (CSV).

Cleaning: Xử lý giá trị Null, Infinity.

Encoding: Chuyển đổi nhãn (Label Encoding) và các đặc trưng phân loại (One-Hot Encoding).

Scaling: Chuẩn hóa dữ liệu số bằng MinMaxScaler để đưa về khoảng [0, 1] giúp mô hình hội tụ nhanh hơn.

Bước 2: Huấn luyện mô hình (src/train-src)
Các file train_*_balance.ipynb thực hiện quy trình huấn luyện chuẩn:

Reshape Data:

Với MLP: Input dạng 2D (samples, features).

Với CNN/LSTM/GRU: Input dạng 3D (samples, time_steps, features).

Xây dựng Model (TensorFlow/Keras):

Ví dụ cấu trúc CNN-LSTM trong code:

Python

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(64))
model.add(Dense(n_classes, activation='softmax'))
Compile & Fit: Sử dụng Adam optimizer và sparse_categorical_crossentropy (hoặc categorical) loss function.

Lưu Model: Model tốt nhất được lưu vào thư mục models/.

Bước 3: Đánh giá (results)
Code tự động sinh ra các biểu đồ đánh giá:

Confusion Matrix: Để xem độ chính xác trên từng loại tấn công cụ thể.

Accuracy/Loss History: Để kiểm tra hiện tượng Overfitting/Underfitting.

📊 Kết quả (Results)
Các biểu đồ kết quả được lưu trong thư mục results/.

CNN-LSTM thường cho kết quả tốt nhất nhờ khả năng học đặc trưng hỗn hợp.

MLP có tốc độ train nhanh nhất nhưng độ chính xác thấp hơn trên các mẫu tấn công phức tạp.

👥 Tác giả
Bùi Trọng Phúc