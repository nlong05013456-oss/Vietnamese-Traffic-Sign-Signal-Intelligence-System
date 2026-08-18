# 🚦 BÁO CÁO TỔNG KẾT TIẾN ĐỘ DỰ ÁN (PROJECT PROGRESS REPORT)
**Dự án:** Vietnamese Traffic Intelligence System (ADAS & MLOps)  
**Kỹ sư thực hiện:** Machine Learning Engineer  
**Trạng thái:** Hoàn tất 100% Giai đoạn Chuẩn bị Dữ liệu, EDA, Kiến trúc Hệ thống, Huấn luyện Baseline Classifier & Đóng gói MLOps.

---

## 1. 🎯 TỔNG QUAN HỆ THỐNG & KIẾN TRÚC ĐA NHIỆM

Hệ thống được thiết kế theo tư duy **Data-Centric AI + Modular Architecture**, chia tách bài toán phức tạp thành 3 module độc lập được đồng bộ qua chuỗi thời gian:

```
                       CAMERA / VIDEO STREAM
                                │
                                ▼
                      ┌──────────────────┐
                      │   Frame Input     │
                      └────────┬─────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
       🪧 Sign             🚦 Light           ⏱️ Countdown
       Detector            Detector           Classifier
       (YOLO11n)           (YOLO11n)       (Digit Localizer + CNN)
            │                  │                  │
            ▼                  ▼                  ▼
       7 VN Classes        RED / GREEN         0 to 9 Digits
      (P.127, W.205...)   (Task-Aligned)      (Left-Right Merge)
            │                  │                  │
            └───────────┬──────┴──────────────────┘
                        ▼
            ┌────────────────────────┐
            │   Temporal Tracker     │  <- (Làm mượt trạng thái thời gian)
            │   & State Smoothing    │
            └───────────┬────────────┘
                        │
                        ▼
            ┌────────────────────────┐
            │   Driver HUD Overlay   │  <- (Giao diện Cảnh báo Lái xe)
            │   & Structured JSON    │
            └───────────┬────────────┘
                        │
                        ▼
            ┌────────────────────────┐
            │    FastAPI + Docker    │  <- (MLOps Production Serving)
            └────────────────────────┘
```

---

## 2. 🧹 GIAI ĐOẠN 1: DATA ENGINEERING & DATA-CENTRIC AI

### A. Kiểm toán dữ liệu tự động (Data Audit)
* Đã quét toàn diện **$10.252$ file ảnh và nhãn** bằng công cụ `src/data_audit.py`.
* Kết quả: **$0$ ảnh hỏng (0-byte/corrupt)**, $100\%$ ảnh mở được.

### B. Khắc phục lỗi nhãn nghiêm trọng qua Visual QA
* Phát hiện bộ dữ liệu mở bên thứ ba bị gán sai nhãn (folder `8` chứa ảnh số `2`, folder `9` chứa ảnh số `3`).
* Đã xóa sạch toàn bộ nhãn lỗi và khởi tạo lại bộ dữ liệu chữ số LED 7 đoạn chuẩn xác $100\%$ gồm đủ 10 lớp ($0 \to 9$), mỗi lớp $250$ ảnh với các màu LED Đỏ, Xanh, Vàng.

### C. Chuẩn hóa format & Chống rò rỉ dữ liệu (Anti-Data Leakage)
* Chuyển đổi toàn bộ toạ độ pixel COCO $[x_{min}, y_{min}, w, h]$ sang chuẩn YOLO $[x_c, y_c, w, h] \in [0, 1]$.
* Áp dụng **Group Split theo `street_id`** ($70\%$ Train, $15\%$ Val, $15\%$ Test) để đảm bảo các khung hình từ cùng một tuyến đường quay không bị rò rỉ giữa tập Train và Test.
* **Đóng băng (Freeze) dữ liệu chuẩn hóa tại:** `Data/Processed/`.

| Dataset mục tiêu | Tập Train | Tập Val | Tập Test | Tổng BBox / Ảnh |
| :--- | :---: | :---: | :---: | :---: |
| **🪧 Biển báo Zalo (7 Lớp)** | $2.636$ ảnh ($6.238$ obj) | $734$ ảnh ($1.948$ obj) | $1.130$ ảnh ($2.814$ obj) | **$11.000$ objects** |
| **🚦 Đèn giao thông VN** | $2.513$ ảnh ($4.607$ obj) | $103$ ảnh ($181$ obj) | $50$ ảnh ($95$ obj) | **$4.883$ objects** |
| **⏱️ Chữ số đếm ngược ($0 \to 9$)** | $1.750$ ảnh ($175$/lớp) | $370$ ảnh ($37$/lớp) | $380$ ảnh ($38$/lớp) | **$2.500$ images** |

---

## 3. 📈 GIAI ĐOẠN 2: EXPLORATORY DATA ANALYSIS (EDA CHUYÊN SÂU)

Đã xuất bản 5 biểu đồ phân tích thống kê tại `Data/Reports/eda_charts/` và Notebook `notebooks/01_data_eda_and_validation.ipynb`:

1. **Phân bố 7 lớp biển báo:** Phân bố đều qua các tập, nhóm nhiều nhất là `Nguy hiểm` ($27.7\%$), ít nhất là `Cấm rẽ` ($5.1\%$).
2. **Cân bằng Đèn giao thông:** Đèn Đỏ (**$53.49\%$**) và Đèn Xanh (**$46.51\%$**) đạt tỷ lệ cân bằng chuẩn $1.15 : 1$.
3. **Định lượng thách thức Vật thể Nhỏ (Small Objects):**
   * **$76.05\%$** số biển báo là vật thể nhỏ ($<32\text{px}$).
   * Kích thước trung vị (*Median BBox*): **$17.0\text{px}$** $\longrightarrow$ Minh chứng thực tế cho việc chọn cấu hình FPN và độ phân giải ảnh.
4. **Mức độ mất cân bằng lớp:** Hệ số $5.48 : 1$ (ở mức Moderate, được kiểm soát tốt bởi Task-Aligned Loss).
5. **Đa dạng môi trường (Environmental Diversity):**
   * Tập Biển báo: Ban ngày ($91.3\%$), Ban đêm/thiếu sáng ($5.3\%$), Chói nắng ($3.3\%$).
   * Tập Đèn giao thông: Ban ngày ($79.3\%$), Ban đêm ($14.7\%$), Chói nắng ($6.0\%$).

---

## 4. 🧠 GIAI ĐOẠN 3: MODEL DEVELOPMENT & MULTI-TASK PIPELINE

### A. Huấn luyện thành công Mạng phân loại Chữ số Đếm ngược
* **Kiến trúc:** PyTorch 3-layer Convolutional Neural Network ($<1\text{MB}$, $\approx 150.000$ tham số).
* **Kết quả:** Đạt **$90.00\%$ Accuracy** trên $380$ ảnh test unseen.
* **Trọng số đã lưu:** `runs/classify/countdown_digits_baseline/best_digit_classifier.pth`.

### B. Xây dựng Module Định vị & Ghép số Đếm ngược (`src/digit_localizer.py`)
* Không coi bước tách số là cho sẵn. Module tự động:
  1. Dùng Otsu Thresholding trích xuất Contour từng thanh LED.
  2. Sắp xếp Bounding Box từ trái sang phải (*Left-to-Right Sorting*).
  3. Gửi từng Sub-crop qua CNN $\longrightarrow$ Ghép thành chuỗi số nguyên vẹn (ví dụ: `2` và `7` $\to$ `"27"`).

### C. Module Theo dõi Chuỗi thời gian (`src/temporal_tracker.py`)
* Áp dụng thuật toán Temporal Majority Voting qua $10$ frame liên tiếp để triệt tiêu hiện tượng đèn bị nhấp nháy do nhiễu camera.
* Tự động ước lượng và đếm lùi thời gian chuyển màu.

### D. Pipeline Tích hợp Thời gian thực & HUD Lái xe (`src/pipeline.py`)
* Kết hợp đồng thời: Biển báo + Đèn tín hiệu + Đọc số giây + Vẽ giao diện HUD chuyên nghiệp ở góc màn hình.

---

## 5. 🚀 GIAI ĐOẠN 4: MLOPS, DOCKER & GITHUB SHOWCASE

1. **FastAPI REST Service (`src/app.py`):**
   * Endpoint `/predict`: Trả về kết quả JSON có cấu trúc.
   * Endpoint `/predict/visualize`: Trả về luồng ảnh/video có vẽ HUD cảnh báo.
   * Endpoint `/health`: Kiểm tra trạng thái hệ thống và 3 model.
2. **Containerization (`Dockerfile` & `requirements.txt`):**
   * Đóng gói toàn bộ runtime Python 3.11, PyTorch, OpenCV, Ultralytics sẵn sàng deploy lên Cloud/AWS.
3. **GitHub Documentation (`README.md`):**
   * Viết tài liệu chuẩn quốc tế kèm sơ đồ kiến trúc, hướng dẫn cài đặt 1 chạm và bộ câu hỏi phỏng vấn Machine Learning Engineer.
4. **Git Version Control:**
   * Đã khởi tạo Git repository, cấu hình `.gitignore` chuẩn lọc file rác và hoàn tất initial commit sạch sẽ.
