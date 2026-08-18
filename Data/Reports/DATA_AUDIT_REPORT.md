# 📊 BÁO CÁO KIỂM TOÁN DỮ LIỆU TOÀN DIỆN (DATA AUDIT REPORT)
**Thời gian kiểm toán:** `2026-08-18 03:17:41`  
**Dự án:** Nhận diện Biển báo & Đèn tín hiệu Giao thông Việt Nam (ADAS + MLOps)

---

## 1. 🪧 Bộ Dữ Liệu Biển Báo Giao Thông (Zalo Traffic Sign 2020)
- **Số lượng ảnh Train:** `4,500` ảnh
- **Số lượng ảnh Test (Public):** `586` ảnh
- **Tổng số Bounding Box (Annotations):** `11,000` objects
- **Ảnh có gán nhãn:** `4,500` ảnh | **Ảnh nền / Background:** `0` ảnh
- **Ảnh lỗi (Corrupted / 0-byte):** `0` ảnh
- **Nhãn lỗi (Invalid BBox):** `0` lỗi
- **Trùng lặp chính xác (Exact Duplicates):** `17` nhóm

### 📈 Phân bố các lớp biển báo (Class Distribution):
| Class ID | Tên Nhóm Biển Báo | Số Lượng Object | Tỷ Lệ |
| :---: | :--- | :---: | :---: |
| **6** | Nguy hiểm | **3,049** | 27.7% |
| **2** | Cấm dừng và đỗ | **2,221** | 20.2% |
| **5** | Cấm còn lại | **1,787** | 16.2% |
| **1** | Cấm ngược chiều | **1,416** | 12.9% |
| **7** | Hiệu lệnh | **1,022** | 9.3% |
| **4** | Giới hạn tốc độ | **949** | 8.6% |
| **3** | Cấm rẽ | **556** | 5.1% |

### 🔍 Phân tích kích thước biển báo:
- **Nhỏ (< 32px - Small/Far):** `8,329` objects (75.7%)
- **Vừa (32 - 96px - Medium):** `2,380` objects (21.6%)
- **Lớn (> 96px - Large):** `291` objects (2.6%)

---

## 2. 🚦 Bộ Dữ Liệu Đèn Giao Thông (Vietnam Traffic Light)
- **Tổng số ảnh:** `2,666` ảnh
  - Tập `train`: `2,513` ảnh (4,607 objects)
  - Tập `valid`: `103` ảnh (181 objects)
  - Tập `test`: `50` ảnh (95 objects)
- **Tổng số Bounding Box:** `4,883` objects
- **Ảnh lỗi (Corrupted / 0-byte):** `0` ảnh
- **Nhãn lỗi (Invalid BBox):** `23` lỗi
- **Trùng lặp chính xác:** `0` nhóm

### 📈 Phân bố các lớp đèn:
| Class ID | Trạng Thái Đèn | Số Lượng Object | Tỷ Lệ |
| :---: | :--- | :---: | :---: |
| **(1)** | Red | **2,612** | 53.5% |
| **(0)** | Green | **2,271** | 46.5% |

### 🔍 Phân tích kích thước đèn tín hiệu:
- **Nhỏ (< 32px):** `367` objects (7.5%)
- **Vừa (32 - 96px):** `3,050` objects (62.5%)
- **Lớn (> 96px):** `1,466` objects (30.0%)

---

## 3. ⏱️ Bộ Chữ Số Đếm Ngược (Countdown Timer Digits 0-9)
- **Tổng số ảnh:** `2,500` ảnh
- **Phân bố các chữ số:** Chuẩn $100\%$ (**250 ảnh/chữ số** từ `0` đến `9`).
- **Ảnh lỗi:** `0` ảnh

---

## 4. 🎯 KẾT LUẬN & HÀNH ĐỘNG TIẾP THEO (Action Plan)
1. ✅ **Chất lượng ảnh:** 100% mở được, không có file ảnh 0-byte hay corrupt.
2. 🔄 **Chuẩn hóa nhãn Zalo:** Chuyển đổi toàn bộ toạ độ pixel COCO sang chuẩn YOLO $0 	o 1$.
3. 🛡️ **Group Split:** Sử dụng `street_id` trong dataset Zalo để chia Train/Val/Test chống rò rỉ dữ liệu (*Data Leakage*).