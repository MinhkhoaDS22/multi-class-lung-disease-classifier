# 🏥 MediScan AI

Ứng dụng web phân tích ảnh X-quang ngực bằng trí tuệ nhân tạo.

## 📊 Thông tin Model

| Thuộc tính | Giá trị |
|---|---|
| Architecture | EfficientNet-B5 |
| Input | 256×256 px |
| Accuracy | **91.76%** (TTA×3) |
| Classes | 5 loại bệnh |
| Dataset | VinBigData + COVID-19 Radiography |

### Các loại bệnh phân loại:
- 🔴 **COVID-19** – Viêm phổi do SARS-CoV-2
- 🟠 **Fibrosis** – Xơ phổi
- 🟡 **Lung Opacity** – Mờ phổi
- 🟢 **Normal** – Bình thường
- 🟡 **Viral Pneumonia** – Viêm phổi virus

---

## 🚀 Hướng dẫn chạy

### Bước 1: Khởi động Backend (FastAPI)

```bash
# Double-click file này:
start_backend.bat

# Hoặc chạy thủ công:
cd mediscan_backend
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Backend sẽ chạy tại: **http://localhost:8000**  
API Docs: **http://localhost:8000/docs**

### Bước 2: Khởi động Frontend (Flutter Web)

```bash
# Double-click file này:
start_frontend.bat

# Hoặc chạy thủ công:
cd mediscan_app
flutter run -d chrome --web-port 3000
```

Frontend sẽ chạy tại: **http://localhost:3000**

---

## 📁 Cấu trúc thư mục

```
TTTN/
├── result/
│   └── best_model.pth          ← File model AI (EfficientNet-B5)
├── mediscan_backend/
│   ├── main.py                 ← FastAPI server
│   ├── model.py                ← Model loader & inference
│   └── requirements.txt        ← Python dependencies
├── mediscan_app/
│   ├── lib/
│   │   ├── main.dart           ← Entry point Flutter
│   │   └── screens/
│   │       └── home_screen.dart ← Giao diện chính
│   └── web/
│       └── index.html          ← HTML template
├── start_backend.bat           ← Script chạy backend
└── start_frontend.bat          ← Script chạy frontend
```

---

## ⚙️ Yêu cầu hệ thống

### Backend
- Python ≥ 3.9
- PyTorch ≥ 2.0 (CPU hoặc CUDA)
- VRAM ≥ 4GB (nếu dùng GPU), hoặc RAM ≥ 8GB (CPU)

### Frontend
- Flutter ≥ 3.x
- Chrome browser

---

## 🔌 API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/` | Kiểm tra server |
| GET | `/health` | Trạng thái model |
| POST | `/predict` | Phân tích ảnh X-quang |

### Ví dụ gọi API:
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@xray.jpg"
```

### Response mẫu:
```json
{
  "success": true,
  "predicted_class": "Normal",
  "predicted_name_vi": "Bình thường",
  "confidence": 95.12,
  "results": [
    {"class_key": "Normal", "name_vi": "Bình thường", "percentage": 95.12, ...},
    {"class_key": "COVID", "name_vi": "COVID-19", "percentage": 1.23, ...}
  ]
}
```
