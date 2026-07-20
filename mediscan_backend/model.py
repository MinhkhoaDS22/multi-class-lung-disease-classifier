import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import os

# ==========================================
# CẤU HÌNH MODEL
# ==========================================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result', 'best_model.pth')
IMG_SIZE = 256

# 5 lớp bệnh (đúng thứ tự alphabet như khi train)
CLASSES = ['COVID', 'Fibrosis', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

CLASS_INFO = {
    'COVID': {
        'name_vi': 'COVID-19',
        'description': 'Viêm phổi do virus SARS-CoV-2',
        'severity': 'danger',
        'color': '#FF4757',
    },
    'Fibrosis': {
        'name_vi': 'Xơ phổi',
        'description': 'Sẹo mô phổi (Pulmonary Fibrosis)',
        'severity': 'warning',
        'color': '#FFA502',
    },
    'Lung_Opacity': {
        'name_vi': 'Mờ phổi',
        'description': 'Tổn thương vùng phổi (Lung Opacity)',
        'severity': 'warning',
        'color': '#FF6B35',
    },
    'Normal': {
        'name_vi': 'Bình thường',
        'description': 'Phổi không có dấu hiệu bệnh lý',
        'severity': 'normal',
        'color': '#2ED573',
    },
    'Viral Pneumonia': {
        'name_vi': 'Viêm phổi virus',
        'description': 'Viêm phổi do virus thông thường',
        'severity': 'warning',
        'color': '#ECCC68',
    },
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# TIỀN XỬ LÝ ẢNH (khớp với lúc train TTA)
# ==========================================
inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Augmentation cho TTA (Test-Time Augmentation)
tta_transforms = [
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((int(IMG_SIZE * 1.1), int(IMG_SIZE * 1.1))),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
]


# ==========================================
# SINGLETON MODEL LOADER
# ==========================================
class MediScanModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self):
        if self._loaded:
            return
        print(f"[MediScan] Đang load model từ: {MODEL_PATH}")
        print(f"[MediScan] Thiết bị: {DEVICE}")

        # EfficientNet-B5 (khớp với file best_model.pth)
        self.model = timm.create_model(
            'efficientnet_b5',
            pretrained=False,
            num_classes=len(CLASSES)
        )

        state = torch.load(MODEL_PATH, map_location=DEVICE)
        self.model.load_state_dict(state)
        self.model.to(DEVICE)
        self.model.eval()

        self._loaded = True
        print("[MediScan] ✅ Model đã sẵn sàng!")

    def predict(self, image_bytes: bytes) -> dict:
        """Nhận bytes ảnh, trả về dict xác suất từng lớp bệnh."""
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # TTA: lấy trung bình softmax trên nhiều augmentation
        probs_list = []
        with torch.no_grad():
            for tfm in tta_transforms:
                tensor = tfm(image).unsqueeze(0).to(DEVICE)
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                probs_list.append(probs)

        avg_probs = np.mean(probs_list, axis=0)

        # Tạo kết quả
        results = []
        predicted_class = CLASSES[int(np.argmax(avg_probs))]

        for i, cls in enumerate(CLASSES):
            info = CLASS_INFO[cls]
            results.append({
                'class_key': cls,
                'name_vi': info['name_vi'],
                'description': info['description'],
                'severity': info['severity'],
                'color': info['color'],
                'probability': float(avg_probs[i]),
                'percentage': round(float(avg_probs[i]) * 100, 2),
            })

        # Sắp xếp giảm dần theo xác suất
        results.sort(key=lambda x: x['probability'], reverse=True)

        return {
            'predicted_class': predicted_class,
            'predicted_name_vi': CLASS_INFO[predicted_class]['name_vi'],
            'predicted_severity': CLASS_INFO[predicted_class]['severity'],
            'confidence': round(float(np.max(avg_probs)) * 100, 2),
            'results': results,
            'device': str(DEVICE),
        }


# Global instance
mediscan_model = MediScanModel()
