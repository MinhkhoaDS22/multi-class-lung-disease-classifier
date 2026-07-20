import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import timm

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ (HYPERPARAMETERS)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Thư mục gốc của project
CSV_FILE = os.path.join(BASE_DIR, 'master_data.csv')   # File dữ liệu tổng
RESULT_DIR = os.path.join(BASE_DIR, 'result')          # Thư mục lưu kết quả
BATCH_SIZE = 32                    # Số ảnh trong 1 batch (Giảm xuống 16 nếu máy báo lỗi hết RAM/VRAM)
EPOCHS = 15                        # Số vòng huấn luyện
LEARNING_RATE = 1e-4               # Tốc độ học
IMG_SIZE = 224                     # Kích thước ảnh chuẩn cho DenseNet
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tạo thư mục lưu kết quả nếu chưa có
os.makedirs(RESULT_DIR, exist_ok=True)

# ==========================================
# 2. XÂY DỰNG DATASET & TIỀN XỬ LÝ
# ==========================================
class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform
        
        # Tạo từ điển ánh xạ Nhãn Chữ -> Nhãn Số
        self.classes = sorted(self.data['class_name'].unique())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['file_path']
        label_name = self.data.iloc[idx]['class_name']
        
        # Chuyển relative path thành absolute path
        if not os.path.isabs(img_path):
            img_path = os.path.join(BASE_DIR, img_path)
        
        # Mở ảnh và chuyển về RGB (đề phòng ảnh X-ray chỉ có 1 kênh màu)
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[label_name]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

# Data Augmentation cho tập Train (Giúp mô hình bền bỉ hơn)
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),         # Lật ngang ngẫu nhiên
    transforms.RandomRotation(10),             # Xoay ngẫu nhiên 10 độ
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Chỉ Resize và Normalize cho tập Validation
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. HÀM VẼ BIỂU ĐỒ (VISUALIZATION)
# ==========================================
def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))
    
    # Biểu đồ Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Đồ thị Mất mát (Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Biểu đồ Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Đồ thị Độ chính xác (Accuracy)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'training_curves.png'))
    plt.close()
    print("-> Đã lưu biểu đồ huấn luyện: training_curves.png")

def plot_confusion_matrix(true_labels, pred_labels, classes):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Ma trận nhầm lẫn (Confusion Matrix)')
    plt.ylabel('Nhãn thực tế (True)')
    plt.xlabel('Nhãn dự đoán (Predicted)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'confusion_matrix.png'))
    plt.close()
    print("-> Đã lưu ma trận nhầm lẫn: confusion_matrix.png")

# ==========================================
# 4. CHƯƠNG TRÌNH CHÍNH (MAIN PROCESS)
# ==========================================
def main():
    print(f"--- ĐANG CHẠY TRÊN THIẾT BỊ: {DEVICE} ---")
    
    # Đọc dữ liệu và chia tập Train/Val (Tỷ lệ 80/20)
    df_full = pd.read_csv(CSV_FILE)
    df_train, df_val = train_test_split(df_full, test_size=0.2, random_state=42, stratify=df_full['class_name'])
    
    print(f"Tổng số ảnh Train: {len(df_train)} | Val: {len(df_val)}")
    
    # Khởi tạo Dataset và DataLoader
    train_dataset = ChestXrayDataset(df_train, transform=train_transforms)
    val_dataset = ChestXrayDataset(df_val, transform=val_transforms)
    
    classes = train_dataset.classes
    num_classes = len(classes)
    print(f"Danh sách bệnh ({num_classes} loại): {classes}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Khởi tạo Model DenseNet-121
    model = timm.create_model('densenet121', pretrained=True, num_classes=num_classes)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Lưu trữ lịch sử huấn luyện
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    
    print("\n--- BẮT ĐẦU HUẤN LUYỆN ---")
    for epoch in range(EPOCHS):
        # ---------------- TRAINING ----------------
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        # Lưu lịch sử
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        
        # Lưu mô hình tốt nhất
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            save_path = os.path.join(RESULT_DIR, 'best_densenet_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  -> Mô hình cải thiện! Đã lưu tại: {save_path}")

    # ==========================================
    # 5. ĐÁNH GIÁ VÀ VẼ BIỂU ĐỒ SAU HUẤN LUYỆN
    # ==========================================
    print("\n--- HOÀN TẤT HUẤN LUYỆN, ĐANG TRÍCH XUẤT BÁO CÁO ---")
    plot_training_curves(history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'])
    
    # Load lại model tốt nhất để đánh giá trên tập Val
    model.load_state_dict(torch.load(os.path.join(RESULT_DIR, 'best_densenet_model.pth')))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Vẽ Confusion Matrix
    plot_confusion_matrix(all_labels, all_preds, classes)
    
    # Lưu báo cáo dạng text (Precision, Recall, F1-score)
    report = classification_report(all_labels, all_preds, target_names=classes)
    with open(os.path.join(RESULT_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)
    print("-> Đã lưu báo cáo chi tiết: classification_report.txt")
    print("\n[THÀNH CÔNG] Hãy kiểm tra thư mục 'result' để xem thành quả!")

if __name__ == '__main__':
    main()