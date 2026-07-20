import os
import math
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import timm

# ══════════════════════════════════════════════════════════
# 1. CẤU HÌNH
# ══════════════════════════════════════════════════════════
CSV_FILE   = '/kaggle/input/datasets/khoajakilevi/datasetk/master_data_sampled.csv'
VIN_DIR    = '/kaggle/input/datasets/khoajakilevi/datasetk/VIN'
COVID_DIR  = '/kaggle/input/datasets/khoajakilevi/datasetk/COVID-19_Radiography_Dataset'
RESULT_DIR = '/kaggle/working/result'

# ── Model ──
MODEL_NAME = 'efficientnet_b5'
IMG_SIZE   = 256

# ── Training ──
BATCH_SIZE     = 32
EPOCHS         = 80
LR_BACKBONE    = 1e-4    # LR cho backbone (pretrained features)
LR_HEAD        = 3e-4    # LR cho classifier head (mới khởi tạo)
WEIGHT_DECAY   = 1e-4
LABEL_SMOOTH   = 0.1
WARMUP_EPOCHS  = 5       # Linear warmup
EARLY_STOP_PAT = 20
MIN_DELTA      = 1e-4

# ── MixUp ──
MIXUP_ALPHA = 0.4
MIXUP_PROB  = 0.5        # OFF trong WARMUP_EPOCHS đầu

# ── Cân bằng dataset: cắt TẤT CẢ class về ~1500 ──
# Không dùng WeightedSampler NOR class weights → không bị double-weight
# Chỉ cần dataset cân bằng + CrossEntropy thường là đủ
CAP_PER_CLASS = 1500     # Viral Pneumonia ~1345 → ít hơn cap → giữ nguyên

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(RESULT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════
# 2. CÂN BẰNG DATASET (cắt tất cả class về ≤ CAP_PER_CLASS)
# ══════════════════════════════════════════════════════════
def balance_dataset(df, cap=CAP_PER_CLASS, random_state=42):
    """
    Undersample mỗi class xuống còn tối đa `cap` mẫu.
    Viral Pneumonia (~1345) nhỏ hơn cap → giữ nguyên.
    Kết quả: dataset gần bằng nhau (~7000 mẫu, imbalance ratio < 1.2×).

    LƯU Ý: Không dùng WeightedRandomSampler vì dataset đã cân bằng.
             Không dùng class weight trong loss vì dataset đã cân bằng.
             Dùng cả hai sẽ gây double-weighting → gradient explosion → collapse.
    """
    parts = []
    print('\n[DATASET BALANCING]')
    for cls in sorted(df['class_name'].unique()):
        sub   = df[df['class_name'] == cls]
        n_ori = len(sub)
        if n_ori > cap:
            sub = sub.sample(cap, random_state=random_state)
        parts.append(sub)
        arrow = '✂' if n_ori > cap else '✓'
        print(f'  {cls:22s}: {n_ori:5d} → {len(sub):5d} {arrow}')

    balanced = pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f'\n  Tổng: {len(balanced)} mẫu | '
          f'Ratio max/min = {max(balanced["class_name"].value_counts()) / min(balanced["class_name"].value_counts()):.2f}×')
    return balanced


# ══════════════════════════════════════════════════════════
# 3. DATASET
# ══════════════════════════════════════════════════════════
def resolve_path(rel_path: str) -> str:
    rel_path = rel_path.replace('\\', '/')
    if rel_path.startswith('VIN/'):
        base = os.path.join(VIN_DIR, rel_path[4:])
    elif rel_path.startswith('COVID-19_Radiography_Dataset/'):
        base = os.path.join(COVID_DIR, rel_path[len('COVID-19_Radiography_Dataset/'):])
    else:
        base = rel_path
    if os.path.isfile(base):
        return base
    stem, _ = os.path.splitext(base)
    for ext in ('.jpg', '.png', '.jpeg', '.JPG', '.PNG'):
        cand = stem + ext
        if os.path.isfile(cand):
            return cand
    return base


class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, class_to_idx, transform=None):
        self.data         = dataframe.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform    = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row      = self.data.iloc[idx]
        img_path = resolve_path(row['file_path'])
        label    = self.class_to_idx[row['class_name']]
        image    = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


# ── Transforms ──
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.02),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.12)),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── TTA: 5 transforms ──
tta_tfs = [
    val_tf,
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((int(IMG_SIZE * 1.15), int(IMG_SIZE * 1.15))),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((int(IMG_SIZE * 0.88), int(IMG_SIZE * 0.88))),
        transforms.Pad(int(IMG_SIZE * 0.06)),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((int(IMG_SIZE * 1.15), int(IMG_SIZE * 1.15))),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
]


# ══════════════════════════════════════════════════════════
# 4. MIXUP
# ══════════════════════════════════════════════════════════
def mixup_data(x, y, alpha=0.4):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx   = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ══════════════════════════════════════════════════════════
# 5. UTILITIES
# ══════════════════════════════════════════════════════════
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4, path='best_model.pth'):
        self.patience   = patience
        self.min_delta  = min_delta
        self.path       = path
        self.counter    = 0
        self.best_score = -float('inf')
        self.triggered  = False

    def __call__(self, val_acc, model):
        if val_acc > self.best_score + self.min_delta:
            self.best_score = val_acc
            self.counter    = 0
            torch.save(model.state_dict(), self.path)
            print(f'  ✔ New best val_acc={val_acc:.4f} → saved model')
            return False
        else:
            self.counter += 1
            print(f'  – No improvement {self.counter}/{self.patience} (best={self.best_score:.4f})')
            if self.counter >= self.patience:
                self.triggered = True
                return True
            return False


def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, eta_min_ratio=0.01):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, k1, k2, title in zip(
        axes,
        ['train_loss', 'train_acc'],
        ['val_loss',   'val_acc'],
        ['Loss', 'Accuracy'],
    ):
        ax.plot(history[k1], label='Train')
        ax.plot(history[k2], label='Val')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'training_curves.png'))
    plt.close()


def plot_confusion_matrix(true_labels, pred_labels, classes):
    cm = confusion_matrix(true_labels, pred_labels)
    # Normalize để dễ đọc
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm,      annot=True, fmt='d',    cmap='Blues', ax=axes[0],
                xticklabels=classes, yticklabels=classes)
    axes[0].set_title('Confusion Matrix (counts)')
    sns.heatmap(cm_norm, annot=True, fmt='.2f',  cmap='Blues', ax=axes[1],
                xticklabels=classes, yticklabels=classes)
    axes[1].set_title('Confusion Matrix (normalized)')
    for ax in axes:
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'confusion_matrix.png'))
    plt.close()


# ══════════════════════════════════════════════════════════
# 6. TRAIN / VALIDATE
# ══════════════════════════════════════════════════════════
def run_epoch(model, loader, criterion, optimizer=None, scaler=None,
              is_train=True, use_mixup=False):
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    nan_batches = 0

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if is_train:
            do_mix = use_mixup and (random.random() < MIXUP_PROB)
            if do_mix:
                images, y_a, y_b, lam = mixup_data(images, labels, MIXUP_ALPHA)

            with autocast():
                outputs  = model(images)
                # CrossEntropy: cast logits về float32 để tránh NaN với AMP
                logits32 = outputs.float()
                if do_mix:
                    loss = mixup_criterion(criterion, logits32, y_a, y_b, lam)
                else:
                    loss = criterion(logits32, labels)

            if not torch.isfinite(loss):
                nan_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                _, preds = torch.max(logits32, 1)
            if do_mix:
                correct += (lam * (preds == y_a).sum().item()
                            + (1 - lam) * (preds == y_b).sum().item())
            else:
                correct += (preds == labels).sum().item()

        else:
            with torch.no_grad(), autocast():
                outputs  = model(images)
                logits32 = outputs.float()
                loss     = criterion(logits32, labels)
            _, preds = torch.max(logits32, 1)
            correct += (preds == labels).sum().item()

        total      += labels.size(0)
        total_loss += loss.item() * images.size(0)

    if nan_batches:
        print(f'  [WARN] {nan_batches} batches skipped (NaN loss)')
    if total == 0:
        return float('nan'), 0.0
    return total_loss / total, correct / total


# ══════════════════════════════════════════════════════════
# 7. INFERENCE VỚI TTA
# ══════════════════════════════════════════════════════════
def predict_with_tta(model, df_val, class_to_idx):
    model.eval()
    all_probs = None

    for i, tf in enumerate(tta_tfs):
        ds     = ChestXrayDataset(df_val, class_to_idx, transform=tf)
        loader = DataLoader(ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=2, pin_memory=True)
        probs_list = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(DEVICE)
                with autocast():
                    out = model(images)
                probs_list.append(F.softmax(out.float(), dim=1).cpu())
        probs     = torch.cat(probs_list, dim=0)
        all_probs = probs if all_probs is None else all_probs + probs
        print(f'  TTA {i+1}/{len(tta_tfs)} done')

    preds = all_probs.argmax(dim=1).numpy()

    # True labels
    ds_plain  = ChestXrayDataset(df_val, class_to_idx, transform=val_tf)
    loader_pl = DataLoader(ds_plain, batch_size=BATCH_SIZE * 2, shuffle=False,
                           num_workers=2, pin_memory=True)
    true_labels = []
    for _, lbs in loader_pl:
        true_labels.extend(lbs.numpy())

    return np.array(true_labels), preds


# ══════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════
def main():
    print('=' * 65)
    print(f'  Chest X-ray Classifier — {MODEL_NAME} {IMG_SIZE}px')
    print(f'  Device: {DEVICE}')
    print('=' * 65)

    # ── Dataset: đọc, cân bằng, chia train/val ──
    df_raw      = pd.read_csv(CSV_FILE)
    df_balanced = balance_dataset(df_raw, cap=CAP_PER_CLASS)

    df_train, df_val = train_test_split(
        df_balanced, test_size=0.2, random_state=42,
        stratify=df_balanced['class_name']
    )
    df_train = df_train.reset_index(drop=True)
    df_val   = df_val.reset_index(drop=True)

    all_classes  = sorted(df_balanced['class_name'].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
    num_classes  = len(all_classes)

    print(f'\nTrain: {len(df_train)} | Val: {len(df_val)} | Classes: {all_classes}')
    print(f'\nPhân phối train:')
    print(df_train['class_name'].value_counts().to_string())

    # ── DataLoaders (shuffle=True, KHÔNG dùng WeightedSampler) ──
    # Dataset đã cân bằng → không cần sampler hay class weights
    train_ds = ChestXrayDataset(df_train, class_to_idx, transform=train_tf)
    val_ds   = ChestXrayDataset(df_val,   class_to_idx, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── Model ──
    print(f'\n[MODEL] Loading {MODEL_NAME} pretrained ...')
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'  Params: {n_params:.1f}M')

    # ── Loss: CrossEntropy + LabelSmoothing (KHÔNG có class weights) ──
    # Class weights đã KHÔNG cần thiết vì dataset cân bằng rồi.
    # Dùng class weights khi dataset đã cân bằng + WeightedSampler = double-weight → collapse.
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler    = GradScaler()

    # ── Optimizer: 2 param groups (backbone / head) ──
    # Backbone: LR nhỏ để bảo toàn pretrained features
    # Head: LR lớn hơn vì mới khởi tạo ngẫu nhiên
    head_params     = list(model.classifier.parameters())
    head_ids        = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LR_BACKBONE},
        {'params': head_params,     'lr': LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = get_warmup_cosine_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)

    print(f'\nOptimizer:')
    print(f'  backbone: {sum(p.numel() for p in backbone_params)/1e6:.1f}M params | LR={LR_BACKBONE:.0e}')
    print(f'  head    : {sum(p.numel() for p in head_params)/1e6:.1f}M params | LR={LR_HEAD:.0e}')
    print(f'\nSchedule: warmup {WARMUP_EPOCHS} ep → cosine {EPOCHS - WARMUP_EPOCHS} ep')
    print(f'MixUp: OFF for first {WARMUP_EPOCHS} epochs, prob={MIXUP_PROB} after')

    best_model_path = os.path.join(RESULT_DIR, 'best_model.pth')
    early_stopper   = EarlyStopping(patience=EARLY_STOP_PAT, min_delta=MIN_DELTA,
                                    path=best_model_path)

    # ══════════════════════════════════════════════════════
    # TRAINING LOOP
    # ══════════════════════════════════════════════════════
    print('\n' + '=' * 65)
    print('  TRAINING')
    print('=' * 65)

    history = {k: [] for k in ['train_loss', 'val_loss', 'train_acc', 'val_acc']}

    for epoch in range(EPOCHS):
        use_mix   = epoch >= WARMUP_EPOCHS
        tag       = 'WARMUP' if epoch < WARMUP_EPOCHS else 'TRAIN '

        tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion,
            optimizer=optimizer, scaler=scaler,
            is_train=True, use_mixup=use_mix
        )
        vl_loss, vl_acc = run_epoch(
            model, val_loader, criterion, is_train=False
        )
        scheduler.step()

        for k, v in zip(['train_loss','val_loss','train_acc','val_acc'],
                        [tr_loss, vl_loss, tr_acc, vl_acc]):
            history[k].append(v)

        lr_bb  = optimizer.param_groups[0]['lr']
        lr_hd  = optimizer.param_groups[1]['lr']
        mix    = 'mix=ON ' if use_mix else 'mix=OFF'
        print(f'[{tag}] Ep {epoch+1:3d}/{EPOCHS} | {mix} | '
              f'LR bb={lr_bb:.1e} hd={lr_hd:.1e} | '
              f'Train {tr_loss:.4f}/{tr_acc:.4f} | Val {vl_loss:.4f}/{vl_acc:.4f}')

        if early_stopper(vl_acc, model):
            print(f'\nEarly stopping at epoch {epoch+1}.')
            break

    plot_training_curves(history)

    # ══════════════════════════════════════════════════════
    # EVALUATION với TTA × 5
    # ══════════════════════════════════════════════════════
    print(f'\n{"="*65}')
    print(f'  EVALUATION (TTA × {len(tta_tfs)})')
    print(f'{"="*65}')
    print(f'Loading best model (val_acc={early_stopper.best_score:.4f}) ...')
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    true_labels, pred_labels = predict_with_tta(model, df_val, class_to_idx)
    acc_tta = (true_labels == pred_labels).mean()

    print(f'\n✅ Accuracy (TTA×{len(tta_tfs)}): {acc_tta:.4f} ({acc_tta*100:.2f}%)')

    plot_confusion_matrix(true_labels, pred_labels, all_classes)

    report = classification_report(true_labels, pred_labels, target_names=all_classes, digits=4)
    print('\n[CLASSIFICATION REPORT]\n', report)

    with open(os.path.join(RESULT_DIR, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Model : {MODEL_NAME} {IMG_SIZE}px\n')
        f.write(f'Dataset: balanced cap={CAP_PER_CLASS}/class\n')
        f.write(f'Loss  : CrossEntropy + LabelSmooth={LABEL_SMOOTH} (NO class weights)\n')
        f.write(f'Accuracy (TTA×{len(tta_tfs)}): {acc_tta:.4f} ({acc_tta*100:.2f}%)\n\n')
        f.write(report)

    print(f"\n[DONE] Kết quả lưu tại '{RESULT_DIR}'")


if __name__ == '__main__':
    main()
