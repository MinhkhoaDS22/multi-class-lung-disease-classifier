import pandas as pd
import os
import shutil

# 1. Cấu hình đường dẫn
csv_path = 'vin_filtered_by_name.csv'  # File CSV đã lọc ở Bước 2
source_folder = 'VinBigData Chest X-ray Resized JPG (512x512)/train' # Đường dẫn đến folder 15,000 ảnh 512px
destination_folder = 'VIN' # Folder đích bạn muốn lưu ảnh để làm đồ án

# 2. Tạo folder VIN nếu chưa có
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
    print(f"--- Đã tạo thư mục: {destination_folder} ---")

# 3. Đọc danh sách ID từ file CSV
df = pd.read_csv(csv_path)
# Lấy danh sách ID duy nhất (tránh copy trùng 1 ảnh nhiều lần)
unique_ids = df['image_id'].unique()

print(f"Bắt đầu copy {len(unique_ids)} ảnh...")

# 4. Vòng lặp copy file
count = 0
for img_id in unique_ids:
    # Bạn kiểm tra đuôi file là .png hay .jpg để sửa cho đúng nhé (thường là .png)
    file_name = f"{img_id}.jpg" 
    source_path = os.path.join(source_folder, file_name)
    dest_path = os.path.join(destination_folder, file_name)
    
    # Kiểm tra nếu file tồn tại thì mới copy
    if os.path.exists(source_path):
        shutil.copy(source_path, dest_path)
        count += 1
        if count % 500 == 0:
            print(f"Đã copy xong {count} ảnh...")
    else:
        print(f"Cảnh báo: Không tìm thấy ảnh {file_name}")

print(f"--- HOÀN THÀNH ---")
print(f"Tổng cộng đã copy {count} ảnh vào folder '{destination_folder}'")