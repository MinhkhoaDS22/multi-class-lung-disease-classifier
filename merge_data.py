import pandas as pd
import os

# --- CẤU HÌNH ĐƯỜNG DẪN (HÃY KIỂM TRA KỸ CHỖ NÀY) ---
vin_csv_path = 'vin_filtered_by_name.csv'
vin_img_path = 'VIN/' 
# Đảm bảo đường dẫn này trỏ thẳng vào folder chứa các mục: COVID, Normal, ...
covid_base_path = 'COVID-19_Radiography_Dataset' 

# 1. Đọc dữ liệu Vin
df_vin = pd.read_csv(vin_csv_path)
df_vin['file_path'] = df_vin['image_id'].apply(lambda x: os.path.join(vin_img_path, f"{x}.png"))

# Chuẩn hóa nhãn Vin: 'No finding' -> 'Normal', 'Pulmonary fibrosis' -> 'Fibrosis'
# 'Lung Opacity' giữ nguyên hoặc đổi thành 'Lung_Opacity' cho khớp bộ COVID
name_map = {
    'No finding': 'Normal',
    'Pulmonary fibrosis': 'Fibrosis',
    'Lung Opacity': 'Lung_Opacity'
}
df_vin['class_name'] = df_vin['class_name'].map(name_map).fillna(df_vin['class_name'])
df_vin = df_vin[['file_path', 'class_name']]

# 2. Quét dữ liệu bộ COVID (Thêm lệnh print để kiểm tra)
covid_data = []
covid_folders = ['COVID', 'Normal', 'Viral Pneumonia', 'Lung_Opacity']

print(f"Đang tìm kiếm bộ COVID tại: {os.path.abspath(covid_base_path)}")

for folder in covid_folders:
    # COVID-19 Radiography Database thường có cấu trúc: Folder_Benh/images/anh.png
    img_dir = os.path.join(covid_base_path, folder, 'images')
    
    if os.path.exists(img_dir):
        files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))]
        print(f"--- Tìm thấy {len(files)} ảnh trong folder {folder}")
        for filename in files:
            covid_data.append({
                'file_path': os.path.join(img_dir, filename),
                'class_name': folder
            })
    else:
        print(f"!!! Cảnh báo: Không tìm thấy thư mục: {img_dir}")

df_covid = pd.DataFrame(covid_data)

# 3. Gộp và Lưu
if not df_covid.empty:
    df_final = pd.concat([df_vin, df_covid], ignore_index=True)
    df_final.to_csv('master_data.csv', index=False)
    print("\n--- HOÀN THÀNH ---")
    print(f"Tổng số ảnh: {len(df_final)}")
    print(df_final['class_name'].value_counts())
else:
    print("\n!!! LỖI: Không quét được ảnh nào từ bộ COVID. Hãy kiểm tra lại đường dẫn covid_base_path.")