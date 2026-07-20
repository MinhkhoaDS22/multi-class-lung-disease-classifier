import pandas as pd

# 1. Đọc file csv gốc
df = pd.read_csv('train.csv')

# 2. Định nghĩa danh sách các tên nhãn (class_name) mong muốn
# Bạn có thể thay đổi tên chính xác theo file của bạn
target_names = [
    'Lung Opacity',        # Tương ứng nhãn 7
    'Pulmonary fibrosis',   # Tương ứng nhãn 13
    'No finding'           # Tương ứng nhãn 14
]

# 3. Lọc dữ liệu dựa trên cột class_name
df_filtered = df[df['class_name'].isin(target_names)].copy()

# 4. Bước quan trọng: Loại bỏ các dòng trùng lặp để lấy danh sách ảnh duy nhất
# Vì 1 ảnh có thể có nhiều dòng nhãn giống nhau do nhiều bác sĩ đọc
df_unique_images = df_filtered.drop_duplicates(subset=['image_id', 'class_name'])

# 5. Kiểm tra số lượng ảnh thực tế bạn sẽ có
print("Số lượng ảnh duy nhất cho mỗi nhãn:")
print(df_unique_images['class_name'].value_counts())

# 6. Xuất ra file CSV mới
output_filename = 'vin_filtered_by_name.csv'
df_unique_images.to_csv(output_filename, index=False)

print(f"\nĐã tạo xong file: {output_filename}")