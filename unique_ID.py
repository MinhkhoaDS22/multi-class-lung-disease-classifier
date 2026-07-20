import pandas as pd

df = pd.read_csv('vin_filtered_by_name.csv')
so_luong_id_duy_nhat = df['image_id'].nunique()

print(f"Số lượng ID ảnh duy nhất trong CSV: {so_luong_id_duy_nhat}")