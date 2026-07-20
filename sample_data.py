import pandas as pd

# --- CONFIG ---
INPUT_CSV    = 'master_data.csv'        # file CSV goc (dau vao)
OUTPUT_CSV   = 'master_data_sampled.csv'  # file CSV moi (dau ra)
NORMAL_SAMPLE = 9000                    # so anh Normal muon lay
RANDOM_SEED   = 42                      # seed de ket qua tai lap duoc

# --- DOC DU LIEU GOC ---
df = pd.read_csv(INPUT_CSV)

print("=== Class distribution in original file ===")
print(df['class_name'].value_counts())
print(f"Total: {len(df)}\n")

# --- LAY MAU ---
# Tach Normal ra, lay ngau nhien NORMAL_SAMPLE anh
df_normal = df[df['class_name'] == 'Normal'].sample(
    n=NORMAL_SAMPLE, random_state=RANDOM_SEED
)

# Giu nguyen tat ca cac lop benh con lai
df_others = df[df['class_name'] != 'Normal']

# Gop lai
df_sampled = pd.concat([df_normal, df_others], ignore_index=True)

# Shuffle toan bo de tranh du lieu bi sap xep theo thu tu lop
df_sampled = df_sampled.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# --- LUU CSV MOI ---
df_sampled.to_csv(OUTPUT_CSV, index=False)

print("=== Class distribution in new sampled file ===")
print(df_sampled['class_name'].value_counts())
print(f"Total: {len(df_sampled)}")
print(f"\nDone! Saved to: {OUTPUT_CSV}")
