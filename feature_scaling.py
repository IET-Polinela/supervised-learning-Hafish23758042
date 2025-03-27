import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load dataset
dataset_path = 'dataset_tanpa_outlier.csv'
df = pd.read_csv(dataset_path)

# Pilih hanya kolom numerik untuk scaling
numerical_cols = df.select_dtypes(include=['number']).columns
data = df[numerical_cols]

# Inisialisasi scaler
standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

# Transformasi data
standard_scaled_data = standard_scaler.fit_transform(data)
minmax_scaled_data = minmax_scaler.fit_transform(data)

# Konversi kembali ke DataFrame
standard_scaled_df = pd.DataFrame(standard_scaled_data, columns=numerical_cols)
minmax_scaled_df = pd.DataFrame(minmax_scaled_data, columns=numerical_cols)

# Simpan hasil scaling ke file CSV
standard_scaled_df.to_csv('standard_scaled_data.csv', index=False)
minmax_scaled_df.to_csv('minmax_scaled_data.csv', index=False)

print("Dataset hasil scaling telah disimpan sebagai 'standard_scaled_data.csv' dan 'minmax_scaled_data.csv'.")

# Tampilkan statistik deskriptif sebelum dan sesudah scaling
print("Original Data Statistics:\n", data.describe())
print("\nStandardScaler Data Statistics:\n", standard_scaled_df.describe())
print("\nMinMaxScaler Data Statistics:\n", minmax_scaled_df.describe())

# Plot distribusi sebelum dan sesudah scaling
fig, axes = plt.subplots(len(numerical_cols), 3, figsize=(15, 5 * len(numerical_cols)))

for i, col in enumerate(numerical_cols):
    # Sebelum scaling
    axes[i, 0].hist(data[col], bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[i, 0].set_title(f'Original {col}')
    axes[i, 0].set_xlabel(col)
    axes[i, 0].set_ylabel('Frequency')
    axes[i, 0].grid(True)

    # StandardScaler
    axes[i, 1].hist(standard_scaled_df[col], bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[i, 1].set_title(f'StandardScaler {col}')
    axes[i, 1].set_xlabel(col)
    axes[i, 1].grid(True)

    # MinMaxScaler
    axes[i, 2].hist(minmax_scaled_df[col], bins=30, alpha=0.7, color='red', edgecolor='black')
    axes[i, 2].set_title(f'MinMaxScaler {col}')
    axes[i, 2].set_xlabel(col)
    axes[i, 2].grid(True)

plt.tight_layout()

# Simpan gambar
plt.savefig('scaling_visualization.png')
plt.show()
