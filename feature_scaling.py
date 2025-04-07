import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

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

# Gabungkan semua nilai fitur menjadi satu array untuk masing-masing jenis data
original_values = data.values.flatten()
standard_values = standard_scaled_df.values.flatten()
minmax_values = minmax_scaled_df.values.flatten()

# Plot histogram gabungan
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Original
axes[0].hist(original_values, bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[0].set_title('Original Data Distribution')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')
axes[0].grid(True)

# StandardScaler
axes[1].hist(standard_values, bins=50, color='green', alpha=0.7, edgecolor='black')
axes[1].set_title('StandardScaler Data Distribution')
axes[1].set_xlabel('Value')
axes[1].grid(True)

# MinMaxScaler
axes[2].hist(minmax_values, bins=50, color='red', alpha=0.7, edgecolor='black')
axes[2].set_title('MinMaxScaler Data Distribution')
axes[2].set_xlabel('Value')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('scaling_summary_histograms.png')
plt.show()
