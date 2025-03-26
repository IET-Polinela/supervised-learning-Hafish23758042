import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "cleaned_train.csv"
df = pd.read_csv(file_path)

# Memilih hanya kolom numerik
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns

# Visualisasi boxplot untuk semua fitur numerik
plt.figure(figsize=(20, 15))
for i, feature in enumerate(numerical_features[:12]):  # Menampilkan 12 fitur pertama
    plt.subplot(4, 3, i+1)
    sns.boxplot(y=df[feature])
    plt.title(feature)

plt.tight_layout()

# Menyimpan gambar
visualization_path = "boxplot_visualization.png"
plt.savefig(visualization_path, dpi=300, bbox_inches='tight')

# Menampilkan hasil visualisasi
plt.show()

print(f"Gambar visualisasi disimpan pada {visualization_path}")

# Menentukan outlier menggunakan metode IQR
def detect_outliers_iqr(data, feature):
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]

# Menentukan outlier menggunakan metode Z-score
def detect_outliers_zscore(data, feature, threshold=3):
    mean = data[feature].mean()
    std = data[feature].std()
    return data[(np.abs((data[feature] - mean) / std)) > threshold]

# Mengidentifikasi outlier pada seluruh fitur numerik
outliers_iqr = pd.DataFrame()
outliers_zscore = pd.DataFrame()

for feature in numerical_features:
    outliers_iqr = pd.concat([outliers_iqr, detect_outliers_iqr(df, feature)])
    outliers_zscore = pd.concat([outliers_zscore, detect_outliers_zscore(df, feature)])

# Menghapus duplikasi karena beberapa nilai bisa masuk dalam beberapa kategori outlier
outliers_iqr = outliers_iqr.drop_duplicates()
outliers_zscore = outliers_zscore.drop_duplicates()

# Dataset tanpa outlier
df_no_outliers = df.drop(outliers_iqr.index)

# Dataset hanya dengan outlier
df_only_outliers = df.loc[outliers_iqr.index]

# Menampilkan jumlah data dalam setiap dataset
print("Dataset Asli:", df.shape)
print("Dataset Tanpa Outlier:", df_no_outliers.shape)
print("Dataset Hanya Outlier:", df_only_outliers.shape)

# Menyimpan hasil ke CSV
df_no_outliers.to_csv("dataset_tanpa_outlier.csv", index=False)
df_only_outliers.to_csv("dataset_dengan_outlier.csv", index=False)

print("Dataset telah disimpan sebagai 'dataset_tanpa_outlier.csv' dan 'dataset_dengan_outlier.csv'")
