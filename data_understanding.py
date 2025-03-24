import pandas as pd

# Load dataset
file_path = "dataset_encoded.csv"
df = pd.read_csv(file_path)

# Hitung statistik deskriptif
stats = df.describe(include='all').T  # Transpose agar lebih mudah dibaca

# Tambahkan median secara manual karena tidak ada dalam describe()
if df.select_dtypes(include=['number']).shape[1] > 0:  # Pastikan hanya kolom numerik
    stats['median'] = df.median()

# Tampilkan hasil statistik
target_cols = ['count', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max']
print(stats[target_cols])

# Hitung jumlah dan persentase nilai yang hilang di setiap kolom
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

# Gabungkan hasil dalam DataFrame
missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
missing_data = missing_data[missing_data['Missing Values'] > 0]  # Tampilkan hanya kolom yang memiliki nilai hilang

print("\nNilai yang hilang dalam dataset:")
print(missing_data)

# Jika ada kolom dengan nilai hilang lebih dari 50%, hapus kolom tersebut
threshold = 50  # Persentase batas penghapusan
cols_to_drop = missing_data[missing_data['Percentage'] > threshold].index.tolist()

if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"\nFitur yang dihapus karena terlalu banyak nilai yang hilang (> {threshold}%): {cols_to_drop}")

# Strategi penanganan nilai yang hilang:
# 1. Isi nilai yang hilang untuk fitur numerik dengan median
numerical_cols = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
for col in numerical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# 2. Isi nilai yang hilang untuk fitur kategori berdasarkan kasus spesifik
categorical_fill_none = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                         'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in categorical_fill_none:
    if col in df.columns:
        df[col] = df[col].astype(str).fillna("None")

# 3. Isi nilai yang hilang dengan mode untuk fitur kategori lainnya
for col in df.select_dtypes(include=['object']).columns:
    if df[col].isnull().sum() > 0:  # Hanya jika masih ada nilai hilang
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nDataset telah dibersihkan dari nilai yang hilang.")

# Simpan dataset yang sudah dibersihkan
df.to_csv("dataset_cleaned.csv", index=False)
print("Dataset yang telah dibersihkan disimpan sebagai 'dataset_cleaned.csv'.")
