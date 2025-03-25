import pandas as pd

# Load dataset
file_path = "train.csv"
df = pd.read_csv(file_path)

# Menghitung statistik deskriptif
stats = df.describe().T  # Transpose agar lebih mudah dibaca
stats["median"] = df.median(numeric_only=True)

# Menampilkan informasi statistik
stats = stats[["count", "mean", "median", "std", "min", "25%", "50%", "75%", "max"]]
print(stats)

# Mengecek jumlah nilai yang hilang di setiap kolom
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_data = pd.DataFrame({"Missing Values": missing_values, "Percentage": missing_percentage})
missing_data = missing_data[missing_data["Missing Values"] > 0]
print("\nJumlah nilai yang hilang di setiap kolom:")
print(missing_data)

# Strategi penanganan nilai yang hilang
# - Jika lebih dari 50% data hilang, pertimbangkan untuk menghapus kolom
columns_to_drop = missing_data[missing_data["Percentage"] > 50].index.tolist()
df = df.drop(columns=columns_to_drop)
print("\nKolom yang dihapus karena terlalu banyak nilai yang hilang:", columns_to_drop)

# - Mengisi nilai yang hilang dengan median untuk kolom numerik yang masih ada
target_columns = [col for col in missing_data.index if col in df.columns and df[col].dtype in ['int64', 'float64']]
df[target_columns] = df[target_columns].apply(lambda x: x.fillna(x.median()))

# - Mengisi nilai yang hilang untuk kolom kategorikal dengan modus
categorical_cols = [col for col in missing_data.index if col in df.columns and df[col].dtype == 'object']
df[categorical_cols] = df[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))

print("\nPenanganan nilai yang hilang selesai.")

# Simpan data yang sudah dibersihkan
cleaned_file_path = "cleaned_train.csv"
df.to_csv(cleaned_file_path, index=False)
print(f"\nData yang telah dibersihkan disimpan di: {cleaned_file_path}")
