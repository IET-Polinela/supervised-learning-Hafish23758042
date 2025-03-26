import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('cleaned_train.csv')

# Encoding categorical features
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Save the encoded dataset
df.to_csv('encoded_dataset.csv', index=False)
print("Encoded dataset saved as 'encoded_dataset.csv'")

# Memisahkan fitur independen (X) dan target (Y)
X = df.drop(columns=['SalePrice'])  # Ganti 'target' dengan nama kolom target yang sesuai
Y = df['SalePrice']

# Membagi dataset menjadi training (80%) dan testing (20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Menampilkan informasi hasil pembagian data
print("Training data shape:", X_train.shape, Y_train.shape)
print("Testing data shape:", X_test.shape, Y_test.shape)
