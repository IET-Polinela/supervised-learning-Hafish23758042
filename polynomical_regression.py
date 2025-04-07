import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load datasets
file_outlier = 'dataset_tanpa_outlier.csv'
file_minmax = 'minmax_scaled_data_tanpa_outlier.csv'
file_standard = 'standard_scaled_data.csv'

data_outlier = pd.read_csv(file_outlier)
data_minmax = pd.read_csv(file_minmax)
data_standard = pd.read_csv(file_standard)

# Drop kolom 'Id' dan pisahkan target
X_outlier = data_outlier.drop(columns=['Id', 'SalePrice'])
y_outlier = data_outlier['SalePrice']
X_minmax = data_minmax.drop(columns=['Id', 'SalePrice'])
y_minmax = data_minmax['SalePrice']
X_standard = data_standard.drop(columns=['Id', 'SalePrice'])
y_standard = data_standard['SalePrice']

# Split data
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train_o, X_test_o, y_train_o, y_test_o = split_data(X_outlier, y_outlier)
X_train_m, X_test_m, y_train_m, y_test_m = split_data(X_minmax, y_minmax)
X_train_s, X_test_s, y_train_s, y_test_s = split_data(X_standard, y_standard)

# Fungsi khusus untuk menangani degree 1.5
def custom_polynomial_transform(X, degree):
    if degree == 1.5:
        X = X.copy()
        sqrt_features = np.sqrt(np.abs(X))  # Hindari nilai negatif
        sqrt_features.columns = [f"{col}^0.5" for col in X.columns]
        X_combined = pd.concat([X, sqrt_features], axis=1)
        return X_combined.values
    else:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        return poly.fit_transform(X)

# Fungsi pelatihan dan evaluasi
def train_and_evaluate_polynomial(X_train, X_test, y_train, y_test, degree):
    X_poly_train = custom_polynomial_transform(X_train, degree)
    X_poly_test = custom_polynomial_transform(X_test, degree)

    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    y_pred = model.predict(X_poly_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        'degree': degree,
        'y_true': y_test,
        'y_pred': y_pred,
        'mse': mse,
        'r2': r2,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }

# Jalankan untuk degree 1, 1.5, dan 2
datasets = {
    'Tanpa Outlier': (X_train_o, X_test_o, y_train_o, y_test_o),
    'MinMax Scaled': (X_train_m, X_test_m, y_train_m, y_test_m),
    'Standard Scaled': (X_train_s, X_test_s, y_train_s, y_test_s)
}

results = {
    name: [train_and_evaluate_polynomial(X_train, X_test, y_train, y_test, d) for d in [1, 1.5, 2]]
    for name, (X_train, X_test, y_train, y_test) in datasets.items()
}

# Visualisasi
fig, axes = plt.subplots(3, 3, figsize=(18, 18))
for row, (name, result_list) in enumerate(results.items()):
    for col, result in enumerate(result_list):
        axes[row, col].scatter(result['y_true'], result['y_pred'], alpha=0.7, edgecolor='k')
        axes[row, col].plot([result['y_true'].min(), result['y_true'].max()],
                            [result['y_true'].min(), result['y_true'].max()], 'r--')
        axes[row, col].set_title(f"{name} - Degree {result['degree']}\nMSE: {result['mse']:.2f}, R2: {result['r2']:.2f}")
        axes[row, col].set_xlabel("Actual SalePrice")
        axes[row, col].set_ylabel("Predicted SalePrice")
        axes[row, col].grid(True)

plt.tight_layout()
plt.savefig("polynomial_regression_degree_1_1.5_2.png")
plt.show()

# Ringkasan hasil
summary_data = {}
for name, result_list in results.items():
    for result in result_list:
        summary_data[f"{name} - Degree {result['degree']}"] = {
            'MSE': result['mse'],
            'R2 Score': result['r2'],
            'Jumlah Data Training': result['train_size'],
            'Jumlah Data Testing': result['test_size']
        }

summary_df = pd.DataFrame(summary_data).T

print("Ringkasan Evaluasi Model Polynomial Regression:")
print(summary_df)
print("Visualisasi telah disimpan sebagai 'polynomial_regression_degree_1_1.5_2.png'")
