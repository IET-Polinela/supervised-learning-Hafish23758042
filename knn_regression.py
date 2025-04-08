import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Fungsi untuk memproses satu dataset
def process_knn(file_path, dataset_name):
    # Baca data
    df = pd.read_csv(file_path)

    # Asumsi: kolom terakhir adalah target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # K yang akan dicoba
    k_values = [3, 5, 7]
    results = {}

    # Plot
    plt.figure(figsize=(15, 4))
    for idx, k in enumerate(k_values):
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluasi
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[k] = {'MSE': mse, 'R2': r2}

        # Visualisasi
        plt.subplot(1, 3, idx + 1)
        plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Aktual')
        plt.ylabel('Prediksi')
        plt.title(f'K={k}\nMSE={mse:.4f}, R2={r2:.4f}')
        plt.grid(True)

    plt.suptitle(f'Hasil KNN Regression - {dataset_name}', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return results

# Jalankan untuk setiap dataset
results_standard = process_knn('standard_scaled_data.csv', 'Standard Scaled')
results_minmax = process_knn('minmax_scaled_data_tanpa_outlier.csv', 'MinMax Scaled')
results_original = process_knn('dataset_tanpa_outlier.csv', 'Original (Tanpa Scaling)')

# Tampilkan hasil evaluasi
print("\n=== Hasil Evaluasi ===\n")
def print_results(results, name):
    print(f"Dataset: {name}")
    for k, vals in results.items():
        print(f"K = {k}: MSE = {vals['MSE']:.4f}, R2 = {vals['R2']:.4f}")
    print()

print_results(results_standard, "Standard Scaled")
print_results(results_minmax, "MinMax Scaled")
print_results(results_original, "Original (Tanpa Scaling)")
