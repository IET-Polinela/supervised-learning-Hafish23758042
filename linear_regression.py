import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load datasets
file_outlier = 'dataset_dengan_outlier.csv'
file_scaled = 'minmax_scaled_data.csv'

data_outlier = pd.read_csv(file_outlier)
data_scaled = pd.read_csv(file_scaled)

# Assume the last column is the target variable (y) and the rest are features (X)
X_outlier = data_outlier.iloc[:, :-1]
y_outlier = data_outlier.iloc[:, -1]
X_scaled = data_scaled.iloc[:, :-1]
y_scaled = data_scaled.iloc[:, -1]

# Split data into train and test sets
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_outlier, y_outlier, test_size=0.2, random_state=42)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Train Linear Regression models
model_o = LinearRegression()
model_s = LinearRegression()

model_o.fit(X_train_o, y_train_o)
model_s.fit(X_train_s, y_train_s)

# Predictions
y_pred_o = model_o.predict(X_test_o)
y_pred_s = model_s.predict(X_test_s)

# Calculate MSE and R2 score
mse_o = mean_squared_error(y_test_o, y_pred_o)
r2_o = r2_score(y_test_o, y_pred_o)

mse_s = mean_squared_error(y_test_s, y_pred_s)
r2_s = r2_score(y_test_s, y_pred_s)

print(f"Dataset dengan Outlier: MSE = {mse_o:.4f}, R2 Score = {r2_o:.4f}")
print(f"Dataset Tanpa Outlier & Scaled: MSE = {mse_s:.4f}, R2 Score = {r2_s:.4f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Scatter plot: Actual vs Predicted
sns.scatterplot(x=y_test_o, y=y_pred_o, ax=axes[0, 0], alpha=0.7)
axes[0, 0].set_title("Actual vs Predicted (Outlier)")
axes[0, 0].set_xlabel("Actual Values")
axes[0, 0].set_ylabel("Predicted Values")

sns.scatterplot(x=y_test_s, y=y_pred_s, ax=axes[1, 0], alpha=0.7)
axes[1, 0].set_title("Actual vs Predicted (Scaled)")
axes[1, 0].set_xlabel("Actual Values")
axes[1, 0].set_ylabel("Predicted Values")

# Residual plot
residuals_o = y_test_o - y_pred_o
residuals_s = y_test_s - y_pred_s
sns.scatterplot(x=y_pred_o, y=residuals_o, ax=axes[0, 1], alpha=0.7)
axes[0, 1].axhline(0, color='r', linestyle='--')
axes[0, 1].set_title("Residual Plot (Outlier)")
axes[0, 1].set_xlabel("Predicted Values")
axes[0, 1].set_ylabel("Residuals")

sns.scatterplot(x=y_pred_s, y=residuals_s, ax=axes[1, 1], alpha=0.7)
axes[1, 1].axhline(0, color='r', linestyle='--')
axes[1, 1].set_title("Residual Plot (Scaled)")
axes[1, 1].set_xlabel("Predicted Values")
axes[1, 1].set_ylabel("Residuals")

# Residual distribution plot
sns.histplot(residuals_o, kde=True, ax=axes[0, 2])
axes[0, 2].set_title("Residual Distribution (Outlier)")

sns.histplot(residuals_s, kde=True, ax=axes[1, 2])
axes[1, 2].set_title("Residual Distribution (Scaled)")

plt.tight_layout()
plt.savefig("linear_regression_visualization.png")  # Save the visualization
plt.show()

# Analysis of results
if r2_s > r2_o:
    print("Model dengan data yang telah diproses (tanpa outlier dan scaled) memiliki performa yang lebih baik.")
else:
    print("Model dengan outlier mungkin masih memiliki pengaruh signifikan pada performa.")

print("Visualisasi telah disimpan sebagai 'linear_regression_visualization.png'")
