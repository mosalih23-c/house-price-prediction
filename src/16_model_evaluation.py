import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. Load model and dataset
# =========================================================

model = joblib.load(
    "models/house_price_model.pkl"
)

df = pd.read_csv("data/train.csv")


# =========================================================
# 2. Separate features and target
# =========================================================

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]


# =========================================================
# 3. Same train/test split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# 4. Prediction
# =========================================================

y_pred = model.predict(X_test)


# =========================================================
# 5. Evaluation metrics
# =========================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\n===== Final Model Evaluation =====")

print(f"MAE:  {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R²:   {r2:.4f}")


# =========================================================
# 6. Create results directory
# =========================================================

os.makedirs("results", exist_ok=True)


# =========================================================
# 7. Actual vs Predicted
# =========================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

# Perfect prediction line
min_price = min(y_test.min(), y_pred.min())
max_price = max(y_test.max(), y_pred.max())

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    linestyle="--"
)

plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted Sale Price")

plt.tight_layout()

plt.savefig(
    "results/actual_vs_predicted.png",
    dpi=300
)

plt.show()


# =========================================================
# 8. Residuals
# =========================================================

residuals = y_test - y_pred


plt.figure(figsize=(8, 6))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Sale Price")
plt.ylabel("Residual")
plt.title("Residual Analysis")

plt.tight_layout()

plt.savefig(
    "results/residuals.png",
    dpi=300
)

plt.show()


# =========================================================
# 9. Save evaluation results
# =========================================================

results = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R2"
    ],
    "Value": [
        mae,
        rmse,
        r2
    ]
})

results.to_csv(
    "results/model_metrics.csv",
    index=False
)

print("\nEvaluation files saved to results/")