import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. Load Data
# =========================================================

df = pd.read_csv("data/train.csv")


# =========================================================
# 2. Select Feature and Target
# =========================================================

X = df[["GrLivArea"]]
y = df["SalePrice"]


# =========================================================
# 3. Train / Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# 4. Create Model
# =========================================================

model = LinearRegression()


# =========================================================
# 5. Train Model
# =========================================================

model.fit(X_train, y_train)


# =========================================================
# 6. Make Predictions
# =========================================================

y_pred = model.predict(X_test)


# =========================================================
# 7. Evaluate Model
# =========================================================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


# =========================================================
# 8. Results
# =========================================================

print("\n===== Baseline Model =====")

print(f"MAE:  {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R²:   {r2:.4f}")