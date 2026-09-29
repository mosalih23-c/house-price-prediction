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
# 2. Select Features and Target
# =========================================================

features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "1stFlrSF",
    "YearBuilt",
    "YearRemodAdd",
    "FullBath",
    "TotRmsAbvGrd"
]

X = df[features]
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
# 4. Create and Train Model
# =========================================================

model = LinearRegression()

model.fit(X_train, y_train)


# =========================================================
# 5. Predictions
# =========================================================

y_pred = model.predict(X_test)


# =========================================================
# 6. Evaluation
# =========================================================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


# =========================================================
# 7. Results
# =========================================================

print("\n===== Multiple Features Model =====")

print(f"MAE:  {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R²:   {r2:.4f}")