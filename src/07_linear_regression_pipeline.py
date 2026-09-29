import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. Load dataset
# =========================================================

df = pd.read_csv("data/train.csv")


# =========================================================
# 2. Separate features and target
# =========================================================

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]


# =========================================================
# 3. Identify feature types
# =========================================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


# =========================================================
# 4. Train / Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# 5. Numerical preprocessing
# =========================================================

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])


# =========================================================
# 6. Categorical preprocessing
# =========================================================

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# =========================================================
# 7. Preprocessor
# =========================================================

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])


# =========================================================
# 8. Complete ML Pipeline
# =========================================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])


# =========================================================
# 9. Train
# =========================================================

model.fit(X_train, y_train)


# =========================================================
# 10. Prediction
# =========================================================

y_pred = model.predict(X_test)


# =========================================================
# 11. Evaluation
# =========================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


# =========================================================
# 12. Results
# =========================================================

print("\n===== Linear Regression + Full Preprocessing =====")

print(f"MAE:  {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R²:   {r2:.4f}")