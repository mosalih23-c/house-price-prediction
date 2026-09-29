import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    KFold
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


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
# 8. XGBoost
# =========================================================

xgb = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)


# =========================================================
# 9. Complete Pipeline
# =========================================================

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", xgb)
])


# =========================================================
# 10. Hyperparameter Search Space
# =========================================================

param_distributions = {
    "regressor__n_estimators": [300, 500, 700, 1000],
    "regressor__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "regressor__max_depth": [3, 4, 5, 6, 7],
    "regressor__min_child_weight": [1, 3, 5],
    "regressor__subsample": [0.7, 0.8, 0.9, 1.0],
    "regressor__colsample_bytree": [0.7, 0.8, 0.9, 1.0]
}


# =========================================================
# 11. Cross-Validation
# =========================================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# =========================================================
# 12. Randomized Search
# =========================================================

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=20,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)


# =========================================================
# 13. Train
# =========================================================

print("\nStarting Hyperparameter Tuning...")

search.fit(X_train, y_train)


# =========================================================
# 14. Best Parameters
# =========================================================

print("\n===== Best Parameters =====")

for parameter, value in search.best_params_.items():
    print(f"{parameter}: {value}")


# =========================================================
# 15. Best Cross-Validation Score
# =========================================================

best_cv_rmse = -search.best_score_

print("\n===== Best CV Score =====")
print(f"CV RMSE: {best_cv_rmse:,.2f}")