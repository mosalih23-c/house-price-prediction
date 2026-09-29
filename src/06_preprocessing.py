import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


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
# 7. Combine preprocessing
# =========================================================

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])


# =========================================================
# 8. Fit preprocessing ONLY on training data
# =========================================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


# =========================================================
# 9. Display results
# =========================================================

print("\n===== Preprocessing Results =====")

print(f"Original training shape: {X_train.shape}")
print(f"Original testing shape:  {X_test.shape}")

print(f"\nProcessed training shape: {X_train_processed.shape}")
print(f"Processed testing shape:  {X_test_processed.shape}")

print("\nPreprocessing completed successfully.")