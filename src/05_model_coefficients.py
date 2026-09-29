import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# =========================================================
# 1. Load dataset
# =========================================================

df = pd.read_csv("data/train.csv")


# =========================================================
# 2. Select features
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
# 3. Train / Test split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# 4. Train model
# =========================================================

model = LinearRegression()
model.fit(X_train, y_train)


# =========================================================
# 5. Get coefficients
# =========================================================

coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})


# Sort by absolute coefficient
coefficients["Absolute_Coefficient"] = (
    coefficients["Coefficient"].abs()
)

coefficients = coefficients.sort_values(
    by="Absolute_Coefficient",
    ascending=False
)


# =========================================================
# 6. Display results
# =========================================================

print("\n===== Model Coefficients =====")

print(
    coefficients[
        ["Feature", "Coefficient"]
    ].to_string(index=False)
)