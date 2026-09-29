import pandas as pd
import joblib


# =========================================================
# 1. Load saved model
# =========================================================

model = joblib.load(
    "models/house_price_model.pkl"
)


# =========================================================
# 2. Load dataset
# =========================================================

df = pd.read_csv("data/train.csv")


# =========================================================
# 3. Select one house
# =========================================================

house = df.drop("SalePrice", axis=1).iloc[[0]]


# =========================================================
# 4. Actual price
# =========================================================

actual_price = df["SalePrice"].iloc[0]


# =========================================================
# 5. Prediction
# =========================================================

predicted_price = model.predict(house)[0]


# =========================================================
# 6. Results
# =========================================================

print("\n===== House Price Prediction =====")

print(f"Actual Price:    {actual_price:,.2f}")
print(f"Predicted Price: {predicted_price:,.2f}")

error = abs(actual_price - predicted_price)

print(f"Absolute Error:  {error:,.2f}")