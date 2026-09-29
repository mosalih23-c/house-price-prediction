import pandas as pd
import joblib


# =========================================================
# 1. Load saved model
# =========================================================

model = joblib.load(
    "models/house_price_model.pkl"
)


# =========================================================
# 2. Load original dataset
# =========================================================
# نستخدمه فقط للحصول على أسماء الأعمدة
# وإنشاء input له نفس الـ schema

df = pd.read_csv("data/train.csv")

feature_columns = df.drop(
    "SalePrice",
    axis=1
).columns


# =========================================================
# 3. Create a new house
# =========================================================

new_house = pd.DataFrame(
    [df.drop("SalePrice", axis=1).iloc[0].to_dict()]
)


# =========================================================
# 4. Change some values
# =========================================================

new_house["OverallQual"] = 8
new_house["GrLivArea"] = 2000
new_house["GarageCars"] = 2
new_house["YearBuilt"] = 2010
new_house["FullBath"] = 2


# =========================================================
# 5. Make prediction
# =========================================================

predicted_price = model.predict(new_house)[0]


# =========================================================
# 6. Display result
# =========================================================

print("\n===== New House Prediction =====")

print(f"Predicted Price: {predicted_price:,.2f}")

print("\nHouse Information:")
print(f"Overall Quality: {new_house['OverallQual'].iloc[0]}")
print(f"Living Area:     {new_house['GrLivArea'].iloc[0]}")
print(f"Garage Cars:     {new_house['GarageCars'].iloc[0]}")
print(f"Year Built:      {new_house['YearBuilt'].iloc[0]}")
print(f"Full Bathrooms:  {new_house['FullBath'].iloc[0]}")