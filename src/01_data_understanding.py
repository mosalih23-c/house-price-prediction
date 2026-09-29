import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/train.csv")

print("Dataset loaded successfully!")
print(df.head())
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Statistics:")
print(df["SalePrice"].describe())
# Missing values analysis

missing = df.isnull().sum()

missing = missing[missing > 0].sort_values(ascending=False)

print("\nColumns with missing values:")
print(missing)

print("\nMissing percentage:")
missing_percentage = (missing / len(df)) * 100
print(missing_percentage)

import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# Target Analysis
# =========================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["SalePrice"],
    kde=True
)

plt.title("Distribution of Sale Prices")
plt.xlabel("Sale Price")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("results/saleprice_distribution.png")

plt.show()
# =========================================================
# Correlation with Target
# =========================================================

numeric_df = df.select_dtypes(include=np.number)

correlations = numeric_df.corr()["SalePrice"].sort_values(ascending=False)

print("\nCorrelation with SalePrice:")
print(correlations)
# =========================================================
# Relationship: OverallQual vs SalePrice
# =========================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="OverallQual",
    y="SalePrice"
)

plt.title("Overall Quality vs Sale Price")
plt.xlabel("Overall Quality")
plt.ylabel("Sale Price")

plt.tight_layout()

plt.savefig("results/overallqual_vs_saleprice.png")

plt.show()

# =========================================================
# Relationship: GrLivArea vs SalePrice
# =========================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="GrLivArea",
    y="SalePrice"
)

plt.title("Living Area vs Sale Price")
plt.xlabel("Above Ground Living Area (sq ft)")
plt.ylabel("Sale Price")

plt.tight_layout()

plt.savefig("results/grlivarea_vs_saleprice.png")

plt.show()