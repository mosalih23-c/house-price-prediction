# 🏠 House Price Prediction

Machine Learning project for predicting house prices using the Kaggle House Prices dataset.

The project covers the complete machine learning workflow, including data understanding, exploratory data analysis, preprocessing, model training, model comparison, hyperparameter tuning, evaluation, and prediction.

---

## 📌 Project Overview

The goal of this project is to build a regression model that predicts the sale price of a house based on its characteristics.

The project starts with simple Linear Regression and gradually moves to more advanced ensemble models, including Random Forest and XGBoost.

---

## 📊 Dataset

Dataset: **House Prices - Advanced Regression Techniques**

- Rows: 1,460
- Original columns: 81
- Target variable: `SalePrice`

The dataset contains information about:

- Overall house quality
- Living area
- Basement area
- Garage information
- Number of rooms
- Bathrooms
- Year built
- Year remodeled
- Neighborhood
- Building materials
- And other property characteristics

---

## 🔎 Exploratory Data Analysis

The exploratory analysis includes:

- Dataset structure and data types
- Missing value analysis
- Descriptive statistics
- Target variable distribution
- Correlation analysis
- Relationship between important features and house prices

### Important Features

Some of the strongest numerical relationships with `SalePrice` were:

| Feature | Correlation |
|---|---:|
| OverallQual | 0.791 |
| GrLivArea | 0.709 |
| GarageCars | 0.640 |
| GarageArea | 0.623 |
| TotalBsmtSF | 0.614 |
| 1stFlrSF | 0.606 |
| FullBath | 0.561 |
| TotRmsAbvGrd | 0.534 |
| YearBuilt | 0.523 |
| YearRemodAdd | 0.507 |

---

## 🧹 Data Preprocessing

The preprocessing pipeline handles both numerical and categorical variables.

### Numerical Features

Missing numerical values are handled using:

```text
Median Imputation
