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
### Categorical Features

Missing categorical values are handled using:

```text
Most Frequent Imputation
🤖 Models

Several regression models were evaluated.

1. Linear Regression - Baseline

The baseline model used GrLivArea as the only feature.

Metric	Score
MAE	38,341.20
RMSE	58,471.76
R²	0.5543
2. Linear Regression - Multiple Features

Using several important numerical features:

Metric	Score
MAE	24,949.94
RMSE	39,513.20
R²	0.7965
3. Linear Regression - Full Pipeline

Using all available features with preprocessing:

Metric	Score
MAE	20,485.66
RMSE	31,327.80
R²	0.8720
4. Random Forest Regressor

Configuration:

RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

Results:

Metric	Score
MAE	17,465.29
RMSE	28,554.98
R²	0.8937
5. XGBoost Regressor

Configuration:

XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

Results:

Metric	Score
MAE	15,351.84
RMSE	24,645.83
R²	0.9208
⚙️ Hyperparameter Tuning

RandomizedSearchCV with 5-fold cross-validation was used to tune the XGBoost model.

Best parameters:

n_estimators = 700
learning_rate = 0.03
max_depth = 3
subsample = 0.8
colsample_bytree = 0.9
min_child_weight = 1

Best cross-validation RMSE:

26,544.36

The tuned model was evaluated on the held-out test set:

Metric	Score
MAE	15,543.11
RMSE	24,752.16
R²	0.9201

The original XGBoost configuration achieved slightly better test-set performance, so it was retained as the final model.

📈 Final Model

The final model is an XGBoost regression pipeline containing:

Data
  ↓
Missing Value Imputation
  ↓
One-Hot Encoding
  ↓
XGBoost Regressor
  ↓
Predicted House Price

Final test performance:

Metric	Score
MAE	15,351.84
RMSE	24,645.83
R²	0.9208

The complete preprocessing and model pipeline is saved using joblib.

📊 Model Evaluation

The project includes:

Actual vs Predicted Price
Residual Analysis
SalePrice Distribution
Feature Relationship Visualizations
Model Performance Comparison

The evaluation showed that predictions are generally close to the actual prices, while larger errors are more noticeable for high-priced houses.

🖥️ Prediction Application

A simple Gradio interface was developed for interactive predictions.

The application allows users to enter:

Overall Quality
Living Area
Garage Capacity
Year Built
Full Bathrooms

and returns an estimated house price.

Run the application:

python src/15_app.py
📁 Project Structure
house-price-prediction/
│
├── data/
│   └── train.csv
│
├── models/
│   └── house_price_model.pkl
│
├── results/
│   ├── actual_vs_predicted.png
│   ├── grlivarea_vs_saleprice.png
│   ├── model_metrics.csv
│   ├── overallqual_vs_saleprice.png
│   ├── residuals.png
│   └── saleprice_distribution.png
│
├── src/
│   ├── 01_data_understanding.py
│   ├── 03_baseline_model.py
│   ├── 04_multiple_features.py
│   ├── 05_model_coefficients.py
│   ├── 06_preprocessing.py
│   ├── 07_linear_regression_pipeline.py
│   ├── 08_random_forest.py
│   ├── 09_xgboost.py
│   ├── 10_xgboost_tuning.py
│   ├── 11_evaluate_tuned_xgboost.py
│   ├── 12_save_final_model.py
│   ├── 13_predict.py
│   ├── 14_predict_new_house.py
│   ├── 15_app.py
│   └── 16_model_evaluation.py
│
├── .gitignore
├── README.md
└── requirements.txt
🛠️ Technologies
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
XGBoost
Joblib
Gradio
Git & GitHub
🎯 Key Learning Outcomes

Through this project, I practiced:

Exploratory Data Analysis
Missing Value Handling
Feature Analysis
Categorical Encoding
Machine Learning Pipelines
Regression Models
Random Forest
XGBoost
Hyperparameter Tuning
Cross-Validation
Model Evaluation
Model Persistence
Interactive Prediction
👨‍💻 Author

Mohamed Salih

Information Technology & Computer Science Graduate

GitHub:
https://github.com/mosalih23-c

