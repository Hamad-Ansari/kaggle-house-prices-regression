# 🏠 House Prices – Advanced Regression Techniques

A complete end-to-end machine learning solution for the Kaggle competition  
**House Prices: Advanced Regression Techniques**.

This project focuses on robust feature engineering, advanced preprocessing,
and ensemble-based regression models to accurately predict house prices.

---

## 📌 Problem Statement
Predict the final sale price of residential homes in Ames, Iowa, using
79 explanatory variables describing almost every aspect of residential homes.

Evaluation metric: **RMSE (Root Mean Squared Error)** on log-transformed prices.

---

## 🧠 Approach

### 1. Exploratory Data Analysis
- Target distribution analysis
- Missing value patterns
- Feature interactions (e.g., `OverallQual × GrLivArea`)
- Skewness and outlier handling

### 2. Feature Engineering
- Property age & remodeling age
- Total square footage
- Aggregated bathroom features
- Quality score synthesis
- Rare category encoding
- Log transformation for skewed numerical features

### 3. Preprocessing Pipeline
- Numerical: median imputation + scaling
- Categorical: mode imputation + ordinal encoding
- Unified `ColumnTransformer` for train/test consistency

### 4. Models Used
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor

Cross-validation is used for reliable performance estimation.

---

## 📊 Results

| Model | CV RMSE (mean ± std) |
|------|----------------------|
| Random Forest | Competitive |
| Gradient Boosting | Strong |
| XGBoost | Excellent |
| LightGBM | Best Performance |

(Tree-based boosting models achieved the lowest RMSE.)

---

## 🛠️ Tech Stack
- Python
- NumPy, Pandas
- Scikit-learn
- XGBoost
- LightGBM
- Matplotlib, Seaborn

---

## 📁 Repository Structure
