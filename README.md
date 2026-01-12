# 🏠 House Prices – Advanced Regression Techniques

A complete end-to-end machine learning solution for the Kaggle competition  
**House Prices: Advanced Regression Techniques**.

This project focuses on robust feature engineering, advanced preprocessing,
and ensemble-based regression models to accurately predict house prices.
<img width="1024" height="1536" alt="hous" src="https://github.com/user-attachments/assets/31333946-bbab-42e9-af5b-9047ab4658b8" />

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



│
├── data/
│   ├── raw/                 # Original Kaggle files (not uploaded if large)
│   ├── processed/           # Cleaned / engineered data
│   └── README.md
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_ensembling.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── utils.py
│
├── outputs/
│   ├── figures/
│   └── submissions/
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
👉 If you only have one notebook, place it in:

notebooks/House_Prices_Advanced_Regression.ipynb
That is perfectly acceptable.

3. README.md (Copy–Paste Ready)
README.md
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
notebooks/ → Jupyter notebooks (EDA → Modeling)
src/ → Modular Python code
outputs/ → Figures and Kaggle submissions


---

## 🚀 How to Run

```bash

pip install -r requirements.txt
Open the main notebook:
```
jupyter notebook notebooks/House_Prices_Advanced_Regression.ipynb
🏆 Kaggle Competition
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

👤 Author
Hammad Zahid
Data Scientist | Machine Learning Enthusiast

🔗 LinkedIn: https://www.linkedin.com/in/hammad-zahid-xyz
🐙 GitHub: https://github.com/Hamad-Ansari
