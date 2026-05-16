# 🏦 Loan Approval Prediction — End-to-End ML Project

A complete, production-quality machine learning project built on the Loan Sanction dataset.  
Demonstrates the **full data science workflow** with zero data leakage, clean code, and reproducible results.

---

## 📁 Project Structure

```
loan_project/
├── loan_sanction_train.csv      # Training data (614 rows)
├── loan_sanction_test.csv       # Test data (367 rows)
├── loan_approval_ml.ipynb       # Main notebook (run top-to-bottom)
├── loan_predictions.csv         # Final predictions on test set
├── model_results.csv            # CV & validation scores for all models
└── plots/
    ├── 01_target_distribution.png
    ├── 02_categoricals_vs_target.png
    ├── 03_numeric_distributions.png
    ├── 04_correlation_heatmap.png
    ├── 05_model_comparison.png
    ├── 06_confusion_roc.png
    └── 07_feature_importance.png
```

---

## 🎯 Problem Statement

**Binary classification:** predict whether a loan application will be **approved (Y)** or **rejected (N)** based on applicant demographics, income, loan details, and credit history.

- **Training set:** 614 samples, 12 raw features
- **Test set:** 367 samples
- **Class balance:** ~69% Approved / 31% Rejected

---

## 🔬 Workflow

### 1 · Exploratory Data Analysis
- Missing value audit (Credit History, Self Employed, Loan Amount have gaps)
- Target class distribution — dataset is moderately imbalanced
- Categorical breakdown: credit history and education are strong visual separators
- Numeric distributions: income features are heavily right-skewed → fixed via log transform
- Correlation heatmap across all numeric features

### 2 · Feature Engineering *(leakage-safe — arithmetic only)*

| Feature | Formula | Why |
|---------|----------|-----|
| `TotalIncome` | `ApplicantIncome + CoapplicantIncome` | Household income |
| `Log_TotalIncome` | `log1p(TotalIncome)` | Fixes right skew |
| `Log_LoanAmount` | `log1p(LoanAmount)` | Fixes right skew |
| `EMI` | `LoanAmount / Loan_Amount_Term` | Monthly obligation |
| `LoanToIncome` | `LoanAmount / (TotalIncome + 1)` | Affordability ratio |
| `BalanceIncome` | `TotalIncome - EMI × 1000` | Residual income |

> All features use **arithmetic operations only** — no `mean`, `std`, or `median` — so they are safe to compute before the train/val split.

### 3 · Preprocessing (Zero Data Leakage)

The critical order is:

```
1. encode_labels()   ← deterministic mapping, no statistics
2. add_features()    ← arithmetic only, no statistics
3. train_test_split  ← split BEFORE any statistical fitting
4. fit_transform()   ← imputer + scaler fitted on X_train ONLY
5. transform()       ← X_val and X_test are only ever transformed
```

```python
preprocess = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])

X_train_p = preprocess.fit_transform(X_train)  # ← fit here only
X_val_p   = preprocess.transform(X_val)         # ← transform only
X_test_p  = preprocess.transform(X_test)        # ← transform only
```

### 4 · Model Comparison (5-Fold Stratified CV)

| Model | CV Accuracy | Val Accuracy | Val ROC-AUC |
|-------|------------|--------------|-------------|
| **Random Forest** | 0.7800 ± 0.0475 | 0.8374 | **0.8455** ✅ |
| SVM | 0.7942 ± 0.0422 | 0.8537 | 0.8449 |
| Logistic Regression | 0.7983 ± 0.0457 | 0.8537 | 0.8393 |
| Gradient Boosting | 0.7719 ± 0.0397 | 0.7886 | 0.8028 |
| Decision Tree | 0.7881 ± 0.0474 | 0.8455 | 0.7625 |

**Best model:** Random Forest (highest ROC-AUC = **0.8455**)

### 5 · Evaluation

```
              precision  recall  f1-score  support
   Rejected       0.74    0.74      0.74       38
   Approved       0.88    0.88      0.88       85
   accuracy                         0.84      123
```

**Top predictors (by feature importance):**
1. `Credit_History`
2. `Log_TotalIncome`
3. `LoanToIncome`
4. `Property_Area`
5. `Log_LoanAmount`

---

## 📊 Plots

| # | Plot | What it shows |
|---|------|--------------|
| 01 | Target Distribution | 69% approved — moderate class imbalance |
| 02 | Categoricals vs Target | Credit history is the strongest categorical separator |
| 03 | Numeric Distributions | Income skewness and outliers per class |
| 04 | Correlation Heatmap | Feature-to-feature and feature-to-target correlations |
| 05 | Model Comparison | CV vs validation accuracy + ROC-AUC across all 5 models |
| 06 | Confusion Matrix + ROC | Best model performance breakdown |
| 07 | Feature Importance | Random Forest importances — red = top quartile |

---

## ▶️ How to Run

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn notebook

# 2. Place data files in the same folder as the notebook
#    loan_sanction_train.csv
#    loan_sanction_test.csv

# 3. Launch
jupyter notebook loan_approval_ml.ipynb
```

Run all cells top-to-bottom. Plots are saved to `plots/` and predictions to `loan_predictions.csv`.

---


## 🧰 Tech Stack

`Python 3.x` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn`
