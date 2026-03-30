# Diabetes Prediction — ML Analysis

End-to-end machine learning analysis for binary diabetes prediction using a dataset of 100,000 patient records. Compares Logistic Regression, Random Forest, and Gradient Boosting with full SHAP explainability.

---

## What it does

- **EDA** — distribution analysis, class imbalance check, duplicate removal, categorical encoding
- **Feature engineering** — two derived features: `bmi_age_ratio` and `glucose_hba1c_ratio`
- **Model comparison** — three models evaluated on the same train/test split with classification metrics
- **Hyperparameter tuning** — GridSearchCV with 3-fold CV on Random Forest and Gradient Boosting
- **Explainability** — SHAP TreeExplainer with summary plots, force plots, and dependence plots

---

## Dataset

`diabetes_prediction_dataset.csv` — 100,000 records with 9 features:

| Feature | Type | Notes |
|---|---|---|
| `gender` | categorical | Male/Female (Other removed) |
| `age` | numeric | |
| `hypertension` | binary | |
| `heart_disease` | binary | |
| `smoking_history` | categorical | Encoded as binary (ever smoked = 1) |
| `bmi` | numeric | |
| `HbA1c_level` | numeric | |
| `blood_glucose_level` | numeric | |
| `diabetes` | binary | Target variable (8.5% positive rate) |

---

## Models

| Model | Approach | Tuning |
|---|---|---|
| Logistic Regression | Baseline linear classifier | Default params |
| Random Forest | Ensemble — bagging | GridSearchCV: n_estimators, max_depth, min_samples_leaf |
| Gradient Boosting | Ensemble — boosting | GridSearchCV: n_estimators, max_depth, min_samples_leaf |

All models use a `sklearn.pipeline.Pipeline` with `ColumnTransformer` preprocessing — `StandardScaler` on numeric features, passthrough on binary features.

---

## Explainability (SHAP)

Applied to the best Gradient Boosting model:

- **Summary plot (bar)** — global feature importance ranking
- **Summary plot (beeswarm)** — feature impact distribution across all predictions
- **Force plot** — local explanation for individual predictions
- **Dependence plot** — interaction between BMI and its SHAP values

---

## Stack

`Python 3` `scikit-learn` `pandas` `numpy` `matplotlib` `seaborn` `shap` `XGBoost` `LightGBM`

---

## Quickstart

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn shap xgboost lightgbm kagglehub

# Open the notebook
jupyter notebook diabetis.ipynb
```

Run all cells in order. The dataset CSV is included in the repo — no external download required.

---

## Project structure

```
diabetis.ipynb                    # Main notebook — EDA, modeling, explainability
diabetes_prediction_dataset.csv   # 100,000 patient records
```
