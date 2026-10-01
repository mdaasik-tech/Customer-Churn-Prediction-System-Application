# 📉 Customer Churn Prediction System

Machine Learning based customer churn prediction using **XGBoost**, with an interactive **Streamlit + Plotly** analytics dashboard.

**Author:** Mohammed Aasik
[LinkedIn](https://www.linkedin.com/in/mohammed-aasik-aspiring-machine-learning-engineer-257787433/) · [GitHub](https://github.com/mdaasik-tech)

---

## Overview
Telecom-style companies lose revenue when customers leave. This project trains an XGBoost classifier on customer
account data, predicts **who is likely to churn**, assigns a **Low / Medium / High risk level**, and presents everything
in a dashboard with a live single-customer predictor.

## Problem Statement
Given a customer's profile (contract, services, billing, tenure, charges), predict whether the customer will churn.

## Objective
- Build a reproducible XGBoost pipeline with leak-free preprocessing and hyperparameter tuning.
- Evaluate it honestly with classification metrics.
- Turn predictions into actionable risk levels for a retention team.

## Dataset
`customer_churn_prediction_dataset.csv` (included): 2,000 customers × 15 columns. The app loads it automatically;
you can also upload your own CSV from the sidebar.

| Role | Columns |
|---|---|
| Target | `churn` (Yes / No) |
| Identifier (dropped) | `customer_id` |
| Numerical | `senior_citizen`, `tenure_months`, `monthly_charges`, `total_charges` |
| Categorical | `gender`, `partner`, `dependents`, `internet_service`, `online_security`, `tech_support`, `contract_type`, `paperless_billing`, `payment_method` |

## Data Preprocessing
All steps are inside one scikit-learn `Pipeline`, fitted **only on the training split** (no data leakage):

- **Label Encoding** – `LabelEncoder` converts the target (No/Yes → 0/1).
- **Standard Scaling** – `StandardScaler` for numerical features.
- **One-Hot Encoding** – `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` for categorical features.
- Cleaning: column/text whitespace stripping, `inf` → NaN, numeric-string conversion, duplicate/empty-row removal.

## XGBoost
`XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42, n_jobs=-1)`

## Hyperparameter Tuning
`RandomizedSearchCV` — `n_iter=100`, `cv=5`, `scoring="accuracy"`, `random_state=42` over
`n_estimators`, `max_depth`, `learning_rate`, `min_child_weight`, `gamma`, `subsample`,
`colsample_bytree`, `reg_alpha`, `reg_lambda` (same search space as the notebook).
Train/test split: 80/20, `random_state=42`, stratified.

## Best Parameters
Shown **dynamically** in the app (Overview and Model Information tabs). Nothing is hardcoded.

## Evaluation Metrics
Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, ROC curve, precision-recall curve,
classification report, and a base-vs-tuned comparison. All computed on the held-out test set.

> The project's original notebook run reached about 0.60 accuracy and 0.667 ROC-AUC. The model has modest
> predictive power on this dataset, so treat probabilities as a ranking signal, not a guarantee.

## Churn Risk Logic (replaces PASS / FAIL)
| Churn probability | Risk level |
|---|---|
| ≥ 0.70 | High Risk |
| ≥ 0.40 | Medium Risk |
| < 0.40 | Low Risk |

Thresholds are adjustable in the sidebar.

## Streamlit Features
- Default dataset or CSV upload, with validation and friendly error messages
- Automatic target / identifier / feature-type detection and feature selection
- Sidebar controls: test size, random state, search iterations, risk thresholds
- 7 tabs: Overview · Dataset · Model Performance · Predictions · Customer Predictor · Feature Analysis · Model Information
- Dynamic prediction form generated from the trained model's features
- Feature importance mapped correctly from one-hot columns back to original columns
- Light/dark theme compatible (no hard-coded colours), cached training, CSV downloads

## Project Structure
```text
customer-churn-prediction/
├── app.py
├── customer_churn_prediction_dataset.csv
├── requirements.txt
└── README.md
```

## Installation
```bash
pip install -r requirements.txt
```

## Running the Application
```bash
streamlit run app.py
```
The first run trains the model (searching 100 parameter combinations takes roughly 1–3 minutes depending on your CPU).
Results are cached, so later reruns are instant. Lower **Search iterations** in the sidebar for faster experiments.

## Future Improvements
- Handle class imbalance (`scale_pos_weight`) and tune the decision threshold for recall
- Add SHAP explanations for individual customers
- Save and load the trained model with `joblib`
- Add richer features (usage trends, support tickets) – the biggest lever for better accuracy
- Deploy on Streamlit Community Cloud

## Author
**Mohammed Aasik** – Aspiring Machine Learning Engineer
[LinkedIn](https://www.linkedin.com/in/mohammed-aasik-aspiring-machine-learning-engineer-257787433/) · [GitHub](https://github.com/mdaasik-tech)
