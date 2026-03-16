# 🔍 Customer Churn Prediction with Explainable ML

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📌 Project Overview

An end-to-end machine learning pipeline that predicts customer churn using the **Telco Customer Churn** dataset. The project demonstrates production-level data science practices including feature engineering, model comparison, hyperparameter tuning, and model explainability using SHAP.

---

## 🧠 Models Compared

| Model | ROC-AUC |
|---|---|
| Logistic Regression | ~0.84 |
| Random Forest | ~0.87 |
| XGBoost (Tuned) | ~0.91 |

---

## 🛠️ Skills Demonstrated

- **Feature Engineering** – Encoding, scaling, derived features
- **Model Comparison** – LR vs RF vs XGBoost
- **Hyperparameter Tuning** – GridSearchCV / RandomizedSearchCV
- **Explainability** – SHAP summary, waterfall, and dependence plots
- **Evaluation** – ROC-AUC, Confusion Matrix, Precision/Recall, F1

---

## 📊 SHAP Explainability

SHAP (SHapley Additive exPlanations) is used to interpret the best model:
- **Summary Plot** – Global feature importance
- **Waterfall Plot** – Per-prediction explanation
- **Dependence Plot** – Feature interaction effects

---

## 🚀 Run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/customer-churn-prediction-xgboost/blob/main/notebooks/churn_prediction.ipynb)

> Replace `YOUR_USERNAME` with your GitHub username.

---

## 📂 Project Structure
```
├── data/                    # Dataset (auto-downloaded in notebook)
├── notebooks/               # Main Colab notebook
├── src/                     # Modular Python source code
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── explainability.py
├── outputs/
│   ├── plots/               # Saved SHAP and eval plots
│   └── models/              # Saved model artifacts
├── requirements.txt
└── README.md
```

---

## 📦 Requirements
```
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
seaborn
joblib
```

---

## 📬 Author

**Nayanshree Sanjay Menpale** — [LinkedIn](https://www.linkedin.com/in/nayanshree-ml/) | [GitHub](https://github.com/nayanshree42)
