# 📊 Customer Churn Prediction – End-to-End Data Science Project

## 🔍 Problem Statement

Customer churn is a critical business problem in subscription-based businesses. The goal of this project is to build an end-to-end machine learning system that predicts whether a customer is likely to churn and provides actionable business insights to reduce churn.

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* SHAP
* Matplotlib, Seaborn

---

## 📁 Dataset

Telco Customer Churn Dataset (IBM Sample Dataset)

Target Variable:

* `Churn` (Yes / No)

---

## 🔑 Project Workflow

### 1. Data Understanding & EDA

* Checked missing values, distributions, and class imbalance
* Baseline churn rate ≈ 26.5%
* Explored churn patterns across tenure, contract type, internet service, and charges

### 2. Feature Engineering

* Dropped identifier column (`customerID`)
* Created tenure bins (`tenure_group`)
* Aggregated binary service features into `num_services`
* Encoded categorical features using OneHotEncoder

### 3. Modeling & Evaluation

Built and compared multiple models:

* Logistic Regression (baseline & final model)
* Random Forest
* XGBoost (raw + tuned)

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

Model comparison showed that Logistic Regression achieved comparable ROC-AUC (~0.845) with much better interpretability.

---

## ⚖️ Business-Oriented Threshold Optimization

Instead of using the default 0.5 classification threshold:

* Defined business costs:

  * False Negative (missed churner) = ₹5000
  * False Positive (unnecessary offer) = ₹500

* Optimized threshold to minimize expected business loss

* Selected a lower threshold (0.15) to capture more churners

This improved churn capture while significantly reducing financial loss.

---

## 🔎 Explainability

### Logistic Regression

* Interpreted coefficients using odds ratios
* Identified key churn drivers:

  * Low tenure
  * Month-to-month contracts
  * Fiber optic internet
  * High monthly charges

### XGBoost + SHAP

* Used SHAP summary plots for global feature importance
* Used SHAP force plots for individual-level explanations

---

## 📈 Segment-Level Churn Analysis

Identified high-risk customer segments:

* Month-to-month contract customers
* Tenure < 12 months
* Fiber optic internet users
* High monthly charge customers

Proposed targeted retention strategies such as:

* Contract upgrades
* Discounts for high-risk segments
* Promoting auto-payment and long-term plans

---

## 🏆 Final Model Selection

* Final Model: **Logistic Regression**
* Final Threshold: Business-optimized threshold (not 0.5)
* ROC-AUC: ~0.845
* Recall: ~0.80

Selected for:

* Strong performance
* High interpretability
* Business usability

---

## 📌 Key Learnings

* Model selection should depend on data characteristics, not model complexity
* Business-driven thresholding is critical in churn problems
* Explainability is essential for real-world deployment

---

## 🚀 Future Improvements

* Hyperparameter tuning with cross-validation
* Incorporate customer lifetime value (CLV)
* Deploy as an API for real-time scoring
