---

# 💳 Credit Risk Prediction System

A **Machine Learning project** that predicts whether a loan applicant is **Low Risk** or **High Risk** using structured financial and demographic data.

Built to be **recruiter‑friendly**, **impact‑driven**, and **GitHub‑ready** — with clean code, explainability, and real‑world relevance.

---






🚀 **Live Streamlit App**: [https://your-streamlit-app-link-here](https://your-streamlit-app-link-here)

---

## 📊 At-a-Glance Impact

| Metric          | Value                                                  |
| --------------- | ------------------------------------------------------ |
| Dataset Size    | **1,000+ Credit Applications**                         |
| Models Trained  | **Decision Tree, Random Forest, Extra Trees, XGBoost** |
| Best Accuracy   | **85%+ (Balanced Classes)**                            |
| Target Variable | **Credit Risk (Good / Bad)**                           |
| Business Goal   | **Reduce Default Risk & Improve Loan Decisions**       |



## 🚀 Why This Project Matters

Banks and financial institutions lose **billions of dollars annually** due to loan defaults.

> 🔴 Even a **1% improvement** in credit risk prediction can save **millions** in bad loans.

This project simulates a **real banking credit approval system**, applying machine learning to:

* Reduce default risk
* Improve approval accuracy
* Enable data‑driven lending decisions

---

## 📊 Dataset Overview

* **Customers:** ~1,000 loan applicants
* **Features:** Demographic + Financial attributes
* **Target:** Credit Risk (`Good` / `Bad`)

### Key Features Used

| Feature          | Description            |
| ---------------- | ---------------------- |
| Age              | Applicant age          |
| Sex              | Gender                 |
| Job              | Employment category    |
| Housing          | Own / Rent / Free      |
| Saving Accounts  | Savings status         |
| Checking Account | Checking balance       |
| Credit Amount    | Loan amount requested  |
| Duration         | Loan duration (months) |
| Purpose          | Loan purpose           |

---

## 🧠 Machine Learning Models

Multiple models were trained and evaluated:

| Model         | Strength                |
| ------------- | ----------------------- |
| Decision Tree | Interpretability        |
| Random Forest | Stability & performance |
| Extra Trees   | Reduced variance        |
| XGBoost       | High predictive power   |

> ⚡ **XGBoost** delivered the best overall performance.

---

## 📈 Results & Impact

| Metric             | Score                        |
| ------------------ | ---------------------------- |
| Accuracy           | ~80%+                        |
| Recall (High Risk) | Improved via class balancing |
| Precision          | Optimized using GridSearch   |

### Business Interpretation

* Correctly flags **high‑risk borrowers**
* Minimizes **false approvals**
* Improves lender profitability

---

## 🔧 Feature Engineering

* Label Encoding for categorical variables
* Target encoding for Risk variable
* Class imbalance handled using `class_weight="balanced"`
* Hyperparameter tuning with **GridSearchCV**

---

## 📦 Project Structure

```
credit-risk-prediction/
│
├── data/
│   └── german_credit.csv
│
├── models/
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   └── label_encoders/
│
├── notebooks/
│   └── credit_risk_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── README.md
└── requirements.txt
```

---

## 🛠 Tech Stack

* **Python** 🐍
* **Pandas / NumPy**
* **Scikit‑Learn**
* **XGBoost**
* **Seaborn & Matplotlib**
* **Joblib**

---

## ⚙️ How to Run

```bash
pip install -r requirements.txt
jupyter notebook
```

or train directly:

```bash
python src/train.py
```

---

## 📌 Key Learnings

* Real‑world data is **messy & imbalanced**
* Model accuracy alone is **not enough** — recall matters
* Explainability is crucial in **finance ML**

---

## 🎯 Who Should Look at This?

✔ Recruiters (Data Science / ML / FinTech)
✔ Banks & NBFC analysts
✔ ML engineers learning applied finance
✔ Students preparing for placements

---

## 🌟 Future Improvements

* SHAP explainability
* Model deployment with Streamlit
* Real‑time credit scoring API
* ROC‑AUC optimization

---

## 🙌 Author

**Prathmesh Bunde**
CSE | FinTech | Machine Learning

📌 *If this project helped you, don’t forget to ⭐ the repo!*
