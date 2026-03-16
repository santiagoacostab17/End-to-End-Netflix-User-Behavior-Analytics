# 🎬 Netflix User Behavior Analytics (unfinished)

### End-to-End Data Analysis & Predictive Modeling

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange)
![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-ScikitLearn-green)

---

## 📊 Project Overview

This project analyzes **Netflix user behavior data (~210K records)** to understand engagement, detect churn drivers, and build predictive models.

The goal is to demonstrate a **complete data science workflow**, from raw data to machine learning insights.

---

## 🎯 Objectives

* Measure **user engagement and retention**
* Identify **high-value user segments**
* Detect **key churn drivers**
* Build **predictive churn models**
* Deliver a **clean and reproducible analytics pipeline**

---

## 🗂 Dataset

**Netflix 2025 User Behavior Dataset**

Features include:

* User activity metrics
* Watch time
* Subscription type
* Session behavior
* Engagement patterns
* Churn indicator

Dataset size: **~210,000 rows**

---

## ⚙️ Workflow

### 1️⃣ Data Cleaning

* Missing value handling
* Duplicate removal
* Outlier detection
* Data validation

### 2️⃣ Feature Engineering

* Behavioral metrics
* Temporal features
* Engagement indicators

### 3️⃣ Exploratory Data Analysis

* User segmentation
* Correlation analysis
* Behavioral patterns

### 4️⃣ KPI Metrics

Key metrics computed:

* **Churn Rate**
* **Retention Rate**
* **Average Watch Time**
* **ARPU (Average Revenue Per User)**

### 5️⃣ Predictive Modeling

Models used:

```python
Logistic Regression
Random Forest
Gradient Boosting
```

Goal: **predict churn probability**

---

## 📈 Key Insights

Examples of insights extracted:

* High session frequency → **lower churn probability**
* Low engagement users → **highest churn risk**
* Premium users show **higher retention**

---

## 🛠 Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib / Seaborn**
* **Scikit-Learn**

---

## 📂 Project Structure

```
netflix-user-analytics/

data/
    netflix_user_behavior.csv

notebooks/
    eda.ipynb
    feature_engineering.ipynb

models/
    churn_model.pkl

scripts/
    data_cleaning.py
    train_model.py

README.md
```

---

## 🚀 Future Improvements

* Deploy churn model with **API**
* Add **customer lifetime value prediction**
* Implement **dashboard visualization**

---

## 💡 Author

**Santiago Acosta**
Data Analytics & Machine Learning Projects
