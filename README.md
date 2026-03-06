Netflix User Behavior Analytics
End-to-End Data Analysis & Churn Prediction

Data science project analyzing large-scale user behavior from a streaming platform to understand engagement, retention, and churn risk.

The project implements a complete analytics pipeline, from raw data ingestion to machine learning modeling.

Dataset

Source

Kaggle – Netflix 2025 User Behavior Dataset (~210K records)

The dataset simulates activity within a streaming platform and includes:

User profiles

Watch history

Search activity

Reviews

Recommendation logs

Project Goals

Analyze user engagement patterns

Measure churn and retention

Identify behavioral drivers of churn

Segment users based on engagement

Build predictive churn models

Data Pipeline

Raw Data
↓
Data Validation & Cleaning
↓
Feature Engineering
↓
Exploratory Data Analysis
↓
User Segmentation
↓
Churn Prediction Models

Project Structure
Netflix-User-Analytics/

data/
    raw/
    processed/
    features/

scripts/
    01_data_cleaning.py
    02_feature_engineering.py
    03_churn_modeling.py
    04_user_segmentation.py
    05_retention_analysis.py

notebooks/
    eda.ipynb

README.md
Feature Engineering

User-level behavioral features were created, including:

Total watch time

Average watch duration

Session frequency

Unique content consumption

Search activity

Review engagement

Recency of activity

These features form the churn prediction dataset used for modeling.

Machine Learning

Three models were trained to predict churn:

Logistic Regression

Random Forest

Gradient Boosting

Evaluation metrics:

ROC-AUC

Precision

Recall

F1 Score

User Segmentation

K-Means clustering was applied to identify behavioral segments such as:

Power users

Casual viewers

At-risk users

New users

This reveals different engagement patterns across the platform.

Retention Analysis

Cohort analysis was performed to measure user retention over time.

A retention heatmap visualizes how engagement evolves after users join the platform.

Key Insights

Example findings from the analysis:

Engagement decay predicts churn better than total watch time

Frequent short sessions correlate with higher retention

Early behavioral shifts can signal churn risk weeks in advance

Tech Stack

Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
