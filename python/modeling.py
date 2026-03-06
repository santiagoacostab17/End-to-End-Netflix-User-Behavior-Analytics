import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import os

print("\n===== EXPORTING PROFESSIONAL CHURN SUMMARY =====\n")

# ----------------------------
# Start timer
# ----------------------------
start_time = time.perf_counter()

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("data/features/user_features_churn.csv")
df.drop(columns=[c for c in ["user_id","last_activity"] if c in df.columns], inplace=True)

y = df.pop("churn")
X = df

# ----------------------------
# Convert datetime columns
# ----------------------------
date_cols = X.columns[X.columns.str.contains("date|time")]
for col in date_cols:
    X[col] = pd.to_datetime(X[col], errors="coerce")
    X[col] = X[col].astype('int64') // 10**9

# ----------------------------
# Encode categorical columns
# ----------------------------
cat_cols = X.select_dtypes(include=['object', 'category']).columns
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
X.fillna(0, inplace=True)

# ----------------------------
# Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# Gradient Boosting Model
# ----------------------------
model = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# ----------------------------
# Predictions and risk segments
# ----------------------------
pred_prob = model.predict_proba(X_test)[:,1]
pred_class = (pred_prob > 0.5).astype(int)

results_df = X_test.copy()
results_df["Actual_Churn"] = y_test.values
results_df["Predicted_Churn"] = pred_class
results_df["Churn_Probability"] = pred_prob
results_df["Risk_Segment"] = pd.cut(
    results_df["Churn_Probability"],
    bins=[0,0.33,0.66,1],
    labels=["Low","Medium","High"]
)

# ----------------------------
# Top 10 factors
# ----------------------------
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
top_features = importance.head(10)

# Add top factor values as columns to results_df
for feature in top_features.index:
    results_df[f"Factor_{feature}"] = X_test[feature]

# ----------------------------
# Ensure folder exists
# ----------------------------
os.makedirs("data/results", exist_ok=True)

# ----------------------------
# Save CSV
# ----------------------------
output_path = "data/results/churn_resume.csv"
results_df.to_csv(output_path, index=False)

# ----------------------------
# End timer
# ----------------------------
end_time = time.perf_counter()
elapsed_time = end_time - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

print(f"\nChurn summary CSV saved to: {output_path}")
print(f"Total processing time: {minutes} minutes {seconds} seconds")
print("\nYou can now use this file in Power BI to create your dashboard manually.")
