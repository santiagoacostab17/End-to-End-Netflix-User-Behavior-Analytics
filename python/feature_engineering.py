import pandas as pd
import numpy as np
import os

print("\n===== FEATURE ENGINEERING PIPELINE =====\n")

# -----------------------------
# Paths
# -----------------------------
PROCESSED_PATH = "data/processed/"
FEATURE_PATH = "data/features/"

os.makedirs(FEATURE_PATH, exist_ok=True)

# -----------------------------
# Load cleaned datasets
# -----------------------------
users = pd.read_csv(PROCESSED_PATH + "users_clean.csv")
movies = pd.read_csv(PROCESSED_PATH + "movies_clean.csv")
watch_history = pd.read_csv(PROCESSED_PATH + "watch_history_clean.csv")
search_logs = pd.read_csv(PROCESSED_PATH + "search_logs_clean.csv")
reviews = pd.read_csv(PROCESSED_PATH + "reviews_clean.csv")

# -----------------------------
# Convert date columns
# -----------------------------
watch_history["watch_date"] = pd.to_datetime(watch_history["watch_date"])
search_logs["search_date"] = pd.to_datetime(search_logs["search_date"])

# -----------------------------
# WATCH FEATURES
# -----------------------------
print("Creating watch features...")

watch_features = watch_history.groupby("user_id").agg(
    total_watch_time=("watch_duration_minutes", "sum"),
    avg_watch_time=("watch_duration_minutes", "mean"),
    total_sessions=("session_id", "count"),
    unique_movies=("movie_id", "nunique"),
    avg_completion=("progress_percentage", "mean")
).reset_index()

# -----------------------------
# SESSION FREQUENCY
# -----------------------------
session_freq = watch_history.groupby("user_id").agg(
    first_watch=("watch_date", "min"),
    last_watch=("watch_date", "max")
).reset_index()

session_freq["active_days"] = (
    session_freq["last_watch"] - session_freq["first_watch"]
).dt.days + 1

session_freq["sessions_per_day"] = (
    watch_features["total_sessions"] / session_freq["active_days"]
)

session_freq = session_freq[["user_id", "sessions_per_day"]]

# -----------------------------
# SEARCH FEATURES
# -----------------------------
print("Creating search features...")

search_features = search_logs.groupby("user_id").agg(
    total_searches=("search_query", "count"),
    unique_searches=("search_query", "nunique"),
    avg_search_time=("search_duration_seconds", "mean")
).reset_index()

# -----------------------------
# REVIEW FEATURES
# -----------------------------
print("Creating review features...")

review_features = reviews.groupby("user_id").agg(
    total_reviews=("review_text", "count"),
    avg_sentiment=("sentiment_score", "mean"),
    helpful_votes=("helpful_votes", "sum")
).reset_index()

# -----------------------------
# MERGE USER DATASET
# -----------------------------
print("Merging user dataset...")

df = users.merge(watch_features, on="user_id", how="left")
df = df.merge(session_freq, on="user_id", how="left")
df = df.merge(search_features, on="user_id", how="left")
df = df.merge(review_features, on="user_id", how="left")

df = df.fillna(0)

# -----------------------------
# RECENCY FEATURES
# -----------------------------
print("Creating recency features...")

last_watch = watch_history.groupby("user_id")["watch_date"].max().reset_index()

last_watch.columns = ["user_id", "last_activity"]

df = df.merge(last_watch, on="user_id", how="left")

dataset_last_date = watch_history["watch_date"].max()

df["days_since_last_activity"] = (
    dataset_last_date - pd.to_datetime(df["last_activity"])
).dt.days

# -----------------------------
# CHURN LABEL
# -----------------------------
print("Creating churn label...")

df["churn"] = (df["days_since_last_activity"] > 30).astype(int)

# -----------------------------
# Save features dataset
# -----------------------------
output_file = FEATURE_PATH + "user_features_churn.csv"

df.to_csv(output_file, index=False)

print("Saved feature dataset:", output_file)
print("\nDataset shape:", df.shape)

print("\n===== FEATURE ENGINEERING COMPLETE =====")
