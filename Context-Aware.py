import pandas as pd
import numpy as np

# =========================
# LOAD DATA
# =========================
meals = pd.read_csv("meals.csv")

# =========================
# PREPROCESS
# =========================
meals["expiry"] = pd.to_datetime(meals["expiry"])

# ثابت وقت التشغيل (good for reproducibility)
now = pd.Timestamp.now()

# =========================
# CONTEXT FUNCTION
# =========================
def context_boost(row):

    score = 0.0

    # DISCOUNT
    if row["discount"] == 1:
        score += 0.3

    # EXPIRY
    hours_left = (row["expiry"] - now).total_seconds() / 3600

    if hours_left < 6:
        score += 0.5
    elif hours_left < 12:
        score += 0.3
    elif hours_left < 24:
        score += 0.1

    # PRICE
    if row["price"] < 50:
        score += 0.2
    elif row["price"] < 100:
        score += 0.1

    return score

# =========================
# APPLY
# =========================
meals["context_score"] = meals.apply(context_boost, axis=1)

# =========================
# NORMALIZATION
# =========================
meals["context_score"] = (
    meals["context_score"] - meals["context_score"].min()
) / (
    meals["context_score"].max() - meals["context_score"].min() + 1e-8
)

# =========================
# FUNCTION FOR HYBRID
# =========================
def get_context_scores():
    return meals["context_score"].values

# =========================
# DEBUG CHECK (optional)
# =========================
print(meals[["meal_id", "price", "discount", "context_score"]].head())