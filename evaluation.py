import numpy as np
import pandas as pd
from HYBRID import hybrid_recommend

# =========================
# LOAD DATA
# =========================
interactions = pd.read_csv("interactions.csv")

# =========================
# SPLIT PER USER (IMPORTANT)
# =========================
test_inter = interactions.groupby("user_id").sample(frac=0.2, random_state=42)
train_inter = interactions.drop(test_inter.index)

# =========================
# GROUND TRUTH
# =========================
def actual_items(user_id):
    return set(
        test_inter[test_inter["user_id"] == user_id]["meal_id"]
    )

# =========================
# CACHE (VERY IMPORTANT ⚡)
# =========================
recommendation_cache = {}

# =========================
# PRECISION & RECALL
# =========================
def precision_recall_at_k(user_id, k=10):

    actual = actual_items(user_id)

    if len(actual) == 0:
        return None, None

    # ===== CACHE =====
    if user_id not in recommendation_cache:
        recommendation_cache[user_id] = hybrid_recommend(user_id, topn=k)

    recs_df = recommendation_cache[user_id]

    if recs_df is None or len(recs_df) == 0:
        return 0, 0

    recommended = set(recs_df["meal_id"])

    tp = len(recommended & actual)

    precision = tp / k
    recall = tp / len(actual)

    return precision, recall

# =========================
# EVALUATION
# =========================
precisions = []
recalls = []

# ⚡ خد sample صغير عشان السرعة
sample_users = test_inter["user_id"].dropna().unique()[:30]

print("Users in test:", len(sample_users))

for i, user_id in enumerate(sample_users):

    print(f"Processing user {i}...")

    try:
        p, r = precision_recall_at_k(user_id, k=10)

        if p is not None:
            precisions.append(p)
            recalls.append(r)

    except Exception as e:
        print("Error:", user_id, e)

# =========================
# FINAL RESULT
# =========================
print("Valid users evaluated:", len(precisions))

if len(precisions) > 0:
    print("Average Precision@10:", round(np.mean(precisions), 4))
    print("Average Recall@10   :", round(np.mean(recalls), 4))
else:
    print("⚠️ No valid evaluation data")

print("SCRIPT FINISHED ✅")