from fastapi import FastAPI
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD

app= FastAPI(title="Food Recommendation API 🍔")

# =========================
# LOAD DATA
# =========================
users = pd.read_csv("users.csv")
meals = pd.read_csv("meals.csv")
inter = pd.read_csv("interactions.csv")

# =========================
# PREPARE INTERACTIONS
# =========================
weights = {"view": 1, "click": 3, "rescue": 6}
inter["weight"] = inter["action"].map(weights).fillna(1)

# =========================
# CONTENT-BASED
# =========================
meals["text"] = (
    meals["tags"].fillna("") + " " +
    meals["ingredients"].fillna("") + " " +
    meals["description"].fillna("")
)

tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(meals["text"])
cos_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# =========================
# COLLABORATIVE FILTERING
# =========================
user_item = inter.pivot_table(
    index="user_id",
    columns="meal_id",
    values="weight",
    fill_value=0
)

svd = TruncatedSVD(n_components=20, random_state=42)
user_latent = svd.fit_transform(user_item)
meal_latent = svd.components_.T

# =========================
# CONTEXT SCORES (STATIC SAFE)
# =========================
meals["expiry"] = pd.to_datetime(meals["expiry"])
now = pd.Timestamp.now()


def get_context_scores():

    scores = []

    for _, row in meals.iterrows():

        score = 0.0

        # discount boost
        if row["discount"] == 1:
            score += 0.3

        # expiry boost
        hours_left = (row["expiry"] - now).total_seconds() / 3600

        if hours_left < 6:
            score += 0.5
        elif hours_left < 12:
            score += 0.3
        elif hours_left < 24:
            score += 0.1

        # price boost
        if row["price"] < 50:
            score += 0.2
        elif row["price"] < 100:
            score += 0.1

        scores.append(score)

    scores = np.array(scores)

    # normalization (very important)
    return scores / (scores.max() + 1e-8)

# =========================
# COLLAB SCORES
# =========================
def get_collab_scores(user_id):

    if user_id not in user_item.index:
        return np.zeros(len(meals))

    idx = user_item.index.get_loc(user_id)
    scores = np.dot(meal_latent, user_latent[idx])

    full_scores = np.zeros(len(meals))

    meal_map = {mid: i for i, mid in enumerate(meals["meal_id"])}

    for i, mid in enumerate(user_item.columns):
        if mid in meal_map:
            full_scores[meal_map[mid]] = scores[i]

    return full_scores

# =========================
# CONTENT SCORES
# =========================
def get_content_scores(user_id):

    user_meals = inter[inter["user_id"] == user_id]["meal_id"].unique()

    if len(user_meals) == 0:
        return np.zeros(len(meals))

    idxs = meals[meals["meal_id"].isin(user_meals)].index

    scores = np.mean(cos_sim[:, idxs], axis=1)

    return scores

# =========================
# HYBRID MODEL
# =========================
def hybrid_recommend(user_id, topn=10):

    collab = get_collab_scores(user_id)
    content = get_content_scores(user_id)
    context = get_context_scores()

    # normalize safely
    collab = collab / (collab.max() + 1e-8)
    content = content / (content.max() + 1e-8)

    final = (
        0.5 * collab +
        0.3 * content +
        0.2 * context
    )

    # IMPORTANT: do NOT modify original dataframe
    temp = meals.copy()
    temp["score"] = final

    result = temp.sort_values("score", ascending=False)[
        ["meal_id", "name", "score"]
    ].head(topn)

    return result.to_dict(orient="records")

# =========================
# API ROUTES
# =========================
@app.get("/")
def root():
    return {"message": "API is running 🚀"}

@app.get("/health")
def health():
    return {"status": "OK"}

@app.get("/recommend/{user_id}")
def recommend(user_id: str, topn: int = 10):

    try:
        recs = hybrid_recommend(user_id, topn)

        return {
            "user_id": user_id,
            "recommendations": recs
        }

    except Exception as e:
        return {
            "error": str(e)
        }


@app.get("/test-context")
def test_context():

    scores = get_context_scores()

    return {
        "max": float(scores.max()),
        "min": float(scores.min()),
        "sample": scores[:10].tolist()
    }

@app.get("/test-content/{user_id}")
def test_content(user_id: str):

    scores = get_content_scores(user_id)

    return {
        "max": float(scores.max()),
        "min": float(scores.min()),
        "sample": scores[:10].tolist()
    }

@app.get("/test-collab/{user_id}")
def test_collab(user_id: str):

    scores = get_collab_scores(user_id)

    return {
        "max": float(scores.max()),
        "min": float(scores.min()),
        "sample": scores[:10].tolist()
    }