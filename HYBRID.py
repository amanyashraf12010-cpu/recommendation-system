import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# LOAD
# =========================
users = pd.read_csv("users.csv")
meals = pd.read_csv("meals.csv")
meal_id_to_idx = {
    mid: i for i, mid in enumerate(meals["meal_id"])
}
inter = pd.read_csv("interactions.csv")

# =========================
# PREP
# =========================
weights = {"view": 1, "click": 3, "rescue": 6}
inter["weight"] = inter["action"].map(weights)

# =========================
# POPULARITY
# =========================
popularity = inter.groupby("meal_id")["weight"].sum()
popularity = popularity / (popularity.max() + 1e-8)

# =========================
# CONTENT MODEL
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
# USER-ITEM MATRIX
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
# HELPERS
# =========================
def is_cold_user(user_id):
    return user_id not in user_item.index


def get_collab_scores(user_id):

    if user_id not in user_item.index:
        return np.zeros(len(meals))

    user_idx = user_item.index.get_loc(user_id)
    user_vec = user_latent[user_idx]

    raw_scores = np.dot(meal_latent, user_vec)

    full = np.zeros(len(meals))

    for i, meal_id in enumerate(user_item.columns):
        if meal_id in meal_id_to_idx:
            full[meal_id_to_idx[meal_id]] = raw_scores[i]

    return full

def get_content_scores(user_id):

    user_meals = inter[inter["user_id"] == user_id]["meal_id"].unique()

    if len(user_meals) == 0:
        return np.zeros(len(meals))

    user_idx = meals[meals["meal_id"].isin(user_meals)].index

    scores = np.mean(cos_sim[:, user_idx], axis=1)

    # safety check
    if len(scores) != len(meals):
        scores = np.resize(scores, len(meals))

    return scores

def get_context_scores():

    scores = []

    now = pd.Timestamp.now()

    for _, row in meals.iterrows():

        score = 0

        if row["discount"] == 1:
            score += 0.3

        hours_left = (pd.to_datetime(row["expiry"]) - now).total_seconds() / 3600

        if hours_left < 6:
            score += 0.5
        elif hours_left < 12:
            score += 0.3
        elif hours_left < 24:
            score += 0.1

        if row["price"] < 50:
            score += 0.2
        elif row["price"] < 100:
            score += 0.1

        scores.append(score)

    scores = np.array(scores)
    return scores / (scores.max() + 1e-8)


# =========================
# FINAL HYBRID FUNCTION
# =========================
def hybrid_recommend(user_id, topn=10):

    # -----------------
    # COLD START
    # -----------------
    if is_cold_user(user_id):

        pop_scores = np.array([
            popularity.get(mid, 0) for mid in meals["meal_id"]
        ])

        context = get_context_scores()

        final = 0.6 * pop_scores + 0.4 * context

    # -----------------
    # NORMAL USER
    # -----------------
    else:

        collab = get_collab_scores(user_id)
        content = get_content_scores(user_id)
        context = get_context_scores()

        # safety check (IMPORTANT)
        min_len = min(len(collab), len(content), len(context))

        collab = collab[:min_len]
        content = content[:min_len]
        context = context[:min_len]

        final = (
            0.5 * collab +
            0.3 * content +
            0.2 * context
        )

       
    # =========================
    # OUTPUT (NO GLOBAL CHANGE)
    # =========================
    result = meals.copy()
    result["score"] = final

    return result.sort_values("score", ascending=False)[
        ["meal_id", "name", "score"]
    ].head(topn)


# =========================
# TEST
# =========================
print(hybrid_recommend("u5", topn=10))
print(hybrid_recommend("u_new", topn=10))