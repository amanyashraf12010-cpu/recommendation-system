import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# =========================
# LOAD DATA
# =========================
inter = pd.read_csv("interactions.csv")
meals = pd.read_csv("meals.csv")

# =========================
# CLEAN DATA
# =========================
inter["user_id"] = inter["user_id"].astype(str).str.strip()
inter["meal_id"] = inter["meal_id"].astype(str).str.strip()

# =========================
# IMPUTED WEIGHTS
# =========================
weights = {
    'view': 1,
    'click': 3,
    'rescue': 6
}

inter["weight"] = inter["action"].map(weights)

# fill missing just in case
inter["weight"] = inter["weight"].fillna(1)

# =========================
# USER-ITEM MATRIX
# =========================
user_item_matrix = inter.pivot_table(
    index='user_id',
    columns='meal_id',
    values='weight',
    fill_value=0
)

# =========================
# SVD MODEL
# =========================
svd = TruncatedSVD(n_components=20, random_state=42)

user_latent = svd.fit_transform(user_item_matrix)
meal_latent = svd.components_.T

# =========================
# RECOMMEND FUNCTION
# =========================
def get_collab_scores(user_id):

    # =========================
    # CHECK USER EXISTS
    # =========================
    if user_id not in user_item_matrix.index:
        return np.zeros(len(meals))

    # =========================
    # USER VECTOR
    # =========================
    user_idx = user_item_matrix.index.get_loc(user_id)
    user_vec = user_latent[user_idx]

    # =========================
    # RAW SCORES
    # =========================
    raw_scores = np.dot(meal_latent, user_vec)

    # =========================
    # ALIGN TO FULL MEALS
    # =========================
    full_scores = np.zeros(len(meals))

    meal_id_to_index = {
        mid: i for i, mid in enumerate(meals["meal_id"])
    }

    for i, meal_id in enumerate(user_item_matrix.columns):
        if meal_id in meal_id_to_index:
            full_scores[meal_id_to_index[meal_id]] = raw_scores[i]

    return full_scores