import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# =========================
# LOAD DATA
# =========================
meals = pd.read_csv('meals.csv')
inter = pd.read_csv('interactions.csv')

# =========================
# TEXT FEATURE
# =========================
meals['text'] = (
    meals['tags'].fillna('') + ' ' +
    meals['ingredients'].fillna('') + ' ' +
    meals['description'].fillna('')
)

# =========================
# TF-IDF
# =========================
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf.fit_transform(meals['text'])

# =========================
# COSINE SIMILARITY
# =========================
cos_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# =========================
# INDEX MAP
# =========================
meal_indices = pd.Series(
    meals.index,
    index=meals['meal_id']
).drop_duplicates()

# =========================
# MEAL-TO-MEAL RECOMMENDATION
# =========================
def content_recommend_by_meal(meal_id, topn=10):

    if meal_id not in meal_indices:
        return f"Meal ID {meal_id} not found"

    idx = meal_indices[meal_id]

    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = [i for i in sim_scores if i[0] != idx]

    top_indices = [i[0] for i in sim_scores[:topn]]

    return meals.iloc[top_indices][['meal_id', 'name', 'tags', 'category']]

# =========================
# USER CONTENT SCORES
# =========================
def get_content_scores(user_id):

    user_meals = inter[inter["user_id"] == user_id]["meal_id"].unique()

    if len(user_meals) == 0:
        return np.zeros(len(meals))

    user_indices = meals[meals["meal_id"].isin(user_meals)].index

    scores = np.mean(
        cos_sim[:, user_indices],
        axis=1
    )

    return scores