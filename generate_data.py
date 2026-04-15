import random
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)

# =========================
# CONFIG
# =========================
N_USERS = 1000
N_RESTAURANTS = 80
N_MEALS = 1000
N_INTERACTIONS = 10000

# =========================
# POOLS
# =========================
tags_pool = [
    'vegan','vegetarian','spicy','halal','seafood',
    'dessert','lowcal','kidfriendly','glutenfree'
]

ingredients_pool = [
    'chicken','beef','rice','tomato','cheese','bread',
    'lettuce','fish','potato','onion','garlic','egg'
]

# =========================
# 1) USERS
# =========================
users = []

for i in range(N_USERS):
    prefs = random.sample(tags_pool, k=random.randint(1, 3))
    
    users.append({
        'user_id': f'u{i+1}',
        'lat': 30.0 + random.uniform(-0.6, 0.6),
        'lon': 31.0 + random.uniform(-0.6, 0.6),
        'signup': (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat(),
        'prefs': ','.join(prefs)
    })

users_df = pd.DataFrame(users)

# =========================
# 2) MEALS
# =========================
meals = []

for i in range(N_MEALS):
    rid = f'r{random.randint(1, N_RESTAURANTS)}'
    
    tags = random.sample(tags_pool, k=random.randint(1, 3))
    ingredients = random.sample(ingredients_pool, k=3)
    
    expiry = datetime.now() + timedelta(hours=random.randint(1, 120))
    
    meals.append({
        'meal_id': f'm{i+1}',
        'restaurant_id': rid,
        'name': f'Meal {i+1}',
        'ingredients': ','.join(ingredients),
        'category': random.choice(['main','starter','dessert','drink']),
        'price': round(random.uniform(20, 200), 2),
        'expiry': expiry.isoformat(),
        'discount': random.choice([0, 0, 0, 1]),
        'tags': ','.join(tags),
        'description': f"Delicious {' '.join(tags)} meal with {' '.join(ingredients)}."
    })

meals_df = pd.DataFrame(meals)

# =========================
# 3) INTERACTIONS (SMART LOGIC)
# =========================
actions = []

for _ in range(N_INTERACTIONS):

    # pick user
    user = random.choice(users)
    user_id = user['user_id']
    user_prefs = user['prefs'].split(',')

    # find matching meals
    matched_meals = [
        meal for meal in meals
        if any(tag in meal['tags'] for tag in user_prefs)
    ]

    # choose meal
    if matched_meals:
        meal = random.choice(matched_meals)
        match = True
    else:
        meal = random.choice(meals)
        match = False

    meal_id = meal['meal_id']

    # timestamp
    t = datetime.now() - timedelta(
        days=random.randint(0, 30),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )

    # behavior generation
    if match:
        action = random.choices(
            ['click', 'rescue'],
            weights=[0.7, 0.3]
        )[0]
    else:
        action = random.choices(
            ['view', 'click'],
            weights=[0.85, 0.15]
        )[0]

    actions.append({
        'user_id': user_id,
        'meal_id': meal_id,
        'timestamp': t.isoformat(),
        'action': action
    })

inter_df = pd.DataFrame(actions)

# =========================
# SAVE FILES
# =========================
users_df.to_csv('users.csv', index=False)
meals_df.to_csv('meals.csv', index=False)
inter_df.to_csv('interactions.csv', index=False)

print("✅ Smart datasets created successfully!")