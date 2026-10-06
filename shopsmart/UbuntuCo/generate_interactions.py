"""
generate_interactions.py - SIMULATED user ratings for products.csv.

NOTE FOR YOUR REPORT: this data is synthetic. Each user gets 1-2 favourite
categories, a budget level and a taste for discounts; ratings follow those
tastes plus noise. Replace interactions.csv with real data when available
(columns: user_id, item_id, rating).
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
products = pd.read_csv("products.csv")
cats = products["Category"].unique()
final_price = products["Price"] * (1 - products["DiscountPercent"] / 100)
log_p = np.log(final_price)

rows = []
for u in range(1, 401):
    fav = rng.choice(cats, size=rng.integers(1, 3), replace=False)
    budget = rng.uniform(log_p.min(), log_p.max())          # price the user is comfortable with
    disc_lover = rng.random() < 0.4
    # probability of interacting with each product
    w = np.where(products["Category"].isin(fav), 6.0, 0.6)
    w *= np.exp(-np.abs(log_p - budget) / 2.5)
    w *= 1 + 0.8 * disc_lover * (products["DiscountPercent"] > 0)
    w = w / w.sum()
    n = rng.integers(6, 16)
    for i in rng.choice(len(products), size=n, replace=False, p=w):
        score = 2.6
        score += 1.3 * (products.Category[i] in fav)
        score += 0.4 * disc_lover * (products.DiscountPercent[i] > 0)
        score -= 0.35 * abs(log_p[i] - budget)
        score += rng.normal(0, 0.6)
        rows.append((u, int(products.ProductID[i]), int(np.clip(round(score), 1, 5))))

out = pd.DataFrame(rows, columns=["user_id", "item_id", "rating"])
out.to_csv("interactions.csv", index=False)
print(f"{len(out)} ratings, {out.user_id.nunique()} users, {out.item_id.nunique()} items")
print(out.rating.value_counts().sort_index().to_string())
