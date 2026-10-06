"""
content_model.py - content-based recommender for products.csv
(ProductID, Name, Category, Price, Description).

Each product becomes a feature vector:
  TF-IDF of (name + category + description) + scaled final price + discount
Similarity between products = cosine similarity of these vectors.

Usage:
    python content_model.py                   # demo + evaluation
    python content_model.py --product 1 --n 3
"""
import argparse
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:
    def __init__(self, price_weight=0.3, category_weight=2, discount_weight=0.1):
        self.price_weight = price_weight
        self.discount_weight = discount_weight
        self.category_weight = category_weight

    def fit(self, df):
        self.df = df.reset_index(drop=True)
        # repeat category so it counts more than a single description word
        text = (self.df["Name"] + " "
                + (self.df["Category"] + " ") * self.category_weight
                + self.df["Description"])
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        X_text = self.tfidf.fit_transform(text)

        # numeric features: log-scaled final price (after discount) + discount.
        # log keeps a R12,999 laptop from dwarfing everything else.
        disc = self.df["DiscountPercent"] if "DiscountPercent" in self.df else 0
        final = self.df["Price"].astype(float) * (1 - disc / 100)
        lp = np.log1p(final)
        price = ((lp - lp.min()) / (lp.max() - lp.min())).to_numpy().reshape(-1, 1)
        disc_s = (np.asarray(disc, dtype=float) / 100).reshape(-1, 1) * np.ones((len(self.df), 1))
        X = hstack([X_text, csr_matrix(price * self.price_weight),
                    csr_matrix(disc_s * self.discount_weight)]).tocsr()

        self.sim = cosine_similarity(X)
        np.fill_diagonal(self.sim, 0)          # a product isn't its own recommendation
        self.idx = {pid: i for i, pid in enumerate(self.df["ProductID"])}
        return self

    def similar_to(self, product_id, n=3):
        i = self.idx[product_id]
        top = np.argsort(-self.sim[i])[:n]
        out = self.df.iloc[top][["ProductID", "Name", "Category", "Price", "DiscountPercent"]].copy()
        out["score"] = self.sim[i][top].round(3)
        return out

    def for_user(self, liked_ids, n=3):
        """Recommend based on the products a user liked / bought / viewed."""
        rows = [self.idx[p] for p in liked_ids]
        s = self.sim[rows].mean(axis=0)
        s[rows] = -1                            # drop items already seen
        top = np.argsort(-s)[:n]
        out = self.df.iloc[top][["ProductID", "Name", "Category", "Price", "DiscountPercent"]].copy()
        out["score"] = s[top].round(3)
        return out


def category_precision(model, k=3):
    """Offline check: share of top-k recommendations in the same category as the
    query product (capped by how many same-category products exist)."""
    df, scores = model.df, []
    for pid in df["ProductID"]:
        cat = df.loc[df.ProductID == pid, "Category"].iloc[0]
        same = (df.Category == cat).sum() - 1
        if same == 0:
            continue
        recs = model.similar_to(pid, k)
        hits = (recs["Category"] == cat).sum()
        scores.append(hits / min(k, same))
    return float(np.mean(scores))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="products.csv")
    ap.add_argument("--product", type=int, default=None)
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    m = ContentRecommender().fit(df)

    pids = [a.product] if a.product else [1, 4, 5, 20, 24, 33]
    for pid in pids:
        name = df.loc[df.ProductID == pid, "Name"].iloc[0]
        print(f"\nBecause you viewed: {name}")
        print(m.similar_to(pid, a.n).to_string(index=False))

    print("\nUser who liked Smartphone and Wireless Earbuds:")
    print(m.for_user([24, 31], a.n).to_string(index=False))

    print(f"\nCategory precision@{a.n}: {category_precision(m, a.n):.2f}")
    joblib.dump(m, "content_recommender.joblib")
    print("Saved content_recommender.joblib")
