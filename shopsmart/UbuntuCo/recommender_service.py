"""
recommender_service.py - one import for your Streamlit pages.

    rec = RecommenderService(extra_ratings=real_ratings_df)   # cache it in the app

    rec.for_cart([1, 4])      # items in the cart (guests too)
    rec.similar_to(24)        # "similar products"
    rec.for_user("u_17")      # logged-in customer, or "g_<session>" for a guest
    rec.popular()             # cold start

Training data = simulated seed ratings (interactions.csv, so the model works on
day one) + REAL customer behaviour passed in as `extra_ratings`
(columns user_id, item_id, rating). As real data grows it outweighs the seed.
Every method returns a DataFrame of products with a 'match' column (0-1).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from content_model import ContentRecommender
from Model import Hybrid, Popularity, to_matrix


class RecommenderService:
    def __init__(self, data_dir=None, extra_ratings=None):
        # default: this file's folder, so paths work wherever Streamlit launches from
        d = Path(data_dir) if data_dir else Path(__file__).resolve().parent
        self.products = pd.read_csv(d / "products.csv")
        p = json.load(open(d / "params.json"))

        # 1) simulated seed data
        sim = pd.read_csv(d / "interactions.csv")
        sim["user_id"] = "sim_" + sim["user_id"].astype(str)
        frames = [sim[["user_id", "item_id", "rating"]]]
        self.n_sim_ratings = len(sim)

        # 2) real customer behaviour ('u_3' registered, 'g_ab12' guest)
        self.n_real_ratings = 0
        self.n_real_users = 0
        if extra_ratings is not None and len(extra_ratings):
            e = extra_ratings.copy()
            e["user_id"] = e["user_id"].astype(str)
            e = e[e["item_id"].isin(self.products["ProductID"])]
            if len(e):
                self.n_real_ratings = len(e)
                self.n_real_users = e["user_id"].nunique()
                frames.append(e[["user_id", "item_id", "rating"]])

        df = pd.concat(frames, ignore_index=True)
        df = df.groupby(["user_id", "item_id"], as_index=False)["rating"].max()

        self.uidx = {u: i for i, u in enumerate(sorted(df["user_id"].unique()))}
        self.pidx = {pid: i for i, pid in enumerate(self.products.ProductID)}
        self.R = to_matrix(df, self.uidx, self.pidx)
        self.content = ContentRecommender().fit(self.products)
        self.hybrid = Hybrid(p["alpha"], p["n_factors"]).fit(self.R, content=self.content)
        self.pop = Popularity().fit(self.R)

    # ------------------------------------------------------------ helpers
    def _top(self, scores, n, exclude=()):
        s = np.array(scores, dtype=float)
        s[list(exclude)] = -np.inf
        order = np.argsort(-s)[:n]
        ok = s[np.isfinite(s)]
        span = ok.max() - ok.min()
        out = self.products.iloc[order].copy()
        out["match"] = [(s[i] - ok.min()) / span if span else 0.0 for i in order]
        return out.reset_index(drop=True)

    # ------------------------------------------------------------ public API
    def popular(self, n=5):
        return self._top(self.pop.scores(0), n)

    def similar_to(self, product_id, n=5):
        i = self.pidx[product_id]
        return self._top(self.content.sim[i], n, exclude=[i])

    def for_cart(self, product_ids, n=5):
        ids = [self.pidx[p] for p in product_ids if p in self.pidx]
        if not ids:
            return self.popular(n)
        r = np.zeros(len(self.products))
        r[ids] = 5
        return self._top(self.hybrid.scores_from_vector(r), n, exclude=ids)

    def for_user(self, user_id, n=5):
        user_id = str(user_id)
        if user_id not in self.uidx:
            return self.popular(n)
        u = self.uidx[user_id]
        return self._top(self.hybrid.scores(u), n, exclude=self.R[u].indices)

    # ------------------------------------------------------------ statistics
    def catalog_coverage(self, n=5):
        """Share of the catalog that appears in at least one customer's top-n."""
        seen = set()
        for u in range(self.R.shape[0]):
            top = self._top(self.hybrid.scores(u), n, exclude=self.R[u].indices)
            seen.update(top["ProductID"])
        return len(seen) / len(self.products)

    def sparsity(self):
        return 1 - self.R.nnz / (self.R.shape[0] * self.R.shape[1])


_service = None


def get_service():
    global _service
    if _service is None:
        _service = RecommenderService()
    return _service