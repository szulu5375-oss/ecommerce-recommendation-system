"""
recommender_service.py - one import for your Streamlit pages.

    from recommender_service import get_service
    rec = get_service()                          # wrap in st.cache_resource in your app

    rec.for_cart([1, 4])                         # Cart page: "You might also like"
    rec.similar_to(24)                           # Product page: "Similar products"
    rec.for_user(17)                             # Account page (logged-in user id in interactions.csv)
    rec.popular()                                # Guest / empty cart (cold start)

Every method returns a DataFrame of products (+ a 'match' column, 0-1)
that you can render with your existing product-card code.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from content_model import ContentRecommender
from Model import Hybrid, Popularity, to_matrix


class RecommenderService:
    def __init__(self, data_dir=None):
        # default: the folder containing this file, so it works no matter
        # where Streamlit Cloud launches the app from
        d = Path(data_dir) if data_dir else Path(__file__).resolve().parent
        self.products = pd.read_csv(d / "products.csv")
        df = pd.read_csv(d / "interactions.csv")
        p = json.load(open(d / "params.json"))
        self.uidx = {u: i for i, u in enumerate(sorted(df.user_id.unique()))}
        self.pidx = {pid: i for i, pid in enumerate(self.products.ProductID)}
        self.R = to_matrix(df, self.uidx, self.pidx)
        self.content = ContentRecommender().fit(self.products)
        self.hybrid = Hybrid(p["alpha"], p["n_factors"]).fit(self.R, content=self.content)
        self.pop = Popularity().fit(self.R)

    def _top(self, scores, n, exclude=()):
        s = np.array(scores, dtype=float)
        s[list(exclude)] = -np.inf
        order = np.argsort(-s)[:n]
        ok = s[np.isfinite(s)]
        span = ok.max() - ok.min()
        out = self.products.iloc[order].copy()
        out["match"] = [(s[i] - ok.min()) / span if span else 0.0 for i in order]
        return out.reset_index(drop=True)

    def popular(self, n=5):
        return self._top(self.pop.scores(0), n)

    def similar_to(self, product_id, n=5):
        i = self.pidx[product_id]
        return self._top(self.content.sim[i], n, exclude=[i])

    def for_cart(self, product_ids, n=5):
        """Works for guests: treats cart items as 'liked' (rating 5)."""
        ids = [self.pidx[p] for p in product_ids if p in self.pidx]
        if not ids:
            return self.popular(n)
        r = np.zeros(len(self.products))
        r[ids] = 5
        return self._top(self.hybrid.scores_from_vector(r), n, exclude=ids)

    def for_user(self, user_id, n=5):
        if user_id not in self.uidx:
            return self.popular(n)
        u = self.uidx[user_id]
        return self._top(self.hybrid.scores(u), n, exclude=self.R[u].indices)


_service = None
def get_service():
    global _service
    if _service is None:
        _service = RecommenderService()
    return _service
