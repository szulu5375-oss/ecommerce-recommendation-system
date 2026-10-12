import joblib
import numpy as np
import pandas as pd

CF_WEIGHT = 0.7


class Recommender:
    def __init__(self, path="artifacts.joblib"):
        a = joblib.load(path)
        self.__dict__.update(a)
        # case-insensitive lookup: StockCode or product Description -> Description
        self.lookup = {k.lower(): v for k, v in self.sku_to_product.items()}
        self.lookup.update({p.lower(): p for p in self.content_scores})

    def list_types(self):
        return [{"product": p, "category": self.product_category[p],
                 "stock_code": self.product_sku[p]} for p in self.content_scores]

    def _score(self, cart):
        """Score ALL product types. Returns (scores dict, set of types in the cart)."""
        counts, in_cart, unknown = {}, set(), []
        for item, qty in cart:
            product = self.lookup.get(item.strip().lower())
            if product is None or product not in self.product_category:
                unknown.append(item)
                continue
            in_cart.add(product)
            cat = self.product_category[product]
            counts[cat] = counts.get(cat, 0) + qty
        if unknown:
            raise ValueError(f"Unknown items: {unknown}")

        vec = np.array([counts.get(c, 0) for c in self.categories], dtype=float)
        vec /= vec.sum()

        dist, idx = self.knn.kneighbors([vec])
        sim = 1 - dist[0]
        votes = pd.Series(sim, index=self.neighbour_pref[idx[0]]).groupby(level=0).sum()
        votes = votes / votes.sum() if votes.sum() > 0 else votes

        scores = {
            p: CF_WEIGHT * float(votes.get(p, 0.0)) + (1 - CF_WEIGHT) * cs
            for p, cs in self.content_scores.items()
        }
        return scores, in_cart

    def recommend(self, cart, top_n=3):
        scores, in_cart = self._score(cart)
        scores = {p: s for p, s in scores.items() if p not in in_cart}
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        return [{"product": p, "stock_code": self.product_sku[p],
                 "category": self.product_category[p], "score": round(s, 4)}
                for p, s in top]

    def score_all(self, cart):
        scores, in_cart = self._score(cart)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [{"product": p, "category": self.product_category[p],
                 "score": round(s, 4), "in_cart": p in in_cart}
                for p, s in ranked]