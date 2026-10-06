"""
Model.py - train, tune and evaluate recommenders on products.csv + interactions.csv

Models: Popularity (baseline), Item-based CF, SVD (matrix factorisation),
        Content-based (TF-IDF + price + discount), Hybrid (SVD + Content).
Metrics: Precision@K, Recall@K, F1@K, RMSE/MSE (rating prediction).

Run:  python Model.py
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from content_model import ContentRecommender

K = 5
LIKE = 4          # rating >= 4 counts as "relevant"
SEED = 42


# ------------------------------------------------------------ data
def holdout(df, frac, seed):
    """Hold out `frac` of each user's ratings (users with >=5 ratings only)."""
    rng = np.random.default_rng(seed)
    size = df.groupby("user_id")["item_id"].transform("size")
    mask = (rng.random(len(df)) < frac) & (size >= 5)
    return df[~mask], df[mask]


def to_matrix(df, uidx, iidx):
    return csr_matrix((df.rating, (df.user_id.map(uidx), df.item_id.map(iidx))),
                      shape=(len(uidx), len(iidx)))


# ------------------------------------------------------------ models
def minmax(s):
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)


class Popularity:
    def fit(self, R, **_):
        # mean rating weighted by how many people rated it (Bayesian average)
        cnt = np.asarray((R > 0).sum(0)).ravel()
        tot = np.asarray(R.sum(0)).ravel()
        g = tot.sum() / max(cnt.sum(), 1)
        self.s = (tot + 5 * g) / (cnt + 5)
        return self

    def scores(self, u):
        return self.s


class ItemCF:
    def fit(self, R, **_):
        self.R = R
        self.sim = cosine_similarity(R.T)
        np.fill_diagonal(self.sim, 0)
        return self

    def scores(self, u):
        return self.R[u].toarray().ravel() @ self.sim


class SVD:
    def __init__(self, n_factors=8):
        self.n = n_factors

    def fit(self, R, **_):
        self.mu = R.data.mean()
        Rc = R.astype(float).copy()
        Rc.data -= self.mu
        svd = TruncatedSVD(self.n, random_state=SEED)
        self.U = svd.fit_transform(Rc)
        self.V = svd.components_
        return self

    def scores(self, u):
        return self.U[u] @ self.V + self.mu

    def scores_from_vector(self, r):
        """Fold-in for a NEW user: r = dense rating vector over all items."""
        r = np.asarray(r, dtype=float)
        c = np.where(r > 0, r - self.mu, 0.0)
        return (c @ self.V.T) @ self.V + self.mu

    def rating(self, u, i):
        return float(np.clip(self.scores(u)[i], 1, 5))


class Content:
    """User profile = average similarity to the items they liked in training."""
    def fit(self, R, content=None, **_):
        self.R, self.sim = R, content.sim
        return self

    def scores(self, u):
        return self.scores_from_vector(self.R[u].toarray().ravel())

    def scores_from_vector(self, r):
        liked = np.where(np.asarray(r) >= LIKE)[0]
        if len(liked) == 0:
            return np.zeros(self.sim.shape[0])
        return self.sim[liked].mean(0)


class Hybrid:
    """alpha * SVD + (1-alpha) * Content, both min-max scaled per user."""
    def __init__(self, alpha=0.5, n_factors=8):
        self.alpha, self.svd, self.content = alpha, SVD(n_factors), Content()

    def fit(self, R, content=None, **_):
        self.R = R
        self.svd.fit(R)
        self.content.fit(R, content=content)
        return self

    def scores(self, u):
        return self.scores_from_vector(self.R[u].toarray().ravel())

    def scores_from_vector(self, r):
        return (self.alpha * minmax(self.svd.scores_from_vector(r))
                + (1 - self.alpha) * minmax(self.content.scores_from_vector(r)))


# ------------------------------------------------------------ evaluation
def evaluate(model, R, test, uidx, iidx, k=K):
    rel = test[test.rating >= LIKE].groupby("user_id")["item_id"].apply(set)
    inv = {v: k_ for k_, v in iidx.items()}
    P, Rc = [], []
    for uid, items in rel.items():
        u = uidx[uid]
        s = np.array(model.scores(u), dtype=float)
        s[R[u].indices] = -np.inf                       # skip already-rated items
        top = {inv[j] for j in np.argsort(-s)[:k]}
        h = len(top & items)
        P.append(h / k)
        Rc.append(h / len(items))
    p, r = np.mean(P), np.mean(Rc)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def rating_error(model, test, uidx, iidx):
    e = [model.rating(uidx[r.user_id], iidx[r.item_id]) - r.rating for r in test.itertuples()]
    return float(np.mean(np.square(e)))                 # MSE


# ------------------------------------------------------------ main
if __name__ == "__main__":
    products = pd.read_csv("products.csv")
    df = pd.read_csv("interactions.csv")
    uidx = {u: i for i, u in enumerate(sorted(df.user_id.unique()))}
    iidx = {p: i for i, p in enumerate(products.ProductID)}   # same order as content.sim
    content = ContentRecommender().fit(products)

    trainval, test = holdout(df, 0.2, SEED)           # final test set
    train, val = holdout(trainval, 0.15, SEED + 1)    # validation set for tuning
    print(f"ratings: train {len(train)} | val {len(val)} | test {len(test)}")
    sparsity = 1 - len(df) / (len(uidx) * len(iidx))
    print(f"matrix {len(uidx)} users x {len(iidx)} items, sparsity {sparsity:.1%}\n")

    # ---- tune on validation set
    R_tr = to_matrix(train, uidx, iidx)
    best_n = min([2, 4, 6, 8, 12, 16], key=lambda n: rating_error(
        SVD(n).fit(R_tr), val, uidx, iidx))
    print(f"best SVD factors (val MSE): {best_n}")
    best_a = max([0, .2, .4, .5, .6, .8, 1], key=lambda a: evaluate(
        Hybrid(a, best_n).fit(R_tr, content=content), R_tr, val, uidx, iidx)[2])
    print(f"best hybrid alpha (val F1): {best_a}\n")

    # ---- final: retrain on train+val, score on test
    R = to_matrix(trainval, uidx, iidx)
    models = {
        "Popularity": Popularity(), "ItemCF": ItemCF(), "SVD": SVD(best_n),
        "Content": Content(), f"Hybrid(a={best_a})": Hybrid(best_a, best_n),
    }
    rows = []
    for name, m in models.items():
        m.fit(R, content=content)
        p, r, f1 = evaluate(m, R, test, uidx, iidx)
        mse = rating_error(m, test, uidx, iidx) if name == "SVD" else np.nan
        rows.append((name, p, r, f1, mse))
    res = pd.DataFrame(rows, columns=["Model", f"P@{K}", f"R@{K}", f"F1@{K}", "MSE"])
    print(res.round(3).to_string(index=False))
    res.round(4).to_csv("results.csv", index=False)

    import json
    json.dump({"n_factors": int(best_n), "alpha": float(best_a)}, open("params.json", "w"))
    print("\nSaved results.csv and params.json")
