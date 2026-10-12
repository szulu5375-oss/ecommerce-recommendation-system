import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = "data"
K = 25

# Fill ONLY if none of your tables has a category column.
# Keys must be the exact product Description, values one of your 5 categories.
MANUAL_CATEGORY_MAP = {
    "Wall Clock": "Furniture",
    "USB Cable": "Electronics",
    "White Mug": "Accessories",
    "Desk Lamp": "Furniture",
    "Backpack": "Accessories",
    "Headphones": "Electronics",
    "Blue Pen": "Stationery",
    "Office Chair": "Furniture",
    "Wireless Mouse": "Electronics",
    "T-shirt": "Apparel",
    "Notebook": "Stationery",
}

cf = pd.read_csv(f"{DATA_DIR}/Customer_Features.csv", sep=";")
pf = pd.read_csv(f"{DATA_DIR}/Product_Features.csv", sep=";")
sd = pd.read_csv(f"{DATA_DIR}/StockCode_Detail_Raw.csv", sep=";")

# ---- clean text keys ----
pf["Description"] = pf["Description"].astype(str).str.strip()
sd["Description"] = sd["Description"].astype(str).str.strip()
sd["StockCode"] = sd["StockCode"].astype(str).str.strip()
cf["PreferredProduct"] = cf["PreferredProduct"].astype(str).str.strip()

# ---- customer side: category-share profiles only (a cart can't supply the rest) ----
share_cols = [c for c in cf.columns if c.startswith("CategoryShare_")]
categories = [c.replace("CategoryShare_", "") for c in share_cols]
for c in share_cols + ["PreferredProduct"]:
    assert c == "PreferredProduct" or pd.api.types.is_numeric_dtype(cf[c]), (
        f"{c} was not read as numeric - if decimals use commas, add decimal=',' to read_csv")

customers = cf[cf[share_cols].fillna(0).sum(axis=1) > 0].reset_index(drop=True)  # cosine needs non-zero vectors
X = customers[share_cols].fillna(0).to_numpy()
knn = NearestNeighbors(n_neighbors=min(K, len(X)), metric="cosine", algorithm="brute").fit(X)

# ---- content score (same formula as your original) ----
mm = MinMaxScaler()
pf_s = pf.copy()
pf_s[["PopRankScore", "RevenueScore"]] = mm.fit_transform(pf[["PopularityRank", "TotalRevenue"]])
pf_s["PopRankScore"] = 1 - pf_s["PopRankScore"]
pf_s["ReturnPenalty"] = MinMaxScaler().fit_transform(pf[["ReturnRate"]])
pf_s["content_score"] = (0.5 * pf_s["PopRankScore"] + 0.4 * pf_s["RevenueScore"]
                         - 0.1 * pf_s["ReturnPenalty"])
content_scores = pf_s.set_index("Description")["content_score"].to_dict()

# ---- product -> category ----
def find_category_map():
    for df in (pf, sd):
        col = next((c for c in df.columns
                    if c.lower() in {"category", "productcategory", "product_category"}), None)
        if col:
            d = df.dropna(subset=[col]).drop_duplicates("Description")
            return d.set_index("Description")[col].astype(str).str.strip().to_dict()
    return None

cat_map = find_category_map() or MANUAL_CATEGORY_MAP
canon = {c.lower(): c for c in categories}
product_category = {}
for p in content_scores:
    raw = cat_map.get(p)
    if raw is None or raw.lower() not in canon:
        raise ValueError(
            f"No valid category for product '{p}'. Valid categories: {categories}. "
            "Fill MANUAL_CATEGORY_MAP at the top of this file.")
    product_category[p] = canon[raw.lower()]

# ---- SKU lookups ----
best = sd.sort_values("TotalRevenue", ascending=False).drop_duplicates("Description")
product_sku = best.set_index("Description")["StockCode"].to_dict()
sku_to_product = dict(zip(sd["StockCode"], sd["Description"]))

joblib.dump({
    "categories": categories,
    "knn": knn,
    "neighbour_pref": customers["PreferredProduct"].to_numpy(),
    "content_scores": content_scores,
    "product_category": product_category,
    "product_sku": product_sku,
    "sku_to_product": sku_to_product,
}, "artifacts.joblib")
print("Saved artifacts.joblib")