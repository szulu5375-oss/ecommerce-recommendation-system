from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# DYNAMIC BASE DIRECTORY PATH
# -----------------------------
# Automatically targets the 'shopsmart/UbuntuCo' directory on Streamlit Cloud
BASE_DIR = Path(__file__).resolve().parent

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="ShopSmart Recommendation System",
    page_icon="🛒",
    layout="wide",
)

st.title("🛍️ ShopSmart")
st.subheader("Hybrid E-commerce Product Recommendation Engine")


# -----------------------------
# LOAD ARTIFACTS & DATASETS
# -----------------------------
@st.cache_data
def load_artifacts():
  customers = pd.read_csv(BASE_DIR / "customers_clean.csv")
  sd = pd.read_csv(BASE_DIR / "sd_clean.csv")
  content_scores = joblib.load(BASE_DIR / "content_scores.pkl")
  profile_scaled = joblib.load(BASE_DIR / "profile_scaled.pkl")
  knn = joblib.load(BASE_DIR / "knn_model.pkl")
  return customers, sd, content_scores, profile_scaled, knn


customers, sd, content_scores, profile_scaled, knn = load_artifacts()

def best_stockcode_for(product_description: str) -> str:
    rows = sd[sd["Description"] == product_description]
    if len(rows) == 0:
        return "N/A"
    return rows.sort_values("TotalRevenue", ascending=False).iloc[0]["StockCode"]

def hybrid_recommend(customer_id: int, cf_weight: float = 0.7, top_n: int = 3):
    match = customers.index[customers["CustomerID"] == customer_id]
    if len(match) == 0:
        return None, None

    idx = match[0]

    # Collaborative Filtering distance scoring
    distances, indices = knn.kneighbors([profile_scaled[idx]])
    distances, indices = distances[0][1:], indices[0][1:]
    similarity = 1 - distances

    neighbours = customers.iloc[indices].copy()
    neighbours["similarity"] = similarity

    votes = neighbours.groupby("PreferredProduct")["similarity"].sum()
    cf_scores = votes / votes.sum() if votes.sum() > 0 else votes

    # Combined Scoring
    all_products = content_scores.index
    combined = pd.Series(0.0, index=all_products)
    for product in all_products:
        cf_part = cf_scores.get(product, 0.0)
        content_part = content_scores.get(product, 0.0)
        combined[product] = cf_weight * cf_part + (1 - cf_weight) * content_part

    top = combined.sort_values(ascending=False).head(top_n)

    results = []
    for product, score in top.items():
        sku = best_stockcode_for(product)
        results.append({
            "Product": product,
            "Best SKU": sku,
            "Hybrid Score": round(score, 4)
        })

    known_pref = customers.loc[idx, "PreferredProduct"]
    return pd.DataFrame(results), known_pref

# Sidebar Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["Home", "Get Recommendations", "Customer Database"]
)

if page == "Home":
    st.header("Welcome to ShopSmart")
    st.write("A Hybrid Recommendation Engine combining User-Based Collaborative Filtering & Product Content Scoring.")

elif page == "Get Recommendations":
    st.header("Generate Recommendations")
    col1, col2 = st.columns([1, 2])

    with col1:
        customer_id = st.number_input(
            "Enter Customer ID:",
            min_value=int(customers["CustomerID"].min()),
            max_value=int(customers["CustomerID"].max()),
            value=int(customers["CustomerID"].iloc[0]),
            step=1
        )
        cf_weight = st.slider("Collaborative Filtering Weight", 0.0, 1.0, 0.7, 0.1)
        top_n = st.slider("Top N Recommendations", 1, 5, 3)
        btn = st.button("Generate Recommendations", type="primary")

    with col2:
        if btn:
            recs, true_pref = hybrid_recommend(customer_id, cf_weight, top_n)
            if recs is None:
                st.error("Customer ID not found.")
            else:
                st.subheader(f"Results for Customer #{customer_id}")
                st.info(f"**Known Preferred Product:** {true_pref}")
                st.dataframe(recs, use_container_width=True)

elif page == "Customer Database":
    st.header("Customer Profile Matrix")
    st.dataframe(customers, use_container_width=True)
