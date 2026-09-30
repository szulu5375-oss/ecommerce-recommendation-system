
import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="ShopSmart Recommendation System",
    page_icon="🛒",
    layout="wide"
)

st.title("🛍️ ShopSmart")
st.subheader("E-commerce Product Recommendation System")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    import os 
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR,"products.csv")
    data = pd.read_csv(csv_path)
    return data

df = load_data()

# -----------------------------
# DATA PREPROCESSING
# -----------------------------
df["Description"] = df["Description"].fillna("")
df["Category"] = df["Category"].fillna("")

# Feature Engineering
df["Features"] = df["Category"] + " " + df["Description"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
feature_matrix = vectorizer.fit_transform(df["Features"])

# Similarity Matrix
similarity = cosine_similarity(feature_matrix)

# -----------------------------
# RECOMMENDATION FUNCTION
# (must be defined BEFORE the if/elif page chain)
# -----------------------------
def recommend(product_name, top_n=5):
    index = df[df["Name"] == product_name].index[0]

    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommended = []
    for i in scores[1:top_n + 1]:
        recommended.append(df.iloc[i[0]])

    return pd.DataFrame(recommended)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Choose a page",
    ["Home", "Products", "Recommendations", "Data Analysis"]
)

# -----------------------------
# PAGES
# -----------------------------
if page == "Home":

    st.header("Welcome to ShopSmart")
    st.write("An E-commerce Recommendation System built with Python.")

elif page == "Products":

    st.header("Available Products")

    search = st.text_input("Search product")

    # regex=False so characters like ( or + in a search don't crash it
    results = df[df["Name"].str.contains(search, case=False, regex=False)]

    st.dataframe(results[["Name", "Category", "Price"]])

elif page == "Recommendations":

    st.header("Get Recommendations")

    selected = st.selectbox("Choose a product", df["Name"])

    if st.button("Recommend Similar Products"):
        rec = recommend(selected)
        st.dataframe(rec[["Name", "Category", "Price"]])

elif page == "Data Analysis":

    st.header("Dataset Overview")

    st.dataframe(df)

    st.subheader("Products per Category")

    st.bar_chart(df["Category"].value_counts())
