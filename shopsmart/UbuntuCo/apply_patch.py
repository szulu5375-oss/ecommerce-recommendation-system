"""
apply_patch.py - wires customer logging, the learning recommender and the new
statistics page into shopsmart/UbuntuCo/app.py.

Run ONCE, from the folder that contains app.py:
    python apply_patch.py
A backup is saved as app.py.bak. If any edit can't find its place, nothing is changed.
"""
import shutil
import sys

PATH = "app.py"
src = open(PATH, encoding="utf-8").read()
if "interaction_log" in src:
    sys.exit("app.py already contains the patch - nothing to do.")


def replace_once(text, old, new, label):
    if text.count(old) != 1:
        sys.exit(f"Could not apply edit '{label}' (found {text.count(old)} matches). "
                 "No changes were made.")
    return text.replace(old, new)


def replace_between(text, start, end, new, label):
    i, j = text.find(start), text.find(end)
    if i < 0 or j < 0 or j < i:
        sys.exit(f"Could not apply edit '{label}'. No changes were made.")
    return text[:i] + new + text[j:]


# 1. imports
src = replace_once(
    src,
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n\n"
    "import uuid\n"
    "import interaction_log\n"
    "from recommender_service import RecommenderService\n"
    "from stats_page import render_stats\n",
    "imports",
)

# 2. create tables + session id
src = replace_once(
    src,
    'initialize_database()\nif "authenticated_user" not in st.session_state:',
    'initialize_database()\ninteraction_log.init(DATABASE_PATH)\n'
    'if "session_id" not in st.session_state:\n'
    '    st.session_state.session_id = uuid.uuid4().hex[:12]\n'
    'if "authenticated_user" not in st.session_state:',
    "database + session id",
)

# 3. recommender helpers + new recommend_for_cart
NEW_HELPERS = '''def current_actor():
    """Who is shopping: 'u_<id>' when logged in, otherwise 'g_<session>' (guest)."""
    user = st.session_state.authenticated_user
    return f"u_{user['id']}" if user else f"g_{st.session_state.session_id}"


@st.cache_resource(show_spinner=False, max_entries=2)
def _build_recommender(bucket):
    events = interaction_log.load_events(DATABASE_PATH)
    return RecommenderService(extra_ratings=interaction_log.to_ratings(events))


def get_recommender():
    # retrains automatically after every 5 new customer actions
    return _build_recommender(interaction_log.count_events(DATABASE_PATH) // 5)


def personal_recommendations(n=4):
    recs = get_recommender().for_user(current_actor(), n)
    return recs.drop(columns=["match"]).reset_index(drop=True)


def recommend_for_cart(cart):
    if not cart:
        return df.iloc[0:0].copy()
    recs = get_recommender().for_cart([int(p) for p in cart], n=8)
    return recs.drop(columns=["match"]).reset_index(drop=True)


'''
src = replace_between(
    src, "def recommend_for_cart(cart):", "def product_icon(name, category):",
    NEW_HELPERS, "recommend_for_cart",
)

# 4. log cart additions
src = replace_once(
    src,
    "    st.session_state.cart = cart\n    st.session_state.checkout_confirmed = False\n",
    "    st.session_state.cart = cart\n    st.session_state.checkout_confirmed = False\n"
    "    if amount > 0:\n"
    "        interaction_log.log_event(DATABASE_PATH, current_actor(), int(product_id), \"cart_add\")\n",
    "cart_add logging",
)

# 5. remember cart contents when a PayPal order is created
src = replace_once(
    src,
    "            st.session_state.paypal_order_id = order_id\n",
    "            interaction_log.save_order_items(DATABASE_PATH, order_id, st.session_state.cart)\n"
    "            st.session_state.paypal_order_id = order_id\n",
    "save order items",
)

# 6. turn a paid order into purchase events
src = replace_once(
    src,
    "    else:\n        st.session_state.cart = {}\n",
    "    else:\n"
    "        interaction_log.log_purchases_for_order(DATABASE_PATH, order_id, f\"u_{user['id']}\")\n"
    "        st.session_state.cart = {}\n",
    "purchase logging",
)

# 7. Recommendations page: log views + personal picks
src = replace_once(
    src,
    '    if st.button("Recommend Similar Products"):\n        rec = recommend(selected)\n',
    '    if st.button("Recommend Similar Products"):\n'
    '        interaction_log.log_event(\n'
    '            DATABASE_PATH, current_actor(),\n'
    '            int(df.loc[df["Name"] == selected, "ProductID"].iloc[0]), "view",\n'
    '        )\n'
    '        rec = recommend(selected)\n',
    "view logging",
)
src = replace_once(
    src,
    '        with st.container(key="recommendations-results"):\n            render_product_cards(rec)\n',
    '        with st.container(key="recommendations-results"):\n            render_product_cards(rec)\n'
    '    st.markdown(\'<div class="section-title">Picked for you</div>\', unsafe_allow_html=True)\n'
    '    render_product_cards(personal_recommendations(), key_prefix="for-you-")\n',
    "picked for you",
)

# 8. replace the Data Analysis page
src = replace_between(
    src,
    'elif st.session_state.page == "Data Analysis":',
    'if st.session_state.cart and st.session_state.page in {',
    'elif st.session_state.page == "Data Analysis":\n'
    '    st.header("Store & Recommender Statistics")\n'
    '    render_stats(df, DATABASE_PATH, get_recommender())\n\n',
    "Data Analysis page",
)

shutil.copy(PATH, PATH + ".bak")
open(PATH, "w", encoding="utf-8").write(src)
print("Done. Backup saved as app.py.bak. Now run:  streamlit run app.py")
