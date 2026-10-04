
import base64
import os

import streamlit as st
import pandas as pd
from html import escape

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="ShopSmart Recommendation System",
    page_icon="🛒",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #16182f;
        --navy: #242f49;
        --slate: #384358;
        --peach: #ffa586;
        --red: #b51a28;
        --wine: #541a2e;
        --muted: #697184;
        --page: #fff4ef;
        --border: #f1c3b5;
    }
    .stApp { background: var(--page); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    .stApp button {
        border-color: var(--red);
        background: var(--red);
        color: #fff4ef;
    }
    .stApp button:hover {
        border-color: var(--wine);
        background: var(--wine);
        color: #fff4ef;
    }
    .stApp input, .stApp textarea, .stApp [data-baseweb="select"] > div {
        border-color: var(--border);
    }
    .stApp [data-testid="stDataFrame"],
    .stApp [data-testid="stTable"] {
        border-color: var(--border);
    }
    div.block-container {
        max-width: 1060px;
        padding: 2.1rem 1.25rem 3rem;
    }
    .st-key-home-search {
        width: calc(100% + 2rem);
        margin-left: -1rem;
    }
    .st-key-home-search [data-testid="stTextInput"] input {
        min-height: 5rem;
        padding: 1rem 1.25rem;
        border: 2px solid var(--peach);
        border-radius: 0.85rem;
        background: #fff;
        color: var(--ink);
        font-size: 1.15rem;
        box-shadow: 0 4px 14px rgba(84, 26, 46, 0.12);
    }
    .st-key-home-search [data-testid="stTextInput"] input::placeholder {
        color: #686371;
        opacity: 1;
    }
    .st-key-home-search [data-testid="stTextInput"] input:focus {
        border-color: var(--red);
        box-shadow: 0 0 0 3px rgba(181, 26, 40, 0.17);
    }
    .shop-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        min-height: 3.2rem;
        padding: 0.35rem 0 1rem;
    }
    .shop-brand {
        color: var(--ink);
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.07em;
        line-height: 1.15;
    }
    .shop-brand span { color: var(--red); }
    .header-icon { font-size: 1.4rem; }
    [data-testid="stSidebar"] {
        background: var(--ink);
        border-right: 1px solid var(--border);
        color: #fff4ef;
    }
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stExpandSidebarButton"] {
        display: none;
    }
    [data-testid="stSidebar"] > div {
        padding-top: 1.1rem;
    }
    [data-testid="stSidebar"] .st-key-sidebar-nav-item button,
    [data-testid="stSidebar"] [class*="st-key-sidebar-nav-item-"] button {
        width: 100%;
        min-height: 3.35rem;
        border: 0;
        border-radius: 1.2rem;
        background: transparent;
        color: #e6e8ef;
        text-align: left;
        font-size: 0.95rem;
        font-weight: 600;
        transition: background 0.15s ease, color 0.15s ease;
    }
    [data-testid="stSidebar"] [class*="st-key-sidebar-nav-"] button > div {
        width: 100%;
        justify-content: flex-start !important;
    }
    [data-testid="stSidebar"] [class*="st-key-sidebar-nav-"] button p {
        width: 100%;
        text-align: left;
    }
    [data-testid="stSidebar"] .st-key-sidebar-nav-item button:hover,
    [data-testid="stSidebar"] [class*="st-key-sidebar-nav-item-"] button:hover {
        border: 0;
        background: var(--slate);
        color: var(--peach);
    }
    [data-testid="stSidebar"] [class*="st-key-sidebar-nav-active"] button {
        min-height: 3.8rem;
        border: 2px solid var(--red);
        border-radius: 1.4rem;
        background: var(--red);
        color: #fff4ef;
        box-shadow: 0 3px 0 var(--peach);
    }
    [data-testid="stSidebar"] [class*="st-key-sidebar-nav-active"] button:hover {
        border-color: var(--wine);
        background: var(--wine);
        color: #fff4ef;
    }
    [data-testid="stSidebar"] .shop-brand {
        color: #fff4ef;
    }
    [data-testid="stSidebar"] .shop-brand span {
        color: var(--peach);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #fff4ef;
    }
    .hero {
        min-height: 250px;
        display: flex;
        align-items: center;
        padding: 2rem;
        overflow: hidden;
        border-radius: 1.4rem;
        background-color: var(--peach);
        margin: 0.7rem 0 1.5rem;
    }
    .hero-copy { max-width: 440px; }
    .hero-eyebrow {
        color: var(--wine);
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }
    .hero-title {
        color: var(--ink);
        font-size: clamp(2rem, 5vw, 3.2rem);
        font-weight: 800;
        letter-spacing: -0.06em;
        line-height: 1;
        margin: 0 0 0.8rem;
    }
    .hero-description { color: var(--navy); margin: 0; }
    .section-title {
        color: var(--ink);
        font-size: 1.4rem;
        font-weight: 750;
        letter-spacing: -0.04em;
        margin: 1.2rem 0 0.7rem;
    }
    div[class*="st-key-category-card-"] button {
        position: relative;
        width: 100%;
        height: 7rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-end;
        padding: 0.35rem 0.2rem 0.5rem;
        border: 0;
        border-radius: 0.8rem;
        background: transparent;
        color: var(--ink);
        box-shadow: none;
        transition: background 0.15s ease, transform 0.15s ease;
    }
    div[class*="st-key-category-card-"] button::before {
        content: "";
        position: absolute;
        top: 0.4rem;
        left: 50%;
        width: 48px;
        height: 48px;
        transform: translateX(-50%);
        border-radius: 50%;
        background-position: center;
        background-size: cover;
        background-repeat: no-repeat;
    }
    div[class*="st-key-category-card-"] button p {
        margin: 0;
        color: var(--ink);
        font-size: 0.78rem;
        font-weight: 600;
        line-height: 1.15;
        text-align: center;
    }
    div[class*="st-key-category-card-"] button:hover {
        background: rgba(255,165,134,0.24);
        transform: translateY(-2px);
    }
    div[class*="st-key-category-card-"] button:focus:not(:active) {
        border: 0;
        box-shadow: 0 0 0 2px rgba(181, 26, 40, 0.22);
    }
    div[class*="st-key-category-more"] button {
        width: 100%;
        height: 7rem;
        min-height: 7rem;
        border: 0;
        background: transparent;
        color: var(--red);
        box-shadow: none;
    }
    div[class*="st-key-category-more"] button:hover {
        border: 0;
        background: rgba(255,165,134,0.24);
        color: var(--wine);
        box-shadow: none;
    }
    div[class*="st-key-category-more"] button [data-testid="stIconMaterial"] {
        font-size: 2rem;
    }
    div[class*="st-key-product-card-"] {
        box-sizing: border-box;
        height: 330px;
        min-height: 330px;
        padding: 0.65rem;
        border: 1px solid var(--border);
        border-radius: 0.85rem;
        background: #fffaf7;
    }
    div[class*="st-key-product-card-"] > div[data-testid="stVerticalBlock"] {
        height: 100%;
        gap: 0.35rem;
    }
    div[class*="st-key-product-card-"] {
        display: flex;
        flex-direction: column;
    }
    div[class*="st-key-product-card-"]
    > div[data-testid="stElementContainer"]:last-child {
        margin-top: auto;
    }
    .product-card { width: 100%; }
    .product-icon {
        height: 120px;
        display: grid;
        place-items: center;
        background: #fff;
        border-radius: 0.55rem;
        font-size: 4rem;
        margin-bottom: 0.5rem;
    }
    div[class*="st-key-product-card-"] [data-testid="stImage"] img {
        height: 120px;
        object-fit: contain;
        background: #fff;
        border-radius: 0.55rem;
    }
    .product-name {
        color: var(--ink);
        font-size: 0.95rem;
        font-weight: 650;
        min-height: 1.3em;
        text-align: center;
    }
    .product-category { display: none; }
    .product-price {
        color: var(--ink);
        font-weight: 500;
        margin: 0.35rem 0 0.55rem;
        text-align: center;
    }
    .product-price .sale-price { color: var(--red); font-weight: 800; }
    .product-price .original-price {
        margin-left: 0.35rem;
        color: var(--muted);
        font-size: 0.8rem;
        text-decoration: line-through;
    }
    .discount-badge {
        display: inline-block;
        margin: 0 auto 0.35rem;
        padding: 0.2rem 0.5rem;
        border-radius: 999px;
        background: #ffe2d7;
        color: var(--wine);
        font-size: 0.72rem;
        font-weight: 750;
    }
    div[class*="st-key-product-card-"] button {
        width: 100%;
        min-height: 2.45rem;
        border: 0;
        border-radius: 999px;
        background: var(--red);
        color: #fff;
        font-size: 0.88rem;
        font-weight: 650;
        transition: background 0.15s ease, transform 0.15s ease;
    }
    div[class*="st-key-product-card-"] button:hover {
        border: 0;
        background: var(--wine);
        color: #fff;
        transform: translateY(-1px);
    }
    .cart-quantity { padding-top: 0.35rem; text-align: center; font-weight: 700; }
    .st-key-sidebar-brand .shop-brand { font-size: 2.5rem; }
    @media (max-width: 640px) {
        div.block-container { padding: 0.65rem 0.8rem 2rem; }
        .st-key-home-search { width: 100%; margin-left: 0; }
        .hero { min-height: 205px; padding: 1.2rem; }
        .hero-devices { font-size: 3.8rem; }
        .hero-copy { max-width: 65%; }
        .shop-brand { font-size: 2.5rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "products.csv")
    data = pd.read_csv(csv_path)
    return data


def product_image_path(product_id):
    image_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "Images")
    )
    image_names = [
        f"{product_id}.jpeg",
        f"{product_id}.jpg",
        f"{product_id}.png",
        f"{product_id}.webp",
    ]
    if str(product_id) == "4":
        image_names.append("4.peg.jpeg")

    for image_name in image_names:
        image_path = os.path.join(image_dir, image_name)
        if os.path.isfile(image_path):
            return image_path
    return None


def category_image_path(category):
    image_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "Images")
    )
    image_extensions = (".jpeg", ".jpg", ".png", ".webp")
    category_filename = str(category).casefold()
    for image_name in os.listdir(image_dir):
        filename, extension = os.path.splitext(image_name)
        if (
            filename.casefold() == category_filename
            and extension.casefold() in image_extensions
        ):
            return os.path.join(image_dir, image_name)
    return None


df = load_data()
CATEGORIES = sorted(df["Category"].dropna().astype(str).unique())
CATEGORY_PALETTE = [
    "#16182f",
    "#242f49",
    "#384358",
    "#ffa586",
    "#b51a28",
    "#541a2e",
    "#6b7488",
    "#d75e68",
    "#bd7660",
    "#8a5368",
]
CATEGORY_COLORS = {
    category: CATEGORY_PALETTE[index % len(CATEGORY_PALETTE)]
    for index, category in enumerate(CATEGORIES)
}

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
def recommend_for_cart(cart, top_n=5):
    product_positions = {
        str(product_id): position
        for position, product_id in enumerate(df["ProductID"])
    }
    cart_items = [
        (product_positions[product_id], quantity)
        for product_id, quantity in cart.items()
        if product_id in product_positions and quantity > 0
    ]
    if not cart_items:
        return df.iloc[0:0]

    seed_positions = [position for position, _ in cart_items]
    quantities = [quantity for _, quantity in cart_items]
    scores = similarity[seed_positions].T.dot(quantities)
    cart_positions = set(seed_positions)
    ranked_positions = sorted(
        (position for position in range(len(df)) if position not in cart_positions),
        key=lambda position: scores[position],
        reverse=True,
    )
    return df.iloc[ranked_positions[:top_n]]

def product_icon(name, category):
    name = str(name).lower()
    category = str(category).lower()
    if any(word in name for word in ("headphone", "earbud", "speaker")):
        return "🎧"
    if any(word in name for word in ("mouse", "keyboard")):
        return "🖱️" if "mouse" in name else "⌨️"
    if any(word in name for word in ("phone", "tablet")):
        return "📱"
    if any(word in name for word in ("watch", "clock")):
        return "⌚"
    if any(word in name for word in ("computer", "laptop", "monitor", "webcam")):
        return "💻"
    if any(word in name for word in ("charger", "adapter", "kettle")):
        return "🔌"
    if any(word in name for word in ("dress", "skirt", "blouse", "sweater")):
        return "👗"
    if any(word in name for word in ("shirt", "trouser", "hoodie", "jacket", "jeans")):
        return "👕"
    if any(word in name for word in ("bag", "backpack", "tote", "handbag")):
        return "👜"
    if any(word in name for word in ("shoe", "sneaker", "boot")):
        return "👟"
    if "furniture" in category:
        return "🪑"
    if "home" in category:
        return "🏠"
    if category in ("electronics", "tech devices"):
        return "🔌"
    if "women" in category:
        return "👗"
    if "men" in category or "fashion" in category:
        return "👕"
    if "bag" in category:
        return "👜"
    if "watch" in category:
        return "⌚"
    if "footwear" in category:
        return "👟"
    return "📦"


def render_product_cards(products):
    if products.empty:
        st.info("No products found.")
        return

    columns = st.columns(min(4, len(products)))
    for index, (_, product) in enumerate(products.iterrows()):
        with columns[index % len(columns)]:
            name = escape(str(product["Name"]))
            category = escape(str(product["Category"]))
            icon = product_icon(product["Name"], product["Category"])
            original_price = float(product["Price"])
            discount_percent = float(product.get("DiscountPercent", 0) or 0)
            sale_price = discounted_price(original_price, discount_percent)
            if discount_percent > 0:
                price = (
                    f'<span class="sale-price">R {sale_price:,.2f}</span>'
                    f'<span class="original-price">R {original_price:,.2f}</span>'
                )
                discount_badge = f'<div class="discount-badge">{discount_percent:g}% OFF</div>'
            else:
                price = f"R {original_price:,.2f}"
                discount_badge = ""
            product_id = str(product["ProductID"])
            image_path = product_image_path(product_id)
            icon_markup = "" if image_path else f'<div class="product-icon">{icon}</div>'
            with st.container(key=f"product-card-{product_id}"):
                if image_path:
                    st.image(image_path, use_container_width=True)
                card_html = (
                    f'<div class="product-card">'
                    f'{icon_markup}'
                    f'<div class="product-name">{name}</div>'
                    f'<div class="product-category">{category}</div>'
                    f'{discount_badge}'
                    f'<div class="product-price">{price}</div>'
                    f'</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)
                quantity_in_cart = st.session_state.cart.get(product_id, 0)
                if quantity_in_cart:
                    minus_col, quantity_col, plus_col = st.columns([1, 1.2, 1])
                    with minus_col:
                        st.button(
                            "−",
                            key=f"product_decrease_{product_id}",
                            help=f"Remove one {name} from your cart",
                            on_click=change_cart_quantity,
                            args=(product_id, -1),
                            use_container_width=True,
                        )
                    with quantity_col:
                        st.markdown(
                            f'<div class="cart-quantity">{quantity_in_cart}</div>',
                            unsafe_allow_html=True,
                        )
                    with plus_col:
                        st.button(
                            "+",
                            key=f"product_increase_{product_id}",
                            help=f"Add one more {name} to your cart",
                            on_click=change_cart_quantity,
                            args=(product_id, 1),
                            use_container_width=True,
                        )
                else:
                    st.button(
                        "Add to Cart",
                        key=f"add_{product_id}",
                        use_container_width=True,
                        on_click=change_cart_quantity,
                        args=(product_id, 1),
                    )


def discounted_price(price, discount_percent):
    return float(price) * (1 - float(discount_percent or 0) / 100)


def with_rand_prices(products):
    display_products = products.copy()
    discounts = pd.to_numeric(
        display_products["DiscountPercent"]
        if "DiscountPercent" in display_products
        else pd.Series(0, index=display_products.index),
        errors="coerce",
    ).fillna(0)
    original_prices = pd.to_numeric(display_products["Price"], errors="raise")
    display_products["Original Price"] = original_prices.map(lambda price: f"R {price:,.2f}")
    display_products["Price"] = (
        original_prices * (1 - discounts / 100)
    ).map(lambda price: f"R {price:,.2f}")
    if "DiscountPercent" in display_products:
        display_products["Discount"] = discounts.map(
            lambda discount: f"{discount:g}%" if discount > 0 else "—"
        )
        display_products = display_products.drop(columns=["DiscountPercent"])
    return display_products


def navigate_to_page(page):
    st.session_state.page = page
    st.session_state.category_filter = ""


def navigate_to_category(category):
    navigate_to_page("Products")
    st.session_state.category_filter = category


def toggle_categories():
    st.session_state.show_all_categories = not st.session_state.show_all_categories


def change_cart_quantity(product_id, amount):
    cart = dict(st.session_state.cart)
    new_quantity = cart.get(product_id, 0) + amount
    if new_quantity > 0:
        cart[product_id] = new_quantity
    else:
        cart.pop(product_id, None)
    st.session_state.cart = cart


def open_promotions():
    navigate_to_page("Promotions")


def open_popular_products():
    navigate_to_page("Popular Products")


if "page" not in st.session_state:
    st.session_state.page = "Home"
if "category_filter" not in st.session_state:
    st.session_state.category_filter = ""
elif st.session_state.category_filter not in CATEGORIES:
    st.session_state.category_filter = ""
if "show_all_categories" not in st.session_state:
    st.session_state.show_all_categories = False
if "cart" not in st.session_state:
    st.session_state.cart = {}
pages = [
    "Home",
    "Products",
    "Promotions",
    "Popular Products",
    "Recommendations",
    "Data Analysis",
    "Cart",
]
page_icons = {
    "Home": "home",
    "Products": "shopping_bag",
    "Promotions": "sell",
    "Popular Products": "star",
    "Recommendations": "auto_awesome",
    "Data Analysis": "analytics",
    "Cart": "shopping_cart",
}
st.sidebar.markdown(
    '<div class="shop-brand">Shop<span>Smart</span></div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    '<div style="padding:0.9rem 0 0.4rem;color:#ffa586;font-size:0.72rem;'
    'font-weight:700;letter-spacing:0.12em">NAVIGATION</div>',
    unsafe_allow_html=True,
)
for page in pages:
    nav_key = "sidebar-nav-active" if st.session_state.page == page else "sidebar-nav-item"
    with st.sidebar.container(key=f"{nav_key}-{page.lower().replace(' ', '-')}"):
        st.button(
            page,
            icon=f":material/{page_icons[page]}:",
            key=f"navigate_{page.lower().replace(' ', '_')}",
            on_click=navigate_to_page,
            args=(page,),
            use_container_width=True,
        )

if st.session_state.page == "Home":
    with st.container(key="home-search"):
        search = st.text_input(
            "Search products",
            placeholder="⌕  Search for products, brands and more...",
            label_visibility="collapsed",
        )
    banner_image = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "Images", "Banner.jpeg")
    )
    banner_background = ""
    if os.path.isfile(banner_image):
        with open(banner_image, "rb") as image_file:
            banner_data = base64.b64encode(image_file.read()).decode("ascii")
        banner_background = (
            "background-image: linear-gradient(90deg, #ffa586 0%, "
            "rgba(255,165,134,0.98) 30%, rgba(255,165,134,0.78) 52%, "
            "rgba(22,24,47,0.48) 100%), "
            f"url('data:image/jpeg;base64,{banner_data}');"
        )
    st.markdown(
        f"""
        <div class="hero" style="{banner_background} background-size: cover;
             background-position: 100% center; background-repeat: no-repeat;">
            <div class="hero-copy">
                <div class="hero-eyebrow">Something for every part of life</div>
                <div class="hero-title">Find what makes life better</div>
                <p class="hero-description">Explore electronics, fashion, home essentials, and furniture—all in one place.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Shop by category</div>', unsafe_allow_html=True)
    categories = sorted(df["Category"].dropna().astype(str).unique())
    category_image_rules = []
    for index, category in enumerate(categories):
        image_path = category_image_path(category)
        if image_path:
            extension = os.path.splitext(image_path)[1].lower()
            mime_type = {
                ".jpeg": "image/jpeg",
                ".jpg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
            }[extension]
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("ascii")
            category_image_rules.append(
                f'.st-key-category-card-{index} button::before '
                f'{{ background-image: url("data:{mime_type};base64,{image_data}"); }}'
            )
    if category_image_rules:
        st.markdown(
            f"<style>{''.join(category_image_rules)}</style>",
            unsafe_allow_html=True,
        )
    category_columns = st.columns(6)
    for index, category in enumerate(categories[:5]):
        with category_columns[index]:
            with st.container(key=f"category-card-{index}"):
                st.button(
                    category,
                    key=f"category_{category}",
                    use_container_width=True,
                    on_click=navigate_to_category,
                    args=(category,),
                )
    if len(categories) > 5:
        with category_columns[5]:
            with st.container(key="category-more"):
                st.button(
                    "",
                    icon=(
                        ":material/keyboard_arrow_up:"
                        if st.session_state.show_all_categories
                        else ":material/keyboard_arrow_down:"
                    ),
                    key="toggle_categories",
                    help=(
                        "See fewer categories"
                        if st.session_state.show_all_categories
                        else "See more categories"
                    ),
                    on_click=toggle_categories,
                )
    if st.session_state.show_all_categories:
        extra_categories = categories[5:]
        for row_start in range(0, len(extra_categories), 6):
            extra_category_columns = st.columns(6)
            for column_index, category in enumerate(
                extra_categories[row_start : row_start + 6]
            ):
                category_index = row_start + column_index + 5
                with extra_category_columns[column_index]:
                    with st.container(key=f"category-card-{category_index}"):
                        st.button(
                            category,
                            key=f"category_{category}",
                            use_container_width=True,
                            on_click=navigate_to_category,
                            args=(category,),
                        )

    filtered = df
    if search:
        filtered = filtered[filtered["Name"].str.contains(search, case=False, regex=False)]

    discounted = filtered[
        pd.to_numeric(filtered["DiscountPercent"], errors="coerce").fillna(0) > 0
    ]
    st.markdown('<div class="section-title">Promotions</div>', unsafe_allow_html=True)
    render_product_cards(discounted.head(4))
    if not discounted.empty:
        st.button(
            "View all",
            key="view_all_promotions",
            on_click=open_promotions,
        )
    popular = filtered[
        pd.to_numeric(filtered["DiscountPercent"], errors="coerce").fillna(0) == 0
    ]
    st.markdown('<div class="section-title">Popular products</div>', unsafe_allow_html=True)
    render_product_cards(popular.head(4))
    if not popular.empty:
        st.button(
            "View all",
            key="view_all_popular_products",
            on_click=open_popular_products,
        )

elif st.session_state.page == "Products":
    st.header("Available Products")
    selected_category = st.session_state.category_filter
    if selected_category:
        st.caption(f"Category: {selected_category}")
    search = st.text_input("Search product")
    results = df
    if selected_category:
        results = results[results["Category"] == selected_category]
    if search:
        results = results[results["Name"].str.contains(search, case=False, regex=False)]
    render_product_cards(results.reset_index(drop=True))

elif st.session_state.page == "Popular Products":
    st.header("All Popular Products")
    popular = df[
        pd.to_numeric(df["DiscountPercent"], errors="coerce").fillna(0) == 0
    ].reset_index(drop=True)
    st.caption(f"{len(popular)} products")
    render_product_cards(popular)

elif st.session_state.page == "Promotions":
    st.header("All Promotions")
    discounted = df[
        pd.to_numeric(df["DiscountPercent"], errors="coerce").fillna(0) > 0
    ].reset_index(drop=True)
    st.caption(f"{len(discounted)} discounted products")
    render_product_cards(discounted)

elif st.session_state.page == "Recommendations":
    st.header("Recommended for You")
    cart = st.session_state.cart
    if not cart:
        st.info("Add products to your cart to see personalised recommendations.")
    else:
        rec = recommend_for_cart(cart)
        if rec.empty:
            st.info("No additional products to recommend right now.")
        else:
            st.caption("Based on the products in your cart")
            with st.container(key="recommendations-results"):
                render_product_cards(rec)

elif st.session_state.page == "Cart":
    st.header("Your Cart")
    cart = st.session_state.cart
    if not cart:
        st.info("Your cart is empty. Add products using the Add to Cart buttons.")
    else:
        cart_products = df[df["ProductID"].astype(str).isin(cart)].copy()
        total = 0.0
        for _, product in cart_products.iterrows():
            product_id = str(product["ProductID"])
            quantity = cart[product_id]
            discount_percent = float(product.get("DiscountPercent", 0) or 0)
            unit_price = discounted_price(product["Price"], discount_percent)
            line_total = unit_price * quantity
            total += line_total
            item_col, minus_col, qty_col, plus_col, sum_col = st.columns([4, 0.7, 1.2, 0.7, 1.5])
            with item_col:
                item_price = f"R {unit_price:,.2f} each"
                if discount_percent > 0:
                    item_price += f" (was R {float(product['Price']):,.2f}; {discount_percent:g}% off)"
                st.markdown(
                    f"**{escape(str(product['Name']))}**  \n"
                    f"{escape(str(product['Category']))} · {item_price}"
                )
            with minus_col:
                st.button(
                    "−",
                    key=f"cart_decrease_{product_id}",
                    on_click=change_cart_quantity,
                    args=(product_id, -1),
                    use_container_width=True,
                )
            with qty_col:
                st.markdown(f'<div class="cart-quantity">{quantity}</div>', unsafe_allow_html=True)
            with plus_col:
                st.button(
                    "+",
                    key=f"cart_increase_{product_id}",
                    on_click=change_cart_quantity,
                    args=(product_id, 1),
                    use_container_width=True,
                )
            with sum_col:
                st.markdown(f"**R {line_total:,.2f}**")
            st.divider()

        delivery_fee = 75.0
        st.markdown(f"**Subtotal: R {total:,.2f}**")
        st.markdown(f"**Delivery: R {delivery_fee:,.2f}**")
        st.subheader(f"Total: R {total + delivery_fee:,.2f}")
        if st.button("Checkout", key="checkout_button", use_container_width=True):
            st.success(
                f"Checkout confirmed. Your total, including delivery, is "
                f"R {total + delivery_fee:,.2f}. Your cart has been kept."
            )

elif st.session_state.page == "Data Analysis":
    st.header("Dataset Overview")
    st.dataframe(with_rand_prices(df), use_container_width=True)
    st.subheader("Products per Category")
    category_counts = (
        df["Category"].value_counts().rename_axis("Category")
        .reset_index(name="Products")
    )
    color_encoding = {
        "field": "Category",
        "type": "nominal",
        "scale": {
            "domain": CATEGORIES,
            "range": [CATEGORY_COLORS[category] for category in CATEGORIES],
        },
        "legend": {"title": "Category"},
    }
    bar_chart, pie_chart = st.columns(2)
    with bar_chart:
        st.vega_lite_chart(
            category_counts,
            {
                "mark": "bar",
                "encoding": {
                    "x": {
                        "field": "Category",
                        "type": "nominal",
                        "sort": CATEGORIES,
                        "axis": {"title": "Category", "labelAngle": -45},
                    },
                    "y": {
                        "field": "Products",
                        "type": "quantitative",
                        "axis": {"title": "Number of products"},
                    },
                    "color": color_encoding,
                    "tooltip": [
                        {"field": "Category", "type": "nominal"},
                        {"field": "Products", "type": "quantitative"},
                    ],
                },
            },
            use_container_width=True,
        )
    with pie_chart:
        st.vega_lite_chart(
            category_counts,
            {
                "mark": {"type": "arc", "outerRadius": 115},
                "encoding": {
                    "theta": {"field": "Products", "type": "quantitative"},
                    "color": color_encoding,
                    "tooltip": [
                        {"field": "Category", "type": "nominal"},
                        {"field": "Products", "type": "quantitative"},
                    ],
                },
            },
            use_container_width=True,
        )
    st.subheader("Category Counts Across Categories")
    st.vega_lite_chart(
        category_counts,
        {
            "mark": {"type": "line", "point": True, "color": "#541a2e"},
            "encoding": {
                "x": {
                    "field": "Category",
                    "type": "nominal",
                    "sort": CATEGORIES,
                    "axis": {"title": "Category", "labelAngle": -45},
                },
                "y": {
                    "field": "Products",
                    "type": "quantitative",
                    "axis": {"title": "Number of products"},
                },
                "tooltip": [
                    {"field": "Category", "type": "nominal"},
                    {"field": "Products", "type": "quantitative"},
                ],
            },
        },
        use_container_width=True,
    )
