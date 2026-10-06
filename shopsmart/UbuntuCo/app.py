
import base64
from contextlib import contextmanager
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
from html import escape
from streamlit.errors import StreamlitSecretNotFoundError

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

DATABASE_PATH = os.environ.get(
    "SHOPSMART_DB_PATH",
    os.path.join(os.path.dirname(__file__), "shopsmart.sqlite3"),
)
PASSWORD_HASH_ITERATIONS = 310_000


@contextmanager
def get_database_connection():
    database_directory = os.path.dirname(os.path.abspath(DATABASE_PATH))
    os.makedirs(database_directory, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_database():
    with get_database_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                full_name TEXT NOT NULL,
                password_salt BLOB NOT NULL,
                password_hash BLOB NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS paypal_orders (
                order_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                amount_cents INTEGER NOT NULL,
                status TEXT NOT NULL,
                capture_id TEXT,
                created_at TEXT NOT NULL
            );
            """
        )


def hash_password(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_HASH_ITERATIONS,
    )


def create_user(full_name, email, password):
    normalized_email = email.strip().casefold()
    if not full_name.strip():
        raise ValueError("Enter your name.")
    if "@" not in normalized_email or any(char.isspace() for char in normalized_email):
        raise ValueError("Enter a valid email address.")
    if len(password) < 8:
        raise ValueError("Use a password with at least 8 characters.")

    salt = secrets.token_bytes(16)
    password_hash = hash_password(password, salt)
    with get_database_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO users (email, full_name, password_salt, password_hash, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                normalized_email,
                full_name.strip(),
                salt,
                password_hash,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        return {
            "id": cursor.lastrowid,
            "full_name": full_name.strip(),
            "email": normalized_email,
        }


def authenticate_user(email, password):
    normalized_email = email.strip().casefold()
    with get_database_connection() as connection:
        user = connection.execute(
            """
            SELECT id, email, full_name, password_salt, password_hash
            FROM users WHERE email = ?
            """,
            (normalized_email,),
        ).fetchone()
    if user is None or not hmac.compare_digest(
        hash_password(password, user["password_salt"]),
        user["password_hash"],
    ):
        return None
    return {
        "id": user["id"],
        "email": user["email"],
        "full_name": user["full_name"],
    }


def get_paypal_configuration():
    secrets_path = os.path.join(
        os.path.dirname(__file__),
        ".streamlit",
        "secrets.toml",
    )
    if os.path.isfile(secrets_path):
        with open(secrets_path, "rb") as secrets_file:
            paypal_secrets = tomllib.load(secrets_file).get("paypal", {})
    else:
        try:
            paypal_secrets = st.secrets["paypal"]
        except (KeyError, StreamlitSecretNotFoundError):
            paypal_secrets = {}

    client_id = (
        os.environ.get("PAYPAL_CLIENT_ID") or paypal_secrets.get("client_id", "")
    ).strip()
    client_secret = (
        os.environ.get("PAYPAL_CLIENT_SECRET")
        or paypal_secrets.get("client_secret", "")
    ).strip()
    environment = (
        os.environ.get("PAYPAL_ENVIRONMENT")
        or paypal_secrets.get("environment", "sandbox")
    ).lower()
    return_url = os.environ.get("PAYPAL_RETURN_URL") or paypal_secrets.get(
        "return_url", ""
    )
    cancel_url = os.environ.get("PAYPAL_CANCEL_URL") or paypal_secrets.get(
        "cancel_url", ""
    )
    missing = [
        setting
        for setting, value in (
            ("client_id", client_id),
            ("client_secret", client_secret),
            ("return_url", return_url),
            ("cancel_url", cancel_url),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "PayPal Sandbox is not configured. Set "
            + ", ".join(f"paypal.{setting}" for setting in missing)
            + " in Streamlit secrets before accepting payments."
        )
    if client_id.startswith("YOUR_PAYPAL_") or client_secret.startswith(
        "YOUR_PAYPAL_"
    ):
        raise ValueError(
            "Replace the PayPal credential placeholders in "
            ".streamlit/secrets.toml with the Client ID and Secret from the "
            "same PayPal Sandbox REST app."
        )
    if environment not in {"sandbox", "live"}:
        raise ValueError("PAYPAL_ENVIRONMENT must be either 'sandbox' or 'live'.")

    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "api_base": (
            "https://api-m.sandbox.paypal.com"
            if environment == "sandbox"
            else "https://api-m.paypal.com"
        ),
        "return_url": return_url,
        "cancel_url": cancel_url,
    }


def paypal_api_request(method, path, payload=None):
    configuration = get_paypal_configuration()
    credentials = base64.b64encode(
        f"{configuration['client_id']}:{configuration['client_secret']}".encode("utf-8")
    ).decode("ascii")
    token_request = urllib.request.Request(
        f"{configuration['api_base']}/v1/oauth2/token",
        data=b"grant_type=client_credentials",
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(token_request, timeout=20) as response:
            access_token = json.loads(response.read())["access_token"]
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"PayPal authentication failed: {details}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not connect to PayPal: {error.reason}") from error
    except (KeyError, json.JSONDecodeError) as error:
        raise RuntimeError("PayPal returned an invalid authentication response.") from error

    request_data = None
    headers = {"Authorization": f"Bearer {access_token}"}
    if payload is not None:
        request_data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    api_request = urllib.request.Request(
        f"{configuration['api_base']}{path}",
        data=request_data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(api_request, timeout=20) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"PayPal request failed: {details}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not connect to PayPal: {error.reason}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError("PayPal returned an invalid response.") from error


def create_paypal_order(user_id, amount_cents):
    configuration = get_paypal_configuration()
    amount = f"{amount_cents / 100:.2f}"
    order = paypal_api_request(
        "POST",
        "/v2/checkout/orders",
        {
            "intent": "CAPTURE",
            "purchase_units": [
                {
                    "amount": {
                        "currency_code": "ZAR",
                        "value": amount,
                    }
                }
            ],
            "application_context": {
                "return_url": configuration["return_url"],
                "cancel_url": configuration["cancel_url"],
                "user_action": "PAY_NOW",
                "shipping_preference": "NO_SHIPPING",
            },
        },
    )
    order_id = order.get("id")
    approval_url = next(
        (
            link["href"]
            for link in order.get("links", [])
            if link.get("rel") == "approve" and link.get("href")
        ),
        None,
    )
    if not order_id or not approval_url:
        raise RuntimeError("PayPal did not return a valid order approval link.")

    with get_database_connection() as connection:
        connection.execute(
            """
            INSERT INTO paypal_orders (order_id, user_id, amount_cents, status, created_at)
            VALUES (?, ?, ?, 'created', ?)
            """,
            (
                order_id,
                user_id,
                amount_cents,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    return order_id, approval_url


def capture_paypal_order(order_id, user_id):
    with get_database_connection() as connection:
        order = connection.execute(
            """
            SELECT amount_cents, status FROM paypal_orders
            WHERE order_id = ? AND user_id = ?
            """,
            (order_id, user_id),
        ).fetchone()
    if order is None:
        raise ValueError("This PayPal order does not belong to your account.")
    if order["status"] == "captured":
        return
    if order["status"] != "created":
        raise ValueError("This PayPal order is no longer available for payment.")

    result = paypal_api_request(
        "POST",
        f"/v2/checkout/orders/{urllib.parse.quote(order_id, safe='')}/capture",
        {},
    )
    if result.get("status") != "COMPLETED":
        raise RuntimeError("PayPal has not completed this payment.")
    capture_id = (
        result.get("purchase_units", [{}])[0]
        .get("payments", {})
        .get("captures", [{}])[0]
        .get("id")
    )
    with get_database_connection() as connection:
        connection.execute(
            """
            UPDATE paypal_orders
            SET status = 'captured', capture_id = ?
            WHERE order_id = ? AND user_id = ? AND status = 'created'
            """,
            (capture_id, order_id, user_id),
        )


def mark_paypal_order_cancelled(order_id, user_id):
    if not order_id or not user_id:
        return
    with get_database_connection() as connection:
        connection.execute(
            """
            UPDATE paypal_orders SET status = 'cancelled'
            WHERE order_id = ? AND user_id = ? AND status = 'created'
            """,
            (order_id, user_id),
        )


def cart_total_cents(cart):
    cart_products = df[df["ProductID"].astype(str).isin(cart)]
    subtotal = sum(
        discounted_price(
            product["Price"],
            float(product.get("DiscountPercent", 0) or 0),
        )
        * cart[str(product["ProductID"])]
        for _, product in cart_products.iterrows()
    )
    delivery_fee = 75.0
    return int(round((subtotal + delivery_fee) * 100))


def process_paypal_return():
    action = st.query_params.get("paypal")
    if action == "cancelled":
        user = st.session_state.authenticated_user
        mark_paypal_order_cancelled(
            st.query_params.get("token") or st.session_state.get("paypal_order_id"),
            user["id"] if user else None,
        )
        st.session_state.paypal_order_id = None
        st.session_state.paypal_approval_url = None
        st.session_state.show_checkout_suggestions = False
        st.session_state.paypal_notice = (
            "info",
            "PayPal checkout was cancelled. Your cart is unchanged.",
        )
        st.query_params.clear()
        return

    order_id = st.query_params.get("token")
    if not order_id:
        return
    user = st.session_state.authenticated_user
    if user is None:
        st.session_state.paypal_notice = (
            "error",
            "Log in to the account used for this PayPal checkout to confirm payment.",
        )
        st.query_params.clear()
        return
    try:
        capture_paypal_order(order_id, user["id"])
    except (RuntimeError, ValueError) as error:
        st.session_state.paypal_notice = ("error", str(error))
    else:
        st.session_state.cart = {}
        st.session_state.page = "Home"
        st.session_state.checkout_confirmed = False
        st.session_state.paypal_notice = (
            "success",
            "Payment confirmed by PayPal. Your order is complete.",
        )
    st.session_state.paypal_order_id = None
    st.session_state.paypal_approval_url = None
    st.session_state.show_checkout_suggestions = False
    st.query_params.clear()


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
        width: 100%;
        box-sizing: border-box;
        padding: 0.3rem;
        border: 2px solid var(--peach);
        border-radius: 0.9rem;
        background: #fffaf7;
        box-shadow: 0 3px 12px rgba(84, 26, 46, 0.1);
    }
    .st-key-home-search [data-testid="stTextInput"] {
        width: 100%;
    }
    .st-key-home-search [data-testid="stTextInput"] > div {
        width: 100%;
    }
    .st-key-home-search [data-testid="stTextInput"] input {
        min-height: 3.25rem;
        border: 1px solid var(--border);
        border-radius: 0.75rem;
        background: #fff;
        color: var(--ink);
    }
    .st-key-home-search [data-testid="stTextInput"] input:focus {
        border-color: var(--wine);
        box-shadow: 0 0 0 2px rgba(181, 26, 40, 0.14);
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
def recommend_for_cart(cart, top_n=None):
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
        return df.iloc[0:0].copy()

    cart_positions = {position for position, _ in cart_items}
    seed_positions = [position for position, _ in cart_items]
    quantities = [quantity for _, quantity in cart_items]
    scores = similarity[seed_positions].T.dot(quantities)
    candidates = [
        (index, float(score))
        for index, score in enumerate(scores)
        if index not in cart_positions and score > 0
    ]
    candidates.sort(key=lambda candidate: (-candidate[1], candidate[0]))
    if top_n is not None:
        candidates = candidates[:top_n]
    return df.iloc[[index for index, _ in candidates]].copy()


def recommend_for_products(products, top_n=8):
    product_positions = {
        str(product_id): position
        for position, product_id in enumerate(df["ProductID"])
    }
    source_positions = [
        product_positions[str(product_id)]
        for product_id in products["ProductID"]
        if str(product_id) in product_positions
    ]
    if not source_positions:
        return df.iloc[0:0].copy()

    scores = similarity[source_positions, :].max(axis=0)
    source_position_set = set(source_positions)
    candidates = [
        (index, float(score))
        for index, score in enumerate(scores)
        if index not in source_position_set and score > 0
    ]
    candidates.sort(key=lambda candidate: (-candidate[1], candidate[0]))
    return df.iloc[[index for index, _ in candidates[:top_n]]].copy()


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


def render_product_cards(products, key_prefix="", show_cart_actions=True):
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
            with st.container(key=f"product-card-{key_prefix}{product_id}"):
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
                if show_cart_actions:
                    quantity_in_cart = st.session_state.cart.get(product_id, 0)
                    if quantity_in_cart:
                        minus_col, quantity_col, plus_col = st.columns([1, 1.2, 1])
                        with minus_col:
                            st.button(
                                "−",
                                key=f"{key_prefix}product_decrease_{product_id}",
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
                                key=f"{key_prefix}product_increase_{product_id}",
                                help=f"Add one more {name} to your cart",
                                on_click=change_cart_quantity,
                                args=(product_id, 1),
                                use_container_width=True,
                            )
                    else:
                        st.button(
                            "Add to Cart",
                            key=f"{key_prefix}add_{product_id}",
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
    st.session_state.checkout_confirmed = False


def open_promotions():
    navigate_to_page("Promotions")


def open_popular_products():
    navigate_to_page("Popular Products")


@st.dialog("Before you check out", width="large")
def checkout_suggestions_dialog():
    st.write("These similar products might go well with your cart.")
    approval_url = st.session_state.paypal_approval_url
    suggestions = recommend_for_cart(st.session_state.cart)
    if suggestions.empty:
        st.info("No similar products to recommend right now.")
    else:
        render_product_cards(
            suggestions.head(4),
            key_prefix="checkout-suggestion-",
            show_cart_actions=not bool(approval_url),
        )

    if approval_url:
        st.link_button(
            "Open PayPal to pay",
            approval_url,
            use_container_width=True,
        )
        st.caption(
            "Add or select your card securely on PayPal. ShopSmart does not "
            "collect or store card numbers or security codes."
        )
    elif st.button(
        "Continue to PayPal",
        key="continue_to_paypal_button",
        use_container_width=True,
    ):
        try:
            order_id, approval_url = create_paypal_order(
                st.session_state.authenticated_user["id"],
                cart_total_cents(st.session_state.cart),
            )
        except (RuntimeError, ValueError) as error:
            st.error(str(error))
        else:
            st.session_state.paypal_order_id = order_id
            st.session_state.paypal_approval_url = approval_url
            st.rerun()
    if st.button("Keep shopping", key="keep_shopping_button"):
        mark_paypal_order_cancelled(
            st.session_state.paypal_order_id,
            st.session_state.authenticated_user["id"],
        )
        st.session_state.show_checkout_suggestions = False
        st.session_state.paypal_order_id = None
        st.session_state.paypal_approval_url = None
        st.rerun()


if "page" not in st.session_state:
    st.session_state.page = "Home"
if "last_rendered_page" not in st.session_state:
    st.session_state.last_rendered_page = st.session_state.page
initialize_database()
if "authenticated_user" not in st.session_state:
    st.session_state.authenticated_user = None
if "category_filter" not in st.session_state:
    st.session_state.category_filter = ""
elif st.session_state.category_filter not in CATEGORIES:
    st.session_state.category_filter = ""
if "show_all_categories" not in st.session_state:
    st.session_state.show_all_categories = False
if "cart" not in st.session_state:
    st.session_state.cart = {}
if "show_checkout_suggestions" not in st.session_state:
    st.session_state.show_checkout_suggestions = False
if "checkout_confirmed" not in st.session_state:
    st.session_state.checkout_confirmed = False
if "checkout_after_auth" not in st.session_state:
    st.session_state.checkout_after_auth = False
if "paypal_order_id" not in st.session_state:
    st.session_state.paypal_order_id = None
if "paypal_approval_url" not in st.session_state:
    st.session_state.paypal_approval_url = None
if (
    st.session_state.page == "Account"
    and st.session_state.authenticated_user is None
):
    st.title("Create an account to check out" if st.session_state.checkout_after_auth else "Welcome to ShopSmart")
    signup_tab, login_tab = st.tabs(["Sign up", "Log in"])

    with signup_tab:
        with st.form("signup_form"):
            full_name = st.text_input("Full name", key="signup_full_name")
            signup_email = st.text_input("Email", key="signup_email")
            signup_password = st.text_input(
                "Password",
                type="password",
                key="signup_password",
                help="Use at least 8 characters.",
            )
            confirm_password = st.text_input(
                "Confirm password",
                type="password",
                key="signup_confirm_password",
            )
            signup_submitted = st.form_submit_button(
                "Create account",
                use_container_width=True,
            )
        if signup_submitted:
            if signup_password != confirm_password:
                st.error("The passwords do not match.")
            else:
                try:
                    st.session_state.authenticated_user = create_user(
                        full_name,
                        signup_email,
                        signup_password,
                    )
                except ValueError as error:
                    st.error(str(error))
                except sqlite3.IntegrityError:
                    st.error("An account with that email already exists.")
                else:
                    st.session_state.page = (
                        "Cart" if st.session_state.checkout_after_auth else "Home"
                    )
                    if st.session_state.checkout_after_auth:
                        st.session_state.checkout_after_auth = False
                        st.session_state.show_checkout_suggestions = True
                    st.rerun()

    with login_tab:
        with st.form("login_form"):
            login_email = st.text_input("Email", key="login_email")
            login_password = st.text_input(
                "Password",
                type="password",
                key="login_password",
            )
            login_submitted = st.form_submit_button(
                "Log in",
                use_container_width=True,
            )
        if login_submitted:
            user = authenticate_user(login_email, login_password)
            if user is None:
                st.error("Email or password is incorrect.")
            else:
                st.session_state.authenticated_user = user
                st.session_state.page = (
                    "Cart" if st.session_state.checkout_after_auth else "Home"
                )
                if st.session_state.checkout_after_auth:
                    st.session_state.checkout_after_auth = False
                    st.session_state.show_checkout_suggestions = True
                st.rerun()

    if st.button("Continue browsing as guest", key="continue_as_guest"):
        st.session_state.page = "Home"
        st.session_state.checkout_after_auth = False
        st.rerun()
elif st.session_state.page == "Account":
    st.title("Your account")
    st.write(f"Signed in as {st.session_state.authenticated_user['email']}.")

if st.session_state.authenticated_user is not None:
    process_paypal_return()

pages = [
    "Home",
    "Products",
    "Promotions",
    "Popular Products",
    "Recommendations",
    "Data Analysis",
    "Cart",
    "Account",
]
page_icons = {
    "Home": "home",
    "Products": "shopping_bag",
    "Promotions": "sell",
    "Popular Products": "star",
    "Recommendations": "auto_awesome",
    "Data Analysis": "analytics",
    "Cart": "shopping_cart",
    "Account": "account_circle",
}
st.sidebar.markdown(
    '<div class="shop-brand">Shop<span>Smart</span></div>',
    unsafe_allow_html=True,
)
if st.session_state.authenticated_user is None:
    st.sidebar.caption("Browsing as guest")
else:
    st.sidebar.caption(f"Signed in as {st.session_state.authenticated_user['full_name']}")
    if st.sidebar.button("Log out", key="logout_button", use_container_width=True):
        mark_paypal_order_cancelled(
            st.session_state.paypal_order_id,
            st.session_state.authenticated_user["id"],
        )
        st.session_state.authenticated_user = None
        st.session_state.page = "Home"
        st.session_state.paypal_order_id = None
        st.session_state.paypal_approval_url = None
        st.session_state.show_checkout_suggestions = False
        st.session_state.checkout_confirmed = False
        st.session_state.checkout_after_auth = False
        st.rerun()
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

if "paypal_notice" in st.session_state:
    notice_type, notice_message = st.session_state.pop("paypal_notice")
    getattr(st, notice_type)(notice_message)

if st.session_state.page != st.session_state.last_rendered_page:
    st.html(
        """
        <script>
        const scrollPageToTop = () => {
            window.scrollTo({top: 0, left: 0, behavior: "instant"});
            const appView = document.querySelector(
                '[data-testid="stAppViewContainer"]'
            );
            if (appView) {
                appView.scrollTo({top: 0, left: 0, behavior: "instant"});
            }
        };
        scrollPageToTop();
        window.setTimeout(scrollPageToTop, 100);
        </script>
        """,
        unsafe_allow_javascript=True,
    )
    st.session_state.last_rendered_page = st.session_state.page

if st.session_state.page == "Home":
    st.markdown("**Search product**")
    with st.container(key="home-search"):
        search = st.text_input(
            "Search product",
            label_visibility="collapsed",
            key="home_search_input",
        )
    if search.strip():
        search_results = df[
            df["Name"].str.contains(search.strip(), case=False, regex=False)
        ].reset_index(drop=True)
        st.markdown(
            f'<div class="section-title">Search results ({len(search_results)})</div>',
            unsafe_allow_html=True,
        )
        if search_results.empty:
            st.info("No matching products found.")
        else:
            render_product_cards(
                search_results,
                key_prefix="home-search-result-",
            )

        related_products = recommend_for_products(search_results)
        st.markdown(
            '<div class="section-title">Related products</div>',
            unsafe_allow_html=True,
        )
        if related_products.empty:
            st.info("No related products found.")
        else:
            render_product_cards(
                related_products,
                key_prefix="home-search-related-",
            )
        st.stop()

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
    if not st.session_state.cart:
        st.info("Add products to your cart to see personalised recommendations.")
    else:
        rec = recommend_for_cart(st.session_state.cart, top_n=5)
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
        st.divider()
        similar_products = recommend_for_cart(cart)
        st.markdown(
            '<div class="section-title">Similar products for your cart</div>',
            unsafe_allow_html=True,
        )
        render_product_cards(
            similar_products,
            key_prefix="cart-similar-",
        )
        if st.session_state.checkout_confirmed:
            st.success(
                f"Checkout confirmed. Your total, including delivery, is "
                f"R {total + delivery_fee:,.2f}. Your cart has been kept."
            )
        elif st.button("Checkout", key="checkout_button", use_container_width=True):
            if st.session_state.authenticated_user is None:
                st.session_state.checkout_after_auth = True
                st.session_state.page = "Account"
            else:
                st.session_state.show_checkout_suggestions = True
            st.rerun()

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

if st.session_state.cart and st.session_state.page in {
    "Home",
    "Products",
    "Promotions",
    "Popular Products",
    "Recommendations",
}:
    st.markdown(
        '<div class="section-title">Similar products for your cart</div>',
        unsafe_allow_html=True,
    )
    render_product_cards(
        recommend_for_cart(st.session_state.cart),
        key_prefix="shopping-similar-",
    )

if st.session_state.show_checkout_suggestions:
    checkout_suggestions_dialog()
