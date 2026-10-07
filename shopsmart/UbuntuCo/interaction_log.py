"""
interaction_log.py - record REAL customer behaviour in ShopSmart's SQLite database.

Events and the implicit "rating" the recommender learns from:
    view      (opened Recommendations for a product)  -> 2
    cart_add  (added to cart)                         -> 4
    purchase  (PayPal payment completed)              -> 5
A customer is "u_<id>" when logged in, otherwise "g_<session id>" (guest).
"""
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

import pandas as pd

EVENT_WEIGHTS = {"view": 2, "cart_add": 4, "purchase": 5}


def _connect(db_path):
    con = sqlite3.connect(db_path, timeout=10)
    con.row_factory = sqlite3.Row
    return con


def _now():
    return datetime.now(timezone.utc).isoformat()


def init(db_path):
    with closing(_connect(db_path)) as con, con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY,
                actor TEXT NOT NULL,
                product_id INTEGER NOT NULL,
                event TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_interactions_actor ON interactions(actor);
            CREATE TABLE IF NOT EXISTS order_items (
                order_id TEXT NOT NULL,
                product_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL,
                logged INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (order_id, product_id)
            );
            """
        )


def log_event(db_path, actor, product_id, event):
    """Never raises: logging must not break shopping."""
    if event not in EVENT_WEIGHTS:
        return False
    try:
        with closing(_connect(db_path)) as con, con:
            con.execute(
                "INSERT INTO interactions (actor, product_id, event, created_at) VALUES (?,?,?,?)",
                (actor, int(product_id), event, _now()),
            )
        return True
    except (sqlite3.Error, ValueError):
        return False


def count_events(db_path):
    try:
        with closing(_connect(db_path)) as con:
            return con.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
    except sqlite3.Error:
        return 0


def load_events(db_path):
    cols = ["actor", "product_id", "event", "created_at"]
    try:
        with closing(_connect(db_path)) as con:
            df = pd.read_sql_query(
                "SELECT actor, product_id, event, created_at FROM interactions ORDER BY id", con
            )
    except (sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame(columns=cols)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    return df


def to_ratings(events):
    """Collapse events to one implicit rating per customer/product (strongest signal wins)."""
    if events is None or events.empty:
        return pd.DataFrame(columns=["user_id", "item_id", "rating"])
    r = events.assign(rating=events["event"].map(EVENT_WEIGHTS))
    r = r.groupby(["actor", "product_id"], as_index=False)["rating"].max()
    return r.rename(columns={"actor": "user_id", "product_id": "item_id"})


def save_order_items(db_path, order_id, cart):
    """Remember what was in the cart for a PayPal order (cart: {product_id: qty})."""
    try:
        with closing(_connect(db_path)) as con, con:
            for pid, qty in cart.items():
                con.execute(
                    "INSERT OR REPLACE INTO order_items (order_id, product_id, quantity, logged) "
                    "VALUES (?,?,?,0)",
                    (order_id, int(pid), int(qty)),
                )
    except (sqlite3.Error, ValueError):
        pass


def log_purchases_for_order(db_path, order_id, actor):
    """Turn a paid order into 'purchase' events. Safe to call twice."""
    try:
        with closing(_connect(db_path)) as con, con:
            rows = con.execute(
                "SELECT product_id FROM order_items WHERE order_id=? AND logged=0", (order_id,)
            ).fetchall()
            for row in rows:
                con.execute(
                    "INSERT INTO interactions (actor, product_id, event, created_at) "
                    "VALUES (?,?,?,?)",
                    (actor, row["product_id"], "purchase", _now()),
                )
            con.execute("UPDATE order_items SET logged=1 WHERE order_id=?", (order_id,))
    except sqlite3.Error:
        pass


def captured_orders(db_path):
    try:
        with closing(_connect(db_path)) as con:
            df = pd.read_sql_query(
                "SELECT order_id, user_id, amount_cents, created_at FROM paypal_orders "
                "WHERE status='captured'",
                con,
            )
    except (sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame(columns=["order_id", "user_id", "amount_cents", "created_at"])
    return df
