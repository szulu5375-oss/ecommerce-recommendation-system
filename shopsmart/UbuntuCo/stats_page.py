"""
stats_page.py - the "Data Analysis" page: store, customer and recommender statistics.

    from stats_page import render_stats
    render_stats(df, DATABASE_PATH, get_recommender())
"""
import os

import pandas as pd
import streamlit as st

import interaction_log as il

BASE = os.path.dirname(os.path.abspath(__file__))

# ---- ShopSmart palette -------------------------------------------------
INK, NAVY, SLATE = "#16182f", "#242f49", "#384358"
PEACH, RED, WINE = "#ffa586", "#b51a28", "#541a2e"
PAGE, BORDER = "#fff4ef", "#f1c3b5"
CAT_PALETTE = [RED, NAVY, PEACH, WINE, SLATE, INK, "#d75e68", "#6b7488", "#bd7660"]
EVENT_COLOURS = {"view": PEACH, "cart_add": RED, "purchase": WINE}

_CONFIG = {
    "background": PAGE,
    "view": {"stroke": None},
    "axis": {
        "labelColor": SLATE, "titleColor": INK, "gridColor": BORDER,
        "domainColor": BORDER, "tickColor": BORDER, "labelFontSize": 11,
        "titleFontSize": 12,
    },
}

_CSS = f"""
<style>
[data-testid="stMetric"] {{
    background: #fffaf7; border: 1px solid {BORDER}; border-left: 5px solid {RED};
    border-radius: 0.85rem; padding: 0.8rem 1rem;
}}
[data-testid="stMetricLabel"] p {{ color: {SLATE}; font-weight: 600; }}
[data-testid="stMetricValue"] {{ color: {WINE}; font-weight: 800; }}
[data-baseweb="tab-list"] {{ gap: 0.4rem; }}
[data-baseweb="tab"] p {{ color: {SLATE}; font-weight: 650; }}
button[aria-selected="true"] p {{ color: {RED} !important; }}
[data-baseweb="tab-highlight"] {{ background-color: {RED} !important; }}
[data-baseweb="tab-border"] {{ background-color: {BORDER} !important; }}
.stats-note {{ color: {SLATE}; font-size: 0.85rem; margin: -0.3rem 0 0.8rem; }}
.stApp h2, .stApp h3, .stApp h4 {{ color: {INK}; letter-spacing: -0.03em; }}
.stats-banner {{
    background: linear-gradient(90deg, {INK} 0%, {NAVY} 45%, {WINE} 100%);
    color: #fff4ef; border-radius: 1.1rem; padding: 1.2rem 1.5rem; margin-bottom: 1rem;
    border-bottom: 4px solid {PEACH};
}}
.stats-banner .t {{ font-size: 1.5rem; font-weight: 800; letter-spacing: -0.04em; }}
.stats-banner .s {{ color: {PEACH}; font-size: 0.9rem; margin-top: 0.15rem; }}
</style>
"""


def _vega(data, mark, encoding, height=260):
    st.vega_lite_chart(
        data,
        {"height": height, "mark": mark, "encoding": encoding, "config": _CONFIG},
        use_container_width=True,
        theme=None,
    )


def bar(series, color=RED, color_map=None, horizontal=False, order=None):
    """Bar chart in the ShopSmart palette. color_map: {label: hex} colours each bar."""
    data = series.rename("value").rename_axis("label").reset_index()
    data["label"] = data["label"].astype(str)
    labels = order or list(data["label"])
    cat = {"field": "label", "type": "nominal", "title": None,
           "sort": order if order else ("-x" if horizontal else "-y")}
    val = {"field": "value", "type": "quantitative", "title": None}
    if pd.api.types.is_integer_dtype(series.dtype):
        val["axis"] = {"tickMinStep": 1, "format": "d"}
    if color_map:
        colour = {
            "field": "label", "type": "nominal", "legend": None,
            "scale": {"domain": labels,
                      "range": [color_map.get(l, SLATE) for l in labels]},
        }
    else:
        colour = {"value": color}
    enc = {
        ("y" if horizontal else "x"): cat,
        ("x" if horizontal else "y"): val,
        "color": colour,
        "tooltip": [{"field": "label", "title": "Item"}, {"field": "value", "title": "Value"}],
    }
    mark = {"type": "bar", "cornerRadiusEnd": 5}
    height = max(120, 30 * len(data) + 30) if horizontal else 260
    _vega(data, mark, enc, height)


def line(series):
    data = series.rename("value").rename_axis("day").reset_index()
    data["day"] = pd.to_datetime(data["day"]).dt.strftime("%Y-%m-%d")
    enc = {
        "x": {"field": "day", "type": "temporal", "title": None, "timeUnit": "yearmonthdate",
              "axis": {"format": "%d %b", "tickCount": {"interval": "day", "step": 1}}},
        "y": {"field": "value", "type": "quantitative", "title": None,
              "axis": {"tickMinStep": 1, "format": "d"}},
        "tooltip": [{"field": "day", "title": "Day"}, {"field": "value", "title": "Actions"}],
    }
    mark = {"type": "line", "color": RED, "strokeWidth": 3,
            "point": {"filled": True, "fill": WINE, "size": 70}}
    _vega(data, mark, enc, 240)


def _category_colours(products):
    cats = sorted(products["Category"].dropna().astype(str).unique())
    return {c: CAT_PALETTE[i % len(CAT_PALETTE)] for i, c in enumerate(cats)}


def _final_price(p):
    return p["Price"] * (1 - p["DiscountPercent"].fillna(0) / 100)


def _kpi_row(items):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


def _tab_overview(products, events, orders, rec):
    registered = events[events["actor"].str.startswith("u_")]["actor"].nunique() if len(events) else 0
    guests = events[events["actor"].str.startswith("g_")]["actor"].nunique() if len(events) else 0
    discounted = int((products["DiscountPercent"].fillna(0) > 0).sum())
    revenue = orders["amount_cents"].sum() / 100 if len(orders) else 0

    st.subheader("Store")
    _kpi_row([
        ("Products", len(products)),
        ("Categories", products["Category"].nunique()),
        ("On promotion", discounted),
        ("Average price", f"R {_final_price(products).mean():,.0f}"),
    ])
    st.subheader("Customers (real activity)")
    _kpi_row([
        ("Registered customers", registered),
        ("Guest sessions", guests),
        ("Interactions logged", len(events)),
        ("Paid orders", len(orders)),
        ("Revenue", f"R {revenue:,.2f}"),
    ])
    if len(events) == 0:
        st.info(
            "No customer activity has been recorded yet. Add products to your cart or "
            "complete a checkout and this page fills in automatically."
        )

    with st.expander("How the recommender learns from customers"):
        st.markdown(
            """
**1. Record.** Every time a customer adds a product to the cart, opens a product on the
Recommendations page, or pays for an order, one row is saved in the `interactions` table.
Logged-in customers are saved as `u_<id>`, guests as `g_<session>`.

**2. Convert to ratings.** Each action becomes an implicit rating:
view = 2, added to cart = 4, purchased = 5 (the strongest action wins).

**3. Train.** The ratings are combined with the simulated starter data and fed to the
hybrid model (SVD collaborative filtering + content-based similarity). The model retrains
automatically as new activity arrives.

**4. Recommend.** Registered customers and guests who have acted get personal picks;
brand-new visitors get popular products; the cart page uses the items in the cart.
"""
        )
        _kpi_row([
            ("Simulated ratings (seed)", f"{rec.n_sim_ratings:,}"),
            ("Real ratings", f"{rec.n_real_ratings:,}"),
            ("Real customers in model", rec.n_real_users),
        ])


def _tab_catalog(products):
    p = products.assign(FinalPrice=_final_price(products))
    st.subheader("Products per category")
    cmap = _category_colours(products)
    bar(p["Category"].value_counts(), color_map=cmap)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Average price per category (R)")
        bar(p.groupby("Category")["FinalPrice"].mean().round(0), color_map=cmap)
    with c2:
        st.subheader("Promotions by category")
        promo = (p["DiscountPercent"].fillna(0) > 0).groupby(p["Category"]).sum()
        bar(promo, color_map=cmap)
    st.subheader("Price distribution")
    bins = pd.cut(
        p["FinalPrice"],
        [0, 500, 1000, 2000, 5000, 20000],
        labels=["< R500", "R500-1k", "R1k-2k", "R2k-5k", "> R5k"],
    )
    order = ["< R500", "R500-1k", "R1k-2k", "R2k-5k", "> R5k"]
    bar(bins.value_counts().reindex(order), color=NAVY, order=order)
    with st.expander("Full product table"):
        st.dataframe(
            p.assign(FinalPrice=p["FinalPrice"].round(2)).drop(columns=["Description"]),
            hide_index=True,
            use_container_width=True,
        )


def _tab_customers(products, events):
    st.subheader("Real customer data")
    if events.empty:
        st.info("No real activity yet - this section shows live data as soon as customers shop.")
    else:
        ev = events.merge(
            products[["ProductID", "Name", "Category"]],
            left_on="product_id", right_on="ProductID", how="left",
        )
        ev["Customer type"] = ev["actor"].str.startswith("u_").map({True: "Registered", False: "Guest"})

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Actions by type**")
            bar(ev["event"].value_counts(), color_map=EVENT_COLOURS)
        with c2:
            st.markdown("**Registered vs guest activity**")
            bar(ev["Customer type"].value_counts(), color_map={"Registered": NAVY, "Guest": PEACH})

        st.markdown("**Activity per day**")
        daily = ev.dropna(subset=["created_at"]).groupby(ev["created_at"].dt.date).size()
        line(daily)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Most added to cart**")
            bar(ev[ev["event"] == "cart_add"]["Name"].value_counts().head(10), color=RED, horizontal=True)
        with c4:
            st.markdown("**Most purchased**")
            bought = ev[ev["event"] == "purchase"]["Name"].value_counts().head(10)
            if bought.empty:
                st.caption("No completed purchases yet.")
            else:
                bar(bought, color=WINE, horizontal=True)

        st.markdown("**Customer interest by category** (weighted by action strength)")
        weights = ev["event"].map(il.EVENT_WEIGHTS)
        bar(weights.groupby(ev["Category"]).sum(), color_map=_category_colours(products))

        st.markdown("**Conversion funnel** (unique customers)")
        funnel = ev.groupby("event")["actor"].nunique().reindex(["view", "cart_add", "purchase"]).fillna(0)
        bar(funnel, color_map=EVENT_COLOURS, order=["view", "cart_add", "purchase"])

        st.markdown("**Latest 50 actions**")
        latest = ev.sort_values("created_at", ascending=False).head(50)
        st.dataframe(
            latest[["created_at", "actor", "event", "Name", "Category"]],
            hide_index=True, use_container_width=True,
        )
        st.download_button(
            "Download all activity (CSV)",
            ev[["created_at", "actor", "event", "product_id", "Name", "Category"]].to_csv(index=False),
            file_name="shopsmart_activity.csv",
            mime="text/csv",
        )

    with st.expander("Simulated starter data (demo only - not real customers)"):
        sim_path = os.path.join(BASE, "interactions.csv")
        if os.path.isfile(sim_path):
            sim = pd.read_csv(sim_path).merge(
                products[["ProductID", "Name", "Category"]],
                left_on="item_id", right_on="ProductID", how="left",
            )
            st.caption(f"{len(sim):,} simulated ratings from {sim['user_id'].nunique()} simulated users")
            st.markdown("**Rating distribution**")
            bar(sim["rating"].value_counts().sort_index(), color=SLATE, order=[str(i) for i in range(1, 6)])
            st.markdown("**Average rating per category**")
            bar(sim.groupby("Category")["rating"].mean().round(2), color_map=_category_colours(products))
            st.markdown("**Most-rated products**")
            bar(sim["Name"].value_counts().head(10), color=NAVY, horizontal=True)


def _tab_recommender(products, rec):
    st.subheader("Model comparison")
    st.caption(
        "Measured on a held-out test set of the simulated ratings "
        "(Model.py). Precision/Recall/F1 at K=5; MSE is rating-prediction error."
    )
    results_path = os.path.join(BASE, "results.csv")
    if os.path.isfile(results_path):
        res = pd.read_csv(results_path)
        st.dataframe(res, hide_index=True, use_container_width=True)
        st.markdown("**F1@5 by model** (higher is better)")
        f1 = res.set_index("Model")["F1@5"]
        bar(f1, color_map={m: (RED if m.startswith("Hybrid") else SLATE) for m in f1.index})
    else:
        st.warning("Run `python Model.py` to create results.csv.")

    st.subheader("Training data")
    total = rec.n_sim_ratings + rec.n_real_ratings
    share = rec.n_real_ratings / total if total else 0
    _kpi_row([
        ("Ratings in model", f"{total:,}"),
        ("Real share", f"{share:.1%}"),
        ("Matrix sparsity", f"{rec.sparsity():.1%}"),
        ("Catalog coverage (top 5)", f"{rec.catalog_coverage(5):.0%}"),
    ])
    st.caption(
        "Coverage = share of the catalog recommended to at least one customer. "
        "Sparsity = share of customer/product pairs with no interaction."
    )

    st.subheader("Try it")
    picked = st.multiselect("Pretend a customer has these items in their cart", list(products["Name"]))
    ids = [int(products.loc[products["Name"] == n, "ProductID"].iloc[0]) for n in picked]
    out = rec.for_cart(ids, 5) if ids else rec.popular(5)
    label = "Recommended for this cart" if ids else "Most popular (no cart yet)"
    st.markdown(f"**{label}**")
    show = out[["Name", "Category", "Price", "match"]].rename(columns={"match": "Match"})
    show["Match"] = (show["Match"] * 100).round(0).astype(int).astype(str) + "%"
    st.dataframe(show, hide_index=True, use_container_width=True)


def render_stats(products, db_path, recommender):
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="stats-banner"><div class="t">Shop<span style="color:#ffa586">Smart</span> Analytics</div>'
        '<div class="s">Store, customer and recommender statistics</div></div>',
        unsafe_allow_html=True,
    )
    events = il.load_events(db_path)
    orders = il.captured_orders(db_path)
    tabs = st.tabs(["Overview", "Catalog", "Customers", "Recommender"])
    with tabs[0]:
        _tab_overview(products, events, orders, recommender)
    with tabs[1]:
        _tab_catalog(products)
    with tabs[2]:
        _tab_customers(products, events)
    with tabs[3]:
        _tab_recommender(products, recommender)