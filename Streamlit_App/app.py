"""
app.py — HomeValue AI · Prayagraj
Neo-brutalist map dashboard for estimating house prices across Prayagraj
localities, with famous places, budget-aware map colouring, compare mode and
an insights tab. Prices are in Indian Rupees.
"""

import os
import json
import datetime

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    INK, YELLOW, PINK, TEAL, BLUE, GREEN, RED, CREAM, STATUS_META,
    format_inr, short_inr, indian_grouping, budget_status, price_tier,
    get_quality_traits, explain_price, render_house_svg, puzzle_reveal_html,
    theme_css,
)

st.set_page_config(page_title="HomeValue AI · Prayagraj", page_icon="🏠", layout="wide")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "Model", "house_price_model.pkl")
META_PATH = os.path.join(SCRIPT_DIR, "..", "Model", "model_metadata.json")
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "Dataset", "house_data_cleaned.csv")
LOC_PATH = os.path.join(SCRIPT_DIR, "..", "Dataset", "locality_info.csv")
LM_PATH = os.path.join(SCRIPT_DIR, "..", "Dataset", "landmarks.csv")


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta


@st.cache_data
def load_tables():
    return (pd.read_csv(DATA_PATH), pd.read_csv(LOC_PATH), pd.read_csv(LM_PATH))


model, meta = load_model()
df, loc_df, lm_df = load_tables()
loc_df = loc_df.reset_index(drop=True)
best_model_name = meta["best_model"]
best_r2 = meta["metrics"][best_model_name]["R2"]
best_mae = meta["metrics"][best_model_name]["MAE"]

stats = {
    "area_median": float(df["area_sqft"].median()),
    "rate_median": float((df["price"] / df["area_sqft"]).median()),
    "price_quantiles": {q: float(df["price"].quantile(q)) for q in [0.25, 0.5, 0.75, 0.9]},
}
LOC_NAMES = sorted(loc_df["location"].tolist())
OVERVIEW = "overview"

# ---------------------------------------------------------------------------
# SESSION STATE + CALLBACKS
# ---------------------------------------------------------------------------
for k, v in {"tilt": False, "reveal_counter": 0, "reveal_counter_cmp": 0,
             "has_predicted": False, "has_compared": False, "find_place": OVERVIEW}.items():
    st.session_state.setdefault(k, v)


def _reveal():
    st.session_state["has_predicted"] = True
    st.session_state["reveal_counter"] += 1


def _toggle_tilt():
    st.session_state["tilt"] = not st.session_state["tilt"]


def _reset_view():
    st.session_state["find_place"] = OVERVIEW
    st.session_state["tilt"] = False


st.markdown(theme_css(), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# MODEL HELPERS
# ---------------------------------------------------------------------------
def build_input_df(area_sqft, bedrooms, bathrooms, age_years, distance_to_city_km,
                   garage, garden, lift, near_school, location):
    row = {
        "area_sqft": area_sqft, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "age_years": age_years, "distance_to_city_km": distance_to_city_km,
        "total_rooms": bedrooms + bathrooms,
        "amenity_score": int(garage) + int(garden) + int(lift) + int(near_school),
        "is_new": int(age_years <= 5),
        "garage": int(garage), "garden": int(garden), "lift": int(lift),
        "near_school": int(near_school), "location": location,
    }
    return pd.DataFrame([row]), row


def loc_row(name):
    return loc_df[loc_df["location"] == name].iloc[0]


@st.cache_data
def predict_all(area, bed, bath, age, garage, garden, lift, school):
    rows = [build_input_df(area, bed, bath, age, r.distance_to_city_km,
                           garage, garden, lift, school, r.location)[1]
            for r in loc_df.itertuples()]
    return model.predict(pd.DataFrame(rows))


def predict_one(specs):
    """specs = (area, bed, bath, age, locality, garage, garden, lift, school)"""
    area, bed, bath, age, loc, garage, garden, lift, school = specs
    d = float(loc_row(loc)["distance_to_city_km"])
    X, row = build_input_df(area, bed, bath, age, d, garage, garden, lift, school, loc)
    return float(model.predict(X)[0]), row


# ---------------------------------------------------------------------------
# MAP
# ---------------------------------------------------------------------------
def map_view(find_place):
    if find_place == OVERVIEW:
        return 25.452, 81.853, 11.1
    kind, name = find_place.split("::")
    if kind == "loc":
        r = loc_row(name)
        return float(r["lat"]), float(r["lon"]), 13.3
    r = lm_df[lm_df["name"] == name].iloc[0]
    return float(r["lat"]), float(r["lon"]), 14.2


def build_map(map_df, visible, show_lm, find_place, tilt, focus_loc, focus_text):
    fig = go.Figure()
    vis = map_df[map_df["status"].isin(visible)]

    if focus_loc is not None:
        r = loc_row(focus_loc)
        fig.add_trace(go.Scattermap(lat=[r["lat"]], lon=[r["lon"]], mode="markers",
                                    marker=dict(size=42, color=PINK, opacity=0.75),
                                    hoverinfo="skip", showlegend=False))

    if not vis.empty:
        fig.add_trace(go.Scattermap(lat=vis["lat"], lon=vis["lon"], mode="markers",
                                    marker=dict(size=21, color=INK), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scattermap(
            lat=vis["lat"], lon=vis["lon"], mode="markers+text", text=vis["location"],
            textposition="top center", textfont=dict(size=11, color=INK),
            marker=dict(size=15, color=[STATUS_META[s]["color"] for s in vis["status"]]),
            customdata=list(zip(vis["rate_per_sqft"], vis["pred"].map(short_inr),
                                vis["status"].map(lambda s: STATUS_META[s]["label"]), vis["tier"])),
            hovertemplate=("<b>%{text}</b> (%{customdata[3]})<br>Base rate: ₹%{customdata[0]:,}/sq ft"
                           "<br>Estimate for your specs: <b>%{customdata[1]}</b><br>%{customdata[2]}<extra></extra>"),
            showlegend=False))

    if show_lm:
        fig.add_trace(go.Scattermap(lat=lm_df["lat"], lon=lm_df["lon"], mode="markers",
                                    marker=dict(size=19, color=INK), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scattermap(
            lat=lm_df["lat"], lon=lm_df["lon"], mode="markers+text", text=lm_df["name"],
            textposition="bottom center", textfont=dict(size=11, color="#1B3A8A"),
            marker=dict(size=13, color=BLUE),
            customdata=list(zip(lm_df["note"], lm_df["nearest_locality"])),
            hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>Nearest locality: %{customdata[1]}<extra></extra>",
            showlegend=False))

    if focus_loc is not None and focus_text:
        r = loc_row(focus_loc)
        fig.add_trace(go.Scattermap(lat=[r["lat"]], lon=[r["lon"]], mode="text", text=[focus_text],
                                    textposition="bottom center", textfont=dict(size=14, color=INK),
                                    hoverinfo="skip", showlegend=False))

    lat, lon, zoom = map_view(find_place)
    fig.update_layout(
        map=dict(style="open-street-map", center=dict(lat=lat, lon=lon), zoom=zoom,
                 pitch=55 if tilt else 0, bearing=-15 if tilt else 0),
        height=660, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor=CREAM, bordercolor=INK, font=dict(color=INK, size=13)),
    )
    return fig


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="nn-header">
  <div>
    <p class="nn-title">🏠 HomeValue AI · Prayagraj</p>
    <p class="nn-sub">House price estimator for Prayagraj only · {best_model_name} · avg. error ≈ {short_inr(best_mae)} · R² {best_r2:.2f}</p>
  </div>
  <span class="nn-badge">{len(loc_df)} LOCALITIES · {len(lm_df)} FAMOUS PLACES</span>
</div>
""", unsafe_allow_html=True)

tab_map, tab_compare, tab_insights = st.tabs(["🗺️ Map & Estimate", "⚖️ Compare", "📊 Insights"])

# ===========================================================================
# TAB 1 — MAP & ESTIMATE
# ===========================================================================
with tab_map:
    col_l, col_r = st.columns([1, 2.7], gap="large")

    with col_l:
        with st.container(key="filters_panel"):
            st.markdown('<div class="panel-title">▼ Filters</div>', unsafe_allow_html=True)

            place_opts = ([OVERVIEW] + [f"loc::{n}" for n in LOC_NAMES]
                          + [f"lm::{n}" for n in lm_df["name"]])

            def fmt_place(v):
                if v == OVERVIEW:
                    return "Prayagraj overview"
                kind, name = v.split("::")
                return f"{name} ({'locality' if kind == 'loc' else 'famous place'})"

            st.selectbox("Find a place", place_opts, format_func=fmt_place, key="find_place")

            st.markdown('<div class="panel-sub">Your budget</div>', unsafe_allow_html=True)
            budget_lakh = st.slider("Budget (₹ lakh)", 10, 500, 75, step=5, key="budget")

            st.markdown('<div class="panel-sub">Property</div>', unsafe_allow_html=True)
            area = st.number_input("Area (sq ft)", min_value=400, max_value=6000, value=1200, step=50, key="e_area")
            c1, c2 = st.columns(2)
            with c1:
                bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6], index=1, key="e_bed")
            with c2:
                bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4], index=1, key="e_bath")
            age = st.slider("Age of building (years)", 0, 60, 8, key="e_age")
            location = st.selectbox("Locality for estimate", LOC_NAMES,
                                    index=LOC_NAMES.index("Civil Lines"), key="e_loc")

            st.markdown('<div class="panel-sub">Amenities</div>', unsafe_allow_html=True)
            a1, a2 = st.columns(2)
            with a1:
                garage = st.checkbox("Parking", value=True, key="e_garage")
                lift = st.checkbox("Lift", value=False, key="e_lift")
            with a2:
                garden = st.checkbox("Garden", value=False, key="e_garden")
                near_school = st.checkbox("Near school", value=True, key="e_school")

            st.markdown('<div class="panel-sub">Map layers</div>', unsafe_allow_html=True)
            show_within = st.checkbox("Within budget", value=True, key="f_within")
            show_stretch = st.checkbox("Stretch (up to +25%)", value=True, key="f_stretch")
            show_over = st.checkbox("Over budget", value=True, key="f_over")
            show_lm = st.checkbox("Famous places", value=True, key="f_lm")

            st.write("")
            st.button("Reveal my estimate", width="stretch", key="btn_reveal", on_click=_reveal)
            st.button("2.5D view", width="stretch", key="btn_tilt", on_click=_toggle_tilt)
            st.button("Reset view", width="stretch", key="btn_reset", on_click=_reset_view)

    # --- compute everything the map needs --------------------------------
    budget = budget_lakh * 1e5
    preds = predict_all(area, bedrooms, bathrooms, age, garage, garden, lift, near_school)
    map_df = loc_df.copy()
    map_df["pred"] = preds
    map_df["status"] = [budget_status(p, budget) for p in preds]
    counts = map_df["status"].value_counts().to_dict()
    n_w, n_s, n_o = counts.get("within", 0), counts.get("stretch", 0), counts.get("over", 0)
    total = len(map_df)

    est, est_row = predict_one((area, bedrooms, bathrooms, age, location, garage, garden, lift, near_school))
    visible = [s for s, on in (("within", show_within), ("stretch", show_stretch), ("over", show_over)) if on]
    focus_text = f"YOU: {short_inr(est)}" if st.session_state["has_predicted"] else None
    focus_loc = location if st.session_state["has_predicted"] else None

    with col_r:
        with st.container(key="map_wrap"):
            fig = build_map(map_df, visible, show_lm, st.session_state["find_place"],
                            st.session_state["tilt"], focus_loc, focus_text)
            st.plotly_chart(fig, config={"displayModeBar": False, "scrollZoom": True}, key="main_map")

            with st.container(key="legend_card"):
                st.markdown(f"""
                <div class="lg-title">LEGEND</div>
                <div class="lg-row"><span class="lg-dot" style="background:{RED}"></span>Over budget</div>
                <div class="lg-row"><span class="lg-dot" style="background:{YELLOW}"></span>Stretch (up to +25%)</div>
                <div class="lg-row"><span class="lg-dot" style="background:{GREEN}"></span>Within budget</div>
                <div class="lg-row"><span class="lg-sq"></span>Famous place</div>
                <div class="lg-row"><span class="lg-dot" style="background:#fff"></span>Your estimate (pink ring)</div>
                """, unsafe_allow_html=True)

            with st.container(key="stats_card"):
                pw, ps = n_w / total * 100, n_s / total * 100
                st.markdown(f"""
                <div class="st-title">STATS · for ₹{budget_lakh} L</div>
                <div class="st-grid">
                  <div class="st-rows">
                    <div class="st-row" style="background:#fff">TOTAL <b>{total}</b></div>
                    <div class="st-row" style="background:{RED}">OVER <b>{n_o}</b></div>
                    <div class="st-row" style="background:{YELLOW}">STRETCH <b>{n_s}</b></div>
                    <div class="st-row" style="background:{GREEN}">WITHIN <b>{n_w}</b></div>
                  </div>
                  <div class="st-donutbox"><div class="donut" style="background:conic-gradient({GREEN} 0 {pw:.1f}%, {YELLOW} {pw:.1f}% {pw+ps:.1f}%, {RED} {pw+ps:.1f}% 100%)"></div></div>
                </div>
                """, unsafe_allow_html=True)

        st.caption("Pins show approximate locality centres. Colours update live with your budget and property specs.")

        # --- info for the place picked in "Find a place" -------------------
        fp = st.session_state["find_place"]
        if fp != OVERVIEW:
            kind, name = fp.split("::")
            if kind == "loc":
                r = loc_row(name)
                p = float(map_df.loc[map_df["location"] == name, "pred"].iloc[0])
                st.markdown(f'<div class="place-info"><b>{name}</b> · {r["tier"]} · about ₹{int(r["rate_per_sqft"]):,}/sq ft · '
                            f'{r["distance_to_city_km"]:.1f} km from Civil Lines<br>{r["note"]}<br>'
                            f'Estimate for your specs here: <b>{format_inr(p)}</b> ({short_inr(p)})</div>', unsafe_allow_html=True)
            else:
                r = lm_df[lm_df["name"] == name].iloc[0]
                nr = loc_row(r["nearest_locality"])
                p = float(map_df.loc[map_df["location"] == r["nearest_locality"], "pred"].iloc[0])
                st.markdown(f'<div class="place-info"><b>{name}</b> · {r["note"]}<br>Nearest locality: {r["nearest_locality"]} '
                            f'(about ₹{int(nr["rate_per_sqft"]):,}/sq ft). Your specs there: <b>{short_inr(p)}</b></div>',
                            unsafe_allow_html=True)

        # --- estimate + house + afford table ------------------------------
        e1, e2 = st.columns([1, 1], gap="medium")
        with e1:
            with st.container(key="card_estimate"):
                st.markdown(f'<div class="card-title">Estimate · {location}</div>', unsafe_allow_html=True)
                if st.session_state["has_predicted"]:
                    tier_name, tier_level = price_tier(est, stats)
                    rate = float(loc_row(location)["rate_per_sqft"])
                    st.markdown(puzzle_reveal_html(
                        format_inr(est), f"{tier_name} · ≈ {short_inr(est)} · typical error ±{short_inr(best_mae)}",
                        f"est{st.session_state['reveal_counter']}"), unsafe_allow_html=True)
                    for b in explain_price(est_row, stats, rate):
                        st.markdown(f"- {b}")
                    if est <= budget:
                        st.caption(f"✅ Within your ₹{budget_lakh} L budget.")
                    else:
                        st.caption(f"⚠️ About {short_inr(est - budget)} above your ₹{budget_lakh} L budget.")
                else:
                    st.markdown("Pick a locality and press **Reveal my estimate**.")
        with e2:
            with st.container(key="card_house"):
                st.markdown('<div class="card-title">Your house, visualised</div>', unsafe_allow_html=True)
                if st.session_state["has_predicted"]:
                    _, lvl = price_tier(est, stats)
                    st.markdown(render_house_svg(lvl, garage, garden, lift), unsafe_allow_html=True)
                    traits = get_quality_traits(loc_row(location)["note"], garage, garden, lift,
                                                near_school, age, float(loc_row(location)["distance_to_city_km"]))
                    st.markdown("".join(f'<span class="pill">{t}</span>' for t in traits), unsafe_allow_html=True)
                else:
                    st.markdown("Your house appears here after the estimate is revealed.")

        with st.container(key="card_afford"):
            st.markdown('<div class="card-title">Where can you buy this home? (all Prayagraj localities)</div>', unsafe_allow_html=True)
            tbl = map_df.sort_values("pred")[["location", "tier", "rate_per_sqft", "pred", "status"]].copy()
            tbl["Estimate"] = tbl["pred"].map(lambda v: f"{format_inr(v)}  ({short_inr(v)})")
            tbl["Status"] = tbl["status"].map(lambda s: STATUS_META[s]["label"])
            tbl["Base rate"] = tbl["rate_per_sqft"].map(lambda v: f"₹{int(v):,}/sq ft")
            tbl = tbl.rename(columns={"location": "Locality", "tier": "Segment"})[
                ["Locality", "Segment", "Base rate", "Estimate", "Status"]]
            st.dataframe(tbl, hide_index=True, width="stretch", height=330)

# ===========================================================================
# TAB 2 — COMPARE
# ===========================================================================
with tab_compare:
    st.markdown("Compare two properties anywhere in Prayagraj.")
    colA, colB = st.columns(2, gap="large")

    def property_form(prefix, default_area, default_loc):
        with st.container(key=f"card_form_{prefix}"):
            st.markdown(f'<div class="card-title">Property {prefix}</div>', unsafe_allow_html=True)
            ar = st.number_input("Area (sq ft)", 400, 6000, default_area, step=50, key=f"{prefix}_area")
            c1, c2 = st.columns(2)
            with c1:
                bed = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6], index=2, key=f"{prefix}_bed")
                ag = st.slider("Age (yrs)", 0, 60, 10, key=f"{prefix}_age")
            with c2:
                bath = st.selectbox("Bathrooms", [1, 2, 3, 4], index=1, key=f"{prefix}_bath")
                lc = st.selectbox("Locality", LOC_NAMES, index=LOC_NAMES.index(default_loc), key=f"{prefix}_loc")
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                g = st.checkbox("Parking", True, key=f"{prefix}_garage")
            with k2:
                ga = st.checkbox("Garden", False, key=f"{prefix}_garden")
            with k3:
                li = st.checkbox("Lift", False, key=f"{prefix}_lift")
            with k4:
                sc = st.checkbox("School", True, key=f"{prefix}_school")
        return (ar, bed, bath, ag, lc, g, ga, li, sc)

    with colA:
        specs_a = property_form("A", 1400, "Civil Lines")
    with colB:
        specs_b = property_form("B", 1400, "Naini")

    def _cmp():
        st.session_state["has_compared"] = True
        st.session_state["reveal_counter_cmp"] += 1

    st.button("Compare properties", width="stretch", key="cmp_btn", on_click=_cmp)

    if st.session_state["has_compared"]:
        pa, ra = predict_one(specs_a)
        pb, rb = predict_one(specs_b)
        a_wins = pa >= pb
        rc = st.session_state["reveal_counter_cmp"]
        rA, rB = st.columns(2, gap="large")
        with rA:
            with st.container(key="card_res_a"):
                st.markdown(f'<div class="card-title">Property A · {specs_a[4]}</div>', unsafe_allow_html=True)
                st.markdown(puzzle_reveal_html(format_inr(pa), f"≈ {short_inr(pa)}", f"cmpA{rc}"), unsafe_allow_html=True)
                if a_wins:
                    st.markdown('<span class="win-badge">🏆 Higher value</span>', unsafe_allow_html=True)
        with rB:
            with st.container(key="card_res_b"):
                st.markdown(f'<div class="card-title">Property B · {specs_b[4]}</div>', unsafe_allow_html=True)
                st.markdown(puzzle_reveal_html(format_inr(pb), f"≈ {short_inr(pb)}", f"cmpB{rc}"), unsafe_allow_html=True)
                if not a_wins:
                    st.markdown('<span class="win-badge">🏆 Higher value</span>', unsafe_allow_html=True)

        hi, lo = (ra, rb) if a_wins else (rb, ra)
        rate_hi, rate_lo = float(loc_row(hi["location"])["rate_per_sqft"]), float(loc_row(lo["location"])["rate_per_sqft"])
        reasons = []
        if abs(hi["area_sqft"] - lo["area_sqft"]) > 50:
            reasons.append((abs(hi["area_sqft"] - lo["area_sqft"]) * rate_lo,
                            f"{abs(hi['area_sqft'] - lo['area_sqft']):.0f} sq ft {'more' if hi['area_sqft'] > lo['area_sqft'] else 'less'} area"))
        if hi["location"] != lo["location"]:
            reasons.append((abs(rate_hi - rate_lo) * min(hi["area_sqft"], lo["area_sqft"]),
                            f"{hi['location']} (₹{rate_hi:,.0f}/sq ft) vs {lo['location']} (₹{rate_lo:,.0f}/sq ft)"))
        if abs(lo["age_years"] - hi["age_years"]) > 3:
            reasons.append((abs(lo["age_years"] - hi["age_years"]) * 0.007 * max(pa, pb),
                            f"being {abs(lo['age_years'] - hi['age_years'])} years {'newer' if lo['age_years'] > hi['age_years'] else 'older'}"))
        amen = lambda r: r["garage"] + r["garden"] + r["lift"] + r["near_school"]
        if amen(hi) != amen(lo):
            reasons.append((abs(amen(hi) - amen(lo)) * 3e5, f"{abs(amen(hi) - amen(lo))} {'more' if amen(hi) > amen(lo) else 'fewer'} amenities"))
        reasons.sort(key=lambda r: -r[0])
        main = reasons[0][1] if reasons else "small differences in specifications"
        st.markdown(f'<div class="callout">Property {"A" if a_wins else "B"} costs <b>{format_inr(abs(pa - pb))}</b> '
                    f'({short_inr(abs(pa - pb))}) more, mainly due to {main}.</div>', unsafe_allow_html=True)

# ===========================================================================
# TAB 3 — INSIGHTS
# ===========================================================================
with tab_insights:
    k1, k2, k3, k4 = st.columns(4)
    for col, (label, value) in zip([k1, k2, k3, k4], [
        ("Median price", short_inr(stats["price_quantiles"][0.5])),
        ("Median area", f"{stats['area_median']:,.0f} sq ft"),
        ("Median rate", f"₹{stats['rate_median']:,.0f}/sq ft"),
        ("Localities", f"{len(loc_df)}"),
    ]):
        col.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>',
                     unsafe_allow_html=True)
    st.write("")

    def style_fig(fig, h=340):
        fig.update_layout(height=h, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="#fff", font=dict(family="Space Grotesk", color=INK))
        return fig

    with st.container(key="card_rate"):
        st.markdown('<div class="card-title">Base price per sq ft by locality (₹)</div>', unsafe_allow_html=True)
        tmp = loc_df.sort_values("rate_per_sqft")
        fb = px.bar(tmp, x="rate_per_sqft", y="location", orientation="h", color="tier",
                    color_discrete_map={"Premium": RED, "Mid-range": YELLOW, "Affordable": GREEN},
                    template="plotly_white")
        fb.update_traces(marker_line_color=INK, marker_line_width=2)
        fb.update_layout(xaxis_title="₹ per sq ft", yaxis_title="", legend_title="")
        st.plotly_chart(style_fig(fb, 620), key="bar_rate")
        st.caption("Rates come from 99acres / Housing.com locality pages (mid-2026) where available; "
                   "the two sites often disagree, so treat these as working figures. "
                   "Localities marked 'estimate' in locality_info.csv had no clear public figure.")

    i1, i2 = st.columns(2, gap="large")
    with i1:
        with st.container(key="card_hist"):
            st.markdown('<div class="card-title">Price distribution (₹ lakh)</div>', unsafe_allow_html=True)
            d2 = df.assign(price_lakh=df["price"] / 1e5)
            fh = px.histogram(d2, x="price_lakh", nbins=40, template="plotly_white", color_discrete_sequence=[BLUE])
            fh.update_traces(marker_line_color=INK, marker_line_width=1.5)
            fh.update_layout(xaxis_title="Price (₹ lakh)", yaxis_title="Count")
            st.plotly_chart(style_fig(fh), key="hist")
    with i2:
        with st.container(key="card_scatter"):
            st.markdown('<div class="card-title">Area vs price by segment</div>', unsafe_allow_html=True)
            d3 = df.merge(loc_df[["location", "tier"]], on="location").assign(price_lakh=lambda x: x["price"] / 1e5)
            fs = px.scatter(d3.sample(min(900, len(d3)), random_state=1), x="area_sqft", y="price_lakh", color="tier",
                            color_discrete_map={"Premium": RED, "Mid-range": YELLOW, "Affordable": GREEN},
                            template="plotly_white", opacity=0.8)
            fs.update_layout(xaxis_title="Area (sq ft)", yaxis_title="Price (₹ lakh)", legend_title="")
            st.plotly_chart(style_fig(fs), key="scatter")

    with st.container(key="card_lm"):
        st.markdown('<div class="card-title">Famous places and what nearby homes cost</div>', unsafe_allow_html=True)
        lm_tbl = lm_df.merge(loc_df[["location", "rate_per_sqft"]], left_on="nearest_locality", right_on="location")
        lm_tbl["Base rate nearby"] = lm_tbl["rate_per_sqft"].map(lambda v: f"₹{int(v):,}/sq ft")
        lm_tbl = lm_tbl.rename(columns={"name": "Place", "note": "About", "nearest_locality": "Nearest locality"})
        st.dataframe(lm_tbl[["Place", "About", "Nearest locality", "Base rate nearby"]], hide_index=True,
                     width="stretch")

    with st.expander("Dataset preview and model performance"):
        prev = df.head(25).copy()
        prev["price"] = prev["price"].map(lambda v: f"₹{indian_grouping(round(v))}")
        st.dataframe(prev, width="stretch", height=300)
        mdf = pd.DataFrame(meta["metrics"]).T
        mdf["MAE (₹ lakh)"] = (mdf["MAE"] / 1e5).round(2)
        mdf["RMSE (₹ lakh)"] = (mdf["RMSE"] / 1e5).round(2)
        mdf["R²"] = mdf["R2"].round(3)
        st.dataframe(mdf[["MAE (₹ lakh)", "RMSE (₹ lakh)", "R²"]], width="stretch")
        st.caption("The best model is chosen by lowest MAE (typical error in rupees).")

st.write("")
st.markdown(f"""
<div style="text-align:center; font-family:'Space Mono',monospace; font-size:0.72rem; padding:10px 0 24px 0;">
HomeValue AI · Prayagraj · {best_model_name} · Indicative estimates from a synthetic dataset calibrated to 2026 listing rates, not a valuation · {datetime.date.today():%d %b %Y}
</div>""", unsafe_allow_html=True)
