"""
app.py — HomeValue AI · Prayagraj
Neo-brutalist map dashboard for estimating house prices across Prayagraj
localities, with famous places, budget-aware map colouring, compare mode and
an insights tab. Prices are in Indian Rupees.
"""

import os
import json
import math
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


# ---------------------------------------------------------------------------
# ROOMS FROM AREA ("basic idea" of bedrooms / bathrooms for any land size)
# Same relationship the training data was built on (~1 bedroom per 480 sq ft,
# bathrooms ~ 0.75 x bedrooms), growing smoothly for very large areas.
# ---------------------------------------------------------------------------
def suggest_rooms(area_sqft):
    area = max(float(area_sqft), 0.0)
    if area <= 2900:
        beds = math.ceil(area / 480 - 0.5)
    else:
        beds = math.ceil(2900 / 480 - 0.5) + math.ceil((area - 2900) / 1500 - 0.5)
    beds = max(1, int(beds))
    baths = max(1, int(math.floor(beds * 0.75 + 0.5)))
    label = f"{beds} BHK large home / bungalow" if beds >= 7 else f"{beds} BHK"
    return {"bedrooms": beds, "bathrooms": baths, "label": label}

def max_rooms(area_sqft):
    """Most bedrooms / bathrooms that can sensibly fit: ~400 sq ft per bedroom, ~500 per bathroom."""
    area = max(float(area_sqft), 0.0)
    return max(1, int(area // 400)), max(1, int(area // 500))


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


# The model only saw homes up to ~5,000 sq ft, 6 bedrooms and 4 bathrooms.
# Bigger inputs are clamped for the model and the price is scaled with area.
AREA_CAP, BED_CAP, BATH_CAP = 5000, 6, 4
AREA_FLOOR = 400          # smallest size in the training data (~350 sq ft)
MIN_HOME_SQFT = 300       # below this nothing liveable can be built (smallest studio / 1 RK)


def clamp_for_model(area, bed, bath):
    a = min(max(area, AREA_FLOOR), AREA_CAP)
    return a, min(bed, BED_CAP), min(bath, BATH_CAP), area / a


def loc_row(name):
    return loc_df[loc_df["location"] == name].iloc[0]


@st.cache_data
def predict_all(area, bed, bath, age, garage, garden, lift, school):
    m_area, m_bed, m_bath, scale = clamp_for_model(area, bed, bath)
    rows = [build_input_df(m_area, m_bed, m_bath, age, r.distance_to_city_km,
                           garage, garden, lift, school, r.location)[1]
            for r in loc_df.itertuples()]
    return model.predict(pd.DataFrame(rows)) * scale


def predict_one(specs):
    """specs = (area, bed, bath, age, locality, garage, garden, lift, school)"""
    area, bed, bath, age, loc, garage, garden, lift, school = specs
    d = float(loc_row(loc)["distance_to_city_km"])
    m_area, m_bed, m_bath, scale = clamp_for_model(area, bed, bath)
    X, _ = build_input_df(m_area, m_bed, m_bath, age, d, garage, garden, lift, school, loc)
    _, row = build_input_df(area, bed, bath, age, d, garage, garden, lift, school, loc)
    return float(model.predict(X)[0]) * scale, row


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
    <p class="nn-sub">House price estimates across {len(loc_df)} Prayagraj localities · prices in ₹</p>
  </div>
  <span class="nn-badge">{best_model_name} · R² {best_r2:.3f} · MAE ₹{indian_grouping(best_mae)}</span>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# INPUT DEFAULTS + CALLBACKS
# ---------------------------------------------------------------------------
def _sync_rooms():
    s = suggest_rooms(st.session_state["in_area"])
    mb, mt = max_rooms(st.session_state["in_area"])
    st.session_state["in_beds"] = min(s["bedrooms"], mb)
    st.session_state["in_baths"] = min(s["bathrooms"], mt)


_d = suggest_rooms(1200)
for k, v in {"in_area": 1200, "in_beds": _d["bedrooms"], "in_baths": _d["bathrooms"],
             "in_age": 8, "in_garage": True, "in_garden": False, "in_lift": False,
             "in_school": True, "in_budget": 7500000, "in_loc": "Civil Lines"}.items():
    st.session_state.setdefault(k, v)


def _reveal_cmp():
    st.session_state["has_compared"] = True
    st.session_state["reveal_counter_cmp"] += 1


def place_label(code):
    if code == OVERVIEW:
        return "🗺️ Whole city overview"
    kind, name = code.split("::")
    return f"📍 {name}" if kind == "loc" else f"⭐ {name}"


PLACE_CODES = ([OVERVIEW] + [f"loc::{n}" for n in LOC_NAMES]
               + [f"lm::{n}" for n in sorted(lm_df["name"].tolist())])

left, right = st.columns([1, 3], gap="large")

# ---------------------------------------------------------------------------
# LEFT: FILTERS PANEL
# ---------------------------------------------------------------------------
with left:
    with st.container(key="filters_panel"):
        st.markdown('<div class="panel-title">Your property</div>', unsafe_allow_html=True)
        area = st.number_input("Area (sq ft)", min_value=1, max_value=50000,
                               step=50, key="in_area", on_change=_sync_rooms)
        valid_main = area >= MIN_HOME_SQFT
        if not valid_main:
            st.error(f"A home can't be built in {area} sq ft. Minimum is {MIN_HOME_SQFT} sq ft "
                     f"(a small studio / 1 RK).")
        sug = suggest_rooms(area)
        max_b, max_t = max_rooms(area)
        st.session_state["in_beds"] = min(max(st.session_state["in_beds"], 1), max_b)
        st.session_state["in_baths"] = min(max(st.session_state["in_baths"], 1), max_t)
        st.caption(f"Typical for this size: {sug['label']}, {sug['bathrooms']} bath · "
                   f"max possible: {max_b} bed / {max_t} bath")
        c1, c2 = st.columns(2)
        beds = c1.number_input("Bedrooms", min_value=1, max_value=max_b, key="in_beds")
        baths = c2.number_input("Bathrooms", min_value=1, max_value=max_t, key="in_baths")
        age = st.slider("Age of building (years)", 0, 60, key="in_age")
        loc_sel = st.selectbox("Locality", LOC_NAMES, key="in_loc")

        st.markdown('<div class="panel-sub">Amenities</div>', unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        garage = a1.checkbox("Garage", key="in_garage")
        garden = a2.checkbox("Garden", key="in_garden")
        lift = a1.checkbox("Lift", key="in_lift")
        school = a2.checkbox("Near school", key="in_school")

        st.markdown('<div class="panel-sub">Budget</div>', unsafe_allow_html=True)
        budget = st.number_input("Your budget (₹)", min_value=100000, max_value=1000000000,
                                 step=100000, key="in_budget")
        st.caption(f"= {short_inr(budget)}")

        st.button("Reveal my estimate", key="btn_reveal", on_click=_reveal, width="stretch",
                  disabled=not valid_main)

specs_main = (float(area), int(beds), int(baths), int(age), loc_sel,
              bool(garage), bool(garden), bool(lift), bool(school))

# Live per-locality estimates for the map
preds = predict_all(float(max(area, MIN_HOME_SQFT)), int(beds), int(baths), int(age),
                    bool(garage), bool(garden), bool(lift), bool(school))
map_df = loc_df.copy()
map_df["pred"] = preds
map_df["status"] = [budget_status(p, budget) for p in preds]
map_df["tier"] = [price_tier(p, stats)[0] for p in preds]

# ---------------------------------------------------------------------------
# RIGHT: TABS
# ---------------------------------------------------------------------------
with right:
    tab_map, tab_cmp, tab_ins, tab_emi = st.tabs(["Map", "Compare", "Insights", "EMI calculator"])

    # ======================= MAP TAB =======================
    with tab_map:
        if not valid_main:
            st.error(f"Enter an area of at least {MIN_HOME_SQFT} sq ft to see estimates on the map.")
        else:
            t1, t2, t3, t4, t5 = st.columns([3, 2, 1.2, 1.2, 1.2])
            find_place = t1.selectbox("Find a place", PLACE_CODES, key="find_place",
                                      format_func=place_label)
            visible = t2.multiselect("Show", list(STATUS_META.keys()), default=list(STATUS_META.keys()),
                                     format_func=lambda s: STATUS_META[s]["label"])
            show_lm = t3.checkbox("Landmarks", value=True)
            t4.write("")
            t4.button("Toggle 3D tilt", key="btn_tilt", on_click=_toggle_tilt, width="stretch")
            t5.write("")
            t5.button("Reset view", key="btn_reset", on_click=_reset_view, width="stretch")

            price_main, row_main = predict_one(specs_main)
            focus_loc = loc_sel if st.session_state["has_predicted"] else None
            focus_text = short_inr(price_main) if focus_loc else None

            with st.container(key="map_wrap"):
                fig = build_map(map_df, visible, show_lm, find_place,
                                st.session_state["tilt"], focus_loc, focus_text)
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

                with st.container(key="legend_card"):
                    st.markdown(
                        '<div class="lg-title">LEGEND</div>'
                        + "".join(f'<div class="lg-row"><span class="lg-dot" style="background:{m["color"]}"></span>{m["label"]}</div>'
                                  for m in STATUS_META.values())
                        + '<div class="lg-row"><span class="lg-sq"></span>Famous place</div>',
                        unsafe_allow_html=True)

                counts = map_df["status"].value_counts().to_dict()
                n_w, n_s, n_o = counts.get("within", 0), counts.get("stretch", 0), counts.get("over", 0)
                tot = max(n_w + n_s + n_o, 1)
                p1, p2 = n_w / tot * 100, (n_w + n_s) / tot * 100
                with st.container(key="stats_card"):
                    st.markdown(f"""
    <div class="st-title">YOUR BUDGET {short_inr(budget)}</div>
    <div class="st-grid">
    <div class="st-rows">
    <div class="st-row" style="background:{GREEN}">Within <b>{n_w}</b></div>
    <div class="st-row" style="background:{YELLOW}">Stretch <b>{n_s}</b></div>
    <div class="st-row" style="background:{RED}">Over <b>{n_o}</b></div>
    </div>
    <div class="st-donutbox"><div class="donut" style="background:conic-gradient({GREEN} 0 {p1:.1f}%, {YELLOW} {p1:.1f}% {p2:.1f}%, {RED} {p2:.1f}% 100%)"></div></div>
    </div>
    """, unsafe_allow_html=True)

            if find_place != OVERVIEW:
                kind, name = find_place.split("::")
                if kind == "loc":
                    r = loc_row(name)
                    p = float(map_df.loc[map_df["location"] == name, "pred"].iloc[0])
                    st.markdown(f'<div class="place-info"><b>{name}</b> ({r["tier"]}) — {r["note"]}<br>'
                                f'Base rate ₹{indian_grouping(r["rate_per_sqft"])}/sq ft · '
                                f'{r["distance_to_city_km"]} km from Civil Lines · '
                                f'Estimate for your specs: <b>{format_inr(p)}</b></div>', unsafe_allow_html=True)
                else:
                    r = lm_df[lm_df["name"] == name].iloc[0]
                    st.markdown(f'<div class="place-info"><b>{name}</b> — {r["note"]}<br>'
                                f'Nearest locality: {r["nearest_locality"]}</div>', unsafe_allow_html=True)

            # ---- estimate result ----
            if st.session_state["has_predicted"]:
                loc_info = loc_row(loc_sel)
                tier_name, tier_lvl = price_tier(price_main, stats)
                status = budget_status(price_main, budget)
                traits = get_quality_traits(loc_info["note"], garage, garden, lift, school,
                                            age, float(loc_info["distance_to_city_km"]))
                why = explain_price(row_main, stats, float(loc_info["rate_per_sqft"]))
                rc1, rc2 = st.columns([3, 2])
                with rc1:
                    with st.container(key="card_result"):
                        st.markdown('<div class="card-title">Estimated price</div>', unsafe_allow_html=True)
                        st.markdown(puzzle_reveal_html(format_inr(price_main),
                                                       f"{tier_name} · {loc_sel} · ≈ ₹{indian_grouping(price_main / area)}/sq ft",
                                                       key=f"r{st.session_state['reveal_counter']}"),
                                    unsafe_allow_html=True)
                        st.markdown(f'<span class="win-badge" style="background:{STATUS_META[status]["color"]}">'
                                    f'{STATUS_META[status]["label"]}</span>', unsafe_allow_html=True)
                        low, high = price_main - best_mae, price_main + best_mae
                        st.caption(f"Typical model error ±{short_inr(best_mae)} → likely range "
                                   f"{short_inr(max(low, 0))} – {short_inr(high)}")
                        st.markdown('<div class="panel-sub">Why this estimate</div>', unsafe_allow_html=True)
                        for b in why:
                            st.markdown(f"- {b}")
                        st.markdown("".join(f'<span class="pill">{t}</span>' for t in traits),
                                    unsafe_allow_html=True)
                        if area > AREA_CAP or beds > BED_CAP or baths > BATH_CAP:
                            st.caption("Your size is beyond the training data, so the price is scaled up from "
                                       "the largest homes the model has seen. Treat it as a rough guide.")
                with rc2:
                    with st.container(key="card_house"):
                        st.markdown('<div class="card-title">Your home</div>', unsafe_allow_html=True)
                        st.markdown(render_house_svg(tier_lvl, garage, garden, lift), unsafe_allow_html=True)
            else:
                st.info("Set your property details on the left and press **Reveal my estimate**.")

    # ======================= COMPARE TAB =======================
    with tab_cmp:
        def spec_block(tag, title, defaults):
            with st.container(key=f"card_spec_{tag}"):
                st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
                loc = st.selectbox("Locality", LOC_NAMES, index=LOC_NAMES.index(defaults["loc"]), key=f"{tag}_loc")
                ar = st.number_input("Area (sq ft)", 1, 50000, defaults["area"], 50, key=f"{tag}_area")
                if ar < MIN_HOME_SQFT:
                    st.error(f"Minimum is {MIN_HOME_SQFT} sq ft — nothing liveable fits in {ar} sq ft.")
                s = suggest_rooms(ar)
                mb, mt = max_rooms(ar)
                st.caption(f"Max possible for this size: {mb} bed / {mt} bath")
                x1, x2 = st.columns(2)
                bd = x1.number_input("Bedrooms", 1, mb, min(s["bedrooms"], mb), key=f"{tag}_bed_{ar}")
                bt = x2.number_input("Bathrooms", 1, mt, min(s["bathrooms"], mt), key=f"{tag}_bath_{ar}")
                ag = st.slider("Age (years)", 0, 60, defaults["age"], key=f"{tag}_age")
                y1, y2 = st.columns(2)
                g1 = y1.checkbox("Garage", defaults["garage"], key=f"{tag}_garage")
                g2 = y2.checkbox("Garden", defaults["garden"], key=f"{tag}_garden")
                g3 = y1.checkbox("Lift", False, key=f"{tag}_lift")
                g4 = y2.checkbox("Near school", True, key=f"{tag}_school")
            return (float(ar), int(bd), int(bt), int(ag), loc, bool(g1), bool(g2), bool(g3), bool(g4))

        ca, cb = st.columns(2, gap="large")
        with ca:
            spec_a = spec_block("a", "Property A",
                                {"loc": "Civil Lines", "area": 1500, "age": 5, "garage": True, "garden": True})
        with cb:
            spec_b = spec_block("b", "Property B",
                                {"loc": "Naini", "area": 1500, "age": 15, "garage": False, "garden": False})

        st.button("Compare properties", key="cmp_btn", on_click=_reveal_cmp)

        if st.session_state["has_compared"] and min(spec_a[0], spec_b[0]) < MIN_HOME_SQFT:
            st.error(f"Both properties need at least {MIN_HOME_SQFT} sq ft to be compared.")
        elif st.session_state["has_compared"]:
            pa, _ = predict_one(spec_a)
            pb, _ = predict_one(spec_b)
            ra, rb = st.columns(2, gap="large")
            for col, nm, p, sp in ((ra, "Property A", pa, spec_a), (rb, "Property B", pb, spec_b)):
                with col:
                    with st.container(key=f"card_res_{nm[-1].lower()}"):
                        win = (p >= max(pa, pb))
                        badge = '<span class="win-badge">Higher value</span>' if win else ""
                        st.markdown(f'<div class="card-title">{nm} · {sp[4]} {badge}</div>', unsafe_allow_html=True)
                        st.markdown(puzzle_reveal_html(format_inr(p), f"{sp[0]:,.0f} sq ft · {sp[1]} bed · {sp[2]} bath",
                                                       key=f"c{nm[-1].lower()}{st.session_state['reveal_counter_cmp']}"),
                                    unsafe_allow_html=True)
            diff = abs(pa - pb)
            hi, lo = ("A", "B") if pa >= pb else ("B", "A")
            pct = diff / max(min(pa, pb), 1) * 100
            reasons = []
            if spec_a[4] != spec_b[4]:
                rate_a = float(loc_row(spec_a[4])["rate_per_sqft"])
                rate_b = float(loc_row(spec_b[4])["rate_per_sqft"])
                if rate_a != rate_b:
                    reasons.append(f"{spec_a[4] if rate_a > rate_b else spec_b[4]} has the higher locality rate")
            if spec_a[0] != spec_b[0]:
                reasons.append("larger area on " + ("A" if spec_a[0] > spec_b[0] else "B"))
            if spec_a[3] != spec_b[3]:
                reasons.append("newer building on " + ("A" if spec_a[3] < spec_b[3] else "B"))
            na, nb = sum(spec_a[5:]), sum(spec_b[5:])
            if na != nb:
                reasons.append("more amenities on " + ("A" if na > nb else "B"))
            why_txt = "; ".join(reasons) if reasons else "the two are almost identical"
            st.markdown(f'<div class="callout">Property {hi} is worth about <b>{format_inr(diff)}</b> '
                        f'({pct:.0f}%) more than Property {lo}. Main drivers: {why_txt}.</div>',
                        unsafe_allow_html=True)

    # ======================= INSIGHTS TAB =======================
    with tab_ins:
        k1, k2, k3, k4 = st.columns(4)
        for col, label, val in (
            (k1, "Homes in dataset", f"{len(df):,}"),
            (k2, "Median price", short_inr(stats["price_quantiles"][0.5])),
            (k3, "Median rate", f"₹{indian_grouping(stats['rate_median'])}/sq ft"),
            (k4, "Model R²", f"{best_r2:.3f}")):
            col.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div>'
                         f'<div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)

        g1, g2 = st.columns(2)
        with g1:
            with st.container(key="card_hist"):
                st.markdown('<div class="card-title">Price distribution</div>', unsafe_allow_html=True)
                f1 = px.histogram(df, x="price", nbins=40, color_discrete_sequence=[BLUE])
                f1.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="#fff", xaxis_title="Price (₹)", yaxis_title="Homes")
                st.plotly_chart(f1, width="stretch")
        with g2:
            with st.container(key="card_loc"):
                st.markdown('<div class="card-title">Median ₹/sq ft by locality</div>', unsafe_allow_html=True)
                rate = (df.assign(rate=df["price"] / df["area_sqft"]).groupby("location")["rate"]
                        .median().sort_values().reset_index())
                f2 = px.bar(rate, x="rate", y="location", orientation="h", color_discrete_sequence=[PINK])
                f2.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="#fff", xaxis_title="₹ per sq ft", yaxis_title="")
                st.plotly_chart(f2, width="stretch")

        with st.container(key="card_corr"):
            st.markdown('<div class="card-title">Feature correlation</div>', unsafe_allow_html=True)
            corr = df.select_dtypes("number").corr()
            f3 = px.imshow(corr, text_auto=".2f", color_continuous_scale=[[0, TEAL], [0.5, CREAM], [1, RED]],
                           zmin=-1, zmax=1, aspect="auto")
            f3.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(f3, width="stretch")

        with st.expander("Dataset preview"):
            st.dataframe(df.head(50), width="stretch")

    # ======================= EMI TAB =======================
    with tab_emi:
        default_price = int(round(price_main, -3)) if valid_main else 5000000
        e1, e2 = st.columns([1, 2], gap="large")
        with e1:
            with st.container(key="card_emi_in"):
                st.markdown('<div class="card-title">Loan details</div>', unsafe_allow_html=True)
                if not valid_main:
                    st.warning("Enter a valid area on the left to start from your estimated price.")
                price_in = st.number_input("Property price (₹)", min_value=100000, max_value=1000000000,
                                           value=max(default_price, 100000), step=100000, key="emi_price")
                down_pct = st.slider("Down payment (%)", 0, 90, 20, key="emi_down")
                rate = st.slider("Interest rate (% per year)", 5.0, 15.0, 8.5, 0.05, key="emi_rate")
                years = st.slider("Tenure (years)", 1, 30, 20, key="emi_years")
                income = st.number_input("Monthly income (₹, optional)", min_value=0, value=0,
                                         step=5000, key="emi_income")

        down = price_in * down_pct / 100
        loan = price_in - down
        n = years * 12
        r = rate / 12 / 100
        emi_val = loan / n if r == 0 else loan * r * (1 + r) ** n / ((1 + r) ** n - 1)
        total_pay = emi_val * n
        total_int = total_pay - loan

        with e2:
            k1, k2, k3, k4 = st.columns(4)
            for col, label, val in (
                (k1, "Monthly EMI", format_inr(emi_val)),
                (k2, "Loan amount", short_inr(loan)),
                (k3, "Total interest", short_inr(total_int)),
                (k4, "Total payable", short_inr(total_pay))):
                col.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div>'
                             f'<div class="kpi-value" style="font-size:1.05rem">{val}</div></div>',
                             unsafe_allow_html=True)

            if income > 0:
                ratio = emi_val / income * 100
                msg = (f"EMI is <b>{ratio:.0f}%</b> of your monthly income. "
                       + ("Comfortable (under 40%)." if ratio <= 40 else
                          "High — lenders usually prefer EMI under 40-50% of income."))
                st.markdown(f'<div class="callout">{msg}</div>', unsafe_allow_html=True)
            if price_in > budget:
                st.markdown(f'<div class="callout">This price is {short_inr(price_in - budget)} above your '
                            f'budget of {short_inr(budget)}. A larger down payment may help.</div>',
                            unsafe_allow_html=True)

            g1, g2 = st.columns(2)
            with g1:
                with st.container(key="card_emi_pie"):
                    st.markdown('<div class="card-title">Where your money goes</div>', unsafe_allow_html=True)
                    pie = px.pie(names=["Down payment", "Loan principal", "Interest"],
                                 values=[down, loan, total_int], hole=0.5,
                                 color_discrete_sequence=[TEAL, BLUE, PINK])
                    pie.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(pie, width="stretch")
            with g2:
                with st.container(key="card_emi_bal"):
                    st.markdown('<div class="card-title">Outstanding loan by year</div>', unsafe_allow_html=True)
                    bal, rows = loan, [(0, loan)]
                    for y in range(1, years + 1):
                        for _ in range(12):
                            bal = bal * (1 + r) - emi_val
                        rows.append((y, max(bal, 0)))
                    bal_df = pd.DataFrame(rows, columns=["Year", "Balance"])
                    ln = px.area(bal_df, x="Year", y="Balance", color_discrete_sequence=[BLUE])
                    ln.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                                     plot_bgcolor="#fff", yaxis_title="₹ outstanding")
                    st.plotly_chart(ln, width="stretch")
            st.caption("EMI = P × r × (1+r)ⁿ / ((1+r)ⁿ − 1), r = monthly rate, n = months. "
                       "Indicative only — banks add processing fees and may change rates.")


st.caption(f"HomeValue AI · {best_model_name} model trained on a synthetic Prayagraj dataset · "
           f"{datetime.date.today().year} · estimates are indicative, not valuations.")
