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

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

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
    """Most bedrooms / bathrooms that can sensibly fit (~400 sq ft per bedroom, ~500 per
    bathroom), never more than the model has seen in its training data."""
    area = max(float(area_sqft), 0.0)
    return max(1, min(int(area // 400), BED_CAP)), max(1, min(int(area // 500), BATH_CAP))


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "Model")
META_PATH = os.path.join(MODEL_DIR, "model_metadata.json")
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "Dataset", "house_data_cleaned.csv")
LOC_PATH = os.path.join(SCRIPT_DIR, "..", "Dataset", "locality_info.csv")
LM_PATH = os.path.join(SCRIPT_DIR, "..", "Dataset", "landmarks.csv")


@st.cache_resource
def load_model():
    """XGBoost booster from its JSON export (works across XGBoost versions) + metadata."""
    with open(META_PATH) as f:
        meta = json.load(f)
    booster = xgb.Booster()
    booster.load_model(os.path.join(MODEL_DIR, meta["model_file"]))
    return booster, meta


@st.cache_data
def load_tables():
    return (pd.read_csv(DATA_PATH), pd.read_csv(LOC_PATH), pd.read_csv(LM_PATH))


model, meta = load_model()
df, loc_df, lm_df = load_tables()
loc_df = loc_df.reset_index(drop=True)
best_model_name = meta["best_model"]
best_r2 = meta["metrics"][best_model_name]["R2"]
best_mae = meta["metrics"][best_model_name]["MAE"]
best_mape = meta["metrics"][best_model_name]["MAPE"]

# What the model has actually seen. Anything beyond this is scaled, not learned.
_rng = meta["training_ranges"]
AREA_FLOOR, AREA_CAP = int(_rng["area_sqft"][0]), int(_rng["area_sqft"][1])
BED_CAP, BATH_CAP = int(_rng["bedrooms"][1]), int(_rng["bathrooms"][1])
MIN_HOME_SQFT = 300       # below this nothing liveable can be built (smallest studio / 1 RK)

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
             "snap": None, "show_afford": False, "has_compared": False,
             "find_place": OVERVIEW}.items():
    st.session_state.setdefault(k, v)


def _reveal():
    """Freeze the current inputs as 'the estimate'. It only changes when pressed again."""
    ss = st.session_state
    ss["snap"] = (float(ss["in_area"]), int(ss["in_beds"]), int(ss["in_baths"]), int(ss["in_age"]),
                  ss["in_loc"], bool(ss["in_garage"]), bool(ss["in_garden"]), bool(ss["in_lift"]),
                  bool(ss["in_school"]))
    ss["reveal_counter"] += 1


def _toggle_afford():
    st.session_state["show_afford"] = not st.session_state["show_afford"]


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


_PP = meta["preprocessing"]
_NUM_COLS = meta["feature_cols_numeric"]
_MEAN, _SCALE = np.array(_PP["numeric_mean"]), np.array(_PP["numeric_scale"])
_CATS = np.array(_PP["location_categories"])
_DIST = loc_df.set_index("location")["distance_to_city_km"]


def predict_rows(area, bed, bath, age, garage, garden, lift, school, location):
    """Prices (₹) for one or many homes. Every argument is a scalar or an array.

    Sizes outside what the model has seen are clamped for the model and the price is
    scaled with area (a 24,000 sq ft home costs about twice a 12,000 sq ft one)."""
    area, bed, bath, age, garage, garden, lift, school, location = np.broadcast_arrays(
        *[np.atleast_1d(np.asarray(v)) for v in (area, bed, bath, age, garage, garden, lift, school, location)])
    area = area.astype(float)
    m_area = np.clip(area, AREA_FLOOR, AREA_CAP)
    m_bed, m_bath = np.minimum(bed, BED_CAP), np.minimum(bath, BATH_CAP)
    g, ga, li, sc = (np.asarray(v).astype(int) for v in (garage, garden, lift, school))
    frame = pd.DataFrame({
        "area_sqft": m_area, "bedrooms": m_bed, "bathrooms": m_bath,
        "age_years": age, "distance_to_city_km": _DIST.reindex(location).to_numpy(float),
        "total_rooms": m_bed + m_bath, "amenity_score": g + ga + li + sc,
        "is_new": (np.asarray(age) <= 5).astype(int),
        "garage": g, "garden": ga, "lift": li, "near_school": sc, "location": location})
    num = (frame[_NUM_COLS].to_numpy(float) - _MEAN) / _SCALE
    cat = (frame["location"].to_numpy()[:, None] == _CATS[None, :]).astype(float)
    return np.expm1(model.predict(xgb.DMatrix(np.hstack([num, cat])))) * (area / m_area)


@st.cache_data
def predict_all(area, bed, bath, age, garage, garden, lift, school):
    return predict_rows(area, bed, bath, age, garage, garden, lift, school, loc_df["location"].to_numpy())


def predict_one(specs):
    """specs = (area, bed, bath, age, locality, garage, garden, lift, school)"""
    area, bed, bath, age, loc, garage, garden, lift, school = specs
    price = float(predict_rows(area, bed, bath, age, garage, garden, lift, school, loc)[0])
    _, row = build_input_df(area, bed, bath, age, float(_DIST[loc]), garage, garden, lift, school, loc)
    return price, row


# ---------------------------------------------------------------------------
# WHAT CAN MY BUDGET BUY?  (largest home each locality offers for a budget)
# ---------------------------------------------------------------------------
AFFORD_MAX = 50000        # same limit as the area box
AFFORD_AREAS = np.concatenate([np.arange(MIN_HOME_SQFT, AREA_CAP + 1, 50),
                               np.arange(AREA_CAP + 250, AFFORD_MAX + 1, 250)])
_aff_rooms = [(min(suggest_rooms(x)["bedrooms"], max_rooms(x)[0]),
               min(suggest_rooms(x)["bathrooms"], max_rooms(x)[1])) for x in AFFORD_AREAS]
AFFORD_BEDS = np.array([r[0] for r in _aff_rooms])
AFFORD_BATHS = np.array([r[1] for r in _aff_rooms])


@st.cache_data
def what_budget_buys(budget, age, garage, garden, lift, school):
    """For every locality: the largest home (50 sq ft steps, rooms suggested for each size)
    whose estimated price is within `budget`. Returns one row per locality."""
    rows = []
    for name in loc_df["location"]:
        p = predict_rows(AFFORD_AREAS, AFFORD_BEDS, AFFORD_BATHS, age, garage, garden, lift, school, name)
        p = np.maximum.accumulate(p)              # a bigger home never costs less
        ok = np.nonzero(p <= budget)[0]
        base = {"Locality": name, "Smallest home price": float(p[0])}
        if ok.size == 0:
            rows.append({**base, "Area": np.nan, "Beds": np.nan, "Baths": np.nan,
                         "Price": np.nan, "At limit": False})
        else:
            i = int(ok[-1])
            rows.append({**base, "Area": float(AFFORD_AREAS[i]), "Beds": int(AFFORD_BEDS[i]),
                         "Baths": int(AFFORD_BATHS[i]), "Price": float(p[i]),
                         "At limit": bool(i == len(AFFORD_AREAS) - 1)})
    return pd.DataFrame(rows)


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


def fit_view(points, width_px=900, height_px=660, pad=0.5):
    """Centre + zoom that keeps every (lat, lon) point on screen."""
    import math
    lats = [p[0] for p in points]; lons = [p[1] for p in points]
    lat0, lat1, lon0, lon1 = min(lats), max(lats), min(lons), max(lons)
    clat, clon = (lat0 + lat1) / 2, (lon0 + lon1) / 2
    span_lat = max(lat1 - lat0, 0.004); span_lon = max(lon1 - lon0, 0.004)
    cosf = math.cos(math.radians(clat))
    z_lon = math.log2(width_px * 360 / (512 * span_lon))
    z_lat = math.log2(height_px * 360 * cosf / (512 * span_lat))
    return clat, clon, max(10.0, min(13.5, min(z_lon, z_lat) - pad))


def build_map(map_df, visible, show_lm, find_place, tilt, focus_loc, focus_text):
    fig = go.Figure()
    vis = map_df[map_df["status"].isin(visible)]

    if focus_loc is not None:
        r = loc_row(focus_loc)
        fig.add_trace(go.Scattermap(lat=[r["lat"]], lon=[r["lon"]], mode="markers",
                                    marker=dict(size=42, color=PINK, opacity=0.75),
                                    hoverinfo="skip", showlegend=False))

    if not vis.empty:
        # dark outline layer under the coloured dots
        fig.add_trace(go.Scattermap(lat=vis["lat"].astype(float).tolist(), lon=vis["lon"].astype(float).tolist(),
                                    mode="markers", marker=dict(size=21, color=INK),
                                    hoverinfo="skip", showlegend=False))
        # one trace per budget status, each with ONE plain colour (most reliable across plotly.js versions)
        for status, meta in STATUS_META.items():
            sub = vis[vis["status"] == status]
            if sub.empty:
                continue
            fig.add_trace(go.Scattermap(
                lat=sub["lat"].astype(float).tolist(), lon=sub["lon"].astype(float).tolist(),
                mode="markers+text", text=sub["location"].tolist(),
                textposition="top center", textfont=dict(size=11, color=INK),
                marker=dict(size=15, color=meta["color"]),
                customdata=[[int(r), str(p), str(meta["label"]), str(t)]
                            for r, p, t in zip(sub["rate_per_sqft"], sub["pred"].map(short_inr), sub["tier"])],
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
    if find_place == OVERVIEW and not vis.empty and len(vis) < len(map_df):
        pts = list(zip(vis["lat"].astype(float), vis["lon"].astype(float)))
        if focus_loc is not None:
            r = loc_row(focus_loc); pts.append((float(r["lat"]), float(r["lon"])))
        lat, lon, zoom = fit_view(pts)
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
  <span class="nn-badge">{best_model_name} · R² {best_r2:.3f} · typical error ±{best_mape*100:.1f}%</span>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# INPUT DEFAULTS + CALLBACKS
# ---------------------------------------------------------------------------
def _apply_suggested(area_key, beds_key, baths_key):
    area = st.session_state[area_key]
    s, (mb, mt) = suggest_rooms(area), max_rooms(area)
    st.session_state[beds_key] = min(s["bedrooms"], mb)
    st.session_state[baths_key] = min(s["bathrooms"], mt)


def _on_area_change(area_key, beds_key, baths_key, manual_key):
    """Area changed: follow the suggestion, but never overwrite rooms the person chose."""
    if not st.session_state[manual_key]:
        _apply_suggested(area_key, beds_key, baths_key)


def _mark_manual(manual_key):
    st.session_state[manual_key] = True


def _reset_rooms(area_key, beds_key, baths_key, manual_key):
    st.session_state[manual_key] = False
    _apply_suggested(area_key, beds_key, baths_key)


def _clamp_rooms(area, beds_key, baths_key):
    mb, mt = max_rooms(area)
    st.session_state[beds_key] = min(max(st.session_state[beds_key], 1), mb)
    st.session_state[baths_key] = min(max(st.session_state[baths_key], 1), mt)


_d = suggest_rooms(1200)
for k, v in {"in_area": 1200, "in_beds": _d["bedrooms"], "in_baths": _d["bathrooms"], "in_manual": False,
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
        area = st.number_input("Area (sq ft)", min_value=1, max_value=50000, step=50, key="in_area",
                               on_change=_on_area_change, args=("in_area", "in_beds", "in_baths", "in_manual"))
        valid_main = area >= MIN_HOME_SQFT
        if not valid_main:
            st.error(f"A home can't be built in {area} sq ft. Minimum is {MIN_HOME_SQFT} sq ft "
                     f"(a small studio / 1 RK).")
        elif area > AREA_CAP:
            st.info(f"{area:,} sq ft is bigger than any home the model was trained on ({AREA_CAP:,} sq ft). "
                    f"The price is scaled up with area and rooms stop at {BED_CAP} bed / {BATH_CAP} bath, "
                    f"so treat it as a rough guide.")
        sug = suggest_rooms(area)
        max_b, max_t = max_rooms(area)
        _clamp_rooms(area, "in_beds", "in_baths")
        sug_b, sug_t = min(sug["bedrooms"], max_b), min(sug["bathrooms"], max_t)
        st.caption(f"Typical for this size: {sug_b} BHK, {sug_t} bath"
                   + (" (model limit)" if (sug_b, sug_t) != (sug["bedrooms"], sug["bathrooms"]) else "") + " · "
                   f"max possible: {max_b} bed / {max_t} bath"
                   + (" · rooms kept as you set them" if st.session_state["in_manual"] else ""))
        c1, c2 = st.columns(2)
        beds = c1.number_input("Bedrooms", min_value=1, max_value=max_b, key="in_beds",
                               on_change=_mark_manual, args=("in_manual",))
        baths = c2.number_input("Bathrooms", min_value=1, max_value=max_t, key="in_baths",
                                on_change=_mark_manual, args=("in_manual",))
        if st.session_state["in_manual"]:
            st.button("↺ Use suggested rooms", key="btn_rooms_reset", on_click=_reset_rooms,
                      args=("in_area", "in_beds", "in_baths", "in_manual"), width="stretch")
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
    tab_map, tab_cmp, tab_emi, tab_ins = st.tabs(["Map", "Compare", "EMI calculator", "Insights"])

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
            if not visible:
                visible = list(STATUS_META.keys())
                t2.caption("Nothing selected, showing all")
            show_lm = t3.checkbox("Landmarks", value=True)
            t4.write("")
            t4.button("Toggle 3D tilt", key="btn_tilt", on_click=_toggle_tilt, width="stretch")
            t5.write("")
            t5.button("Reset view", key="btn_reset", on_click=_reset_view, width="stretch")

            for _k in visible:
                _names = map_df.loc[map_df["status"] == _k, "location"].tolist()
                if _names and len(visible) < len(STATUS_META):
                    st.caption(f"{STATUS_META[_k]['label']} ({len(_names)}): " + ", ".join(_names))

            snap = st.session_state["snap"]
            if snap is not None:
                price_snap, row_snap = predict_one(snap)
                focus_loc, focus_text = snap[4], short_inr(price_snap)
            else:
                focus_loc = focus_text = None

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

            # ---- what can my budget buy + CSV download ----
            csv_df = map_df[["location", "pred", "status"]].rename(columns={"location": "Locality"})
            csv_df.insert(1, "Locality segment", csv_df["Locality"].map(loc_df.set_index("location")["tier"]))
            csv_df.insert(2, "Base rate (INR per sq ft)", csv_df["Locality"].map(loc_df.set_index("location")["rate_per_sqft"]))
            csv_df.insert(3, "Distance from Civil Lines (km)", csv_df["Locality"].map(_DIST))
            csv_df["Estimated price (INR)"] = csv_df.pop("pred").round(0).astype(int)
            csv_df["Budget status"] = csv_df.pop("status").map(lambda s: STATUS_META[s]["label"])
            for col, val in (("Budget (INR)", int(budget)), ("Area (sq ft)", int(area)), ("Bedrooms", int(beds)),
                             ("Bathrooms", int(baths)), ("Age (years)", int(age)), ("Garage", int(garage)),
                             ("Garden", int(garden)), ("Lift", int(lift)), ("Near school", int(school))):
                csv_df[col] = val
            csv_df = csv_df.sort_values("Estimated price (INR)").reset_index(drop=True)

            bb1, bb2 = st.columns(2)
            bb1.button("Hide what my budget buys" if st.session_state["show_afford"] else "What can my budget buy?",
                       key="btn_afford", on_click=_toggle_afford, width="stretch")
            bb2.download_button("Download all-locality estimates", data=csv_df.to_csv(index=False).encode("utf-8"),
                                file_name=f"prayagraj_estimates_{int(area)}sqft.csv", mime="text/csv",
                                key="btn_csv", width="stretch")

            if st.session_state["show_afford"]:
                res = what_budget_buys(float(budget), int(age), bool(garage), bool(garden), bool(lift), bool(school))
                mine = res[res["Locality"] == loc_sel].iloc[0]
                can = res.dropna(subset=["Area"]).sort_values(["Area", "Price"], ascending=False)
                with st.container(key="card_afford"):
                    st.markdown(f'<div class="card-title">What {short_inr(budget)} can buy</div>', unsafe_allow_html=True)
                    if pd.notna(mine["Area"]):
                        plus = "+" if mine["At limit"] else ""
                        rough = "~" if mine["Area"] > AREA_CAP else ""
                        st.markdown(
                            f'In **{loc_sel}**, {short_inr(budget)} buys a home of up to about '
                            f'**{rough}{int(mine["Area"]):,}{plus} sq ft** ({int(mine["Beds"])} bed · {int(mine["Baths"])} bath), '
                            f'estimated at {short_inr(mine["Price"])}.')
                        if mine["Area"] > AREA_CAP:
                            st.caption(f"Above {AREA_CAP:,} sq ft the model has no data: the price is scaled with "
                                       f"area and rooms are capped at {BED_CAP} bed / {BATH_CAP} bath. Rough guide.")
                        if mine["At limit"]:
                            st.caption(f"{AFFORD_MAX:,} sq ft is the largest size this app accepts.")
                    else:
                        st.error(f"{short_inr(budget)} is not enough for even the smallest home ({MIN_HOME_SQFT} sq ft) "
                                 f"in {loc_sel}, which is about {short_inr(mine['Smallest home price'])} there.")
                        if len(can):
                            st.markdown(f"**But it can buy a home in {len(can)} other localities:**")
                    if len(can):
                        tbl = pd.DataFrame({
                            "Locality": can["Locality"].values,
                            "Segment": can["Locality"].map(loc_df.set_index("location")["tier"]).values,
                            "Largest home": [f"{'~' if x > AREA_CAP else ''}{int(x):,}{'+' if lim else ''} sq ft" for x, lim in zip(can["Area"], can["At limit"])],
                            "Rooms": [f"{int(b)} bed · {int(t)} bath" for b, t in zip(can["Beds"], can["Baths"])],
                            "Est. price": [short_inr(x) for x in can["Price"]],
                            "": ["◀ your locality" if n == loc_sel else "" for n in can["Locality"]],
                        })
                        st.dataframe(tbl, hide_index=True, width="stretch", height=min(60 + 35 * len(tbl), 420))
                    else:
                        st.warning(f"{short_inr(budget)} is not enough for a {MIN_HOME_SQFT} sq ft home anywhere in Prayagraj "
                                   f"with these age and amenity settings.")
                    n_out = len(res) - len(can)
                    st.caption(f"Uses your building age ({age} yrs) and amenities; bedrooms and bathrooms are suggested "
                               f"for each size. {n_out} of {len(res)} localities are out of reach. Estimates, not offers.")

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

            # ---- estimate result (frozen when you press the button) ----
            if snap is not None:
                s_area, s_beds, s_baths, s_age, s_loc, s_gar, s_gdn, s_lift, s_sch = snap
                loc_info = loc_row(s_loc)
                tier_name, tier_lvl = price_tier(price_snap, stats)
                status = budget_status(price_snap, budget)
                traits = get_quality_traits(loc_info["note"], s_gar, s_gdn, s_lift, s_sch,
                                            s_age, float(loc_info["distance_to_city_km"]))
                why = explain_price(row_snap, stats, float(loc_info["rate_per_sqft"]))
                if snap != specs_main:
                    st.warning("You changed the inputs after this estimate. Press **Reveal my estimate** "
                               "to update it.")
                rc1, rc2 = st.columns([3, 2])
                with rc1:
                    with st.container(key="card_result"):
                        st.markdown('<div class="card-title">Estimated price</div>', unsafe_allow_html=True)
                        st.markdown(puzzle_reveal_html(format_inr(price_snap),
                                                       f"{tier_name} · {s_loc} · ≈ ₹{indian_grouping(price_snap / s_area)}/sq ft",
                                                       key=f"r{st.session_state['reveal_counter']}"),
                                    unsafe_allow_html=True)
                        st.caption(f"For: {s_area:,.0f} sq ft · {s_beds} bed · {s_baths} bath · {s_age} yrs old")
                        st.markdown(f'<span class="win-badge" style="background:{STATUS_META[status]["color"]}">'
                                    f'{STATUS_META[status]["label"]}</span>', unsafe_allow_html=True)
                        low, high = price_snap * (1 - best_mape), price_snap * (1 + best_mape)
                        st.caption(f"Typical model error ±{best_mape*100:.1f}% → likely range "
                                   f"{short_inr(max(low, 0))} – {short_inr(high)}")
                        st.markdown('<div class="panel-sub">Why this estimate</div>', unsafe_allow_html=True)
                        for b in why:
                            st.markdown(f"- {b}")
                        st.markdown("".join(f'<span class="pill">{t}</span>' for t in traits),
                                    unsafe_allow_html=True)
                        if s_area > AREA_CAP:
                            st.caption(f"{s_area:,.0f} sq ft is beyond the largest home the model has seen "
                                       f"({AREA_CAP:,} sq ft), so the price is scaled up from that. "
                                       f"Treat it as a rough guide.")
                        elif s_area < AREA_FLOOR:
                            st.caption(f"{s_area:,.0f} sq ft is smaller than the homes the model has seen "
                                       f"(from {AREA_FLOOR} sq ft), so the price is scaled down. Rough guide only.")
                with rc2:
                    with st.container(key="card_house"):
                        st.markdown('<div class="card-title">Your home</div>', unsafe_allow_html=True)
                        st.markdown(render_house_svg(tier_lvl, s_gar, s_gdn, s_lift), unsafe_allow_html=True)
            else:
                st.info("Set your property details on the left and press **Reveal my estimate**.")

    # ======================= COMPARE TAB =======================
    with tab_cmp:
        def spec_block(tag, title, defaults):
            ss = st.session_state
            ak, bk, tk, mk = f"{tag}_area", f"{tag}_bed", f"{tag}_bath", f"{tag}_manual"
            ss.setdefault(ak, defaults["area"])
            ss.setdefault(mk, False)
            if bk not in ss:
                _apply_suggested(ak, bk, tk)
            with st.container(key=f"card_spec_{tag}"):
                st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
                loc = st.selectbox("Locality", LOC_NAMES, index=LOC_NAMES.index(defaults["loc"]), key=f"{tag}_loc")
                ar = st.number_input("Area (sq ft)", 1, 50000, step=50, key=ak,
                                     on_change=_on_area_change, args=(ak, bk, tk, mk))
                if ar < MIN_HOME_SQFT:
                    st.error(f"Minimum is {MIN_HOME_SQFT} sq ft — nothing liveable fits in {ar} sq ft.")
                mb, mt = max_rooms(ar)
                _clamp_rooms(ar, bk, tk)
                st.caption(f"Max possible for this size: {mb} bed / {mt} bath"
                           + (" · rooms kept as you set them" if ss[mk] else ""))
                x1, x2 = st.columns(2)
                bd = x1.number_input("Bedrooms", 1, mb, key=bk, on_change=_mark_manual, args=(mk,))
                bt = x2.number_input("Bathrooms", 1, mt, key=tk, on_change=_mark_manual, args=(mk,))
                if ss[mk]:
                    st.button("↺ Use suggested rooms", key=f"{tag}_rooms_reset", on_click=_reset_rooms,
                              args=(ak, bk, tk, mk), width="stretch")
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
        # Price follows the estimate you revealed; down payment is a % of that price (default 20%).
        est_price = int(round(price_snap, -3)) if snap is not None else None
        if st.session_state.get("_emi_est") != est_price:
            st.session_state["_emi_est"] = est_price
            st.session_state["emi_price"] = est_price
        st.session_state.setdefault("emi_price", est_price)
        st.session_state.setdefault("emi_down_pct", 20)

        e1, e2 = st.columns([1, 2], gap="large")
        with e1:
            with st.container(key="card_emi_in"):
                st.markdown('<div class="card-title">Loan details</div>', unsafe_allow_html=True)
                price_in = st.number_input("Property price (₹)", min_value=100000, max_value=1000000000,
                                           step=100000, key="emi_price",
                                           placeholder="Reveal an estimate or type a price")
                if est_price is not None and price_in == est_price:
                    st.caption(f"Taken from your estimate for {snap[4]}")
                down_pct = st.slider("Down payment (% of price)", 0, 100, key="emi_down_pct")
                if price_in is not None:
                    st.caption(f"= {short_inr(price_in * down_pct / 100)} down, "
                               f"{short_inr(price_in * (100 - down_pct) / 100)} loan")
                rate = st.slider("Interest rate (% per year)", 5.0, 15.0, 8.5, 0.05, key="emi_rate")
                years = st.slider("Tenure (years)", 1, 30, 20, key="emi_years")
                income = st.number_input("Monthly income (₹, optional)", min_value=0, value=0,
                                         step=5000, key="emi_income")

        with e2:
            if price_in is None:
                st.info("Go to the **Map** tab, enter your property details and press **Reveal my estimate** — "
                        "the price will appear here automatically. You can also type a price yourself.")
            else:
                down = float(price_in) * down_pct / 100
                loan = float(price_in) - down
                n = years * 12
                r = rate / 12 / 100
                if loan <= 0:
                    emi_val = total_pay = total_int = 0.0
                else:
                    emi_val = loan / n if r == 0 else loan * r * (1 + r) ** n / ((1 + r) ** n - 1)
                    total_pay = emi_val * n
                    total_int = total_pay - loan

                k1, k2, k3, k4 = st.columns(4)
                for col, label, val in (
                    (k1, "Monthly EMI", format_inr(emi_val)),
                    (k2, "Loan amount", short_inr(loan)),
                    (k3, "Total interest", short_inr(total_int)),
                    (k4, "Total payable", short_inr(total_pay))):
                    col.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div>'
                                 f'<div class="kpi-value" style="font-size:1.05rem">{val}</div></div>',
                                 unsafe_allow_html=True)

                if loan <= 0:
                    st.markdown(f'<div class="callout">A 100% down payment covers the full price of '
                                f'{short_inr(price_in)} — no loan needed.</div>', unsafe_allow_html=True)
                else:
                    st.caption(f"Price {short_inr(price_in)} − down payment {short_inr(down)} "
                               f"({down / price_in * 100:.0f}%) = loan {short_inr(loan)}")
                    if income > 0:
                        ratio = emi_val / income * 100
                        msg = (f"EMI is <b>{ratio:.0f}%</b> of your monthly income. "
                               + ("Comfortable (under 40%)." if ratio <= 40 else
                                  "High — lenders usually prefer EMI under 40-50% of income."))
                        st.markdown(f'<div class="callout">{msg}</div>', unsafe_allow_html=True)

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
