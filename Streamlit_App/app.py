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
