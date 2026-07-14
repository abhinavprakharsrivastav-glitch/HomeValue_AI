"""
app.py — HomeValue AI
A glassmorphic, Indian-localized house price estimator with:
  • Animated particle / constellation background
  • Puzzle-style price reveal
  • Rule-based "why this price" explanations
  • Parametric house illustration based on the predicted price tier
  • Compare mode (two properties side-by-side, gold-glow winner)
  • Insights tab with live Plotly charts + dataset preview
  • Dark / light theme toggle
"""

import os
import json
import datetime

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from utils import (
    format_inr, short_inr, usd_to_inr, indian_grouping,
    LOCATION_DISPLAY, display_locations, to_raw_location, LOCATION_ICON,
    get_quality_traits, explain_price, price_tier,
    render_house_svg, puzzle_reveal_html, particle_background_component,
    theme_css,
)

st.set_page_config(page_title="HomeValue AI", page_icon="🏠", layout="wide")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "Model", "house_price_model.pkl")
META_PATH = os.path.join(SCRIPT_DIR, "..", "Model", "model_metadata.json")
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "Dataset", "house_data_cleaned.csv")


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta


@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_data
def dataset_stats(df):
    return {
        "area_median": float(df["area_sqft"].median()),
        "area_mean": float(df["area_sqft"].mean()),
        "distance_median": float(df["distance_to_city_km"].median()),
        "age_median": float(df["age_years"].median()),
        "price_quantiles": {q: float(df["price"].quantile(q)) for q in [0.25, 0.5, 0.75, 0.9]},
    }


model, meta = load_model()
df = load_dataset()
stats = dataset_stats(df)
locations = meta["locations"]
best_model_name = meta["best_model"]
best_r2 = meta["metrics"][best_model_name]["R2"]

# ---------------------------------------------------------------------------
# SESSION STATE DEFAULTS
# ---------------------------------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True
if "reveal_counter" not in st.session_state:
    st.session_state["reveal_counter"] = 0
if "reveal_counter_cmp" not in st.session_state:
    st.session_state["reveal_counter_cmp"] = 0

# ---------------------------------------------------------------------------
# BACKGROUND + THEME
# ---------------------------------------------------------------------------
components.html(particle_background_component(st.session_state["dark_mode"]), height=0)
st.markdown(theme_css(st.session_state["dark_mode"]), unsafe_allow_html=True)


def build_input_df(area_sqft, bedrooms, bathrooms, age_years, distance_to_city_km,
                    garage, garden, pool, near_school, location):
    total_rooms = bedrooms + bathrooms
    amenity_score = int(garage) + int(garden) + int(pool) + int(near_school)
    is_new = int(age_years <= 5)
    row = {
        "area_sqft": area_sqft, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "age_years": age_years, "distance_to_city_km": distance_to_city_km,
        "total_rooms": total_rooms, "amenity_score": amenity_score, "is_new": is_new,
        "garage": int(garage), "garden": int(garden), "pool": int(pool),
        "near_school": int(near_school), "location": location,
    }
    return pd.DataFrame([row]), row


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
today = datetime.date.today().strftime("%d %b %Y")
h_left, h_right = st.columns([5, 1])
with h_left:
    st.markdown(f"""
    <div class="hva-header">
        <div>
            <p class="hva-title">🏠 HomeValue AI</p>
            <p class="hva-subtitle">Smart house price estimation for the Indian market · {best_model_name} model · R² {best_r2:.3f}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
with h_right:
    toggle_label = "☀️ Light" if st.session_state["dark_mode"] else "🌙 Dark"
    if st.button(toggle_label, use_container_width=True, key="theme_toggle_btn"):
        st.session_state["dark_mode"] = not st.session_state["dark_mode"]
        st.rerun()

tab_estimate, tab_compare, tab_insights = st.tabs(["🏠 Estimate", "⚖️ Compare", "📊 Insights"])

# ===========================================================================
# TAB 1 — ESTIMATE
# ===========================================================================
with tab_estimate:
    left, right = st.columns([1.25, 1], gap="large")

    with left:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">Property Specifications</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                area_sqft = st.number_input("Area (sq ft)", min_value=200, max_value=10000, value=1800, step=50, key="e_area")
                bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6], index=2, key="e_bed")
                bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4], index=1, key="e_bath")
            with c2:
                age_years = st.slider("Age (years)", 0, 80, 10, key="e_age")
                distance_to_city_km = st.slider("Distance to city (km)", 0.0, 50.0, 5.0, step=0.5, key="e_dist")
                location_display = st.selectbox("Locality", display_locations(locations), key="e_loc")
                location = to_raw_location(location_display, locations)

        st.write("")
        with st.container(border=True):
            st.markdown('<div class="eyebrow">Amenities & Filters</div>', unsafe_allow_html=True)
            a1, a2, a3, a4 = st.columns(4)
            with a1:
                garage = st.checkbox("Garage", value=True, key="e_garage")
            with a2:
                garden = st.checkbox("Garden", value=True, key="e_garden")
            with a3:
                pool = st.checkbox("Pool", value=False, key="e_pool")
            with a4:
                near_school = st.checkbox("Near school", value=True, key="e_school")

            budget_lakh = st.slider(
                "Your budget (₹ Lakh) — optional, for reference",
                min_value=10, max_value=2000, value=350, step=10, key="e_budget"
            )

        st.write("")
        predict_clicked = st.button("🔮 Reveal My Estimate", use_container_width=True, key="e_predict_btn")

    with right:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">Estimated Value</div>', unsafe_allow_html=True)

            if predict_clicked:
                st.session_state["has_predicted"] = True
                st.session_state["reveal_counter"] += 1

            if st.session_state.get("has_predicted"):
                # Recompute on every rerun (not just on click) so the price
                # stays in sync as the user tweaks area/bedrooms/etc. after
                # the first reveal — sliders/number_input changes rerun the
                # script but do NOT set predict_clicked=True, so gating the
                # prediction behind that flag freezes the price until the
                # button is pressed again.
                input_df, row = build_input_df(
                    area_sqft, bedrooms, bathrooms, age_years, distance_to_city_km,
                    garage, garden, pool, near_school, location
                )
                prediction_usd = float(model.predict(input_df)[0])
                st.session_state["last_input"] = input_df
                st.session_state["last_row"] = row
                st.session_state["last_pred"] = prediction_usd

            if "last_pred" in st.session_state:
                pred = st.session_state["last_pred"]
                row = st.session_state["last_row"]
                tier_name, tier_level = price_tier(pred, stats)

                reveal_key = f"est{st.session_state['reveal_counter']}"
                st.markdown(
                    puzzle_reveal_html(
                        format_inr(pred),
                        f"{tier_name} · based on {best_model_name} regression · ≈ {short_inr(pred)}",
                        reveal_key,
                    ),
                    unsafe_allow_html=True,
                )

                bullets = explain_price(row, stats)
                st.markdown("**Why this estimate:**")
                for b in bullets:
                    st.markdown(f"- {b}")

                over_budget = usd_to_inr(pred) > st.session_state["e_budget"] * 1_00_000
                if over_budget:
                    st.caption(f"⚠️ This is above your ₹{st.session_state['e_budget']} L budget.")
                else:
                    st.caption(f"✅ This fits within your ₹{st.session_state['e_budget']} L budget.")
            else:
                st.markdown("""
                <div style="text-align:center; padding: 30px 10px; opacity:0.75; font-family:'JetBrains Mono',monospace; font-size:0.85rem;">
                Fill in the specifications and press<br>"Reveal My Estimate" to see your prediction.
                </div>
                """, unsafe_allow_html=True)

        if "last_pred" in st.session_state:
            st.write("")
            with st.container(border=True):
                st.markdown('<div class="eyebrow">Your House, Visualised</div>', unsafe_allow_html=True)
                row = st.session_state["last_row"]
                tier_name, tier_level = price_tier(st.session_state["last_pred"], stats)
                svg = render_house_svg(
                    tier_level, garage=row["garage"], garden=row["garden"],
                    pool=row["pool"], uptown_gate=(row["location"] == "Uptown"),
                    dark_mode=st.session_state["dark_mode"],
                )
                st.markdown(f'<div class="house-illustration">{svg}</div>', unsafe_allow_html=True)

                traits = get_quality_traits(
                    row["location"], row["garage"], row["garden"], row["pool"],
                    row["near_school"], row["age_years"], row["distance_to_city_km"]
                )
                pills = "".join(f'<span class="trait-pill">{LOCATION_ICON.get(row["location"], "🏠")} {t}</span>' for t in traits)
                st.markdown(pills, unsafe_allow_html=True)

    if "last_input" in st.session_state:
        st.write("")
        with st.expander("View raw input sent to the model"):
            st.dataframe(st.session_state["last_input"].T.rename(columns={0: "Value"}), use_container_width=True)

# ===========================================================================
# TAB 2 — COMPARE
# ===========================================================================
with tab_compare:
    st.markdown("Set up two properties and see which one comes out on top.")
    colA, colB = st.columns(2, gap="large")

    def property_form(prefix, default_area, default_loc_idx):
        with st.container(border=True):
            st.markdown(f'<div class="eyebrow">Property {prefix}</div>', unsafe_allow_html=True)
            area = st.number_input("Area (sq ft)", 200, 10000, default_area, step=50, key=f"{prefix}_area")
            c1, c2 = st.columns(2)
            with c1:
                bed = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6], index=2, key=f"{prefix}_bed")
                age = st.slider("Age (yrs)", 0, 80, 10, key=f"{prefix}_age")
            with c2:
                bath = st.selectbox("Bathrooms", [1, 2, 3, 4], index=1, key=f"{prefix}_bath")
                dist = st.slider("Distance (km)", 0.0, 50.0, 5.0, step=0.5, key=f"{prefix}_dist")
            loc_display = st.selectbox("Locality", display_locations(locations), index=default_loc_idx, key=f"{prefix}_loc")
            loc = to_raw_location(loc_display, locations)
            a1, a2, a3, a4 = st.columns(4)
            with a1:
                garage = st.checkbox("Garage", value=True, key=f"{prefix}_garage")
            with a2:
                garden = st.checkbox("Garden", value=True, key=f"{prefix}_garden")
            with a3:
                pool = st.checkbox("Pool", value=False, key=f"{prefix}_pool")
            with a4:
                school = st.checkbox("School", value=True, key=f"{prefix}_school")
        return area, bed, bath, age, dist, loc, garage, garden, pool, school

    with colA:
        specs_a = property_form("A", 1800, 1)
    with colB:
        specs_b = property_form("B", 2200, 4)

    compare_clicked = st.button("⚖️ Compare Properties", use_container_width=True, key="cmp_btn")

    if compare_clicked:
        st.session_state["has_compared"] = True
        st.session_state["reveal_counter_cmp"] += 1

    if st.session_state.get("has_compared"):
        # Recompute on every rerun so edits to either property update the
        # comparison immediately, instead of only on the button click.
        df_a, row_a = build_input_df(*specs_a)
        df_b, row_b = build_input_df(*specs_b)
        pred_a = float(model.predict(df_a)[0])
        pred_b = float(model.predict(df_b)[0])
        st.session_state["cmp_result"] = {
            "row_a": row_a, "row_b": row_b, "pred_a": pred_a, "pred_b": pred_b
        }

    if "cmp_result" in st.session_state:
        res = st.session_state["cmp_result"]
        row_a, row_b = res["row_a"], res["row_b"]
        pred_a, pred_b = res["pred_a"], res["pred_b"]
        a_wins = pred_a >= pred_b

        rc = st.session_state["reveal_counter_cmp"]

        # Streamlit's st.container(key=...) tags the container's wrapper div
        # with a `.st-key-<key>` class we can target with real CSS — a plain
        # HTML <div> would NOT actually wrap it in the DOM (Streamlit renders
        # markdown and containers as siblings, not parent/child).
        st.markdown(f"""
        <style>
        .st-key-cmp_card_a {{ {"border: 2px solid var(--hva-accent2) !important; border-radius: 16px !important; animation: goldPulse 2.2s ease-in-out infinite !important;" if a_wins else ""} }}
        .st-key-cmp_card_b {{ {"border: 2px solid var(--hva-accent2) !important; border-radius: 16px !important; animation: goldPulse 2.2s ease-in-out infinite !important;" if not a_wins else ""} }}
        </style>
        """, unsafe_allow_html=True)

        resA, resB = st.columns(2, gap="large")
        with resA:
            with st.container(border=True, key="cmp_card_a"):
                st.markdown('<div class="eyebrow">Property A</div>', unsafe_allow_html=True)
                st.markdown(
                    puzzle_reveal_html(format_inr(pred_a), f"≈ {short_inr(pred_a)}", f"cmpA{rc}"),
                    unsafe_allow_html=True,
                )
                if a_wins:
                    st.markdown("🏆 **Higher estimated value**")

        with resB:
            with st.container(border=True, key="cmp_card_b"):
                st.markdown('<div class="eyebrow">Property B</div>', unsafe_allow_html=True)
                st.markdown(
                    puzzle_reveal_html(format_inr(pred_b), f"≈ {short_inr(pred_b)}", f"cmpB{rc}"),
                    unsafe_allow_html=True,
                )
                if not a_wins:
                    st.markdown("🏆 **Higher estimated value**")

        # --- difference callout: find the largest contributing factor ---
        diff_usd = abs(pred_a - pred_b)
        higher, lower = (row_a, row_b) if a_wins else (row_b, row_a)
        higher_label = "A" if a_wins else "B"

        reasons = []
        area_diff = higher["area_sqft"] - lower["area_sqft"]
        if abs(area_diff) > 50:
            reasons.append((abs(area_diff) * 60, f"+{abs(area_diff):.0f} sq ft more area"))
        if higher["location"] != lower["location"]:
            reasons.append((40000, f"being in {LOCATION_DISPLAY.get(higher['location'], higher['location'])} vs {LOCATION_DISPLAY.get(lower['location'], lower['location'])}"))
        age_diff = lower["age_years"] - higher["age_years"]
        if abs(age_diff) > 3:
            reasons.append((abs(age_diff) * 800, f"being {abs(age_diff)} years {'newer' if age_diff > 0 else 'older'}"))
        amenity_diff = (higher["garage"] + higher["garden"] + higher["pool"] + higher["near_school"]) - \
                       (lower["garage"] + lower["garden"] + lower["pool"] + lower["near_school"])
        if amenity_diff != 0:
            reasons.append((abs(amenity_diff) * 15000, f"{abs(amenity_diff)} more amenities"))

        reasons.sort(key=lambda r: -r[0])
        main_reason = reasons[0][1] if reasons else "small overall differences in specifications"

        st.markdown(
            f'<div class="diff-callout">Property {higher_label} is <b>{format_inr(diff_usd)}</b> '
            f'({short_inr(diff_usd)}) more, mainly due to {main_reason}.</div>',
            unsafe_allow_html=True,
        )

# ===========================================================================
# TAB 3 — INSIGHTS
# ===========================================================================
with tab_insights:
    st.markdown('<div class="eyebrow">Key Metrics</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    kpi_defs = [
        ("Median Price", short_inr(stats["price_quantiles"][0.5]), df["price"]),
        ("Median Area", f"{stats['area_median']:,.0f} sq ft", df["area_sqft"]),
        ("Avg. Distance", f"{stats['distance_median']:.1f} km", df["distance_to_city_km"]),
        ("Listings", f"{len(df):,}", df["price"]),
    ]
    for col, (label, value, series) in zip([k1, k2, k3, k4], kpi_defs):
        with col:
            spark = go.Figure(go.Scatter(
                y=series.sample(min(60, len(series)), random_state=1).reset_index(drop=True),
                mode="lines", line=dict(color="#E3B23C", width=2), fill="tozeroy",
                fillcolor="rgba(227,178,60,0.15)",
            ))
            spark.update_layout(
                height=60, margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>', unsafe_allow_html=True)
            st.plotly_chart(spark, use_container_width=True, config={"displayModeBar": False}, key=f"spark_{label}")
            st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    plot_template = "plotly_dark" if st.session_state["dark_mode"] else "plotly_white"

    ic1, ic2 = st.columns(2, gap="large")
    with ic1:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">Price Distribution (₹)</div>', unsafe_allow_html=True)
            df_inr = df.copy()
            df_inr["price_inr"] = df_inr["price"] * usd_to_inr(1)
            fig_hist = px.histogram(df_inr, x="price_inr", nbins=40, template=plot_template,
                                     color_discrete_sequence=["#8ECAE6"])
            fig_hist.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    xaxis_title="Price (₹)", yaxis_title="Count")
            st.plotly_chart(fig_hist, use_container_width=True, key="hist_chart")

    with ic2:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">Correlation Heatmap</div>', unsafe_allow_html=True)
            numeric_cols = ["area_sqft", "bedrooms", "bathrooms", "age_years", "distance_to_city_km", "price"]
            corr = df[numeric_cols].corr().round(2)
            fig_heat = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                                  template=plot_template, aspect="auto")
            fig_heat.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_heat, use_container_width=True, key="heat_chart")

    with st.container(border=True):
        st.markdown('<div class="eyebrow">Price by Locality</div>', unsafe_allow_html=True)
        loc_avg = df.groupby("location")["price"].mean().reset_index()
        loc_avg["price_inr"] = loc_avg["price"] * usd_to_inr(1)
        loc_avg["locality"] = loc_avg["location"].map(lambda l: LOCATION_DISPLAY.get(l, l))
        fig_bar = px.bar(loc_avg.sort_values("price_inr"), x="price_inr", y="locality", orientation="h",
                          template=plot_template, color="price_inr", color_continuous_scale="Sunset")
        fig_bar.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               xaxis_title="Average Price (₹)", yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True, key="bar_chart")

    with st.container(border=True):
        st.markdown('<div class="eyebrow">Dataset Preview</div>', unsafe_allow_html=True)
        preview = df.head(25).copy()
        preview["price"] = (preview["price"] * usd_to_inr(1)).map(lambda v: f"₹{indian_grouping(round(v))}")
        preview["location"] = preview["location"].map(lambda l: LOCATION_DISPLAY.get(l, l))
        st.dataframe(preview, use_container_width=True, height=320)

    with st.expander("Model performance comparison (held-out test set)"):
        metrics_df = pd.DataFrame(meta["metrics"]).T.round(2)
        st.dataframe(metrics_df, use_container_width=True)
        st.caption("MAE / RMSE shown in USD (as produced during training) — lower is better. R² closer to 1 is better.")

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.write("")
st.markdown(f"""
<div style="text-align:center; opacity:0.55; font-family:'JetBrains Mono',monospace; font-size:0.72rem; padding: 10px 0 24px 0;">
HomeValue AI · {best_model_name} regression · R² {best_r2:.3f} · Prices shown in ₹ at an illustrative rate of 1 USD ≈ ₹{usd_to_inr(1):.0f} · {today}
</div>
""", unsafe_allow_html=True)
