"""
utils.py — HomeValue AI (Prayagraj edition)
Helpers: rupee formatting, distance, budget status, rule-based explanations,
parametric SVG house, puzzle price reveal and the neo-brutalist theme CSS.
All prices are in Indian Rupees (the dataset is already in Rs).
"""

import math
import random

CITY_CENTRE = (25.4480, 81.8432)  # Civil Lines, used as "city centre"

# Palette (matches the dashboard reference)
INK = "#111111"
YELLOW = "#F4C430"
PINK = "#F28AB2"
TEAL = "#5FD3C4"
BLUE = "#3D7BD9"
GREEN = "#2FBF5E"
RED = "#E8553A"
CREAM = "#FFF6DC"

STATUS_META = {
    "within":  {"label": "Within budget",        "color": GREEN},
    "stretch": {"label": "Stretch (up to +25%)", "color": YELLOW},
    "over":    {"label": "Over budget",          "color": RED},
}


# ---------------------------------------------------------------------------
# CURRENCY
# ---------------------------------------------------------------------------
def indian_grouping(n) -> str:
    """12,34,567 style digit grouping."""
    s = str(int(n))
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


def format_inr(amount: float) -> str:
    return f"₹{indian_grouping(round(amount))}"


def short_inr(amount: float) -> str:
    if amount >= 1_00_00_000:
        return f"₹{amount / 1_00_00_000:.2f} Cr"
    if amount >= 1_00_000:
        return f"₹{amount / 1_00_000:.1f} L"
    return f"₹{indian_grouping(round(amount))}"


# ---------------------------------------------------------------------------
# GEO
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# BUDGET STATUS / PRICE TIER
# ---------------------------------------------------------------------------
def budget_status(pred: float, budget: float) -> str:
    if pred <= budget:
        return "within"
    if pred <= budget * 1.25:
        return "stretch"
    return "over"


def price_tier(prediction, stats):
    q = stats["price_quantiles"]
    if prediction < q[0.25]:
        return "Budget Home", 1
    if prediction < q[0.50]:
        return "Mid-Range Home", 2
    if prediction < q[0.75]:
        return "Comfort Home", 3
    if prediction < q[0.90]:
        return "Premium Home", 4
    return "Luxury Home", 5


# ---------------------------------------------------------------------------
# TRAITS + EXPLANATION
# ---------------------------------------------------------------------------
def get_quality_traits(note, garage, garden, lift, near_school, age_years, distance_km):
    traits = [note]
    if garage:
        traits.append("Dedicated parking")
    if garden:
        traits.append("Garden / lawn space")
    if lift:
        traits.append("Lift in the building")
    if near_school:
        traits.append("School within walking distance")
    if age_years <= 5:
        traits.append("Newly built, low maintenance")
    elif age_years >= 35:
        traits.append("Older property, may need renovation")
    if distance_km <= 2:
        traits.append("Very close to Civil Lines")
    elif distance_km >= 8:
        traits.append("Far from the city centre, quieter")
    return traits[:6]


def explain_price(inputs: dict, stats: dict, loc_rate: float):
    """Transparent rule-based 'why this price' bullets (not SHAP)."""
    bullets = []
    area, med = inputs["area_sqft"], stats["area_median"]
    if area >= med * 1.25:
        bullets.append(f"↑ Bigger than a typical home ({area:,.0f} vs ~{med:,.0f} sq ft median)")
    elif area <= med * 0.75:
        bullets.append(f"↓ Smaller than a typical home ({area:,.0f} vs ~{med:,.0f} sq ft median)")

    rate_med = stats["rate_median"]
    if loc_rate >= rate_med * 1.25:
        bullets.append(f"↑ {inputs['location']} is a premium locality (about ₹{loc_rate:,.0f}/sq ft vs ₹{rate_med:,.0f} city median)")
    elif loc_rate <= rate_med * 0.8:
        bullets.append(f"↓ {inputs['location']} is an affordable locality (about ₹{loc_rate:,.0f}/sq ft vs ₹{rate_med:,.0f} city median)")
    else:
        bullets.append(f"→ {inputs['location']} is priced close to the city median (about ₹{loc_rate:,.0f}/sq ft)")

    age = inputs["age_years"]
    if age <= 5:
        bullets.append("↑ New construction adds a premium")
    elif age >= 30:
        bullets.append("↓ Older building brings the price down")

    d = inputs["distance_to_city_km"]
    if d <= 2:
        bullets.append("↑ Within 2 km of Civil Lines")
    elif d >= 8:
        bullets.append("↓ More than 8 km from Civil Lines")

    n_amen = inputs["garage"] + inputs["garden"] + inputs["lift"] + inputs["near_school"]
    if n_amen >= 3:
        bullets.append(f"↑ {n_amen}/4 amenities present")
    elif n_amen <= 1:
        bullets.append(f"↓ Only {n_amen}/4 amenities present")

    return bullets[:6]


# ---------------------------------------------------------------------------
# PARAMETRIC HOUSE SVG
# ---------------------------------------------------------------------------
def render_house_svg(tier_level, garage, garden, lift):
    wall_colors = ["#D9B78F", "#E0BE86", "#E8C67C", "#F0CE72", "#F4D667"]
    roof_colors = ["#8A5A3C", "#9A5A36", "#A9522F", "#B8482B", "#C63F27"]
    wall, roof = wall_colors[tier_level - 1], roof_colors[tier_level - 1]
    two_story = tier_level >= 4
    win_rows = 2 if two_story else 1
    win_cols = 3 if tier_level >= 3 else 2

    W, H = 340, 220
    house_w = 150 + tier_level * 12
    house_h = 130 if two_story else 90
    hx = (W - house_w) / 2
    hy = 150 - house_h

    windows = ""
    for r in range(win_rows):
        for c in range(win_cols):
            wx = hx + 14 + c * ((house_w - 28 - 20) / max(win_cols - 1, 1))
            wy = hy + 14 + r * (house_h / win_rows)
            windows += (f'<rect x="{wx:.1f}" y="{wy:.1f}" width="20" height="20" fill="#8ED1F2" '
                        f'stroke="{INK}" stroke-width="2.5"/>'
                        f'<line x1="{wx+10:.1f}" y1="{wy:.1f}" x2="{wx+10:.1f}" y2="{wy+20:.1f}" stroke="{INK}" stroke-width="1.5"/>')

    dx, dy_ = hx + house_w / 2 - 13, hy + house_h - 40
    door = (f'<rect x="{dx:.1f}" y="{dy_:.1f}" width="26" height="40" fill="#7A4A2A" stroke="{INK}" stroke-width="2.5"/>'
            f'<circle cx="{dx+21:.1f}" cy="{dy_+20:.1f}" r="1.8" fill="{YELLOW}" stroke="{INK}" stroke-width="0.8"/>')
    roof_svg = (f'<polygon points="{hx-10},{hy} {hx+house_w/2},{hy-45} {hx+house_w+10},{hy}" '
                f'fill="{roof}" stroke="{INK}" stroke-width="3"/>')

    garage_svg = ""
    if garage:
        gw, gh = 55, 55
        gx, gy = hx - gw + 6, 150 - gh
        garage_svg = (f'<rect x="{gx:.1f}" y="{gy:.1f}" width="{gw}" height="{gh}" fill="{wall}" stroke="{INK}" stroke-width="2.5"/>'
                      f'<rect x="{gx+6:.1f}" y="{gy+gh-30:.1f}" width="{gw-12}" height="26" fill="#8A8A8A" stroke="{INK}" stroke-width="2"/>'
                      f'<polygon points="{gx-4},{gy} {gx+gw/2:.1f},{gy-20} {gx+gw+4},{gy}" fill="{roof}" stroke="{INK}" stroke-width="2.5"/>')

    lift_svg = ""
    if lift:
        lift_svg = (f'<rect x="{hx+house_w-26:.1f}" y="{hy-62:.1f}" width="22" height="30" fill="#BBBBBB" stroke="{INK}" stroke-width="2.5"/>'
                    f'<text x="{hx+house_w-15:.1f}" y="{hy-42:.1f}" font-size="8" font-weight="700" text-anchor="middle" fill="{INK}">LIFT</text>')

    garden_svg = ""
    if garden:
        for tx in (hx - 34 if not garage else hx - 70, hx + house_w + 24):
            garden_svg += (f'<rect x="{tx-3:.1f}" y="165" width="6" height="20" fill="#7A4A2A" stroke="{INK}" stroke-width="1.5"/>'
                           f'<circle cx="{tx:.1f}" cy="158" r="16" fill="#3FAE66" stroke="{INK}" stroke-width="2.5"/>')

    gate_svg = ""
    if tier_level >= 4:
        gate_svg = (f'<rect x="20" y="195" width="{W-40}" height="6" fill="#777" stroke="{INK}" stroke-width="1.5"/>'
                    f'<rect x="20" y="180" width="5" height="21" fill="#777" stroke="{INK}" stroke-width="1.5"/>'
                    f'<rect x="{W-25}" y="180" width="5" height="21" fill="#777" stroke="{INK}" stroke-width="1.5"/>')

    return f"""
<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;max-width:360px;border:3px solid {INK};background:#BDE7F7;">
  <circle cx="290" cy="35" r="16" fill="{YELLOW}" stroke="{INK}" stroke-width="2.5"/>
  <rect x="0" y="150" width="{W}" height="{H-150}" fill="#9BD98F" stroke="{INK}" stroke-width="2.5"/>
  {gate_svg}{garden_svg}{garage_svg}{roof_svg}{lift_svg}
  <rect x="{hx:.1f}" y="{hy:.1f}" width="{house_w}" height="{house_h}" fill="{wall}" stroke="{INK}" stroke-width="3"/>
  {windows}{door}
</svg>
"""


# ---------------------------------------------------------------------------
# PUZZLE PRICE REVEAL
# ---------------------------------------------------------------------------
def puzzle_reveal_html(value_text: str, caption_text: str, key: str):
    tiles, i = "", 0
    n_cols, n_rows = 6, 3
    for r in range(n_rows):
        for c in range(n_cols):
            rnd = random.Random(hash(key) + i)
            delay = i * 0.045 + rnd.uniform(0, 0.05)
            ang = random.Random(hash(key) + i + 100).uniform(-140, 140)
            dx = math.cos(math.radians(ang)) * 140
            dy = math.sin(math.radians(ang)) * 140
            tiles += (f'<div class="pt pt-{key}" style="left:{c/n_cols*100:.3f}%; top:{r/n_rows*100:.3f}%; '
                      f'width:{100/n_cols:.3f}%; height:{100/n_rows:.3f}%; animation-delay:{delay:.3f}s; '
                      f'--fx:{dx:.1f}px; --fy:{dy:.1f}px; --fr:{ang:.1f}deg;"></div>')
            i += 1
    return f"""
<div class="pz-wrap">
  <div class="pz-value">{value_text}</div>
  <div class="pz-caption">{caption_text}</div>
  <div class="pz-tiles">{tiles}</div>
</div>
<style>
.pz-wrap {{ position:relative; text-align:center; padding:16px 8px; overflow:hidden; background:#fff;
           border:3px solid {INK}; box-shadow:4px 4px 0 {INK}; margin-bottom:10px; }}
.pz-value {{ font-family:'Space Mono',monospace; font-weight:700; font-size:2.3rem; color:{INK}; position:relative; z-index:1; }}
.pz-caption {{ font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:0.78rem; margin-top:4px; position:relative; z-index:1; }}
.pz-tiles {{ position:absolute; inset:0; z-index:2; pointer-events:none; }}
.pt {{ position:absolute; background:{INK}; border:1px solid {YELLOW}; }}
.pt-{key} {{ animation: fly-{key} 0.7s cubic-bezier(.4,0,.2,1) forwards; }}
@keyframes fly-{key} {{
  0%   {{ opacity:1; transform:translate(0,0) rotate(0) scale(1); }}
  100% {{ opacity:0; transform:translate(var(--fx),var(--fy)) rotate(var(--fr)) scale(0.3); }}
}}
</style>
"""


# ---------------------------------------------------------------------------
# NEO-BRUTALIST THEME
# ---------------------------------------------------------------------------
def theme_css():
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"], [data-testid="stApp"] {{ font-family:'Space Grotesk',sans-serif; color:{INK}; }}
[data-testid="stAppViewContainer"], [data-testid="stApp"] {{
    background-color:{CREAM} !important;
    background-image: radial-gradient({INK}22 1px, transparent 1px); background-size: 18px 18px;
}}
[data-testid="stHeader"] {{ background:transparent !important; }}
[data-testid="stSidebar"] {{ display:none; }}
.block-container {{ padding-top:1.4rem; max-width:1500px; }}
h1,h2,h3,p,span,label,li {{ color:{INK}; }}

/* ---------- header ---------- */
.nn-header {{ background:{YELLOW}; border:3px solid {INK}; box-shadow:6px 6px 0 {INK};
    padding:14px 22px; margin-bottom:22px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; }}
.nn-title {{ font-weight:700; font-size:1.9rem; margin:0; letter-spacing:0.02em; text-transform:uppercase; }}
.nn-sub {{ font-family:'Space Mono',monospace; font-size:0.75rem; margin:2px 0 0 0; }}
.nn-badge {{ background:{INK}; color:{YELLOW} !important; padding:4px 10px; font-family:'Space Mono',monospace; font-size:0.72rem; font-weight:700; }}

/* ---------- panels / cards ---------- */
.st-key-filters_panel {{ background:{YELLOW}; border:3px solid {INK}; box-shadow:6px 6px 0 {INK}; padding:16px 16px 10px 16px; }}
div[class*="st-key-card_"] {{ background:#fff; border:3px solid {INK}; box-shadow:6px 6px 0 {INK}; padding:14px 16px; margin-bottom:6px; }}
.panel-title {{ font-weight:700; font-size:1.05rem; letter-spacing:0.08em; text-transform:uppercase;
    border-bottom:3px solid {INK}; padding-bottom:6px; margin-bottom:10px; }}
.panel-sub {{ font-weight:700; font-size:0.78rem; letter-spacing:0.1em; text-transform:uppercase;
    margin:12px 0 4px 0; border-bottom:2px solid {INK}; padding-bottom:2px; }}
.card-title {{ font-weight:700; font-size:0.85rem; letter-spacing:0.12em; text-transform:uppercase;
    border-bottom:3px solid {INK}; padding-bottom:5px; margin-bottom:10px; }}

/* ---------- map frame + overlay cards ---------- */
.st-key-map_wrap {{ position:relative; background:{TEAL}; border:3px solid {INK}; box-shadow:8px 8px 0 {INK}; padding:10px; }}
.st-key-legend_card {{ position:absolute !important; top:24px; right:24px; width:215px !important; z-index:50;
    background:{PINK}; border:3px solid {INK}; box-shadow:4px 4px 0 {INK}; padding:8px 12px 18px 12px; }}
.st-key-stats_card {{ position:absolute !important; right:24px; bottom:26px; width:250px !important; z-index:50;
    background:{TEAL}; border:3px solid {INK}; box-shadow:4px 4px 0 {INK}; padding:8px 10px; }}
.lg-title, .st-title {{ font-weight:700; font-size:0.8rem; letter-spacing:0.1em; border-bottom:2px solid {INK}; margin-bottom:6px; padding-bottom:2px; }}
.lg-row {{ display:flex; align-items:center; gap:8px; font-size:0.74rem; font-weight:600; margin:3px 0; }}
.lg-dot {{ width:13px; height:13px; border-radius:50%; border:2px solid {INK}; display:inline-block; flex:none; }}
.lg-sq {{ width:12px; height:12px; border:2px solid {INK}; background:{BLUE}; display:inline-block; flex:none; }}
.st-grid {{ display:flex; gap:8px; }}
.st-rows {{ flex:1; }}
.st-row {{ display:flex; justify-content:space-between; align-items:center; border:2px solid {INK}; padding:2px 8px;
    margin:4px 0; font-weight:700; font-size:0.74rem; box-shadow:2px 2px 0 {INK}; }}
.st-row b {{ font-family:'Space Mono',monospace; font-size:0.95rem; }}
.st-donutbox {{ width:78px; background:{YELLOW}; border:2px solid {INK}; box-shadow:2px 2px 0 {INK}; display:flex; align-items:center; justify-content:center; padding:4px; }}
.donut {{ width:58px; height:58px; border-radius:50%; border:2px solid {INK}; position:relative; }}
.donut::after {{ content:''; position:absolute; inset:14px; background:{YELLOW}; border:2px solid {INK}; border-radius:50%; }}

/* ---------- inputs ---------- */
[data-testid="stWidgetLabel"] p {{ font-weight:700 !important; font-size:0.78rem !important; text-transform:uppercase; letter-spacing:0.06em; }}
[data-baseweb="input"], [data-baseweb="select"] > div {{ border:3px solid {INK} !important; border-radius:0 !important; background:#fff !important; }}
[data-baseweb="base-input"] {{ background:#fff !important; }}
[data-testid="stNumberInput"] input, [data-baseweb="select"] div {{ color:{INK} !important; font-weight:600; }}
[data-baseweb="popover"] li, [data-baseweb="popover"] ul {{ background:#fff !important; color:{INK} !important; font-weight:600; }}
[data-testid="stCheckbox"] label p {{ font-weight:700 !important; font-size:0.82rem !important; text-transform:none; letter-spacing:0; }}
[data-baseweb="checkbox"] > span:first-child {{ border:3px solid {INK} !important; border-radius:0 !important; background:#fff !important; }}
[data-baseweb="checkbox"] input:checked + div, [data-baseweb="checkbox"][aria-checked="true"] > span:first-child {{ background:{INK} !important; }}
[data-testid="stSlider"] [role="slider"] {{ background:{INK} !important; border:3px solid {INK} !important; border-radius:0 !important; box-shadow:2px 2px 0 {PINK} !important; }}
[data-testid="stSlider"] [data-testid="stSliderThumbValue"], [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {{ color:{INK} !important; font-family:'Space Mono',monospace; font-weight:700; }}

.stButton > button {{ border:3px solid {INK} !important; border-radius:0 !important; font-weight:700 !important; letter-spacing:0.08em;
    text-transform:uppercase; color:#fff !important; box-shadow:4px 4px 0 {INK}; transition:transform .08s, box-shadow .08s; padding:0.55rem 0.8rem !important; }}
.stButton > button p {{ color:#fff !important; font-weight:700 !important; }}
.stButton > button:hover {{ transform:translate(2px,2px); box-shadow:2px 2px 0 {INK}; }}
.stButton > button:active {{ transform:translate(4px,4px); box-shadow:0 0 0 {INK}; }}
.st-key-btn_reveal button, .st-key-btn_tilt button, .st-key-cmp_btn button {{ background:{BLUE} !important; }}
.st-key-btn_reset button {{ background:{GREEN} !important; }}

/* ---------- tabs / expander / dataframe ---------- */
[role="tablist"], [data-baseweb="tab-list"] {{ gap:8px; border-bottom:3px solid {INK} !important; }}
[role="tab"] {{ background:#fff !important; border:3px solid {INK} !important; border-bottom:none !important; border-radius:0 !important;
    padding:6px 16px !important; height:auto !important; margin-right:6px; }}
[role="tab"][aria-selected="true"] {{ background:{YELLOW} !important; }}
[role="tab"] p {{ font-weight:700 !important; text-transform:uppercase; letter-spacing:0.06em; font-size:0.85rem !important; }}
.react-aria-SelectionIndicator, [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{ display:none !important; }}
[data-testid="stExpander"] {{ border:3px solid {INK} !important; border-radius:0 !important; background:#fff !important; }}
[data-testid="stDataFrame"] {{ border:3px solid {INK}; }}

/* ---------- small bits ---------- */
.pill {{ display:inline-block; padding:3px 9px; margin:3px 5px 3px 0; border:2px solid {INK}; background:{CREAM}; font-size:0.72rem; font-weight:600; box-shadow:2px 2px 0 {INK}; }}
.kpi {{ background:#fff; border:3px solid {INK}; box-shadow:5px 5px 0 {INK}; padding:10px 14px; }}
.kpi-label {{ font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; }}
.kpi-value {{ font-family:'Space Mono',monospace; font-weight:700; font-size:1.45rem; }}
.callout {{ border:3px solid {INK}; background:{PINK}; padding:10px 14px; font-weight:600; font-size:0.9rem; box-shadow:4px 4px 0 {INK}; margin-top:12px; }}
.place-info {{ border:3px solid {INK}; background:{CREAM}; padding:8px 12px; font-size:0.8rem; font-weight:600; margin-top:8px; box-shadow:3px 3px 0 {INK}; }}
.win-badge {{ display:inline-block; background:{YELLOW}; border:3px solid {INK}; padding:2px 10px; font-weight:700; box-shadow:3px 3px 0 {INK}; }}
</style>
"""
