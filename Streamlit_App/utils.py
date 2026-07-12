"""
utils.py — HomeValue AI
Helper functions: currency formatting, Indian locality mapping,
rule-based "why this price" explanations, parametric SVG house
illustration, animated particle/constellation background, and
theme (dark/light) CSS generation.
"""

import random

# ---------------------------------------------------------------------------
# CURRENCY
# ---------------------------------------------------------------------------
# The bundled dataset is a synthetic training set with prices in USD.
# We apply a fixed, clearly-labelled conversion rate to present every
# figure in Indian Rupees. Change USD_TO_INR if you want a different rate.
USD_TO_INR = 83.0


def usd_to_inr(usd_amount: float) -> float:
    return usd_amount * USD_TO_INR


def indian_grouping(n: int) -> str:
    """Format an integer with Indian digit grouping (e.g. 12,34,567)."""
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


def format_inr(usd_amount: float, short: bool = False) -> str:
    """Full Indian-formatted rupee string, e.g. ₹2,73,10,483."""
    inr = usd_to_inr(usd_amount)
    if short:
        return short_inr(usd_amount)
    return f"₹{indian_grouping(round(inr))}"


def short_inr(usd_amount: float) -> str:
    """Compact Lakh / Crore style, e.g. ₹1.24 Cr or ₹42.3 L."""
    inr = usd_to_inr(usd_amount)
    if inr >= 1_00_00_000:
        return f"₹{inr / 1_00_00_000:.2f} Cr"
    if inr >= 1_00_000:
        return f"₹{inr / 1_00_000:.2f} L"
    return f"₹{indian_grouping(round(inr))}"


# ---------------------------------------------------------------------------
# INDIAN-STYLE LOCALITY NAMES
# ---------------------------------------------------------------------------
# The model was trained on these exact category strings, so we never change
# the underlying values sent to the model — only how they are displayed.
LOCATION_DISPLAY = {
    "City Center": "City Center (CBD)",
    "Downtown": "Downtown (Old City / Commercial Hub)",
    "Uptown": "Uptown (Posh Colony)",
    "Suburb": "Suburb (Residential Township)",
    "Rural": "Rural / Outskirts",
}
LOCATION_ICON = {
    "City Center": "🏙️",
    "Downtown": "🏬",
    "Uptown": "🏡",
    "Suburb": "🏘️",
    "Rural": "🌾",
}


def display_locations(locations):
    return [LOCATION_DISPLAY.get(loc, loc) for loc in locations]


def to_raw_location(display_value: str, locations):
    for loc in locations:
        if LOCATION_DISPLAY.get(loc, loc) == display_value:
            return loc
    return display_value


# ---------------------------------------------------------------------------
# LOCALITY "QUALITY" TRAITS
# ---------------------------------------------------------------------------
LOCATION_TRAITS = {
    "City Center": [
        "Near markets & shopping hubs",
        "Excellent metro / bus connectivity",
        "Close to hospitals & offices",
    ],
    "Downtown": [
        "Heart of commercial activity",
        "Walking distance to markets & shops",
        "High rental demand in this pocket",
    ],
    "Uptown": [
        "Premium, low-density neighbourhood",
        "Gated-community feel, upscale surroundings",
        "Sought after by premium buyers",
    ],
    "Suburb": [
        "Family-friendly residential township",
        "More open space & easier parking",
        "Quieter than the city core",
    ],
    "Rural": [
        "Peaceful, low-noise surroundings",
        "Larger plot potential",
        "Ideal for a farmhouse-style home",
    ],
}


def get_quality_traits(location, garage, garden, pool, near_school, age_years, distance_km):
    traits = list(LOCATION_TRAITS.get(location, []))
    if garage:
        traits.append("Dedicated covered parking")
    if garden:
        traits.append("Private garden / lawn space")
    if pool:
        traits.append("Swimming pool — a premium feature")
    if near_school:
        traits.append("Walking distance to a school")
    if age_years <= 5:
        traits.append("Newly built — low maintenance expected")
    elif age_years >= 35:
        traits.append("Older property — may need some renovation")
    if distance_km <= 3:
        traits.append("Very close to the city — high convenience")
    elif distance_km >= 20:
        traits.append("Far from the city — more affordable, more peaceful")
    return traits[:6]


# ---------------------------------------------------------------------------
# RULE-BASED "WHY THIS PRICE" EXPLANATION
# ---------------------------------------------------------------------------
def explain_price(inputs: dict, stats: dict):
    """
    Produce short bullet points explaining why the estimate is relatively
    high or low, by comparing the given inputs against dataset statistics.
    This is a transparent, rule-based explainer (not a formal SHAP
    decomposition) meant to give an intuitive sense of the main drivers.
    """
    bullets = []

    area = inputs["area_sqft"]
    area_med = stats["area_median"]
    if area >= area_med * 1.2:
        bullets.append(f"↑ Larger than a typical home ({area:,.0f} sq ft vs ~{area_med:,.0f} sq ft median) — pushes the price up")
    elif area <= area_med * 0.8:
        bullets.append(f"↓ Smaller than a typical home ({area:,.0f} sq ft vs ~{area_med:,.0f} sq ft median) — brings the price down")

    dist = inputs["distance_to_city_km"]
    if dist <= 3:
        bullets.append("↑ Very close to the city centre — proximity adds a premium")
    elif dist >= 18:
        bullets.append("↓ Located far from the city centre — distance lowers the price")

    age = inputs["age_years"]
    if age <= 5:
        bullets.append("↑ Newly built property — recency adds a premium")
    elif age >= 40:
        bullets.append("↓ Older property — age brings the price down")

    loc = inputs["location"]
    premium_locations = {"City Center", "Uptown", "Downtown"}
    if loc in premium_locations:
        bullets.append(f"↑ {LOCATION_DISPLAY.get(loc, loc)} is a premium locality — location adds significant value")
    else:
        bullets.append(f"↓ {LOCATION_DISPLAY.get(loc, loc)} is a more affordable locality — keeps the price moderate")

    amenity_score = int(inputs["garage"]) + int(inputs["garden"]) + int(inputs["pool"]) + int(inputs["near_school"])
    if inputs["pool"]:
        bullets.append("↑ Private pool is a high-value amenity")
    if amenity_score >= 3:
        bullets.append(f"↑ {amenity_score}/4 amenities present — extra features add to the estimate")
    elif amenity_score <= 1:
        bullets.append(f"↓ Only {amenity_score}/4 amenities present — fewer features keep the estimate lower")

    rooms = inputs["bedrooms"] + inputs["bathrooms"]
    if rooms >= 8:
        bullets.append(f"↑ {inputs['bedrooms']} bedrooms + {inputs['bathrooms']} bathrooms is spacious — more rooms add value")
    elif rooms <= 3:
        bullets.append(f"↓ Compact layout ({inputs['bedrooms']} bed / {inputs['bathrooms']} bath) — fewer rooms keep the price lower")

    if not bullets:
        bullets.append("This home sits close to the dataset average on most factors — a fairly typical estimate")

    return bullets[:6]


# ---------------------------------------------------------------------------
# PRICE TIER (for the house illustration + labelling)
# ---------------------------------------------------------------------------
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
    return "Luxury Villa", 5


# ---------------------------------------------------------------------------
# PARAMETRIC HOUSE SVG (no external images / internet needed)
# ---------------------------------------------------------------------------
def render_house_svg(tier_level, garage, garden, pool, uptown_gate, dark_mode=True):
    """Build a small parametric SVG house that scales in complexity with
    the price tier (1=budget .. 5=luxury) and reflects chosen amenities."""

    sky_top = "#0A1F38" if dark_mode else "#BFE3F5"
    sky_bot = "#16283A" if dark_mode else "#EAF6FF"
    ground = "#1B3A2E" if dark_mode else "#BEE3C6"
    wall_colors = ["#8E7A63", "#A9906F", "#C7A76C", "#D9B672", "#E7C873"]
    roof_colors = ["#5A3A2A", "#6E4530", "#7C4A2A", "#8B3E2A", "#A13B2A"]
    wall = wall_colors[tier_level - 1]
    roof = roof_colors[tier_level - 1]
    gold = "#E3B23C"
    two_story = tier_level >= 4
    win_rows = 2 if two_story else 1
    win_cols = 3 if tier_level >= 3 else 2

    W, H = 340, 220
    house_w = 150 + tier_level * 12
    house_h = 90 if not two_story else 130
    house_x = (W - house_w) / 2
    house_y = 150 - house_h

    windows = ""
    win_w, win_h = 20, 20
    for r in range(win_rows):
        for c in range(win_cols):
            wx = house_x + 14 + c * ((house_w - 28) / max(win_cols - 1, 1)) - win_w / 2
            wx = house_x + 14 + c * ((house_w - 28 - win_w) / max(win_cols - 1, 1))
            wy = house_y + 14 + r * (house_h / max(win_rows, 1))
            windows += (
                f'<rect x="{wx:.1f}" y="{wy:.1f}" width="{win_w}" height="{win_h}" '
                f'rx="2" fill="{gold}" opacity="0.85" stroke="#3a2a1a" stroke-width="1.5"/>'
                f'<line x1="{wx + win_w/2:.1f}" y1="{wy:.1f}" x2="{wx + win_w/2:.1f}" y2="{wy+win_h:.1f}" stroke="#3a2a1a" stroke-width="1"/>'
            )

    door_w, door_h = 26, 40
    door_x = house_x + house_w / 2 - door_w / 2
    door_y = house_y + house_h - door_h
    door = (
        f'<rect x="{door_x:.1f}" y="{door_y:.1f}" width="{door_w}" height="{door_h}" '
        f'rx="3" fill="#4A2E1E" stroke="#2B1B10" stroke-width="1.5"/>'
        f'<circle cx="{door_x + door_w - 5:.1f}" cy="{door_y + door_h/2:.1f}" r="1.6" fill="{gold}"/>'
    )

    roof_pts = f"{house_x-10},{house_y} {house_x+house_w/2},{house_y-45} {house_x+house_w+10},{house_y}"
    roof_svg = f'<polygon points="{roof_pts}" fill="{roof}" stroke="#2B1B10" stroke-width="2"/>'

    garage_svg = ""
    if garage:
        gw, gh = 55, 55
        gx, gy = house_x - gw + 6, 150 - gh
        garage_svg = (
            f'<rect x="{gx:.1f}" y="{gy:.1f}" width="{gw}" height="{gh}" fill="{wall}" stroke="#2B1B10" stroke-width="2"/>'
            f'<rect x="{gx+6:.1f}" y="{gy+gh-30:.1f}" width="{gw-12}" height="26" rx="2" fill="#5b5049" stroke="#2B1B10" stroke-width="1.5"/>'
            f'<polygon points="{gx-4},{gy} {gx+gw/2:.1f},{gy-20} {gx+gw+4},{gy}" fill="{roof}" stroke="#2B1B10" stroke-width="1.5"/>'
        )

    pool_svg = ""
    if pool:
        pool_svg = (
            f'<ellipse cx="{house_x + house_w + 45:.1f}" cy="188" rx="34" ry="14" '
            f'fill="#3AA6C9" opacity="0.85" stroke="#1c6b85" stroke-width="2"/>'
            f'<ellipse cx="{house_x + house_w + 45:.1f}" cy="188" rx="24" ry="8" fill="#8EE3F5" opacity="0.6"/>'
        )

    garden_svg = ""
    if garden:
        tree_x = [house_x - 34, house_x + house_w + 20]
        for tx in tree_x:
            garden_svg += (
                f'<rect x="{tx-3:.1f}" y="165" width="6" height="20" fill="#4A2E1E"/>'
                f'<circle cx="{tx:.1f}" cy="158" r="16" fill="#3E8E5B"/>'
                f'<circle cx="{tx-9:.1f}" cy="164" r="11" fill="#4CA968"/>'
                f'<circle cx="{tx+9:.1f}" cy="164" r="11" fill="#4CA968"/>'
            )

    gate_svg = ""
    if uptown_gate or tier_level >= 4:
        gate_svg = (
            f'<rect x="20" y="195" width="{W-40}" height="6" fill="#7a6a52"/>'
            f'<rect x="20" y="180" width="5" height="21" fill="#7a6a52"/>'
            f'<rect x="{W-25}" y="180" width="5" height="21" fill="#7a6a52"/>'
        )

    chimney = ""
    if tier_level >= 2:
        chimney = f'<rect x="{house_x+house_w-30:.1f}" y="{house_y-30:.1f}" width="12" height="30" fill="#5a4a3a" stroke="#2B1B10" stroke-width="1.5"/>'

    stars = ""
    if dark_mode:
        rnd = random.Random(tier_level * 17)
        for _ in range(18):
            sx, sy = rnd.uniform(5, W - 5), rnd.uniform(5, 110)
            stars += f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{rnd.uniform(0.6,1.6):.1f}" fill="#C7ECFB" opacity="{rnd.uniform(0.4,0.9):.2f}"/>'

    sun_moon = (
        f'<circle cx="285" cy="35" r="16" fill="{"#EAF3FF" if dark_mode else "#FFD34D"}" opacity="0.9"/>'
    )

    svg = f"""
<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;max-width:360px;">
  <defs>
    <linearGradient id="sky{tier_level}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{sky_top}"/>
      <stop offset="100%" stop-color="{sky_bot}"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="{W}" height="{H}" fill="url(#sky{tier_level})"/>
  {stars}
  {sun_moon}
  <rect x="0" y="150" width="{W}" height="{H-150}" fill="{ground}"/>
  {gate_svg}
  {garden_svg}
  {garage_svg}
  {roof_svg}
  <rect x="{house_x:.1f}" y="{house_y:.1f}" width="{house_w}" height="{house_h}" fill="{wall}" stroke="#2B1B10" stroke-width="2.5"/>
  {chimney}
  {windows}
  {door}
  {pool_svg}
</svg>
"""
    return svg


# ---------------------------------------------------------------------------
# PUZZLE-STYLE PRICE REVEAL (CSS tiles that fly apart to reveal the number)
# ---------------------------------------------------------------------------
def puzzle_reveal_html(value_text: str, caption_text: str, key: str, accent="#E3B23C"):
    import math
    tiles = ""
    n_cols, n_rows = 6, 3
    i = 0
    for r in range(n_rows):
        for c in range(n_cols):
            rnd = random.Random(hash(key) + i)
            delay = (i * 0.045) + rnd.uniform(0, 0.05)
            angle_deg = random.Random(hash(key) + i + 100).uniform(-140, 140)
            dist = 140
            dx = math.cos(math.radians(angle_deg)) * dist
            dy = math.sin(math.radians(angle_deg)) * dist
            tiles += (
                f'<div class="puzzle-tile pt-{key}" style="'
                f'left:{c/n_cols*100:.3f}%; top:{r/n_rows*100:.3f}%; '
                f'width:{100/n_cols:.3f}%; height:{100/n_rows:.3f}%; '
                f'animation-delay:{delay:.3f}s; --fly-x:{dx:.1f}px; --fly-y:{dy:.1f}px; '
                f'--fly-rot:{angle_deg:.1f}deg;"></div>'
            )
            i += 1

    html = f"""
<div class="puzzle-wrap" id="puzzle-{key}">
  <div class="puzzle-value">{value_text}</div>
  <div class="puzzle-caption">{caption_text}</div>
  <div class="puzzle-tiles">{tiles}</div>
</div>
<style>
.puzzle-wrap {{
    position: relative;
    text-align: center;
    padding: 18px 10px;
    overflow: hidden;
    border-radius: 10px;
}}
.puzzle-value {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.6rem;
    color: {accent};
    letter-spacing: 0.01em;
    position: relative;
    z-index: 1;
}}
.puzzle-caption {{
    position: relative;
    z-index: 1;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    opacity: 0.75;
    margin-top: 4px;
}}
.puzzle-tiles {{
    position: absolute;
    inset: 0;
    z-index: 2;
    pointer-events: none;
}}
.pt-{key} {{
    position: absolute;
    background: linear-gradient(135deg, #1c3352, #0e1f38);
    border: 1px solid rgba(142,202,230,0.25);
    animation: puzzleFly-{key} 0.7s cubic-bezier(.4,0,.2,1) forwards;
}}
@keyframes puzzleFly-{key} {{
    0%   {{ opacity: 1; transform: translate(0,0) rotate(0deg) scale(1); }}
    100% {{ opacity: 0; transform: translate(var(--fly-x), var(--fly-y)) rotate(var(--fly-rot)) scale(0.3); }}
}}
</style>
"""
    return html


# ---------------------------------------------------------------------------
# PARTICLE / CONSTELLATION BACKGROUND (canvas injected behind the app)
# ---------------------------------------------------------------------------
def particle_background_component(dark_mode=True):
    particle_color = "142,202,230" if dark_mode else "60,110,150"
    line_color = "142,202,230" if dark_mode else "90,140,180"
    bg_color = "transparent"
    n_particles = 70

    js = f"""
<script>
(function() {{
    const doc = window.parent.document;
    const old = doc.getElementById('hva-particle-canvas');
    if (old) {{ old.remove(); }}
    if (window.parent.__hvaParticleRAF) {{
        window.parent.cancelAnimationFrame(window.parent.__hvaParticleRAF);
    }}

    const canvas = doc.createElement('canvas');
    canvas.id = 'hva-particle-canvas';
    canvas.style.position = 'fixed';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100vw';
    canvas.style.height = '100vh';
    canvas.style.zIndex = '-1';
    canvas.style.pointerEvents = 'none';
    doc.body.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    function resize() {{
        canvas.width = window.parent.innerWidth;
        canvas.height = window.parent.innerHeight;
    }}
    resize();
    window.parent.addEventListener('resize', resize);

    const N = {n_particles};
    let particles = [];
    for (let i = 0; i < N; i++) {{
        particles.push({{
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.35,
            vy: (Math.random() - 0.5) * 0.35,
            r: Math.random() * 1.6 + 0.6
        }});
    }}

    function tick() {{
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        for (let p of particles) {{
            p.x += p.vx; p.y += p.vy;
            if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
            if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba({particle_color},0.85)';
            ctx.fill();
        }}
        for (let i = 0; i < N; i++) {{
            for (let j = i + 1; j < N; j++) {{
                const a = particles[i], b = particles[j];
                const dx = a.x - b.x, dy = a.y - b.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 130) {{
                    ctx.beginPath();
                    ctx.moveTo(a.x, a.y);
                    ctx.lineTo(b.x, b.y);
                    ctx.strokeStyle = 'rgba({line_color},' + (1 - dist / 130) * 0.35 + ')';
                    ctx.lineWidth = 0.6;
                    ctx.stroke();
                }}
            }}
        }}
        window.parent.__hvaParticleRAF = window.parent.requestAnimationFrame(tick);
    }}
    tick();
}})();
</script>
"""
    return js


# ---------------------------------------------------------------------------
# THEME CSS (dark / light glassmorphism)
# ---------------------------------------------------------------------------
def theme_css(dark_mode=True):
    if dark_mode:
        bg = "#0A1220"
        text = "#E8F0F7"
        muted = "#9FB3C8"
        glass_bg = "rgba(20, 32, 54, 0.55)"
        glass_border = "rgba(142, 202, 230, 0.22)"
        accent = "#8ECAE6"
        accent2 = "#E3B23C"
        input_bg = "rgba(255,255,255,0.06)"
        shadow = "0 8px 32px rgba(0,0,0,0.45)"
    else:
        bg = "#F3F6FA"
        text = "#16283A"
        muted = "#5A6B7D"
        glass_bg = "rgba(255, 255, 255, 0.55)"
        glass_border = "rgba(22, 40, 58, 0.12)"
        accent = "#1D6FA5"
        accent2 = "#B9791E"
        input_bg = "rgba(255,255,255,0.85)"
        shadow = "0 8px 28px rgba(30,60,90,0.12)"

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

:root {{
    --hva-bg: {bg};
    --hva-text: {text};
    --hva-muted: {muted};
    --hva-glass: {glass_bg};
    --hva-border: {glass_border};
    --hva-accent: {accent};
    --hva-accent2: {accent2};
    --hva-input: {input_bg};
    --hva-shadow: {shadow};
}}

[data-testid="stAppViewContainer"], [data-testid="stApp"] {{
    background-color: {bg} !important;
}}
[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stSidebar"] {{ display: none; }}
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: {text}; }}

@keyframes fadeUp {{ from {{ opacity: 0; transform: translateY(14px); }} to {{ opacity: 1; transform: translateY(0); }} }}
@keyframes glowPulse {{
    0%, 100% {{ box-shadow: 0 0 12px rgba(142,202,230,0.25); }}
    50% {{ box-shadow: 0 0 26px rgba(142,202,230,0.45); }}
}}
@keyframes goldPulse {{
    0%, 100% {{ box-shadow: 0 0 16px rgba(227,178,60,0.45), 0 0 2px rgba(227,178,60,0.6); }}
    50% {{ box-shadow: 0 0 30px rgba(227,178,60,0.75), 0 0 6px rgba(227,178,60,0.8); }}
}}
@keyframes slideInImg {{
    from {{ opacity: 0; transform: translateX(30px) scale(0.94); }}
    to   {{ opacity: 1; transform: translateX(0) scale(1); }}
}}
@keyframes flashPop {{
    0%   {{ filter: brightness(2.4); opacity: 0; transform: scale(0.9); }}
    40%  {{ filter: brightness(1.5); opacity: 1; }}
    100% {{ filter: brightness(1); opacity: 1; transform: scale(1); }}
}}

/* Glass header */
.hva-header {{
    border: 1px solid var(--hva-border);
    border-radius: 16px;
    padding: 20px 28px;
    margin-bottom: 24px;
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;
    background: var(--hva-glass);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    box-shadow: var(--hva-shadow);
    animation: fadeUp 0.5s ease;
}}
.hva-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700; font-size: 2rem; margin: 0;
    background: linear-gradient(90deg, var(--hva-accent), var(--hva-accent2));
    -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.hva-subtitle {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
    color: var(--hva-muted); letter-spacing: 0.06em; margin-top: 4px;
}}

/* Glass cards / containers */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: var(--hva-glass) !important;
    border: 1px solid var(--hva-border) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    padding: 8px;
    transition: box-shadow 0.25s ease, transform 0.25s ease;
    animation: fadeUp 0.6s ease;
}}
[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    box-shadow: var(--hva-shadow), 0 0 18px rgba(142,202,230,0.18);
    transform: translateY(-2px);
}}
[data-testid="stVerticalBlockBorderWrapper"] * {{ color: var(--hva-text); }}

.hva-winner {{
    border: 2px solid var(--hva-accent2) !important;
    animation: goldPulse 2.2s ease-in-out infinite !important;
}}

.eyebrow {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    letter-spacing: 0.14em; text-transform: uppercase; color: var(--hva-accent2);
    margin-bottom: 8px; border-bottom: 1px dashed var(--hva-border); padding-bottom: 6px;
}}

/* Inputs */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stTextInput"] input {{
    background: var(--hva-input) !important;
    border: 1px solid var(--hva-border) !important;
    border-radius: 8px !important;
    color: var(--hva-text) !important;
}}
[data-testid="stSlider"] [role="slider"] {{ background-color: var(--hva-accent2) !important; box-shadow: 0 0 8px rgba(227,178,60,0.5) !important; }}
[data-testid="stSlider"] > div > div > div > div {{ background: linear-gradient(90deg, var(--hva-accent), var(--hva-accent2)) !important; }}

/* Buttons */
.stButton > button {{
    background: linear-gradient(90deg, var(--hva-accent), var(--hva-accent2)) !important;
    color: #0A1220 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em;
    padding: 0.7rem 1rem !important;
    box-shadow: 0 4px 18px rgba(142,202,230,0.3);
    transition: all 0.2s ease;
}}
.stButton > button:hover {{ transform: translateY(-2px); box-shadow: 0 8px 26px rgba(227,178,60,0.4); }}
.stButton > button:active {{ transform: translateY(0) scale(0.98); }}

/* KPI sparkline cards */
.kpi-card {{
    background: var(--hva-glass); border: 1px solid var(--hva-border); border-radius: 14px;
    padding: 14px 16px; backdrop-filter: blur(10px); animation: fadeUp 0.5s ease;
}}
.kpi-label {{ font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--hva-muted); text-transform: uppercase; letter-spacing: 0.08em; }}
.kpi-value {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.4rem; color: var(--hva-text); margin-top: 2px; }}

/* House illustration wrap */
.house-illustration {{
    animation: slideInImg 0.6s ease;
    text-align: center;
}}
.house-illustration img, .house-illustration svg {{ animation: flashPop 0.7s ease; border-radius: 12px; }}

/* Diff callout */
.diff-callout {{
    border-left: 3px solid var(--hva-accent2);
    background: var(--hva-glass);
    padding: 10px 14px; border-radius: 8px; font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem; margin-top: 10px; animation: fadeUp 0.5s ease;
}}

.trait-pill {{
    display: inline-block; padding: 4px 10px; margin: 3px 4px 3px 0;
    border: 1px solid var(--hva-border); border-radius: 999px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    background: rgba(142,202,230,0.08); color: var(--hva-text);
}}

[data-testid="stExpander"] {{ border: 1px solid var(--hva-border) !important; background: var(--hva-glass) !important; border-radius: 12px !important; }}
[data-testid="stTabs"] button[role="tab"] {{ font-family: 'Space Grotesk', sans-serif; font-weight: 600; }}
[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}
</style>
"""
