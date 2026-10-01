"""
generate_dataset.py  (Prayagraj edition)
----------------------------------------
Builds a synthetic-but-calibrated house price dataset for PRAYAGRAJ ONLY.

Outputs (all written next to this script):
  house_data.csv      raw training data (prices in Indian Rupees, with some
                      missing values / duplicates / outliers for the
                      preprocessing phase of the capstone)
  locality_info.csv   one row per locality: centre (lat, lon), base rate
                      in Rs per sq ft, and a short note
  landmarks.csv       famous places of Prayagraj shown on the map

How the prices were calibrated
  'listing' rates below were taken from 99acres / Housing.com locality
  pages (mid-2026). Those sites disagree with each other by a lot for some
  localities (e.g. Jhalwa), so each rate is a rounded working figure, not an
  official valuation. 'estimate' rates had no clear public figure and were
  set by judgement relative to neighbouring localities.
  Coordinates are approximate locality centres, used only to draw the map
  and to compute distance to the city centre (Civil Lines).
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CITY_CENTRE = (25.4480, 81.8432)  # Civil Lines


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# name, lat, lon, Rs/sq ft, rate_source, listing weight, note
LOCALITIES = [
    ("Civil Lines",   25.4480, 81.8432, 10500, "listing",  1.3, "Main commercial and business hub of the city"),
    ("George Town",   25.4575, 81.8500, 11500, "listing",  1.0, "Upper-class planned colony (est. 1901) with many hospitals"),
    ("Tagore Town",   25.4605, 81.8600,  9000, "estimate", 0.9, "Established premium residential colony"),
    ("Stanley Road",  25.4440, 81.8380,  9850, "listing",  0.6, "Well-connected central road close to Civil Lines"),
    ("Lukerganj",     25.4530, 81.8260,  8000, "listing",  0.9, "Established neighbourhood north of the Junction station"),
    ("Ashok Nagar",   25.4620, 81.8300,  7000, "listing",  0.9, "Established residential area with good livability"),
    ("Sulem Sarai",   25.4900, 81.8900,  7000, "listing",  1.0, "Fast-growing area, strong price rise in last 3 years"),
    ("Allahpur",      25.4553, 81.8717,  6200, "listing",  1.0, "Close to Anand Bhawan and the Sangam"),
    ("Mumfordganj",   25.4650, 81.8480,  6300, "estimate", 0.8, "Old British-era residential neighbourhood"),
    ("Teliyarganj",   25.4800, 81.8630,  6200, "estimate", 0.9, "Near MNNIT Allahabad, popular with students and staff"),
    ("Katra",         25.4475, 81.8555,  6500, "estimate", 0.8, "Major commercial market near Civil Lines"),
    ("Colonelganj",   25.4415, 81.8500,  6500, "estimate", 0.6, "Dense central locality"),
    ("Phaphamau",     25.5339, 81.8539,  6000, "listing",  0.8, "Satellite township on the Lucknow highway"),
    ("Chowk",         25.4368, 81.8341,  5500, "estimate", 0.5, "Historic old-city market around the Clock Tower"),
    ("Daraganj",      25.4330, 81.8830,  5200, "estimate", 0.6, "Oldest suburb, on the Ganga bank near Sangam"),
    ("Naini",         25.3960, 81.8700,  5200, "listing",  1.4, "Across the Yamuna, mid-segment and fast developing"),
    ("Dhoomanganj",   25.4730, 81.8570,  5000, "estimate", 0.9, "Good rental yields, dense residential area"),
    ("Subedarganj",   25.4480, 81.8130,  5000, "estimate", 0.7, "West side, near Subedarganj station"),
    ("Kydganj",       25.4370, 81.8560,  4800, "estimate", 0.6, "Densely populated central locality"),
    ("Alopibagh",     25.4290, 81.8640,  4400, "estimate", 0.5, "Near Alopi Devi Mandir, close to Sangam"),
    ("Muthiganj",     25.4300, 81.8430,  4200, "estimate", 0.5, "Old-city locality"),
    ("Jhalwa",        25.3700, 81.8500,  4200, "listing",  1.2, "Affordable, popular with students and professionals"),
    ("Bamrauli",      25.4400, 81.7750,  4000, "listing",  0.8, "West side near Prayagraj airport"),
    ("Jhunsi",        25.4300, 81.9300,  3600, "estimate", 0.9, "Large township on the Ganga bank, east of Sangam"),
]

LANDMARKS = [
    ("Triveni Sangam",               25.4270, 81.8880, "Confluence of Ganga, Yamuna and Saraswati; Kumbh Mela site"),
    ("Allahabad Fort",               25.4285, 81.8740, "Akbar-era fort at the Sangam"),
    ("Bade Hanuman Mandir",          25.4282, 81.8805, "Famous reclining Hanuman temple near the fort"),
    ("Khusro Bagh",                  25.4388, 81.8275, "Mughal-era garden and tombs"),
    ("Chowk Clock Tower",            25.4368, 81.8341, "Historic Ghantaghar (1913) in the old-city market"),
    ("Prayagraj Junction",           25.4465, 81.8275, "Main railway station"),
    ("All Saints Cathedral",         25.4498, 81.8420, "Patthar Girja, Civil Lines"),
    ("Chandrashekhar Azad Park",     25.4510, 81.8490, "Formerly Alfred Park; large city park"),
    ("Anand Bhawan",                 25.4597, 81.8615, "Nehru family home, now a museum"),
    ("University of Allahabad",      25.4610, 81.8550, "One of India's oldest universities"),
    ("MNNIT Allahabad",              25.4925, 81.8634, "Motilal Nehru National Institute of Technology"),
]

loc_df = pd.DataFrame(LOCALITIES, columns=["location", "lat", "lon", "rate_per_sqft",
                                           "rate_source", "weight", "note"])
loc_df["distance_to_city_km"] = haversine_km(loc_df.lat, loc_df.lon, *CITY_CENTRE).round(2)
loc_df["tier"] = pd.cut(loc_df.rate_per_sqft, [0, 5499, 7999, 1e9],
                        labels=["Affordable", "Mid-range", "Premium"]).astype(str)
loc_out = loc_df.drop(columns=["weight"])
loc_out.to_csv(os.path.join(SCRIPT_DIR, "locality_info.csv"), index=False)

lm_df = pd.DataFrame(LANDMARKS, columns=["name", "lat", "lon", "note"])
nearest = []
for _, r in lm_df.iterrows():
    d = haversine_km(r.lat, r.lon, loc_df.lat.values, loc_df.lon.values)
    nearest.append(loc_df.location.iloc[int(np.argmin(d))])
lm_df["nearest_locality"] = nearest
lm_df.to_csv(os.path.join(SCRIPT_DIR, "landmarks.csv"), index=False)

# ---------------------------------------------------------------------------
# HOUSES
# ---------------------------------------------------------------------------
N = 3000
w = loc_df.weight / loc_df.weight.sum()
idx = np.random.choice(len(loc_df), N, p=w)
loc = loc_df.iloc[idx].reset_index(drop=True)

area = np.clip(np.random.lognormal(np.log(1250), 0.42, N), 350, 5000).round(0)
bedrooms = np.clip(np.round(area / 480 + np.random.normal(0, 0.6, N)), 1, 6).astype(int)
bathrooms = np.clip(np.round(bedrooms * 0.75 + np.random.normal(0, 0.6, N)), 1, 4).astype(int)
age = np.clip(np.random.gamma(2.0, 7.0, N), 0, 60).round().astype(int)

rate = loc.rate_per_sqft.values
premium_share = np.interp(rate, [3600, 6000, 11500], [0.08, 0.25, 0.55])
garage = (np.random.rand(N) < 0.60).astype(int)
garden = (np.random.rand(N) < 0.35).astype(int)
lift = (np.random.rand(N) < premium_share).astype(int)
near_school = (np.random.rand(N) < 0.60).astype(int)

# small street-level jitter around the locality centre (~400 m)
lat = loc.lat.values + np.random.normal(0, 0.0035, N)
lon = loc.lon.values + np.random.normal(0, 0.0035, N)
dist = haversine_km(lat, lon, *CITY_CENTRE).round(2)

f = rate / 6000.0  # scale fixed add-ons with how expensive the locality is
age_factor = 1 - np.minimum(age * 0.007, 0.32) + 0.06 * (age <= 5)
layout_factor = 1 + 0.02 * (bedrooms - 3) + 0.015 * (bathrooms - 2)
price = area * rate * age_factor * layout_factor
price = price + garage * 4.0e5 * f + garden * 3.0e5 * f + lift * 2.5e5 * f
price = price * np.where(near_school == 1, 1.02, 1.0)
price = price * np.random.lognormal(0, 0.07, N)
price = np.clip(price, 8e5, None).round(0)

df = pd.DataFrame({
    "area_sqft": area, "location": loc.location.values, "bedrooms": bedrooms,
    "bathrooms": bathrooms, "age_years": age, "garage": garage, "garden": garden,
    "lift": lift, "near_school": near_school, "distance_to_city_km": dist,
    "price": price,
})

# --- messiness for the preprocessing phase --------------------------------
for col, frac in [("bathrooms", 0.03), ("garage", 0.02), ("area_sqft", 0.01)]:
    i = df.sample(frac=frac, random_state=1).index
    df.loc[i, col] = np.nan
df = pd.concat([df, df.sample(20, random_state=2)], ignore_index=True)
oi = df.sample(8, random_state=3).index
df.loc[oi, "area_sqft"] = df.loc[oi, "area_sqft"] * 5
df.loc[oi[:4], "price"] = df.loc[oi[:4], "price"] * 4
df = df.sample(frac=1, random_state=7).reset_index(drop=True)
df.to_csv(os.path.join(SCRIPT_DIR, "house_data.csv"), index=False)

print("Dataset generated:", df.shape, "| localities:", len(loc_df), "| landmarks:", len(lm_df))
print((df.price / 1e5).describe().round(1))
