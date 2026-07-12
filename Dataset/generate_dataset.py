"""
generate_dataset.py
--------------------
Generates a realistic synthetic House Price dataset for the AIML Summer
Internship 2026 Capstone Project (Project 2: House Price Prediction System).

Why synthetic? This environment has no live access to Kaggle. The dataset
below mirrors the structure and relationships of real Kaggle house-price
datasets (area, location, rooms, age, amenities -> price), including
realistic noise, a few missing values, a few duplicates, and a few outliers
-- so every Phase 3 (Data Preprocessing) step in the PDF has something real
to fix. To use a real Kaggle dataset instead, just drop a CSV with the same
column names into Dataset/house_data.csv and skip this script.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

N = 2000

locations = ["Downtown", "Suburb", "Rural", "City Center", "Uptown"]
location_multiplier = {
    "Downtown": 1.35,
    "City Center": 1.45,
    "Uptown": 1.15,
    "Suburb": 1.00,
    "Rural": 0.70,
}

df = pd.DataFrame({
    "area_sqft": np.random.normal(1800, 650, N).clip(400, 6000).round(0),
    "location": np.random.choice(locations, N, p=[0.20, 0.15, 0.15, 0.30, 0.20]),
    "bedrooms": np.random.choice([1, 2, 3, 4, 5, 6], N, p=[0.05, 0.20, 0.30, 0.25, 0.15, 0.05]),
    "bathrooms": np.random.choice([1, 2, 3, 4], N, p=[0.25, 0.40, 0.25, 0.10]),
    "age_years": np.random.randint(0, 60, N),
    "garage": np.random.choice([0, 1], N, p=[0.35, 0.65]),
    "garden": np.random.choice([0, 1], N, p=[0.45, 0.55]),
    "pool": np.random.choice([0, 1], N, p=[0.85, 0.15]),
    "near_school": np.random.choice([0, 1], N, p=[0.4, 0.6]),
    "distance_to_city_km": np.random.exponential(8, N).clip(0.2, 50).round(2),
})

# --- Ground-truth price formula with noise ------------------------------
base_price = (
    df["area_sqft"] * 120
    + df["bedrooms"] * 8000
    + df["bathrooms"] * 6000
    - df["age_years"] * 900
    + df["garage"] * 9000
    + df["garden"] * 5000
    + df["pool"] * 15000
    + df["near_school"] * 7000
    - df["distance_to_city_km"] * 1200
    + 40000
)
loc_mult = df["location"].map(location_multiplier)
noise = np.random.normal(0, 18000, N)
df["price"] = (base_price * loc_mult + noise).clip(lower=25000).round(0)

# --- Inject realistic messiness for the Data Preprocessing phase --------
# 1) Missing values
for col, frac in [("bathrooms", 0.03), ("garage", 0.02), ("area_sqft", 0.01)]:
    idx = df.sample(frac=frac, random_state=1).index
    df.loc[idx, col] = np.nan

# 2) Duplicate rows
dupes = df.sample(20, random_state=2)
df = pd.concat([df, dupes], ignore_index=True)

# 3) Outliers
outlier_idx = df.sample(8, random_state=3).index
df.loc[outlier_idx, "area_sqft"] = df.loc[outlier_idx, "area_sqft"] * 5
df.loc[outlier_idx[:4], "price"] = df.loc[outlier_idx[:4], "price"] * 4

df = df.sample(frac=1, random_state=7).reset_index(drop=True)
df.to_csv(os.path.join(SCRIPT_DIR, "house_data.csv"), index=False)

print("Dataset generated:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
