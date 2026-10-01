"""
train_pipeline.py
------------------
House Price Prediction System - full ML pipeline
(mirrors Phases 3-7 of the Capstone Project Guidelines)

Phase 3: Data Preprocessing
Phase 4: Exploratory Data Analysis (EDA)
Phase 5: Feature Engineering
Phase 6: Model Building (Linear Regression, Random Forest, XGBoost)
Phase 7: Model Evaluation (MAE, MSE, RMSE, R2)

Run:  python train_pipeline.py   (prices are in Indian Rupees)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(SCRIPT_DIR)  # parent of Notebook/ = project root
DATASET = f"{BASE}/Dataset/house_data.csv"
MODEL_DIR = f"{BASE}/Model"
DOC_DIR = f"{BASE}/Documentation"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DOC_DIR, exist_ok=True)
os.makedirs(f"{DOC_DIR}/eda_plots", exist_ok=True)

sns.set_style("whitegrid")

# =========================================================================
# PHASE 3: DATA PREPROCESSING
# =========================================================================
print("=" * 70)
print("PHASE 3: DATA PREPROCESSING")
print("=" * 70)

df = pd.read_csv(DATASET)
print(f"Raw shape: {df.shape}")

# 1. Remove duplicates
before = df.shape[0]
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {before - df.shape[0]} duplicate rows")

# 2. Handle missing values (median for numeric)
missing_before = df.isnull().sum().sum()
for col in ["area_sqft", "bathrooms", "garage"]:
    df[col] = df[col].fillna(df[col].median())
print(f"Filled {missing_before} missing values using median imputation")

# 3. Treat outliers using IQR capping on area_sqft and price
def cap_outliers_iqr(series, k=1.5):
    # k=1.5 is the textbook value; we use k=3 for price (see below) so that
    # genuine premium-locality homes are kept and only extreme values are capped.
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return series.clip(lower, upper)

df["area_sqft"] = cap_outliers_iqr(df["area_sqft"], k=3.0)
df["price"] = cap_outliers_iqr(df["price"], k=3.0)
print("Outliers capped (IQR method, k=3) on: area_sqft, price")

# 4. Encoding categorical features happens inside the sklearn Pipeline
#    (OneHotEncoder on 'location') so it's learned only on training data
#    and reused consistently at inference time in the Streamlit app.

df.to_csv(f"{BASE}/Dataset/house_data_cleaned.csv", index=False)
print(f"Cleaned shape: {df.shape}")
print("Saved: Dataset/house_data_cleaned.csv\n")

# =========================================================================
# PHASE 4: EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================================
print("=" * 70)
print("PHASE 4: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

numeric_cols = ["area_sqft", "bedrooms", "bathrooms", "age_years",
                 "distance_to_city_km", "price"]

# Univariate: histograms
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flat, numeric_cols):
    sns.histplot(df[col], kde=True, ax=ax, color="steelblue")
    ax.set_title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig(f"{DOC_DIR}/eda_plots/histograms.png", dpi=110)
plt.close()

# Univariate/Bivariate: boxplots (price by location)
plt.figure(figsize=(14, 6))
order = df.groupby("location")["price"].median().sort_values().index
sns.boxplot(data=df, x="location", y="price", order=order, palette="Set2")
plt.title("Price (Rs) by Prayagraj locality")
plt.xticks(rotation=70)
plt.tight_layout()
plt.savefig(f"{DOC_DIR}/eda_plots/boxplot_price_location.png", dpi=110)
plt.close()

# Bivariate: scatter plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(data=df, x="area_sqft", y="price", ax=axes[0], alpha=0.5)
axes[0].set_title("Area vs Price")
sns.scatterplot(data=df, x="distance_to_city_km", y="price", ax=axes[1], alpha=0.5, color="orange")
axes[1].set_title("Distance to City vs Price")
plt.tight_layout()
plt.savefig(f"{DOC_DIR}/eda_plots/scatterplots.png", dpi=110)
plt.close()

# Correlation heatmap
plt.figure(figsize=(8, 6))
corr = df[numeric_cols + ["garage", "garden", "lift", "near_school"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{DOC_DIR}/eda_plots/correlation_heatmap.png", dpi=110)
plt.close()

print("Saved EDA plots to Documentation/eda_plots/")
print(df[numeric_cols].describe().round(1))
print()

# =========================================================================
# PHASE 5: FEATURE ENGINEERING
# =========================================================================
print("=" * 70)
print("PHASE 5: FEATURE ENGINEERING")
print("=" * 70)

# New meaningful features
df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
df["price_per_sqft_proxy"] = df["area_sqft"] / df["total_rooms"].replace(0, 1)
df["amenity_score"] = df["garage"] + df["garden"] + df["lift"] + df["near_school"]
df["is_new"] = (df["age_years"] <= 5).astype(int)

print("Created features: total_rooms, price_per_sqft_proxy, amenity_score, is_new")

# Feature selection: drop the engineered proxy that leaks structure of the
# target too directly, keep the rest as meaningful, non-redundant signals.
feature_cols_numeric = [
    "area_sqft", "bedrooms", "bathrooms", "age_years",
    "distance_to_city_km", "total_rooms", "amenity_score", "is_new",
    "garage", "garden", "lift", "near_school",
]
feature_cols_categorical = ["location"]
target_col = "price"

X = df[feature_cols_numeric + feature_cols_categorical]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}\n")

# Preprocessing pipeline: scale numeric, one-hot encode categorical
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), feature_cols_numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_categorical),
])

# =========================================================================
# PHASE 6: MODEL BUILDING (3 models)
# =========================================================================
print("=" * 70)
print("PHASE 6: MODEL BUILDING")
print("=" * 70)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9, random_state=42),
}

results = {}
pipelines = {}

for name, model in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
    pipelines[name] = pipe
    print(f"{name:20s} | MAE: {mae:10,.0f} | RMSE: {rmse:10,.0f} | R2: {r2:.4f}")

print()

# =========================================================================
# PHASE 7: MODEL EVALUATION
# =========================================================================
print("=" * 70)
print("PHASE 7: MODEL EVALUATION & COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results).T.sort_values("R2", ascending=False)
print(results_df.round(4))

# Pick the winner by MAE (average error in rupees). R2/RMSE are dominated by
# a handful of extreme-price rows, MAE reflects the typical prediction error.
best_model_name = results_df["MAE"].idxmin()
best_pipeline = pipelines[best_model_name]
print(f"\nBest model: {best_model_name} (MAE = Rs {results_df.loc[best_model_name, 'MAE']:,.0f}, "
      f"R2 = {results_df.loc[best_model_name, 'R2']:.4f})")

# Comparative bar chart
plt.figure(figsize=(8, 5))
results_df["R2"].plot(kind="bar", color=["#2ca02c", "#1f77b4", "#ff7f0e"])
plt.title("Model Comparison - R2 Score")
plt.ylabel("R2 Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{DOC_DIR}/eda_plots/model_comparison.png", dpi=110)
plt.close()

# Actual vs Predicted for best model
preds_best = best_pipeline.predict(X_test)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, preds_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Price (Rs)")
plt.ylabel("Predicted Price (Rs)")
plt.title(f"Actual vs Predicted - {best_model_name}")
plt.tight_layout()
plt.savefig(f"{DOC_DIR}/eda_plots/actual_vs_predicted.png", dpi=110)
plt.close()

# =========================================================================
# SAVE ARTIFACTS
# =========================================================================
joblib.dump(best_pipeline, f"{MODEL_DIR}/house_price_model.pkl")

metadata = {
    "best_model": best_model_name,
    "feature_cols_numeric": feature_cols_numeric,
    "feature_cols_categorical": feature_cols_categorical,
    "locations": sorted(df["location"].unique().tolist()),
    "metrics": results,
}
with open(f"{MODEL_DIR}/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2, default=float)

results_df.round(4).to_csv(f"{DOC_DIR}/model_comparison_results.csv")

print(f"\nSaved model: Model/house_price_model.pkl")
print(f"Saved metadata: Model/model_metadata.json")
print(f"Saved comparison table: Documentation/model_comparison_results.csv")
print("\nPipeline complete.")
