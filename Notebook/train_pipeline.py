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

Training needs:  pip install pandas numpy scikit-learn xgboost matplotlib seaborn
The web app itself only needs xgboost (see Streamlit_App/requirements.txt).

Deployed artifacts (version-proof, no pickle):
  Model/xgb_model.json       XGBoost booster saved with save_model (JSON)
  Model/model_metadata.json  scaling constants, locality list, metrics, ranges
"""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost
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

MAX_AREA = 15000          # nothing bigger than this is a believable single home
DEPLOYED_MODEL = "XGBoost"  # the app uses XGBoost (monotonic age rule + JSON export)

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

# 2. Handle missing values
#    area_sqft is the main price driver and cannot be guessed from a median
#    (that would put a 1,250 sq ft area on a 6,000 sq ft home), so rows
#    without it are dropped. bathrooms / garage are filled with the median.
missing_before = int(df.isnull().sum().sum())
n_no_area = int(df["area_sqft"].isnull().sum())
df = df.dropna(subset=["area_sqft"]).reset_index(drop=True)
for col in ["bathrooms", "garage"]:
    df[col] = df[col].fillna(df[col].median()).round().astype(int)
print(f"Handled {missing_before} missing values: dropped {n_no_area} rows with no area, "
      f"median-filled the rest")

# 3. Outliers: remove rows whose price per sq ft is wildly off for their
#    locality (e.g. an area typed 5x too large) using the IQR rule (k=3) on
#    the log of rate / locality-median rate. Large homes are NOT removed just
#    for being large, because the model must learn big plots and many rooms.
rate = df["price"] / df["area_sqft"]
log_ratio = np.log(rate / rate.groupby(df["location"]).transform("median"))
q1, q3 = log_ratio.quantile(0.25), log_ratio.quantile(0.75)
iqr = q3 - q1
bad = (log_ratio < q1 - 3 * iqr) | (log_ratio > q3 + 3 * iqr) | (df["area_sqft"] > MAX_AREA)
print(f"Removed {int(bad.sum())} outlier rows (implausible price per sq ft or area > {MAX_AREA:,})")
df = df[~bad].reset_index(drop=True)

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

# Prices span ~Rs 10 lakh to ~Rs 15 crore, so every model learns log(price)
# and predictions are converted back to rupees (errors below are in rupees).
def with_log_target(model):
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)

# Monotonic constraint: a building must never get MORE valuable just because
# it is older. Constraints are given per transformed column (numeric columns
# first, then the one-hot locality columns). is_new (age <= 5) is a step
# function of age, so it carries the matching +1 constraint.
n_loc = df["location"].nunique()
constraints = [0] * (len(feature_cols_numeric) + n_loc)
constraints[feature_cols_numeric.index("age_years")] = -1
constraints[feature_cols_numeric.index("is_new")] = +1
xgb_params = dict(n_estimators=500, max_depth=5, learning_rate=0.05,
                  subsample=0.9, colsample_bytree=0.9, random_state=42)

models = {
    "Linear Regression": with_log_target(LinearRegression()),
    "Random Forest": with_log_target(RandomForestRegressor(n_estimators=300, max_depth=14,
                                                           random_state=42, n_jobs=-1)),
    "XGBoost": with_log_target(XGBRegressor(monotone_constraints=tuple(constraints), **xgb_params)),
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
    mape = float(np.mean(np.abs(preds - y_test) / y_test))

    results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "MAPE": mape}
    pipelines[name] = pipe
    print(f"{name:20s} | MAE: {mae:12,.0f} | RMSE: {rmse:12,.0f} | R2: {r2:.4f} | MAPE: {mape*100:.1f}%")

print()

# Sanity check for the monotonic age rule (constrained vs. unconstrained XGBoost)
def count_age_violations(pipe, rows, ages=range(0, 61)):
    bad = 0
    for _, r in rows.iterrows():
        grid = pd.DataFrame([r] * len(ages)).reset_index(drop=True)
        grid["age_years"] = list(ages)
        grid["is_new"] = (grid["age_years"] <= 5).astype(int)
        p = pipe.predict(grid)
        bad += int((np.diff(p) > 1.0).any())     # price rose with age by more than Rs 1
    return bad

sample_rows = X_test.sample(150, random_state=0)
unconstrained = Pipeline(steps=[("preprocessor", preprocessor),
                                ("model", with_log_target(XGBRegressor(**xgb_params)))]).fit(X_train, y_train)
v_con = count_age_violations(pipelines["XGBoost"], sample_rows)
v_unc = count_age_violations(unconstrained, sample_rows)
print(f"Age monotonicity check on 150 homes: constrained XGBoost violations = {v_con}, "
      f"unconstrained = {v_unc}")
assert v_con == 0, "constrained model still raises price with age"
print()

# =========================================================================
# PHASE 7: MODEL EVALUATION
# =========================================================================
print("=" * 70)
print("PHASE 7: MODEL EVALUATION & COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results).T.sort_values("R2", ascending=False)
print(results_df.round(4))

# MAPE (typical % error) is the fairest yardstick now that prices run from
# lakhs to crores; MAE / RMSE / R2 are reported alongside it.
lowest_error = results_df["MAPE"].idxmin()
print(f"\nLowest error: {lowest_error} (MAPE = {results_df.loc[lowest_error, 'MAPE']*100:.1f}%, "
      f"MAE = Rs {results_df.loc[lowest_error, 'MAE']:,.0f})")
best_model_name = DEPLOYED_MODEL
best_pipeline = pipelines[best_model_name]
print(f"Deployed model: {best_model_name} (monotonic age rule, JSON export)")

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
# SAVE ARTIFACTS  (XGBoost JSON + plain-JSON preprocessing, no pickle)
# =========================================================================
prep = best_pipeline.named_steps["preprocessor"]
xgb_fitted = best_pipeline.named_steps["model"].regressor_
xgb_fitted.get_booster().save_model(f"{MODEL_DIR}/xgb_model.json")
for stale in ("house_price_model.pkl",):          # old pickle is no longer used
    if os.path.exists(f"{MODEL_DIR}/{stale}"):
        os.remove(f"{MODEL_DIR}/{stale}")

scaler = prep.named_transformers_["num"]
categories = [str(c) for c in prep.named_transformers_["cat"].categories_[0]]
metadata = {
    "best_model": best_model_name,
    "deployed_model": best_model_name,
    "lowest_error_model": lowest_error,
    "model_file": "xgb_model.json",
    "xgboost_version_trained_with": xgboost.__version__,
    "target_transform": "log1p",
    "monotone_constraints": {"age_years": -1, "is_new": 1},
    "feature_cols_numeric": feature_cols_numeric,
    "feature_cols_categorical": feature_cols_categorical,
    "locations": sorted(df["location"].unique().tolist()),
    "preprocessing": {
        "numeric_mean": [float(v) for v in scaler.mean_],
        "numeric_scale": [float(v) for v in scaler.scale_],
        "location_categories": categories,
    },
    "training_ranges": {
        "area_sqft": [float(df["area_sqft"].min()), float(df["area_sqft"].max())],
        "bedrooms": [int(df["bedrooms"].min()), int(df["bedrooms"].max())],
        "bathrooms": [int(df["bathrooms"].min()), int(df["bathrooms"].max())],
    },
    "metrics": results,
}
with open(f"{MODEL_DIR}/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2, default=float)

# Self-test: reload the JSON the way the app does and compare with sklearn
def app_style_predict(frame):
    meta = json.load(open(f"{MODEL_DIR}/model_metadata.json"))
    pp = meta["preprocessing"]
    num = (frame[meta["feature_cols_numeric"]].to_numpy(float) - np.array(pp["numeric_mean"])) / np.array(pp["numeric_scale"])
    cat = (frame["location"].to_numpy()[:, None] == np.array(pp["location_categories"])[None, :]).astype(float)
    booster = xgboost.Booster()
    booster.load_model(f"{MODEL_DIR}/{meta['model_file']}")
    return np.expm1(booster.predict(xgboost.DMatrix(np.hstack([num, cat]))))

diff = np.abs(app_style_predict(X_test) - best_pipeline.predict(X_test)).max()
print(f"\nJSON model vs sklearn pipeline: max difference = Rs {diff:.4f}")
assert diff < 1.0, "exported model does not match the trained pipeline"

results_df.round(4).to_csv(f"{DOC_DIR}/model_comparison_results.csv", index_label="Model")

# Keep the README table in sync with this exact run
rows = ["| Model | MAE (Rs) | RMSE (Rs) | R² | MAPE |", "|---|---|---|---|---|"]
for name, r in results_df.iterrows():
    label = f"**{name} (deployed)**" if name == DEPLOYED_MODEL else name
    rows.append(f"| {label} | {r['MAE']:,.0f} | {r['RMSE']:,.0f} | {r['R2']:.3f} | {r['MAPE']*100:.1f}% |")
readme_path = f"{BASE}/README.md"
if os.path.exists(readme_path):
    readme = open(readme_path, encoding="utf-8").read()
    block = "<!-- METRICS:START -->\n" + "\n".join(rows) + "\n<!-- METRICS:END -->"
    readme, n = re.subn(r"<!-- METRICS:START -->.*?<!-- METRICS:END -->", lambda m: block, readme, flags=re.S)
    if n:
        open(readme_path, "w", encoding="utf-8").write(readme)
        print("Updated metrics table in README.md")

print(f"Saved model: Model/xgb_model.json")
print(f"Saved metadata: Model/model_metadata.json")
print(f"Saved comparison table: Documentation/model_comparison_results.csv")
print("\nPipeline complete.")
