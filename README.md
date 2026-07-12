# 🏠 HomeValue AI

AIML Summer Internship 2026 — Capstone Project 2
IIHMF, MNNIT Allahabad, Prayagraj

## Objective
Predict house prices based on area, location, number of rooms, age of property,
and available amenities using regression models — presented through an
Indian-localized, glassmorphic Streamlit web app called **HomeValue AI**.

## Folder Structure
```
HomeValue_AI/
├── Dataset/
│   ├── generate_dataset.py       # generates the synthetic dataset
│   ├── house_data.csv            # raw dataset (with missing values, duplicates, outliers)
│   └── house_data_cleaned.csv    # cleaned dataset after preprocessing
├── Notebook/
│   └── train_pipeline.py         # full ML pipeline: preprocessing -> EDA -> feature
│                                    engineering -> model training -> evaluation
├── Model/
│   ├── house_price_model.pkl     # best trained model (sklearn Pipeline, XGBoost)
│   └── model_metadata.json       # feature list, locations, metrics
├── Streamlit_App/
│   ├── app.py                    # HomeValue AI — main deployment web app
│   ├── utils.py                  # currency/locale helpers, explanations, SVG house,
│   │                                particle background, theme CSS, puzzle reveal
│   └── requirements.txt
├── Documentation/
│   ├── eda_plots/                # histograms, boxplots, scatter plots, heatmap
│   └── model_comparison_results.csv
└── README.md
```

## App Features
- **Animated particle / constellation background** (canvas, injected behind the app)
- **Puzzle-style price reveal** — tiles fly apart to reveal the estimate
- **Indian localization** — prices shown in ₹ (Lakh/Crore formatting), localities
  relabelled in an Indian context (e.g. "Uptown" → "Uptown (Posh Colony)")
- **"Why this estimate" bullets** — rule-based explanation comparing your inputs
  to the dataset (bigger than average, closer to city, premium locality, etc.)
- **House illustration** — a parametric SVG house that changes with the price tier
  and your chosen amenities (garage, garden, pool, boundary gate)
- **Locality quality pills** — e.g. "Near markets & shopping hubs" for City Center
- **Compare mode** — two properties side-by-side, gold glowing border on the
  higher-value property, with an auto-generated "why" callout
- **Insights tab** — live Plotly price distribution, correlation heatmap,
  price-by-locality chart, sparkline KPI cards, and a styled dataset preview
- **Dark / light theme toggle**

## How to Run

### 1. (Optional) Regenerate the dataset
```bash
cd Dataset
python3 generate_dataset.py
```
To use a real Kaggle dataset instead, replace `house_data.csv` with a CSV
containing the same columns: `area_sqft, location, bedrooms, bathrooms,
age_years, garage, garden, pool, near_school, distance_to_city_km, price`.

### 2. Train the models
```bash
cd Notebook
python3 train_pipeline.py
```
This runs the complete pipeline (Phases 3–7 of the project methodology) and
saves the best-performing model to `Model/house_price_model.pkl`.

### 3. Launch the web app
```bash
cd Streamlit_App
pip install -r requirements.txt
streamlit run app.py
```
Open the URL shown in the terminal (usually `http://localhost:8501`).

**Note on currency:** the dataset and model are trained on USD prices; the app
applies a fixed, clearly-labelled conversion rate (`USD_TO_INR` in `utils.py`,
default ₹83) to display everything in Rupees. Adjust that constant if you want
a different rate.

## Dataset
Synthetic dataset (2,000 rows) modeled on real-world Kaggle house-price data,
with realistic relationships between features and price, plus injected
missing values, duplicate rows, and outliers so the full Data Preprocessing
phase has real issues to resolve.

**Features:** `area_sqft`, `location`, `bedrooms`, `bathrooms`, `age_years`,
`garage`, `garden`, `pool`, `near_school`, `distance_to_city_km`
**Target:** `price`

## Models Trained
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | ~23,479 | ~39,339 | 0.903 |
| Random Forest | ~22,607 | ~38,441 | 0.907 |
| **XGBoost (best)** | **~20,726** | **~37,015** | **0.914** |

## Tech Stack
- Python, pandas, numpy
- scikit-learn (preprocessing pipeline, Linear Regression, Random Forest)
- XGBoost
- matplotlib, seaborn (EDA)
- Streamlit + Plotly (deployment / interactive charts)

## Testing Notes
The build environment used to assemble this app has no internet access, so
`streamlit`, `plotly`, and `xgboost` could not be installed there and the app
could not be launched in a browser for a live click-through. What was
verified in that environment:
- Both `app.py` and `utils.py` compile cleanly (no syntax errors)
- Every pure-Python helper in `utils.py` (currency formatting, locality
  mapping, price explanations, SVG house generation, puzzle-reveal HTML,
  particle-background script, theme CSS) was unit-tested directly with
  real values
- A real bug was caught by code review and fixed: wrapping a Streamlit
  container in a raw HTML `<div>` does **not** actually nest it in the DOM
  (Streamlit renders containers and markdown as siblings), so the original
  "gold border on the winning property" approach silently would not have
  worked. It now uses `st.container(key=...)` + scoped CSS instead, which
  does reliably target the container.

**Please do a quick click-through after installing requirements** — press
"Reveal My Estimate", try "Compare Properties", and toggle dark/light — to
confirm everything renders as expected on your machine/browser.

## Academic Integrity
This code was scaffolded with AI assistance for learning purposes, per the
project guidelines. Before submission, read through every function, re-run
the pipeline yourself, and be ready to explain each design decision
(preprocessing choices, feature engineering, model selection, metrics).
