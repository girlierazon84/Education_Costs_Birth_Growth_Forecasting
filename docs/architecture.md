# EduForecast Architecture

This document describes the project structure, data flow, and how the forecasting + cost computation pipeline fits together.

## Goals

- Forecast births by Swedish region (historical range starts **1968**).
- Convert births into a **population (age 0–19)** forecast (cohort style).
- Convert population forecasts into **education cost** forecasts (grundskola and gymnasieskola).
- Produce reproducible artifacts (CSVs + plots) and a Streamlit dashboard for exploration.

---

## Repository Layout

Typical structure (your repo matches this concept):

- `src/eduforecast/`
  Core Python package (pipeline, modeling, DB access, costs computations).
- `configs/`
  YAML configuration files (`config.yaml`, `logging.yaml`).
- `data/`
  - `data/raw/` (raw curated inputs, e.g. births per region)
  - `data/external/` (external reference tables, e.g. cost per child)
  - `data/processed/` (SQLite DB and cleaned outputs)
- `artifacts/` (pipeline outputs)
  - `artifacts/forecasts/` (forecast CSVs)
  - `artifacts/metrics/` (evaluation summaries)
  - `artifacts/figures/` (plots)
  - `artifacts/models/` (serialized models, if used)
- `dashboards/`
  Streamlit Multi-Page App (Home + pages).
- `tests/`
  Unit/integration tests.
- `scripts/`, `notebooks/`, `docs/`
  Supporting materials.

---

## Data Sources

### Births (region/year)
Primary input for forecasting births.

Expected columns (standardized in code):
- `Region_Code` (e.g. "01")
- `Region_Name` (optional in raw file; code should map/fill it)
- `Year`
- `Number` (births count as float)

Your raw file currently looks like:
- `Region_Code`, `Year`, `Total_Births`
This is normalized to `Number` in the Streamlit EDA page and should be normalized similarly in the pipeline.

### Cost per child tables
External tables in:
- `data/external/grundskola_costs_per_child.csv`
- `data/external/gymnasieskola_costs_per_child.csv`

Expected columns:
- `Year`
- `Fixed_cost_per_child_kr`
- `Current_cost_per_child_kr`

Important:
- **Fixed vs Current are two different price bases.**
- For totals, choose **one basis** (do **NOT** add Fixed + Current).

---

## Pipeline Flow (High level)

1. **Load / clean historical births**
   - Ensure consistent region coding and year range (start 1968).
   - Persist to SQLite (`birth_data_per_region`) and/or keep as CSV.

2. **Train / select model per region**
   - Candidate models: baseline, regression, RF/XGB, ARIMA/ETS, etc.
   - Evaluate with a metric (e.g. RMSE).
   - Save:
     - Best model per region (`artifacts/metrics/best_models_births.csv`)
     - Summary (`artifacts/metrics/forecast_summary_births.csv`)
     - Optional full score table (`artifacts/metrics/model_scores_births.csv`)

3. **Forecast births 2024–2030**
   - Output:
     - `artifacts/forecasts/birth_forecast_2024_2030.csv`

4. **Population forecast 0–19**
   - Cohort-style transition:
     - Age 0 from births forecast
     - Age a+1 from previous age a (adjusted by mortality/migration if enabled)
   - Output:
     - `artifacts/forecasts/population_0_19_forecast_2024_2030.csv`

5. **Education costs forecast**
   - Convert population to students:
     - grundskola ages 7–16
     - gymnasieskola ages 17–19
   - Multiply by cost per child using the chosen **basis**.
   - Output:
     - `artifacts/forecasts/education_costs_forecast_2024_2030.csv`

---

## Outputs (Artifacts)

### Forecast files
- `birth_forecast_2024_2030.csv`
- `population_0_19_forecast_2024_2030.csv`
- `education_costs_forecast_2024_2030.csv`

### Metrics files
- `forecast_summary_births.csv`
- `best_models_births.csv`
- (optional) `model_scores_births.csv`

---

## Key Design Decisions

### 1) Price basis handling (critical)
Fixed and current are **not additive**. They represent the same cost in different price bases.

Recommended dashboard behavior:
- Show both series for comparison.
- But “Total” charts/KPIs must use **one selected basis**.

### 2) Region names
Some raw sources only have `Region_Code`. For dashboards and reports, ensure `Region_Name` is filled:
- Preferred: join against a region lookup table (SCB mapping).
- Fallback: show `Region_Code` as name.

### 3) Float outputs
All forecasted values should remain **float** to reflect statistical estimates.

---

## Where to Extend Next

- **Backtesting plots** per region (train on earlier years, predict later years).
- **Uncertainty intervals**:
  - model-based PI/CI for births
  - propagate uncertainty through cohort and costs
- **Migration + mortality realism**:
  - incorporate migration_data_per_region
  - incorporate mortality_data_per_region
- **PDF reporting**:
  - auto-generate a PDF with summary tables + charts + methods.
