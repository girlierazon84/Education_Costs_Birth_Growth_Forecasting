# Modeling Notes

This document captures modeling assumptions, evaluation approach, and known limitations for the EduForecast pipeline.

---

## Historical Range

- Births historical data start year: **1968**
- Forecast horizon: typically **2024–2030** (configurable)

---

## Modeling Strategy (Birth Forecast)

### Per-region forecasting
- Models are trained and evaluated **per region**.
- Each region can select a different best model depending on historical patterns.

### Candidate models (examples)
- Naive baseline (last value / drift)
- Linear Regression
- Random Forest / Gradient Boosting / XGBoost (if included)
- ARIMA / SARIMA (for time series)
- Exponential Smoothing (ETS/Holt-Winters)

### Typical evaluation metrics
- RMSE (primary)
- MAE
- SMAPE / MAPE (optional)

### Validation approach
Prefer time-series validation:
- Expanding window or rolling-origin backtesting
- Avoid random shuffles

---

## Cohort Population Forecast (0–19)

### Core logic (simplified)
- Age 0 in year t comes from forecasted births for year t.
- Age a in year t is derived from age a-1 in year t-1.

### What makes this “more realistic”
- Mortality adjustments by age
- Migration adjustments by age/region
- Smoothing extreme jumps

### Current status
- If you are not yet applying mortality/migration, the cohort transitions are deterministic and may overestimate.

---

## Education Costs Forecast

### Student mapping (current default)
- Grundskola: ages 7–16
- Gymnasieskola: ages 17–19

These age bands can be made configurable.

### Cost table use
- Cost per child is taken from external tables (up to 2022 in your file).
- Future years are extrapolated:
  - carry-forward OR
  - growth-rate (e.g. 2.5% per year)

### Fixed vs Current price bases (important)
- **Fixed_Total_Cost_kr**: expressed in a fixed price base (real terms)
- **Current_Total_Cost_kr**: expressed in nominal SEK
- These are **not additive**.
  - If you add them, you double-count the same cost.
- When comparing to SCB totals for a given year (e.g. “55B in 2024”), use:
  - **Current_Total_Cost_kr** (nominal)

---

## Why your totals can differ from SCB

Even if the pipeline is internally consistent, totals can differ because:

1. **Coverage mismatch**
   - SCB’s definition of “gymnasieskola cost” may include/exclude:
     - special programs
     - adult education
     - administrative overhead handled elsewhere
   - Your model assumes costs are proportional to student counts only.

2. **Population vs enrolled students**
   - You forecast population age 17–19 and treat it as “students”.
   - Actual gymnasium enrollment is less than 100% of that cohort.

3. **Cost-per-child table**
   - Your cost-per-child table might not match the same definition as SCB’s “total cost”.
   - If per-child costs include different components, totals will differ.

4. **Migration/mortality not fully modeled**
   - Can cause inflated cohort sizes.

5. **Extrapolation**
   - Growth-rate extrapolation introduces compounding differences.

---

## Recommended Improvements (next steps)

### 1) Add enrollment ratio for gymnasieskola
Create a multiplier per region/year:
- `enrollment_rate_gymn` (e.g. 0.85–0.95)
Then:
- `Forecast_Students_gymn = population_17_19 * enrollment_rate_gymn`

### 2) Add backtesting plots
For each region:
- train up to year T
- predict T+1..T+k
- compare with actual
This catches “suspicious winners”.

### 3) Uncertainty intervals
- births: prediction intervals from model (or residual bootstrap)
- propagate into population and costs with Monte Carlo

### 4) Migration + mortality integration
- Use `migration_data_per_region` and `mortality_data_per_region`
- Apply to cohort transitions:
  - `pop[a,t] = pop[a-1,t-1] * (1 - mort_rate[a-1,t-1]) + net_mig[a,t]`

---

## Sanity Checks

- Compare national totals to known order-of-magnitude benchmarks.
- Ensure dashboards never compute total cost as Fixed + Current.
- Check region totals and outliers for sudden discontinuities.
