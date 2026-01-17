# Education Costs & Birth Growth Forecasting (EduForecast)

A Python forecasting pipeline + Streamlit dashboard for Swedish regional (län) trends:
- **Births forecasting** per region (historical from **1968**)
- **Population forecast (ages 0–19)** using cohort-style logic from births → ages
- **Education cost forecasts** for:
  - **Grundskola (7–16)**
  - **Gymnasieskola (17–19)**

Outputs are written as clean CSV artifacts under `artifacts/` for dashboards and reporting.

---

## Why you saw “110B SEK” for gymnasiet 2024 (and why that’s wrong to compare with SCB)

Your cost output contains **two price bases**:
- **Current (nominal)** total (e.g., “what SCB reports for 2024”)
- **Fixed (real)** total (same spending, but expressed in a fixed-price basis)

✅ For totals, you must choose **one basis** (Current OR Fixed).  
❌ Do **not** add Fixed + Current — that double counts.  
Your Streamlit pages now enforce a **basis selector** to prevent this.

When comparing to SCB totals like “~55B SEK for gymnasiet 2024”, use:
- **Current (nominal)**

---

## Repository layout

```

Education_Costs_Birth_Growth_Forecasting/
├─ .venv/                      # local virtual environment (not committed)
├─ artifacts/                  # outputs: forecasts, metrics, figures, logs
├─ configs/                    # YAML config (years, models, regions, costs settings)
├─ dashboards/                 # Streamlit multi-page app
├─ data/                       # raw/interim/processed/external datasets + sqlite db
├─ docs/                       # project documentation
├─ notebooks/                  # EDA notebooks and experiments
├─ scripts/                    # helper scripts
├─ src/                        # eduforecast package (pipeline + CLI)
├─ tests/                      # pytest tests
├─ .env                        # env vars (keep local; don’t commit secrets)
├─ .gitignore
├─ Makefile
├─ pyproject.toml
└─ README.md

````

---

## Main artifacts produced

After running the pipeline, you should have:

### Forecasts
- `artifacts/forecasts/birth_forecast_2024_2030.csv`
- `artifacts/forecasts/population_0_19_forecast_2024_2030.csv`
- `artifacts/forecasts/education_costs_forecast_2024_2030.csv`

### Metrics (examples)
- `artifacts/metrics/forecast_summary_births.csv`
- `artifacts/metrics/best_models_births.csv`
- (optional) `artifacts/metrics/model_scores_births.csv`

### Optional rollups you generated
- `artifacts/forecasts/education_costs_total_by_region_year.csv`
- `artifacts/forecasts/education_costs_national_totals.csv`

---

## Setup (Windows PowerShell)

### 1) Create + activate virtual environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
````

### 2) Install the project (editable)

From the repo root:

```powershell
pip install -e .
```

---

## Configure

Main config:

* `configs/config.yaml`

Important knobs:

* `modeling.start_year: 1968`
* `forecast.start_year / forecast.end_year`
* `regions.include`
* `costs.extrapolation` and `annual_growth_rate`
* external cost tables:

  * `data/external/grundskola_costs_per_child.csv`
  * `data/external/gymnasieskola_costs_per_child.csv`

---

## Run the pipeline

From the repo root:

```powershell
eduforecast forecast
```

This generates new CSVs in `artifacts/forecasts/` and summary outputs in `artifacts/metrics/`.

---

## Run Streamlit dashboard

Recommended entrypoint:

```powershell
streamlit run dashboards/Home.py
```

Pages:

* **Home**: navigation + file status
* **EDA**: births trends + cost table coverage (1968+)
* **Model Comparison**: best model per region (+ optional scores table)
* **Forecast & Costs**: births + population + cost exploration (**basis selector**)
* **Costs Dashboard** (optional): costs-focused view

---

## Using costs correctly (Current vs Fixed)

The education cost forecast output includes:

* `Fixed_Total_Cost_kr`
* `Current_Total_Cost_kr`

Interpretation:

* **Current_Total_Cost_kr** → compare to SCB totals for that year
* **Fixed_Total_Cost_kr** → compare across years in a constant-price style analysis

Rule:

* Use ONE basis at a time for totals, charts, and KPIs.

---

## Testing

```powershell
pytest
```

---

## Linting (Ruff)

```powershell
ruff check .
```

---

## Makefile shortcuts (if you use them)

Example:

```powershell
make help
make forecast
make dashboard
```

(Exact targets depend on your `Makefile`.)

---

## Roadmap

Order you chose:

1. ✅ Streamlit dashboard
2. PDF report export (summary + charts + method)
3. Model upgrades (backtesting + replace suspicious winners)
4. Improve realism (migration, mortality, cohort transitions, uncertainty intervals)

---

## License

Add a license if you want this to be open-source (MIT/Apache-2.0), or keep it private for portfolio use.
