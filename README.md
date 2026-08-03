# Education Costs & Birth Growth Forecasting (EduForecast)

This repository houses a localized data engineering and analytics framework built to project demographic shifts and public education expenditures across Sweden’s 21 counties (*län*).

The system replaces generic, global trends with regionalized models that isolate unique geographic profiles—such as fast-expanding commuter belts (Stockholm, Uppsala, Halland) versus rural areas facing long-term population decline—without relying on unvetted external inflation tracking tables.

---

## Core Predictive Architecture

The pipeline processes demographic and financial data through four interconnected modules:

1. **Regional Time-Series Optimization**: Models births per region using historical observations starting from **1968**. The engine handles urbanization trends dynamically, applying optimized **Exponential Smoothing (ETS)** with automated trend damping for major municipal hubs. For stable or declining industrial regions, it utilizes a tailored **Damped Drift Model** to prevent long-range linear forecasts from dropping below zero or escalating into impossible figures.
2. **Geographic Guardrails (Strategic Overrides)**: Built-in business logic overrides purely mathematical cross-validation selections in volatile overflow zones like Stockholm, Uppsala, and Halland. This prevents short-term noise from forcing a flat horizontal line (*baseline_naive*), which would result in a severe underestimation of future student populations and municipal infrastructure funding.
3. **Closed Cohort-Component Progression**: Future student enrollment is built step-by-step rather than using blind extrapolation. The system ages existing regional populations forward annually by tracking age-specific survival parameters and net migration vectors (natively supporting negative net migration values for shrinking areas).
4. **Expenditure Escalation Engine**: Projects specific cost frameworks by aggregating population results into strict statutory school categories:
   - **Grundskola (Ages 7–16)**
   - **Gymnasieskola (Ages 17–19)**

---

## Price Bases & Financial Verification (Current vs. Fixed)

The financial outputs evaluate upcoming municipal framework budgets across two alternate economic paths to prevent double-counting or miscalculation:
* **Current (Nominal)**: Compounds real-world unit parameters forward by your configured growth rate (e.g., compounding a 2.5% inflation rate from a capped historical anchor year like 2021). **This is the correct basis to use when validating model outputs directly against actual Skolverket or SCB nominal records.**
* **Fixed (Real)**: Strips out inflation compounding entirely, holding unit costs locked to the anchored history baseline year. This isolates structural student volume changes over time.

⚠️ **Important**: When interacting with the Streamlit metrics, summary charts, and KPI displays, always select **one single basis** at a time. Do **not** sum Fixed and Current metrics together, as this double-counts expenditures (which explains the incorrect 110B SEK total observed in previous unconstrained iterations).

---

## Directory Layout

```text
Education_Costs_Birth_Growth_Forecasting/
├─ artifacts/                  # Generated assets: forecasts, metrics, and plots
│  ├─ figures/                 # Diagnostic cross-validation plots and trend views
│  ├─ forecasts/               # Clean target data arrays mapped directly to dashboards
│  └─ metrics/                 # Detailed error scoring matrices and selection registries
├─ configs/                    # Pipeline configuration layouts
│  ├─ params/                  # Component parameter configurations (features, modeling, costs)
│  └─ config.yaml              # Centralized master runtime parameters and switches
├─ dashboards/                 # High-contrast, minimalist Streamlit analytics web app
│  └─ pages/                   # Tab-navigated standalone evaluation sub-modules
├─ data/                       # Ingestion layers and database tiers
│  ├─ external/                # Historical unit costs per child sheets from Skolverket
│  ├─ processed/               # Compound-indexed production SQLite relational database
│  └─ raw/                     # Multi-decade raw SCB demographic csv extracts
├─ src/                        # Production system package namespace
│  └─ eduforecast/
│     ├─ common/               # Hierarchical YAML configuration loaders and logger setups
│     ├─ costs/                # Float-enforced budget calculators and financial matrices
│     ├─ forecasting/          # Protocol models, legacy shims, and prediction interval math
│     ├─ features/             # Recursive component cohort matrix aging progressions
│     └─ pipelines/            # Decoupled data re-indexing, model training, and forecasting steps
├─ tests/                      # Python testing layout via pytest
├─ pyproject.toml              # Modern Python package management setup
└─ README.md
```

---

## Primary Pipeline Assets

Once the calculations are executed, the following structured files are written under `artifacts/`:

### Forecast Arrays (`artifacts/forecasts/`)
* `birth_forecast_2024_2030.csv` — *Future birth counts per county*
* `population_0_19_forecast_2024_2030.csv` — *Year-by-year age progression matrices*
* `education_costs_forecast_2024_2030.csv` — *Final regional budget projections*

### Evaluation Matrices (`artifacts/metrics/`)
* `best_models_births.csv` — *The final assigned production model tracking ledger*
* `model_scores_births.csv` — *Granular cross-validation metrics (RMSE, MAE, SMAPE)*
* `forecast_summary_births.csv` — *Aggregated forecasting parameters (min/max/mean)*

---

## Local Setup & Environment Configuration

### 1) Initialize the Virtual Environment (Windows PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Install Project Dependencies (Editable Mode)
Run this command from the repository root directory to map your local workspace paths:
```powershell
pip install -e .
```

---

## Pipeline Execution (CLI)

The system provides a clear command-line interface (CLI) via `typer` to trigger separate steps of the execution loop or run everything at once:

### Run the Complete Data Chain
This runs the entire system from raw data processing to final cost evaluation:
```powershell
eduforecast run-all
```

### Run Pipeline Stages Separately
If you only want to process specific steps, you can call them independently:
```powershell
# 1. Clean raw text tables and build the index-optimized SQLite database
eduforecast etl

# 2. Run historical cross-validation and serialize optimized model objects
eduforecast train

# 3. Generate future population counts and compute upcoming education budgets
eduforecast forecast
```

---

## Launch the Interactive Analytics Views

Charts and KPIs are split into intuitive, tab-based layouts using high-contrast dark slate typography (`#334155`) for readable, clean rendering against the white plot backgrounds.

```powershell
streamlit run dashboards/Home.py
```

### Navigational Framework:
* **Home Page**: System file presence logs and pipeline dependency validation status.
* **Exploratory Data Analysis (EDA)**: Deep-dive evaluation of historical birth indicators (1968+) and cost-per-child changes over multiple decades.
* **Model Optimization Matrix**: Full transparency panel displaying algorithm selection profiles, backtest metric histograms, and active demographic override log entries.
* **Projections & Framework Budgets**: Dynamic mapping of student volume changes, unit-cost curves, and overall spending trends equipped with an explicit cost basis toggle.

---

## Quality Assurance & Automated Testing

The system enforces data type validation across all internal operations via custom data structure checks (`validate_df`). These catch malformed row parameters or misaligned integer/float columns before data is passed to downstream pipelines.

```powershell
# Run the automated unit testing layout
pytest

# Enforce uniform styling rules and static analysis
ruff check .
```

---

## Project Roadmap

1.  ✅ **Tab-Based Navigation & Layout Refresh** (Improved UX and typography contrast)
2.  ✅ **Strict Data Contract Verification** (Stabilized for server-side environments)
3.  ⬜ **Automated PDF Executive Reporting** (Summary generation with inline static charts)
4.  ⬜ **Variance Modeling Upgrades** (Integrating rolling historical prediction intervals)

---

## License

This project is configured as a private portfolio asset. For questions regarding reuse, licensing parameters, or deployment blueprints, contact the repository owner.
