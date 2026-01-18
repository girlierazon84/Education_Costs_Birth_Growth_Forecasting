# Data Dictionary

This document describes the main datasets (raw, processed, and artifact outputs) used in EduForecast.

---

## Conventions

- `Region_Code`: string, zero-padded (e.g. `"01"`, `"25"`)
- `Year`: integer (e.g. `1968`, `2024`)
- All forecasted quantities should remain **float** for statistical realism.

---

## Raw Data

### `data/raw/birth_data_per_region.csv`
Birth counts by region and year.

Common input variants:
- `Region_Code`, `Year`, `Total_Births`
- `Region_Code`, `Region_Name`, `Year`, `Number`

Standardized schema (what we want after cleaning):
| Column | Type | Example | Description |
|---|---:|---|---|
| Region_Code | string | "01" | SCB county code |
| Region_Name | string | "Stockholms län" | County name (optional in raw, should be filled) |
| Year | int | 1968 | Year |
| Number | float | 24650.0 | Total births |

---

## External Reference Data

### `data/external/grundskola_costs_per_child.csv`
Per-child cost for grundskola.

Standardized schema:
| Column | Type | Example | Description |
|---|---:|---|---|
| Year | int | 2021 | Year |
| Fixed_cost_per_child_kr | float | 133835.0 | Cost in fixed price base |
| Current_cost_per_child_kr | float | 123500.0 | Cost in current (nominal) SEK |

### `data/external/gymnasieskola_costs_per_child.csv`
Per-child cost for gymnasieskola.

Standardized schema:
| Column | Type | Example | Description |
|---|---:|---|---|
| Year | int | 2021 | Year |
| Fixed_cost_per_child_kr | float | 140120.0 | Cost in fixed price base |
| Current_cost_per_child_kr | float | 129300.0 | Cost in current (nominal) SEK |

Important:
- **Fixed vs Current are two different price bases.**
- Do **NOT** add them together.

---

## Processed / Database Tables (SQLite)

Database path (from config):
- `data/processed/population_education_costs.db`

Your current DB tables:
- `birth_data_per_region`
- `mortality_data_per_region`
- `migration_data_per_region`
- `population_0_19_per_region`

Typical standardized schemas:

### `birth_data_per_region`
| Column | Type | Description |
|---|---|---|
| Region_Code | TEXT | County code |
| Region_Name | TEXT | County name |
| Year | INTEGER | Year |
| Number | REAL | Birth count (float) |

### `mortality_data_per_region` (if used)
| Column | Type | Description |
|---|---|---|
| Region_Code | TEXT | County code |
| Region_Name | TEXT | County name |
| Year | INTEGER | Year |
| Age | INTEGER | Age |
| Number | REAL | Deaths or mortality measure (depends on your ETL definition) |

### `migration_data_per_region` (if used)
| Column | Type | Description |
|---|---|---|
| Region_Code | TEXT | County code |
| Region_Name | TEXT | County name |
| Year | INTEGER | Year |
| Age | INTEGER | Age (if available) |
| Number | REAL | Net migration or migration measure (depends on your ETL definition) |

### `population_0_19_per_region`
| Column | Type | Description |
|---|---|---|
| Region_Code | TEXT | County code |
| Region_Name | TEXT | County name |
| Year | INTEGER | Year |
| Age | INTEGER | Age 0–19 |
| Population | REAL | Population count (float) |

---

## Forecast Artifacts (CSV)

### `artifacts/forecasts/birth_forecast_2024_2030.csv`
| Column | Type | Description |
|---|---:|---|
| Region_Code | string | County code |
| Region_Name | string | County name |
| Year | int | Forecast year |
| Forecast_Births | float | Forecasted births |

### `artifacts/forecasts/population_0_19_forecast_2024_2030.csv`
| Column | Type | Description |
|---|---:|---|
| Region_Code | string | County code |
| Region_Name | string | County name |
| Age | int | 0–19 |
| Year | int | Forecast year |
| Forecast_Population | float | Forecasted population for age |

### `artifacts/forecasts/education_costs_forecast_2024_2030.csv`
| Column | Type | Description |
|---|---:|---|
| Region_Code | string | County code |
| Region_Name | string | County name |
| Year | int | Forecast year |
| School_Type | string | `grundskola` or `gymnasieskola` |
| Forecast_Students | float | Forecasted students (sum of relevant ages) |
| Fixed_Total_Cost_kr | float | Total cost in fixed price base |
| Current_Total_Cost_kr | float | Total cost in current (nominal) SEK |

Important:
- Fixed and Current represent **the same thing in different price bases**.
- For comparisons to SCB totals (e.g. 2024 spending), use **Current_Total_Cost_kr**.

---

## Metrics Artifacts (CSV)

### `artifacts/metrics/best_models_births.csv`
| Column | Type | Description |
|---|---|---|
| Region_Code | string | County code |
| Region_Name | string | County name |
| Best_Model | string | Selected model name |
| Metric | float (optional) | Best score |

### `artifacts/metrics/forecast_summary_births.csv`
Recommended fields:
| Column | Type | Description |
|---|---|---|
| Region_Code | string | County code |
| Region_Name | string | County name |
| Model | string | Best model used |
| RMSE / MAE / SMAPE | float | Metrics |
| Train_Years | string/int | Covered years |

---

## Common Validation Checks

- Region codes are **zero-padded** strings.
- No duplicate keys:
  - births: (Region_Code, Year)
  - population: (Region_Code, Age, Year)
  - costs: (Region_Code, Year, School_Type)
- Costs dashboards should not compute “Total” as Fixed + Current.
