"""src/eduforecast/features/build_features.py"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from eduforecast.features.cohort_pipeline import forecast_population_0_19


def build_population_forecast_features(
    births_forecast: pd.DataFrame,
    *,
    db_path: Path,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Build the population forecast feature table used for cost calculations.

    Inputs
    ------
    births_forecast:
        DataFrame containing at minimum:
            Region_Code, Region_Name (optional), Year, Forecast_Births

    db_path:
        SQLite database path containing historical tables:
            - population_0_19_per_region (seed)
            - mortality_data_per_region
            - migration_data_per_region

    start_year / end_year:
        Forecast horizon for output.

    Returns
    -------
    DataFrame:
        Region_Code, Region_Name, Age, Year, Forecast_Population (floats)
        where Age in [0..19] and Year in [start_year..end_year].
    """
    pop_forecast = forecast_population_0_19(
        births_forecast=births_forecast,
        db_path=db_path,
        start_year=int(start_year),
        end_year=int(end_year),
    )

    # Strong schema normalization (so downstream is stable)
    pop_forecast = pop_forecast.copy()
    pop_forecast["Region_Code"] = pop_forecast["Region_Code"].astype("string").str.strip().str.zfill(2)
    pop_forecast["Region_Name"] = pop_forecast.get("Region_Name", pop_forecast["Region_Code"]).astype(str).str.strip()
    pop_forecast["Age"] = pd.to_numeric(pop_forecast["Age"], errors="coerce").astype("Int64")
    pop_forecast["Year"] = pd.to_numeric(pop_forecast["Year"], errors="coerce").astype("Int64")
    pop_forecast["Forecast_Population"] = pd.to_numeric(pop_forecast["Forecast_Population"], errors="coerce")

    pop_forecast = pop_forecast.dropna(subset=["Region_Code", "Year", "Age", "Forecast_Population"]).copy()
    pop_forecast["Age"] = pop_forecast["Age"].astype(int)
    pop_forecast["Year"] = pop_forecast["Year"].astype(int)

    # enforce bounds (defensive)
    pop_forecast = pop_forecast[
        (pop_forecast["Age"].between(0, 19))
        & (pop_forecast["Year"].between(int(start_year), int(end_year)))
    ].copy()

    # keep as floats ("statistical")
    pop_forecast["Forecast_Population"] = pop_forecast["Forecast_Population"].astype(float)

    return pop_forecast.sort_values(["Region_Code", "Year", "Age"]).reset_index(drop=True)
