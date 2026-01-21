"""src/eduforecast/forecasting/predict_population.py"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from eduforecast.features.cohort_pipeline import forecast_population_0_19


def predict_population_0_19(
    births_forecast: pd.DataFrame,
    *,
    db_path: Path,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Produce population 0–19 forecast using cohort aging + survival + migration baseline.
    """
    return forecast_population_0_19(
        births_forecast=births_forecast,
        db_path=db_path,
        start_year=int(start_year),
        end_year=int(end_year),
    )
