"""src/eduforecast/forecasting/predict_mortality.py"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def predict_mortality_stub(
    *,
    db_path: Path,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Placeholder for mortality forecasting.

    Why keep this file?
    - Keeps a clean, future-proof API in your forecasting package
    - Lets you expand later without refactoring pipelines/dashboards

    Suggested future output schema:
        Region_Code, Region_Name, Age, Year, Forecast_Deaths
    """
    _ = db_path  # reserved for later use
    years = list(range(int(start_year), int(end_year) + 1))
    return pd.DataFrame({"Year": years})
