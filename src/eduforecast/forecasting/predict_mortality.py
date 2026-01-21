"""src/eduforecast/forecasting/predict_mortality.py"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def predict_mortality_stub(*, db_path: Path, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Placeholder for mortality forecasting (kept for stable API).

    Future output schema:
        Region_Code, Region_Name, Age, Year, Forecast_Deaths
    """
    _ = db_path
    years = list(range(int(start_year), int(end_year) + 1))
    return pd.DataFrame({"Year": years})
