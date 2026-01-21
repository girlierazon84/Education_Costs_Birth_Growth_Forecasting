"""tests/unit/test_features.py"""

from __future__ import annotations

import pandas as pd

from eduforecast.preprocessing.clean_births import clean_births
from eduforecast.preprocessing.clean_mortality import clean_mortality
from eduforecast.preprocessing.clean_population import clean_population


def test_clean_births_schema() -> None:
    raw = pd.DataFrame({"Region": ["1"], "År": [2023], "Total_Births": [123]})
    out = clean_births(raw)
    assert list(out.columns) == ["Region_Code", "Region_Name", "Year", "Number"]
    assert out["Region_Code"].iloc[0] == "01"
    assert out["Year"].iloc[0] == 2023
    assert float(out["Number"].iloc[0]) == 123.0


def test_clean_population_schema() -> None:
    raw = pd.DataFrame({"Region": ["01"], "Age": [7], "År": [2023], "Total_Population": [999]})
    out = clean_population(raw)
    assert list(out.columns) == ["Region_Code", "Region_Name", "Age", "Year", "Number"]
    assert out["Age"].iloc[0] == 7
    assert out["Year"].iloc[0] == 2023


def test_clean_mortality_schema() -> None:
    raw = pd.DataFrame({"Region": ["12"], "Age": [0], "År": [2023], "Total_Deaths": [5]})
    out = clean_mortality(raw)
    assert list(out.columns) == ["Region_Code", "Region_Name", "Age", "Year", "Number"]
    assert out["Region_Code"].iloc[0] == "12"
    assert out["Number"].iloc[0] == 5.0
