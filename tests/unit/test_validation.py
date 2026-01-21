"""tests/unit/test_validation.py"""

from __future__ import annotations

import pandas as pd
import pytest

from eduforecast.validation.checks import validate_df
from eduforecast.validation.schemas import COSTS_PER_CHILD_CANONICAL


def test_validate_df_passes_canonical_cost_schema() -> None:
    df = pd.DataFrame(
        {
            "Year": [2023, 2024],
            "Fixed_cost_per_child_kr": [100.0, 110.0],
            "Current_cost_per_child_kr": [200.0, 220.0],
        }
    )
    res = validate_df(df, schema=COSTS_PER_CHILD_CANONICAL, year_col="Year")
    res.raise_if_failed()  # should not raise


def test_validate_df_fails_on_missing_columns() -> None:
    df = pd.DataFrame({"Year": [2023]})
    res = validate_df(df, schema=COSTS_PER_CHILD_CANONICAL, year_col="Year")
    with pytest.raises(Exception):
        res.raise_if_failed()


def test_validate_df_fails_on_duplicates() -> None:
    df = pd.DataFrame(
        {
            "Year": [2023, 2023],
            "Fixed_cost_per_child_kr": [100.0, 100.0],
            "Current_cost_per_child_kr": [200.0, 200.0],
        }
    )
    res = validate_df(df, schema=COSTS_PER_CHILD_CANONICAL, year_col="Year", unique_keys=("Year",))
    with pytest.raises(Exception):
        res.raise_if_failed()


def test_validate_df_fails_on_negative() -> None:
    df = pd.DataFrame(
        {
            "Year": [2023],
            "Fixed_cost_per_child_kr": [-1.0],
            "Current_cost_per_child_kr": [200.0],
        }
    )
    res = validate_df(
        df,
        schema=COSTS_PER_CHILD_CANONICAL,
        year_col="Year",
        nonnegative_cols=("Fixed_cost_per_child_kr", "Current_cost_per_child_kr"),
    )
    with pytest.raises(Exception):
        res.raise_if_failed()
