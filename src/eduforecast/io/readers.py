"""src/eduforecast/io/readers.py"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def read_csv(path: Path, *, dtype: dict[str, Any] | None = None, **kwargs: Any) -> pd.DataFrame:
    """
    Thin CSV reader wrapper.

    - Validates file exists
    - Pass-through to pandas.read_csv
    - Does NOT standardize columns (that belongs in preprocessing/)
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing file:\n{path}")
    return pd.read_csv(path, dtype=dtype, **kwargs)


def read_births_raw(path: Path) -> pd.DataFrame:
    """
    Read births CSV as-is (raw).

    Standardization happens in:
        eduforecast.preprocessing.clean_births.clean_births
    """
    return read_csv(path)


def read_costs_per_child_raw(path: Path) -> pd.DataFrame:
    """
    Read cost-per-child CSV as-is (raw).

    Standardization happens in:
        eduforecast.preprocessing.clean_costs.clean_costs_per_child
    """
    return read_csv(path)


def read_mortality_raw(path: Path) -> pd.DataFrame:
    """
    Read mortality CSV as-is (raw).

    Standardization happens in:
        eduforecast.preprocessing.clean_mortality.clean_mortality
    """
    return read_csv(path)


def read_population_raw(path: Path) -> pd.DataFrame:
    """
    Read population CSV as-is (raw).

    Standardization happens in:
        eduforecast.preprocessing.clean_population.clean_population
    """
    return read_csv(path)


def read_migration_raw(path: Path) -> pd.DataFrame:
    """
    Read migration CSV as-is (raw).

    Migration sometimes uses ';' delimiter, so we try that first.
    Standardization happens in ETL (or you can add clean_migration later).
    """
    try:
        return read_csv(path, sep=";")
    except Exception:
        return read_csv(path)
