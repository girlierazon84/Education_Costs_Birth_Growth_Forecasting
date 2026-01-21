"""src/eduforecast/validation/checks.py"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from eduforecast.validation.schemas import SchemaSpec, assert_schema


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    errors: tuple[str, ...]

    def raise_if_failed(self) -> None:
        if not self.ok:
            msg = "\n".join(self.errors) if self.errors else "Validation failed."
            raise ValueError(msg)


def _as_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _as_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def check_region_code_format(df: pd.DataFrame, col: str = "Region_Code") -> list[str]:
    errs: list[str] = []
    if col not in df.columns:
        return errs
    rc = df[col].astype("string").str.strip()
    bad = ~rc.str.fullmatch(r"\d{2}")
    n_bad = int(bad.sum())
    if n_bad:
        sample = rc[bad].dropna().unique()[:10].tolist()
        errs.append(f"{col}: expected 2-digit codes; bad_count={n_bad}; sample={sample}")
    return errs


def check_year_range(df: pd.DataFrame, *, col: str = "Year", min_year: int | None = None, max_year: int | None = None) -> list[str]:
    errs: list[str] = []
    if col not in df.columns:
        return errs
    y = _as_int_series(df[col])
    if min_year is not None:
        bad = y < int(min_year)
        if int(bad.sum()):
            errs.append(f"{col}: {int(bad.sum())} rows below min_year={min_year}")
    if max_year is not None:
        bad = y > int(max_year)
        if int(bad.sum()):
            errs.append(f"{col}: {int(bad.sum())} rows above max_year={max_year}")
    return errs


def check_age_range(df: pd.DataFrame, *, col: str = "Age", min_age: int = 0, max_age: int = 19) -> list[str]:
    errs: list[str] = []
    if col not in df.columns:
        return errs
    a = _as_int_series(df[col])
    bad = (a < int(min_age)) | (a > int(max_age))
    n_bad = int(bad.sum())
    if n_bad:
        sample = a[bad].dropna().unique()[:10].tolist()
        errs.append(f"{col}: expected [{min_age}..{max_age}]; bad_count={n_bad}; sample={sample}")
    return errs


def check_nonnegative(df: pd.DataFrame, cols: Sequence[str]) -> list[str]:
    errs: list[str] = []
    for c in cols:
        if c not in df.columns:
            continue
        x = _as_float_series(df[c])
        bad = x < 0
        n_bad = int(bad.sum())
        if n_bad:
            errs.append(f"{c}: {n_bad} negative values found")
    return errs


def check_unique_keys(df: pd.DataFrame, keys: Sequence[str]) -> list[str]:
    errs: list[str] = []
    missing = [k for k in keys if k not in df.columns]
    if missing:
        return errs

    dup_mask = df.duplicated(subset=list(keys), keep=False)
    n_dup = int(dup_mask.sum())
    if n_dup:
        sample = df.loc[dup_mask, list(keys)].head(10).to_dict(orient="records")
        errs.append(f"duplicate keys on {list(keys)}; dup_rows={n_dup}; sample={sample}")
    return errs


def validate_df(
    df: pd.DataFrame,
    *,
    schema: SchemaSpec | None = None,
    region_code_col: str | None = None,
    year_col: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    age_col: str | None = None,
    age_min: int = 0,
    age_max: int = 19,
    nonnegative_cols: Sequence[str] = (),
    unique_keys: Sequence[str] = (),
) -> CheckResult:
    """
    Generic validation runner.
    - validates required columns via schema (if provided)
    - validates region code format (if region_code_col)
    - validates year range (if year_col + bounds)
    - validates age range (if age_col)
    - validates nonnegativity and uniqueness (optional)
    """
    errors: list[str] = []

    if schema is not None:
        try:
            assert_schema(df, schema)
        except Exception as e:
            errors.append(str(e))
            # If schema fails, don't attempt downstream checks that may crash
            return CheckResult(ok=False, errors=tuple(errors))

    if region_code_col:
        errors.extend(check_region_code_format(df, col=region_code_col))

    if year_col:
        errors.extend(check_year_range(df, col=year_col, min_year=year_min, max_year=year_max))

    if age_col:
        errors.extend(check_age_range(df, col=age_col, min_age=age_min, max_age=age_max))

    if nonnegative_cols:
        errors.extend(check_nonnegative(df, cols=list(nonnegative_cols)))

    if unique_keys:
        errors.extend(check_unique_keys(df, keys=list(unique_keys)))

    return CheckResult(ok=(len(errors) == 0), errors=tuple(errors))


# ---- Convenience wrappers (match your canonical tables) ----

def validate_births_canonical(df: pd.DataFrame, *, start_year: int | None = None) -> CheckResult:
    from eduforecast.validation.schemas import BIRTHS_CANONICAL
    return validate_df(
        df,
        schema=BIRTHS_CANONICAL,
        region_code_col="Region_Code",
        year_col="Year",
        year_min=start_year,
        nonnegative_cols=("Number",),
        unique_keys=("Region_Code", "Year"),
    )


def validate_population_canonical(df: pd.DataFrame) -> CheckResult:
    from eduforecast.validation.schemas import POPULATION_CANONICAL
    return validate_df(
        df,
        schema=POPULATION_CANONICAL,
        region_code_col="Region_Code",
        year_col="Year",
        age_col="Age",
        age_min=0,
        age_max=19,
        nonnegative_cols=("Number",),
        unique_keys=("Region_Code", "Age", "Year"),
    )


def validate_mortality_canonical(df: pd.DataFrame) -> CheckResult:
    from eduforecast.validation.schemas import MORTALITY_CANONICAL
    return validate_df(
        df,
        schema=MORTALITY_CANONICAL,  # NOTE: if you copy-paste, fix name to MORTALITY_CANONICAL
        region_code_col="Region_Code",
        year_col="Year",
        age_col="Age",
        age_min=0,
        age_max=120,  # deaths can exceed 19 if your file contains all ages
        nonnegative_cols=("Number",),
        unique_keys=("Region_Code", "Age", "Year"),
    )
