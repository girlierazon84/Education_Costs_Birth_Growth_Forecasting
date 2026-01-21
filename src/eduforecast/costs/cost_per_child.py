"""
src/eduforecast/costs/cost_per_child.py

Cost-per-child utilities.

Purpose:
- Load and standardize cost-per-child tables (grundskola / gymnasieskola).
- Provide a clean place to implement cost extrapolation logic later (carry-forward, growth-rate, CPI, etc).

This module is intentionally minimal for now.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from eduforecast.io.readers import read_costs_per_child_raw
from eduforecast.preprocessing.clean_costs import clean_costs_per_child


CostBasis = Literal["fixed", "current"]
ExtrapolationMethod = Literal["carry_forward", "growth_rate"]


@dataclass(frozen=True)
class CostTables:
    grund: pd.DataFrame
    gymn: pd.DataFrame


def load_cost_tables(
    grund_path: Path,
    gymn_path: Path,
    *,
    anchor_max_year: int | None = None,
) -> CostTables:
    """
    Load and standardize grundskola + gymnasieskola cost-per-child tables.

    Returns clean tables with schema:
        Year, Fixed_cost_per_child_kr, Current_cost_per_child_kr
    """
    grund_raw = read_costs_per_child_raw(grund_path)
    gymn_raw = read_costs_per_child_raw(gymn_path)

    grund = clean_costs_per_child(grund_raw)
    gymn = clean_costs_per_child(gymn_raw)

    if anchor_max_year is not None:
        grund = grund[grund["Year"] <= int(anchor_max_year)].copy()
        gymn = gymn[gymn["Year"] <= int(anchor_max_year)].copy()

    return CostTables(grund=grund.reset_index(drop=True), gymn=gymn.reset_index(drop=True))


def cost_schedule_for_years(
    costs: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
    method: ExtrapolationMethod = "carry_forward",
    annual_growth_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Build a cost schedule covering [start_year..end_year].

    Input:
        costs: standardized or raw; will be cleaned.
    Output:
        Year, Fixed_cost_per_child_kr, Current_cost_per_child_kr, Cost_Year

    Logic:
        - carry_forward: for each target Year, use latest known cost year <= Year
        - growth_rate: same as carry_forward, then apply (1+g)^(Year-Cost_Year)
    """
    start_year = int(start_year)
    end_year = int(end_year)
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")

    d = clean_costs_per_child(costs).sort_values("Year").reset_index(drop=True)
    if d.empty:
        raise ValueError("Cost table is empty after cleaning.")

    years = pd.DataFrame({"Year": list(range(start_year, end_year + 1))})

    # As-of backward join: Year gets last known costs <= Year
    base = pd.merge_asof(years, d, on="Year", direction="backward")

    # If forecast starts before first cost year, forward-fill from earliest cost
    if base["Fixed_cost_per_child_kr"].isna().any() or base["Current_cost_per_child_kr"].isna().any():
        base = pd.merge_asof(years, d, on="Year", direction="forward")

    # Determine which historical cost-year was used for each Year
    d2 = d.rename(columns={"Year": "Cost_Year"})
    base = pd.merge_asof(
        years,
        d2,
        left_on="Year",
        right_on="Cost_Year",
        direction="backward",
    )
    if base["Fixed_cost_per_child_kr"].isna().any() or base["Current_cost_per_child_kr"].isna().any():
        base = pd.merge_asof(
            years,
            d2,
            left_on="Year",
            right_on="Cost_Year",
            direction="forward",
        )

    # Apply growth rate relative to the referenced Cost_Year
    if method == "growth_rate":
        yrs = (base["Year"] - base["Cost_Year"]).clip(lower=0).astype(int)
        growth = (1.0 + float(annual_growth_rate)) ** yrs
        for col in ["Fixed_cost_per_child_kr", "Current_cost_per_child_kr"]:
            if col in base.columns:
                base[col] = pd.to_numeric(base[col], errors="coerce") * growth

    elif method != "carry_forward":
        raise ValueError(f"Unknown method: {method}")

    # Ensure stable columns
    cols = ["Year", "Fixed_cost_per_child_kr", "Current_cost_per_child_kr", "Cost_Year"]
    for c in cols:
        if c not in base.columns:
            base[c] = pd.NA

    base["Year"] = pd.to_numeric(base["Year"], errors="coerce").astype("Int64").astype(int)
    base["Cost_Year"] = pd.to_numeric(base["Cost_Year"], errors="coerce").astype("Int64").astype(int)

    return base[cols].sort_values("Year").reset_index(drop=True)
