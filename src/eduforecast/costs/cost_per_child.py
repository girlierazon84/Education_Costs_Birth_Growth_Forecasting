"""
src/eduforecast/costs/cost_per_child.py

Cost-per-child utilities.

Purpose:
- Load and standardize cost-per-child tables (grundskola / gymnasieskola).
- Provide cost extrapolation logic (carry-forward, growth-rate) in one place.
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

    Output schema:
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

    # Ensure expected columns exist (clean_costs_per_child may keep only what exists)
    for col in ["Fixed_cost_per_child_kr", "Current_cost_per_child_kr"]:
        if col not in d.columns:
            d[col] = pd.NA

    years = pd.DataFrame({"Year": list(range(start_year, end_year + 1))})

    # Use merge_asof to pick the latest cost row with Year <= target Year
    sched = pd.merge_asof(years, d, on="Year", direction="backward")

    # If start_year is earlier than first known cost year, forward-fill from earliest
    if sched["Fixed_cost_per_child_kr"].isna().all() and sched["Current_cost_per_child_kr"].isna().all():
        sched = pd.merge_asof(years, d, on="Year", direction="forward")
        sched["Cost_Year"] = int(d["Year"].min())
    else:
        # Determine which cost year was used by recomputing Cost_Year with the same asof join
        d_cost = d[["Year"]].rename(columns={"Year": "Cost_Year"})
        cost_year = pd.merge_asof(years, d_cost, left_on="Year", right_on="Cost_Year", direction="backward")

        # forward fill Cost_Year if needed (when target year < min cost year)
        if cost_year["Cost_Year"].isna().any():
            cost_year = pd.merge_asof(years, d_cost, left_on="Year", right_on="Cost_Year", direction="forward")

        sched["Cost_Year"] = pd.to_numeric(cost_year["Cost_Year"], errors="coerce")

    method = str(method).strip().lower()
    if method == "growth_rate":
        yrs = (pd.to_numeric(sched["Year"], errors="coerce") - pd.to_numeric(sched["Cost_Year"], errors="coerce")).clip(lower=0)
        yrs = yrs.fillna(0).astype(int)
        growth = (1.0 + float(annual_growth_rate)) ** yrs

        for col in ["Fixed_cost_per_child_kr", "Current_cost_per_child_kr"]:
            sched[col] = pd.to_numeric(sched[col], errors="coerce") * growth

    elif method != "carry_forward":
        raise ValueError(f"Unknown method: {method}")

    # Stable types
    sched["Year"] = pd.to_numeric(sched["Year"], errors="coerce").astype("Int64").astype(int)
    sched["Cost_Year"] = pd.to_numeric(sched["Cost_Year"], errors="coerce").astype("Int64").astype(int)

    cols = ["Year", "Fixed_cost_per_child_kr", "Current_cost_per_child_kr", "Cost_Year"]
    return sched[cols].sort_values("Year").reset_index(drop=True)
