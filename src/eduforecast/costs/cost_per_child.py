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
import numpy as np

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

    # Create a local copy to isolate modifications safely
    d = d.copy()

    # ✅ FIX 2: Attach Cost_Year directly to the source dataframe before matching
    d["Cost_Year"] = d["Year"].astype(int)

    # ✅ FIX 1: Enforce explicit floating-point initialization instead of pd.NA
    for col in ["Fixed_cost_per_child_kr", "Current_cost_per_child_kr"]:
        if col not in d.columns:
            d[col] = np.nan
        else:
            d[col] = pd.to_numeric(d[col], errors="coerce").astype(float)

    years = pd.DataFrame({"Year": list(range(start_year, end_year + 1))})

    # Primary match: Locate the latest cost configuration with historical Year <= forecast target Year
    sched = pd.merge_asof(years, d, on="Year", direction="backward")

    # ✅ FIX 2 (Continued): Handle edge cases where target forecast start_year < oldest database year
    if sched["Cost_Year"].isna().any():
        forward_sched = pd.merge_asof(years, d, on="Year", direction="forward")
        missing_mask = sched["Cost_Year"].isna()
        sched.loc[missing_mask, :] = forward_sched.loc[missing_mask, :]

    method = str(method).strip().lower()
    if method == "growth_rate":
        # Calculate years elapsed between the forecast point and the reference cost anchor year
        yrs = (sched["Year"].astype(int) - sched["Cost_Year"].astype(int)).clip(lower=0)
        growth = (1.0 + float(annual_growth_rate)) ** yrs

        # Compounded calculation runs safely on explicit floating-point arrays
        for col in ["Fixed_cost_per_child_kr", "Current_cost_per_child_kr"]:
            sched[col] = sched[col].astype(float) * growth

    elif method != "carry_forward":
        raise ValueError(f"Unknown method: {method}")

    # Solidify strict casting conventions to pass downstream EDU_COSTS_FORECAST schema validators
    sched["Year"] = sched["Year"].astype(int)
    sched["Cost_Year"] = sched["Cost_Year"].astype(int)
    sched["Fixed_cost_per_child_kr"] = pd.to_numeric(sched["Fixed_cost_per_child_kr"], errors="coerce").astype(float)
    sched["Current_cost_per_child_kr"] = pd.to_numeric(sched["Current_cost_per_child_kr"], errors="coerce").astype(float)

    cols = ["Year", "Fixed_cost_per_child_kr", "Current_cost_per_child_kr", "Cost_Year"]
    return sched[cols].sort_values("Year").reset_index(drop=True)
