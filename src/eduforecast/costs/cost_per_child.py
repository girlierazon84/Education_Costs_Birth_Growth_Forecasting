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

from eduforecast.preprocessing.clean_costs import clean_costs_per_child


CostBasis = Literal["fixed", "current"]


@dataclass(frozen=True)
class CostTables:
    grund: pd.DataFrame
    gymn: pd.DataFrame


def load_cost_tables(grund_path: Path, gymn_path: Path, *, anchor_max_year: int | None = None) -> CostTables:
    grund = clean_costs_per_child(pd.read_csv(grund_path))
    gymn = clean_costs_per_child(pd.read_csv(gymn_path))

    if anchor_max_year is not None:
        grund = grund[grund["Year"] <= int(anchor_max_year)].copy()
        gymn = gymn[gymn["Year"] <= int(anchor_max_year)].copy()

    return CostTables(grund=grund, gymn=gymn)


def extrapolate_costs(
    costs: pd.DataFrame,
    *,
    method: Literal["carry_forward", "growth_rate"] = "carry_forward",
    annual_growth_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Optional helper (currently minimal). Provided so you can keep extrapolation logic in one place.
    total_costs.py can call this later if you want.
    """
    d = clean_costs_per_child(costs)

    if method == "carry_forward":
        return d

    if method == "growth_rate":
        d = d.sort_values("Year").reset_index(drop=True)
        last_year = int(d["Year"].max())
        d["Cost_Year"] = d["Year"]

        yrs = (d["Year"] - d["Cost_Year"]).clip(lower=0)
        growth = (1.0 + float(annual_growth_rate)) ** yrs
        for col in ["Fixed_cost_per_child_kr", "Current_cost_per_child_kr"]:
            if col in d.columns:
                d[col] = d[col] * growth
        return d.drop(columns=["Cost_Year"], errors="ignore")

    raise ValueError(f"Unknown method: {method}")
