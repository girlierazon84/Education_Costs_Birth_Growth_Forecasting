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
from typing import Literal, Tuple

import pandas as pd


CostBasis = Literal["fixed", "current"]


@dataclass(frozen=True)
class CostTables:
    grund: pd.DataFrame
    gymn: pd.DataFrame


def load_cost_tables(
    grund_path: Path,
    gymn_path: Path,
) -> CostTables:
    """
    Placeholder loader.

    Expected standardized columns (future):
      - Year (int)
      - Fixed_cost_per_child_kr (float)
      - Current_cost_per_child_kr (float)

    For now, just reads CSVs as-is.
    """
    grund = pd.read_csv(grund_path)
    gymn = pd.read_csv(gymn_path)
    return CostTables(grund=grund, gymn=gymn)


def extrapolate_costs(
    costs: pd.DataFrame,
    *,
    method: Literal["carry_forward", "growth_rate"] = "carry_forward",
    annual_growth_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Placeholder extrapolation (no-op for now).

    Implement later:
    - carry_forward: use last known year’s costs for future years
    - growth_rate: grow from last known year by annual_growth_rate
    """
    _ = (method, annual_growth_rate)
    return costs.copy()
