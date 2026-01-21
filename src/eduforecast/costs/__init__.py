"""src/eduforecast/costs/__init__.py"""

from __future__ import annotations

from .cost_per_child import CostBasis, CostTables, extrapolate_costs, load_cost_tables
from .total_costs import compute_education_costs

__all__ = [
    "CostBasis",
    "CostTables",
    "load_cost_tables",
    "extrapolate_costs",
    "compute_education_costs",
]
