"""src/eduforecast/costs/__init__.py"""

from __future__ import annotations

from .cost_per_child import (
    CostTables,
    ExtrapolationMethod,
    cost_schedule_for_years,
    load_cost_tables,
)
from .total_costs import compute_education_costs

__all__ = [
    "CostTables",
    "ExtrapolationMethod",
    "load_cost_tables",
    "cost_schedule_for_years",
    "compute_education_costs",
]
