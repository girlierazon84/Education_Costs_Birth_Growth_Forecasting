"""src/eduforecast/reporting/__init__.py"""

from __future__ import annotations

from .export import export_report_pack
from .plots import (
    plot_births_forecast_by_region,
    plot_costs_by_region_and_schooltype,
    plot_population_forecast_by_region_ageband,
)
from .tables import (
    make_births_summary_table,
    make_costs_summary_table,
    make_population_summary_table,
)

__all__ = [
    "export_report_pack",
    "plot_births_forecast_by_region",
    "plot_population_forecast_by_region_ageband",
    "plot_costs_by_region_and_schooltype",
    "make_births_summary_table",
    "make_population_summary_table",
    "make_costs_summary_table",
]
