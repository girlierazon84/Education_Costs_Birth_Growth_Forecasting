"""src/eduforecast/reporting/export.py"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from eduforecast.reporting.plots import (
    plot_births_forecast_by_region,
    plot_costs_by_region_and_schooltype,
    plot_population_forecast_by_region_ageband,
)
from eduforecast.reporting.tables import (
    make_births_summary_table,
    make_costs_summary_table,
    make_population_summary_table,
)
from eduforecast.validation.checks import validate_df
from eduforecast.validation.schemas import BIRTHS_FORECAST, EDU_COSTS_FORECAST, POP_0_19_FORECAST


@dataclass(frozen=True)
class ReportPackPaths:
    out_dir: Path
    tables_dir: Path
    figures_dir: Path

    births_summary_csv: Path
    population_summary_csv: Path
    costs_summary_csv: Path


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def export_report_pack(
    *,
    births_forecast: pd.DataFrame,
    pop_forecast: pd.DataFrame,
    costs_forecast: pd.DataFrame,
    out_dir: Path,
    highlight_regions: Iterable[str] = ("01", "12", "14"),
) -> ReportPackPaths:
    """
    Build a reporting "pack":
        - summary CSV tables
        - a few high-signal PNG plots

    This module does NOT do forecasting. It consumes canonical outputs from pipelines.
    """
    out_dir = Path(out_dir)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    _ensure_dir(tables_dir)
    _ensure_dir(figures_dir)

    # Validate inputs (fail early)
    validate_df(births_forecast, schema=BIRTHS_FORECAST, region_code_col="Region_Code", year_col="Year").raise_if_failed()
    validate_df(pop_forecast, schema=POP_0_19_FORECAST, region_code_col="Region_Code", year_col="Year", age_col="Age").raise_if_failed()
    validate_df(costs_forecast, schema=EDU_COSTS_FORECAST, region_code_col="Region_Code", year_col="Year").raise_if_failed()

    # Tables
    births_summary = make_births_summary_table(births_forecast)
    pop_summary = make_population_summary_table(pop_forecast)
    costs_summary = make_costs_summary_table(costs_forecast)

    births_summary_csv = tables_dir / "births_summary.csv"
    population_summary_csv = tables_dir / "population_summary.csv"
    costs_summary_csv = tables_dir / "education_costs_summary.csv"

    births_summary.to_csv(births_summary_csv, index=False)
    pop_summary.to_csv(population_summary_csv, index=False)
    costs_summary.to_csv(costs_summary_csv, index=False)

    # Figures
    highlight_regions = [str(x).strip().zfill(2) for x in highlight_regions]

    plot_births_forecast_by_region(births_forecast, region_codes=highlight_regions, out_dir=figures_dir / "births")

    for rc in highlight_regions:
        plot_population_forecast_by_region_ageband(pop_forecast, region_code=rc, out_dir=figures_dir / "population")
        plot_costs_by_region_and_schooltype(costs_forecast, region_code=rc, out_dir=figures_dir / "costs", basis="fixed")
        plot_costs_by_region_and_schooltype(costs_forecast, region_code=rc, out_dir=figures_dir / "costs", basis="current")

    return ReportPackPaths(
        out_dir=out_dir,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        births_summary_csv=births_summary_csv,
        population_summary_csv=population_summary_csv,
        costs_summary_csv=costs_summary_csv,
    )
