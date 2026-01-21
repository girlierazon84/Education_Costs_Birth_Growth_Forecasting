"""tests/conftest.py"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


@dataclass
class _Obj:
    """Simple attribute container (duck-typed config sections)."""
    __dict__: dict[str, Any]

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


@dataclass
class MinimalAppConfig:
    """
    Duck-typed stand-in for eduforecast.common.config.AppConfig.
    Only includes what pipelines use.
    """
    project_root: Path
    paths: _Obj
    database: _Obj
    forecast: _Obj
    regions: _Obj
    raw: _Obj
    costs: _Obj


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    # Emulate repository root in temp dir
    (tmp_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "external").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts" / "metrics").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts" / "forecasts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts" / "figures").mkdir(parents=True, exist_ok=True)
    return tmp_path


def make_minimal_config(project_root: Path) -> MinimalAppConfig:
    paths = _Obj(
        raw_dir="data/raw",
        processed_dir="data/processed",
        external_dir="data/external",
        metrics_dir="artifacts/metrics",
        forecasts_dir="artifacts/forecasts",
        figures_dir="artifacts/figures",
    )
    database = _Obj(sqlite_path="artifacts/eduforecast_test.db")
    forecast = _Obj(start_year=2024, end_year=2025, interval_level=0.95, export_report_pack=False)
    regions = _Obj(include=None)
    raw = _Obj(costs={})
    costs = _Obj(
        grundskola_cost_table="data/external/grundskola_costs_per_child.csv",
        gymnasieskola_cost_table="data/external/gymnasieskola_costs_per_child.csv",
        anchor_max_year=None,
        extrapolation="carry_forward",
        annual_growth_rate=0.0,
    )
    return MinimalAppConfig(
        project_root=project_root,
        paths=paths,
        database=database,
        forecast=forecast,
        regions=regions,
        raw=raw,
        costs=costs,
    )


@pytest.fixture
def cfg(project_root: Path) -> MinimalAppConfig:
    return make_minimal_config(project_root)


def write_minimal_raw_data(raw_dir: Path) -> None:
    """
    Writes minimal raw CSVs expected by run_etl:
      - birth_data_per_region.csv
      - mortality_data_per_region.csv
      - population_0_16_years.csv
      - population_17_19_years.csv
      - migration_data_per_region.csv
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    births = pd.DataFrame(
        {
            "Region": ["01", "01", "12", "12"],
            "År": [2023, 2024, 2023, 2024],
            "Total_Births": [1000, 1100, 900, 950],
        }
    )
    births.to_csv(raw_dir / "birth_data_per_region.csv", index=False)

    mortality = pd.DataFrame(
        {
            "Region": ["01", "01", "12", "12"],
            "Age": [0, 1, 0, 1],
            "År": [2023, 2023, 2023, 2023],
            "Total_Deaths": [5, 2, 4, 1],
        }
    )
    mortality.to_csv(raw_dir / "mortality_data_per_region.csv", index=False)

    pop_0_16 = pd.DataFrame(
        {
            "Region": ["01", "01", "12", "12"],
            "Age": [7, 16, 7, 16],
            "År": [2023, 2023, 2023, 2023],
            "Total_Population": [10000, 9000, 8000, 7000],
        }
    )
    pop_0_16.to_csv(raw_dir / "population_0_16_years.csv", index=False)

    pop_17_19 = pd.DataFrame(
        {
            "Region": ["01", "12"],
            "Age": [17, 17],
            "År": [2023, 2023],
            "Total_Population": [3000, 2500],
        }
    )
    pop_17_19.to_csv(raw_dir / "population_17_19_years.csv", index=False)

    migration = pd.DataFrame(
        {
            "Region": ["01", "12"],
            "Age": [10, 10],
            "År": [2023, 2023],
            "Total_Migrations": [100, 80],
        }
    )
    migration.to_csv(raw_dir / "migration_data_per_region.csv", index=False)


def write_minimal_cost_tables(external_dir: Path) -> None:
    external_dir.mkdir(parents=True, exist_ok=True)

    grund = pd.DataFrame(
        {
            "Year": [2023, 2024],
            "Fixed_cost_per_child_kr": [100000, 101000],
            "Current_cost_per_child_kr": [105000, 107000],
        }
    )
    grund.to_csv(external_dir / "grundskola_costs_per_child.csv", index=False)

    gymn = pd.DataFrame(
        {
            "Year": [2023, 2024],
            "Fixed_cost_per_child_kr": [120000, 121000],
            "Current_cost_per_child_kr": [126000, 128000],
        }
    )
    gymn.to_csv(external_dir / "gymnasieskola_costs_per_child.csv", index=False)


@pytest.fixture
def minimal_data(cfg: MinimalAppConfig) -> MinimalAppConfig:
    write_minimal_raw_data(cfg.project_root / "data" / "raw")
    write_minimal_cost_tables(cfg.project_root / "data" / "external")
    return cfg
