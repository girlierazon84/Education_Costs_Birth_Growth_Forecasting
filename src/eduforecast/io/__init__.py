"""src/eduforecast/io/__init__.py"""
from .db import connect, ensure_index, read_query, read_table, write_table
from .readers import (
    read_births_raw,
    read_costs_per_child_raw,
    read_csv,
    read_migration_raw,
    read_mortality_raw,
    read_population_raw,
)
from .writers import ensure_parent_dir, write_csv, write_forecast_artifact

__all__ = [
    "connect",
    "ensure_index",
    "read_query",
    "read_table",
    "write_table",
    "read_csv",
    "read_births_raw",
    "read_costs_per_child_raw",
    "read_mortality_raw",
    "read_population_raw",
    "read_migration_raw",
    "ensure_parent_dir",
    "write_csv",
    "write_forecast_artifact",
]
