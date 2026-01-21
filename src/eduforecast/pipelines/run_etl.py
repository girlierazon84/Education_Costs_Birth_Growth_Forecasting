"""src/eduforecast/pipelines/run_etl.py"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from eduforecast.common.config import AppConfig
from eduforecast.io.db import ensure_index, write_table
from eduforecast.io.readers import (
    read_births_raw,
    read_migration_raw,
    read_mortality_raw,
    read_population_raw,
)
from eduforecast.preprocessing.clean_births import clean_births
from eduforecast.preprocessing.clean_mortality import clean_mortality
from eduforecast.preprocessing.clean_population import clean_population

logger = logging.getLogger(__name__)


def _get(cfg_obj: Any, key: str, default: Any = None) -> Any:
    """Support both dict-style and attribute-style config."""
    if cfg_obj is None:
        return default
    if hasattr(cfg_obj, key):
        v = getattr(cfg_obj, key)
        return default if v is None else v
    if isinstance(cfg_obj, dict):
        return cfg_obj.get(key, default)
    return default


def _resolve(cfg: AppConfig, maybe_path: str | Path) -> Path:
    p = Path(maybe_path)
    return p if p.is_absolute() else (cfg.project_root / p).resolve()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _clean_migration(migration: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize migration to:
        Region_Code, Region_Name, Age, Year, Number

    NOTE: kept local for now. If you want, we can move this into:
        eduforecast.preprocessing.clean_migration
    """
    d = migration.copy()
    d.columns = [c.strip() for c in d.columns]

    d = d.rename(
        columns={
            "Region": "Region_Code",
            "region": "Region_Code",
            "Total_Migrations": "Number",
            "total_migrations": "Number",
            "År": "Year",
            "Ar": "Year",
            "year": "Year",
        }
    )

    required = {"Region_Code", "Age", "Year", "Number"}
    missing = required - set(d.columns)
    if missing:
        raise KeyError(f"Migration missing columns {sorted(missing)}. Found: {list(d.columns)}")

    d["Region_Code"] = (
        d["Region_Code"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(2)
    )

    if "Region_Name" not in d.columns:
        d["Region_Name"] = d["Region_Code"]
    d["Region_Name"] = d["Region_Name"].astype(str).str.strip()

    d["Age"] = pd.to_numeric(d["Age"], errors="coerce").astype("Int64")
    d["Year"] = pd.to_numeric(d["Year"], errors="coerce").astype("Int64")
    d["Number"] = pd.to_numeric(d["Number"], errors="coerce").astype(float)

    d = d.dropna(subset=["Age", "Year", "Number"]).copy()
    d["Age"] = d["Age"].astype(int)
    d["Year"] = d["Year"].astype(int)

    return (
        d[["Region_Code", "Region_Name", "Age", "Year", "Number"]]
        .sort_values(["Region_Code", "Age", "Year"])
        .reset_index(drop=True)
    )


def run_etl(cfg: AppConfig) -> None:
    raw_dir = _resolve(cfg, _get(cfg.paths, "raw_dir"))
    processed_dir = _resolve(cfg, _get(cfg.paths, "processed_dir"))
    external_dir = _resolve(cfg, _get(cfg.paths, "external_dir"))

    _ensure_dir(processed_dir)
    _ensure_dir(external_dir)

    sqlite_path = _get(cfg.database, "sqlite_path")
    if not sqlite_path:
        raise ValueError("Missing database.sqlite_path in config")
    db_path = _resolve(cfg, sqlite_path)

    # --- Load raw via IO layer (raw-only) ---
    births_raw = read_births_raw(raw_dir / "birth_data_per_region.csv")
    mortality_raw = read_mortality_raw(raw_dir / "mortality_data_per_region.csv")

    pop_0_16_raw = read_population_raw(raw_dir / "population_0_16_years.csv")
    pop_17_19_raw = read_population_raw(raw_dir / "population_17_19_years.csv")
    pop_all_raw = pd.concat([pop_0_16_raw, pop_17_19_raw], ignore_index=True)

    migration_raw = read_migration_raw(raw_dir / "migration_data_per_region.csv")

    # --- Clean using preprocessing package ---
    births_clean = clean_births(births_raw)
    mortality_clean = clean_mortality(mortality_raw)
    population_clean = clean_population(pop_all_raw)
    migration_clean = _clean_migration(migration_raw)

    # Derived region map (consistent)
    region_map = (
        pd.concat(
            [
                births_clean[["Region_Code", "Region_Name"]],
                mortality_clean[["Region_Code", "Region_Name"]],
                population_clean[["Region_Code", "Region_Name"]],
                migration_clean[["Region_Code", "Region_Name"]],
            ],
            ignore_index=True,
        )
        .drop_duplicates()
        .sort_values("Region_Code")
        .reset_index(drop=True)
    )

    # --- Save processed CSVs ---
    births_clean.to_csv(processed_dir / "births_processed.csv", index=False)
    mortality_clean.to_csv(processed_dir / "mortality_processed.csv", index=False)
    population_clean.to_csv(processed_dir / "population_processed.csv", index=False)
    migration_clean.to_csv(processed_dir / "migration_processed.csv", index=False)
    region_map.to_csv(processed_dir / "region_map.csv", index=False)

    # --- Write SQLite tables (via io.db) ---
    write_table(db_path, "birth_data_per_region", births_clean, if_exists="replace", index=False)
    write_table(db_path, "mortality_data_per_region", mortality_clean, if_exists="replace", index=False)
    write_table(db_path, "population_0_19_per_region", population_clean, if_exists="replace", index=False)
    write_table(db_path, "migration_data_per_region", migration_clean, if_exists="replace", index=False)

    # Helpful indexes for faster reads / joins
    ensure_index(db_path, "birth_data_per_region", ["Region_Code", "Year"], unique=False)
    ensure_index(db_path, "mortality_data_per_region", ["Region_Code", "Age", "Year"], unique=False)
    ensure_index(db_path, "population_0_19_per_region", ["Region_Code", "Age", "Year"], unique=False)
    ensure_index(db_path, "migration_data_per_region", ["Region_Code", "Age", "Year"], unique=False)

    # --- Light sanity checks ---
    assert births_clean["Region_Code"].astype(str).str.len().eq(2).all()
    assert population_clean["Age"].between(0, 19).all()

    logger.info("ETL complete. SQLite written to: %s", db_path)
    logger.info("Processed CSVs written to: %s", processed_dir)
