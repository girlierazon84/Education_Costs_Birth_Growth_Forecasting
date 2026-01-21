"""
src/eduforecast/pipelines/run_etl.py
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from eduforecast.common.config import AppConfig


logger = logging.getLogger(__name__)


def _get(cfg_obj: Any, key: str, default: Any = None) -> Any:
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


def _zfill_region(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    # if numeric-looking, keep digits; otherwise keep as-is
    s = s.str.replace(r"\.0$", "", regex=True)
    return s.str.zfill(2)


def _write_sqlite(df: pd.DataFrame, db_path: Path, table: str) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        df.to_sql(table, con, if_exists="replace", index=False)


def _read_migration(path: Path) -> pd.DataFrame:
    # Prefer ';' but fallback to ','
    try:
        return pd.read_csv(path, sep=";")
    except Exception:
        return pd.read_csv(path)


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

    # --- Load raw ---
    births = pd.read_csv(raw_dir / "birth_data_per_region.csv")
    mortality = pd.read_csv(raw_dir / "mortality_data_per_region.csv")
    pop_0_16 = pd.read_csv(raw_dir / "population_0_16_years.csv")
    pop_17_19 = pd.read_csv(raw_dir / "population_17_19_years.csv")
    migration = _read_migration(raw_dir / "migration_data_per_region.csv")

    # --- Region map (prefer migration for names if present) ---
    if {"Region_Code", "Region_Name"}.issubset(migration.columns):
        region_map = (
            migration[["Region_Code", "Region_Name"]]
            .drop_duplicates()
            .assign(Region_Code=lambda d: _zfill_region(d["Region_Code"]))
        )
    else:
        region_map = pd.DataFrame(columns=["Region_Code", "Region_Name"])

    # --- Births ---
    births_clean = births.copy()
    births_clean = births_clean.rename(columns={"Region": "Region_Code", "Total_Births": "Number"})
    births_clean["Region_Code"] = _zfill_region(births_clean["Region_Code"])
    births_clean["Year"] = pd.to_numeric(births_clean["Year"], errors="coerce").astype("Int64")
    births_clean["Number"] = pd.to_numeric(births_clean["Number"], errors="coerce")

    births_clean = births_clean.dropna(subset=["Year", "Number"]).copy()
    births_clean["Year"] = births_clean["Year"].astype(int)
    births_clean["Number"] = births_clean["Number"].astype(float)

    births_clean = births_clean.merge(region_map, on="Region_Code", how="left")
    births_clean["Region_Name"] = births_clean.get("Region_Name", births_clean["Region_Code"]).astype(str).str.strip()
    births_clean = births_clean[["Region_Code", "Region_Name", "Year", "Number"]].sort_values(["Region_Code", "Year"])

    # --- Mortality ---
    mortality_clean = mortality.copy()
    mortality_clean = mortality_clean.rename(columns={"Region": "Region_Code", "Total_Deaths": "Number"})
    mortality_clean["Region_Code"] = _zfill_region(mortality_clean["Region_Code"])
    mortality_clean["Age"] = pd.to_numeric(mortality_clean["Age"], errors="coerce").astype("Int64")
    mortality_clean["Year"] = pd.to_numeric(mortality_clean["Year"], errors="coerce").astype("Int64")
    mortality_clean["Number"] = pd.to_numeric(mortality_clean["Number"], errors="coerce")

    mortality_clean = mortality_clean.dropna(subset=["Age", "Year", "Number"]).copy()
    mortality_clean["Age"] = mortality_clean["Age"].astype(int)
    mortality_clean["Year"] = mortality_clean["Year"].astype(int)
    mortality_clean["Number"] = mortality_clean["Number"].astype(float)

    mortality_clean = mortality_clean.merge(region_map, on="Region_Code", how="left")
    mortality_clean["Region_Name"] = mortality_clean.get("Region_Name", mortality_clean["Region_Code"]).astype(str).str.strip()
    mortality_clean = mortality_clean[["Region_Code", "Region_Name", "Age", "Year", "Number"]].sort_values(
        ["Region_Code", "Age", "Year"]
    )

    # --- Population 0–19 ---
    pop_all = pd.concat([pop_0_16, pop_17_19], ignore_index=True)
    population_clean = pop_all.copy()
    population_clean = population_clean.rename(columns={"Region": "Region_Code", "Total_Population": "Number"})
    population_clean["Region_Code"] = _zfill_region(population_clean["Region_Code"])
    population_clean["Age"] = pd.to_numeric(population_clean["Age"], errors="coerce").astype("Int64")
    population_clean["Year"] = pd.to_numeric(population_clean["Year"], errors="coerce").astype("Int64")
    population_clean["Number"] = pd.to_numeric(population_clean["Number"], errors="coerce")

    population_clean = population_clean.dropna(subset=["Age", "Year", "Number"]).copy()
    population_clean["Age"] = population_clean["Age"].astype(int)
    population_clean["Year"] = population_clean["Year"].astype(int)
    population_clean["Number"] = population_clean["Number"].astype(float)

    population_clean = population_clean.merge(region_map, on="Region_Code", how="left")
    population_clean["Region_Name"] = population_clean.get("Region_Name", population_clean["Region_Code"]).astype(str).str.strip()
    population_clean = population_clean[["Region_Code", "Region_Name", "Age", "Year", "Number"]].sort_values(
        ["Region_Code", "Age", "Year"]
    )

    # --- Migration ---
    migration_clean = migration.copy()
    migration_clean = migration_clean.rename(columns={"Total_Migrations": "Number"})
    if "Region_Code" not in migration_clean.columns:
        migration_clean = migration_clean.rename(columns={"Region": "Region_Code"})

    migration_clean["Region_Code"] = _zfill_region(migration_clean["Region_Code"])
    migration_clean["Age"] = pd.to_numeric(migration_clean["Age"], errors="coerce").astype("Int64")
    migration_clean["Year"] = pd.to_numeric(migration_clean["Year"], errors="coerce").astype("Int64")
    migration_clean["Number"] = pd.to_numeric(migration_clean["Number"], errors="coerce")

    migration_clean = migration_clean.dropna(subset=["Age", "Year", "Number"]).copy()
    migration_clean["Age"] = migration_clean["Age"].astype(int)
    migration_clean["Year"] = migration_clean["Year"].astype(int)
    migration_clean["Number"] = migration_clean["Number"].astype(float)

    if "Region_Name" not in migration_clean.columns:
        migration_clean = migration_clean.merge(region_map, on="Region_Code", how="left")
    migration_clean["Region_Name"] = migration_clean.get("Region_Name", migration_clean["Region_Code"]).astype(str).str.strip()

    migration_clean = migration_clean[["Region_Code", "Region_Name", "Age", "Year", "Number"]].sort_values(
        ["Region_Code", "Age", "Year"]
    )

    # --- Save processed CSVs ---
    births_clean.to_csv(processed_dir / "births_processed.csv", index=False)
    mortality_clean.to_csv(processed_dir / "mortality_processed.csv", index=False)
    population_clean.to_csv(processed_dir / "population_processed.csv", index=False)
    migration_clean.to_csv(processed_dir / "migration_processed.csv", index=False)
    region_map.sort_values("Region_Code").to_csv(processed_dir / "region_map.csv", index=False)

    # --- Write SQLite tables ---
    _write_sqlite(births_clean, db_path, "birth_data_per_region")
    _write_sqlite(mortality_clean, db_path, "mortality_data_per_region")
    _write_sqlite(population_clean, db_path, "population_0_19_per_region")
    _write_sqlite(migration_clean, db_path, "migration_data_per_region")

    # --- Light sanity checks ---
    assert births_clean["Region_Code"].astype(str).str.len().eq(2).all()
    assert population_clean["Age"].between(0, 19).all()

    logger.info("ETL complete. SQLite written to: %s", db_path)
    logger.info("Processed CSVs written to: %s", processed_dir)
