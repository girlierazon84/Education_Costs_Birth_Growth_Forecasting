"""
scripts/backfill_db.py

Rebuild (backfill) the SQLite database from raw/external sources.

This script is designed to be safe and repeatable:
- Creates the DB if missing
- Drops and recreates core tables
- Loads births + (optionally) migration/mortality if you provide files later

Assumptions:
- config.yaml defines the SQLite path (default: data/processed/population_education_costs.db)
- births raw CSV exists: data/raw/birth_data_per_region.csv
- births raw schema may vary (e.g. Total_Births); we normalize to (Region_Code, Region_Name, Year, Number)
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
import sys

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config file: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def db_path_from_config(cfg: dict) -> Path:
    sqlite_rel = cfg.get("database", {}).get("sqlite_path", "data/processed/population_education_costs.db")
    return PROJECT_ROOT / sqlite_rel


def normalize_births(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize births to columns:
        Region_Code (string, zfill 2)
        Region_Name (string, fallback = Region_Code)
        Year (int)
        Number (float)
    Accepts common variants like Total_Births.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Rename common variants
    rename_map = {}
    for c in df.columns:
        if c in {"Total_Births", "total_births", "Births", "births", "Antal", "Value", "value"}:
            rename_map[c] = "Number"
    df = df.rename(columns=rename_map)

    # Required columns
    if "Region_Code" not in df.columns:
        raise KeyError(f"Births CSV missing Region_Code. Found: {list(df.columns)}")
    if "Year" not in df.columns:
        raise KeyError(f"Births CSV missing Year. Found: {list(df.columns)}")
    if "Number" not in df.columns:
        raise KeyError(f"Births CSV missing Number/Total_Births. Found: {list(df.columns)}")

    df["Region_Code"] = df["Region_Code"].astype("string").str.strip().str.zfill(2)
    if "Region_Name" in df.columns:
        df["Region_Name"] = df["Region_Name"].astype(str).str.strip()
    else:
        df["Region_Name"] = df["Region_Code"].astype(str)

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Number"] = pd.to_numeric(df["Number"], errors="coerce").astype(float)
    df = df.dropna(subset=["Year", "Number"]).copy()
    df["Year"] = df["Year"].astype(int)

    return df[["Region_Code", "Region_Name", "Year", "Number"]]


def backfill_births(con: sqlite3.Connection) -> None:
    births_path = PROJECT_ROOT / "data" / "raw" / "birth_data_per_region.csv"
    if not births_path.exists():
        raise FileNotFoundError(f"Missing births CSV: {births_path}")

    births = pd.read_csv(births_path)
    births = normalize_births(births)

    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS birth_data_per_region;")
    cur.execute(
        """
        CREATE TABLE birth_data_per_region (
            Region_Code TEXT NOT NULL,
            Region_Name TEXT NOT NULL,
            Year INTEGER NOT NULL,
            Number REAL NOT NULL
        );
        """
    )
    con.commit()

    births.to_sql("birth_data_per_region", con, if_exists="append", index=False)

    # Helpful indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_births_region_year ON birth_data_per_region(Region_Code, Year);")
    con.commit()


def main() -> int:
    cfg = load_config()
    db_path = db_path_from_config(cfg)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Backfilling SQLite DB: {db_path}")

    con = sqlite3.connect(db_path)
    try:
        backfill_births(con)
        print("✅ Backfill complete: birth_data_per_region loaded.")
    finally:
        con.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
