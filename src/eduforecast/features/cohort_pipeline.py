"""src/eduforecast/features/cohort_pipeline.py"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def _read_table(db_path: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as con:
        return pd.read_sql(f"SELECT * FROM {table}", con)


def _read_population_0_19(db_path: Path) -> pd.DataFrame:
    """Read the already-combined population 0–19 table."""
    return _read_table(db_path, "population_0_19_per_region")


def build_survival_profile(
    pop_hist: pd.DataFrame,
    deaths_hist: pd.DataFrame,
    last_n_years: int = 5,
) -> pd.DataFrame:
    """
    Build survival rates per Region_Code, Age using the last N historical years available:
        survival(age) ≈ 1 - deaths(age)/population(age)
    Returned survival is used to age cohorts forward.
    """
    pop = pop_hist.copy()
    dea = deaths_hist.copy()

    for df in (pop, dea):
        df["Region_Code"] = df["Region_Code"].astype(str).str.zfill(2)
        df["Region_Name"] = df["Region_Name"].astype(str)
        df["Year"] = df["Year"].astype(int)
        df["Age"] = df["Age"].astype(int)
        df["Number"] = pd.to_numeric(df["Number"], errors="coerce").fillna(0.0)

    # Use last N years per region-age based on population years
    max_year = int(pop["Year"].max())
    use_years = list(range(max_year - last_n_years + 1, max_year + 1))

    pop_n = pop[pop["Year"].isin(use_years)].copy()
    dea_n = dea[dea["Year"].isin(use_years)].copy()

    merged = pop_n.merge(
        dea_n,
        on=["Region_Code", "Region_Name", "Age", "Year"],
        how="left",
        suffixes=("_pop", "_dead"),
    )
    merged["Number_dead"] = merged["Number_dead"].fillna(0.0)

    # survival per row
    with np.errstate(divide="ignore", invalid="ignore"):
        merged["survival"] = 1.0 - (merged["Number_dead"] / merged["Number_pop"])
    merged.loc[~np.isfinite(merged["survival"]), "survival"] = np.nan
    merged["survival"] = merged["survival"].clip(0.0, 1.0)

    # average survival over last N years per region-age
    surv = (
        merged.groupby(["Region_Code", "Region_Name", "Age"], as_index=False)["survival"]
        .mean()
        .rename(columns={"survival": "survival_rate"})
    )

    # fill any missing survival by Age mean (national)
    surv["survival_rate"] = surv["survival_rate"].fillna(
        surv.groupby(["Age"])["survival_rate"].transform("mean")
    )
    surv["survival_rate"] = surv["survival_rate"].fillna(1.0)
    return surv


def build_migration_profile(
    migration_hist: pd.DataFrame,
    start_year: int,
    end_year: int,
    last_n_years: int = 5,
) -> pd.DataFrame:
    """
    Migration is often noisy. For a baseline, hold net migration constant using
    the mean of the last N years available (per region-age), then apply it to all forecast years.
    """
    mig = migration_hist.copy()
    mig["Region_Code"] = mig["Region_Code"].astype(str).str.zfill(2)
    mig["Region_Name"] = mig["Region_Name"].astype(str)
    mig["Year"] = mig["Year"].astype(int)
    mig["Age"] = mig["Age"].astype(int)
    mig["Number"] = pd.to_numeric(mig["Number"], errors="coerce").fillna(0.0)

    max_year = int(mig["Year"].max())
    use_years = list(range(max_year - last_n_years + 1, max_year + 1))
    mig_n = mig[mig["Year"].isin(use_years)].copy()

    prof = (
        mig_n.groupby(["Region_Code", "Region_Name", "Age"], as_index=False)["Number"]
        .mean()
        .rename(columns={"Number": "net_migration_per_year"})
    )

    # Expand to all forecast years
    years = pd.DataFrame({"Year": list(range(start_year, end_year + 1))})
    prof = prof.merge(years, how="cross")
    return prof


def forecast_population_0_19(
    births_forecast: pd.DataFrame,
    db_path: Path,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    births_forecast: Region_Code, Region_Name, Year, Forecast_Births
    Returns: Region_Code, Region_Name, Age, Year, Forecast_Population (floats)
    """
    pop_hist = _read_population_0_19(db_path)
    deaths_hist = _read_table(db_path, "mortality_data_per_region")
    mig_hist = _read_table(db_path, "migration_data_per_region")

    # keep only ages 0-19 for survival/migration usage
    pop_hist = pop_hist[pop_hist["Age"].between(0, 19)].copy()
    deaths_hist = deaths_hist[deaths_hist["Age"].between(0, 19)].copy()
    mig_hist = mig_hist[mig_hist["Age"].between(0, 19)].copy()

    births = births_forecast.copy()
    births["Region_Code"] = births["Region_Code"].astype(str).str.zfill(2)
    births["Year"] = births["Year"].astype(int)
    births["Forecast_Births"] = pd.to_numeric(births["Forecast_Births"], errors="coerce").fillna(0.0)

    # Seed year = latest historical year available
    seed_year = int(pop_hist["Year"].max())
    seed = pop_hist[pop_hist["Year"] == seed_year].copy()
    seed["Region_Code"] = seed["Region_Code"].astype(str).str.zfill(2)
    seed["Forecast_Population"] = pd.to_numeric(seed["Number"], errors="coerce").fillna(0.0)
    seed = seed[["Region_Code", "Region_Name", "Age", "Year", "Forecast_Population"]]

    # Build profiles
    survival = build_survival_profile(pop_hist, deaths_hist, last_n_years=5)
    migration = build_migration_profile(mig_hist, start_year, end_year, last_n_years=5)

    # Index lookups
    births_map = births.set_index(["Region_Code", "Year"])["Forecast_Births"].to_dict()
    surv_map = survival.set_index(["Region_Code", "Age"])["survival_rate"].to_dict()
    mig_map = migration.set_index(["Region_Code", "Age", "Year"])["net_migration_per_year"].to_dict()

    out_all = []
    current = seed.copy()

    for year in range(start_year, end_year + 1):
        next_rows = []
        for (rc, rn), g in current.groupby(["Region_Code", "Region_Name"]):
            # Age 0 = births + migration(age0)
            b0 = float(births_map.get((rc, year), 0.0))
            mig0 = float(mig_map.get((rc, 0, year), 0.0))
            next_rows.append((rc, rn, 0, year, b0 + mig0))

            # Ages 1..19
            for age in range(1, 20):
                prev_age = age - 1
                prev_pop = float(g.loc[g["Age"] == prev_age, "Forecast_Population"].sum())
                s = float(surv_map.get((rc, prev_age), 1.0))
                mig_a = float(mig_map.get((rc, age, year), 0.0))
                next_rows.append((rc, rn, age, year, prev_pop * s + mig_a))

        year_df = pd.DataFrame(
            next_rows,
            columns=["Region_Code", "Region_Name", "Age", "Year", "Forecast_Population"],
        )
        out_all.append(year_df)
        current = year_df

    out = pd.concat(out_all, ignore_index=True).sort_values(["Region_Code", "Year", "Age"])
    return out.reset_index(drop=True)
