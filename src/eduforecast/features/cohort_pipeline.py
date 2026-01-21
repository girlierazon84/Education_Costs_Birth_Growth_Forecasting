"""src/eduforecast/features/cohort_pipeline.py"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eduforecast.io.db import read_table


def _read_population_0_19(db_path: Path) -> pd.DataFrame:
    return read_table(db_path, "population_0_19_per_region")


def build_survival_profile(pop_hist: pd.DataFrame, deaths_hist: pd.DataFrame, last_n_years: int = 5) -> pd.DataFrame:
    pop = pop_hist.copy()
    dea = deaths_hist.copy()

    for df in (pop, dea):
        df["Region_Code"] = df["Region_Code"].astype("string").str.strip().str.zfill(2)
        df["Region_Name"] = df.get("Region_Name", df["Region_Code"]).astype(str).str.strip()
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce").astype("Int64")
        df["Number"] = pd.to_numeric(df["Number"], errors="coerce").fillna(0.0)

    pop = pop.dropna(subset=["Year", "Age"]).copy()
    dea = dea.dropna(subset=["Year", "Age"]).copy()
    pop["Year"] = pop["Year"].astype(int)
    dea["Year"] = dea["Year"].astype(int)
    pop["Age"] = pop["Age"].astype(int)
    dea["Age"] = dea["Age"].astype(int)

    max_year = int(pop["Year"].max())
    use_years = list(range(max_year - int(last_n_years) + 1, max_year + 1))

    pop_n = pop[pop["Year"].isin(use_years)].copy()
    dea_n = dea[dea["Year"].isin(use_years)].copy()

    merged = pop_n.merge(
        dea_n,
        on=["Region_Code", "Region_Name", "Age", "Year"],
        how="left",
        suffixes=("_pop", "_dead"),
    )
    merged["Number_dead"] = merged["Number_dead"].fillna(0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        merged["survival"] = 1.0 - (merged["Number_dead"] / merged["Number_pop"])

    merged.loc[~np.isfinite(merged["survival"]), "survival"] = np.nan
    merged["survival"] = merged["survival"].clip(0.0, 1.0)

    surv = (
        merged.groupby(["Region_Code", "Region_Name", "Age"], as_index=False)["survival"]
        .mean()
        .rename(columns={"survival": "survival_rate"})
    )

    surv["survival_rate"] = surv["survival_rate"].fillna(surv.groupby("Age")["survival_rate"].transform("mean"))
    surv["survival_rate"] = surv["survival_rate"].fillna(1.0)
    return surv


def build_migration_profile(
    migration_hist: pd.DataFrame,
    start_year: int,
    end_year: int,
    last_n_years: int = 5,
) -> pd.DataFrame:
    mig = migration_hist.copy()
    mig["Region_Code"] = mig["Region_Code"].astype("string").str.strip().str.zfill(2)
    mig["Region_Name"] = mig.get("Region_Name", mig["Region_Code"]).astype(str).str.strip()
    mig["Year"] = pd.to_numeric(mig["Year"], errors="coerce").astype("Int64")
    mig["Age"] = pd.to_numeric(mig["Age"], errors="coerce").astype("Int64")
    mig["Number"] = pd.to_numeric(mig["Number"], errors="coerce").fillna(0.0)

    mig = mig.dropna(subset=["Year", "Age"]).copy()
    mig["Year"] = mig["Year"].astype(int)
    mig["Age"] = mig["Age"].astype(int)

    max_year = int(mig["Year"].max())
    use_years = list(range(max_year - int(last_n_years) + 1, max_year + 1))
    mig_n = mig[mig["Year"].isin(use_years)].copy()

    prof = (
        mig_n.groupby(["Region_Code", "Region_Name", "Age"], as_index=False)["Number"]
        .mean()
        .rename(columns={"Number": "net_migration_per_year"})
    )

    years = pd.DataFrame({"Year": list(range(int(start_year), int(end_year) + 1))})
    return prof.merge(years, how="cross")


def forecast_population_0_19(
    births_forecast: pd.DataFrame,
    db_path: Path,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    pop_hist = _read_population_0_19(db_path)
    deaths_hist = read_table(db_path, "mortality_data_per_region")
    mig_hist = read_table(db_path, "migration_data_per_region")

    births = births_forecast.copy()
    births["Region_Code"] = births["Region_Code"].astype("string").str.strip().str.zfill(2)
    births["Year"] = pd.to_numeric(births["Year"], errors="coerce").astype("Int64")
    births["Forecast_Births"] = pd.to_numeric(births["Forecast_Births"], errors="coerce").fillna(0.0)
    births = births.dropna(subset=["Year"]).copy()
    births["Year"] = births["Year"].astype(int)

    seed_year = int(pd.to_numeric(pop_hist["Year"], errors="coerce").max())
    seed = pop_hist[pop_hist["Year"] == seed_year].copy()
    seed["Region_Code"] = seed["Region_Code"].astype("string").str.strip().str.zfill(2)
    seed["Region_Name"] = seed.get("Region_Name", seed["Region_Code"]).astype(str).str.strip()
    seed["Age"] = pd.to_numeric(seed["Age"], errors="coerce").astype(int)
    seed["Forecast_Population"] = pd.to_numeric(seed["Number"], errors="coerce").fillna(0.0).astype(float)
    seed = seed[["Region_Code", "Region_Name", "Age", "Year", "Forecast_Population"]]

    survival = build_survival_profile(pop_hist, deaths_hist, last_n_years=5)
    migration = build_migration_profile(mig_hist, int(start_year), int(end_year), last_n_years=5)

    births_map = births.set_index(["Region_Code", "Year"])["Forecast_Births"].to_dict()
    surv_map = survival.set_index(["Region_Code", "Age"])["survival_rate"].to_dict()
    mig_map = migration.set_index(["Region_Code", "Age", "Year"])["net_migration_per_year"].to_dict()

    out_all: list[pd.DataFrame] = []
    current = seed.copy()

    for year in range(int(start_year), int(end_year) + 1):
        next_rows: list[tuple[str, str, int, int, float]] = []

        for (rc, rn), g in current.groupby(["Region_Code", "Region_Name"]):
            b0 = float(births_map.get((rc, year), 0.0))
            mig0 = float(mig_map.get((rc, 0, year), 0.0))
            next_rows.append((rc, rn, 0, year, b0 + mig0))

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

    out = pd.concat(out_all, ignore_index=True).sort_values(["Region_Code", "Year", "Age"]).reset_index(drop=True)
    return out
