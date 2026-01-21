"""src/eduforecast/costs/total_costs.py"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from eduforecast.costs.cost_per_child import CostTables, cost_schedule_for_years, load_cost_tables


def compute_education_costs(
    pop_forecast: pd.DataFrame,
    grund: pd.DataFrame,
    gymn: pd.DataFrame,
    *,
    extrapolation: str = "carry_forward",
    annual_growth_rate: float = 0.0,
) -> pd.DataFrame:
    """
    pop_forecast expected columns:
        Region_Code, Region_Name, Age, Year, Forecast_Population

    cost tables expected columns (raw or clean):
        Year, Fixed_cost_per_child_kr, Current_cost_per_child_kr

    NOTE:
      Fixed vs Current are alternative bases — DO NOT add them.
    """
    df = pop_forecast.copy()
    df.columns = [c.strip() for c in df.columns]

    required = {"Region_Code", "Age", "Year", "Forecast_Population"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"pop_forecast missing columns {sorted(missing)}. Found: {list(df.columns)}")

    df["Region_Code"] = (
        df["Region_Code"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(2)
    )
    df["Region_Name"] = df.get("Region_Name", df["Region_Code"]).astype(str).str.strip()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").astype("Int64")
    df["Forecast_Population"] = pd.to_numeric(df["Forecast_Population"], errors="coerce").astype(float)
    df = df.dropna(subset=["Year", "Age", "Forecast_Population"]).copy()
    df["Year"] = df["Year"].astype(int)
    df["Age"] = df["Age"].astype(int)

    # derive horizon from pop forecast
    start_year = int(df["Year"].min())
    end_year = int(df["Year"].max())

    grund_ages = set(range(7, 17))   # 7–16
    gymn_ages = set(range(17, 20))   # 17–19

    grund_students = (
        df[df["Age"].isin(grund_ages)]
        .groupby(["Region_Code", "Region_Name", "Year"], as_index=False)["Forecast_Population"]
        .sum()
        .rename(columns={"Forecast_Population": "Forecast_Students"})
    )
    grund_students["School_Type"] = "grundskola"

    gymn_students = (
        df[df["Age"].isin(gymn_ages)]
        .groupby(["Region_Code", "Region_Name", "Year"], as_index=False)["Forecast_Population"]
        .sum()
        .rename(columns={"Forecast_Population": "Forecast_Students"})
    )
    gymn_students["School_Type"] = "gymnasieskola"

    students = pd.concat([grund_students, gymn_students], ignore_index=True)
    students["School_Type"] = students["School_Type"].astype(str).str.strip().str.lower()

    method = str(extrapolation).strip().lower()
    grund_sched = cost_schedule_for_years(
        grund,
        start_year=start_year,
        end_year=end_year,
        method=method if method in {"carry_forward", "growth_rate"} else "carry_forward",
        annual_growth_rate=float(annual_growth_rate),
    )
    gymn_sched = cost_schedule_for_years(
        gymn,
        start_year=start_year,
        end_year=end_year,
        method=method if method in {"carry_forward", "growth_rate"} else "carry_forward",
        annual_growth_rate=float(annual_growth_rate),
    )

    out_parts: list[pd.DataFrame] = []

    for school_type, sched in [("grundskola", grund_sched), ("gymnasieskola", gymn_sched)]:
        s = students[students["School_Type"] == school_type].copy()

        merged = s.merge(sched, on="Year", how="left")

        merged["Fixed_Total_Cost_kr"] = merged["Forecast_Students"] * pd.to_numeric(
            merged["Fixed_cost_per_child_kr"], errors="coerce"
        )
        merged["Current_Total_Cost_kr"] = merged["Forecast_Students"] * pd.to_numeric(
            merged["Current_cost_per_child_kr"], errors="coerce"
        )

        out_parts.append(
            merged[
                [
                    "Region_Code",
                    "Region_Name",
                    "Year",
                    "School_Type",
                    "Forecast_Students",
                    "Fixed_Total_Cost_kr",
                    "Current_Total_Cost_kr",
                ]
            ]
        )

    out = (
        pd.concat(out_parts, ignore_index=True)
        .sort_values(["Region_Code", "Year", "School_Type"])
        .reset_index(drop=True)
    )
    return out
