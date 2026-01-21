"""src/eduforecast/costs/total_costs.py"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from eduforecast.preprocessing.clean_costs import clean_costs_per_child


def load_cost_tables(
    grund_path: Path,
    gymn_path: Path,
    *,
    anchor_max_year: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grund = clean_costs_per_child(pd.read_csv(grund_path))
    gymn = clean_costs_per_child(pd.read_csv(gymn_path))

    if anchor_max_year is not None:
        grund = grund[grund["Year"] <= int(anchor_max_year)].copy()
        gymn = gymn[gymn["Year"] <= int(anchor_max_year)].copy()

    return grund, gymn


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

    cost tables expected columns:
        Year, Fixed_cost_per_child_kr, Current_cost_per_child_kr

    NOTE:
      Fixed vs Current are alternative bases (real vs nominal) — DO NOT add them.
    """
    df = pop_forecast.copy()
    df["Region_Code"] = df["Region_Code"].astype("string").str.strip().str.zfill(2)
    df["Region_Name"] = df.get("Region_Name", df["Region_Code"]).astype(str).str.strip()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").astype("Int64")
    df["Forecast_Population"] = pd.to_numeric(df["Forecast_Population"], errors="coerce").astype(float)
    df = df.dropna(subset=["Year", "Age", "Forecast_Population"]).copy()
    df["Year"] = df["Year"].astype(int)
    df["Age"] = df["Age"].astype(int)

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

    # ensure costs are clean
    grund = clean_costs_per_child(grund)
    gymn = clean_costs_per_child(gymn)

    grund_c = grund.sort_values("Year")[["Year", "Fixed_cost_per_child_kr", "Current_cost_per_child_kr"]].copy()
    gymn_c = gymn.sort_values("Year")[["Year", "Fixed_cost_per_child_kr", "Current_cost_per_child_kr"]].copy()
    grund_c["Cost_Year"] = grund_c["Year"]
    gymn_c["Cost_Year"] = gymn_c["Year"]

    out_parts: list[pd.DataFrame] = []

    for school_type, cost_df in [("grundskola", grund_c), ("gymnasieskola", gymn_c)]:
        s = students[students["School_Type"] == school_type].sort_values("Year").copy()

        merged = pd.merge_asof(s, cost_df, on="Year", direction="backward")

        # If forecasting earlier than first cost year, fallback to earliest cost
        if merged["Fixed_cost_per_child_kr"].isna().any():
            merged = pd.merge_asof(s, cost_df, on="Year", direction="forward")

        if str(extrapolation).lower() == "growth_rate":
            yrs = (merged["Year"] - merged["Cost_Year"]).clip(lower=0)
            growth = (1.0 + float(annual_growth_rate)) ** yrs
            merged["Fixed_cost_per_child_kr"] = merged["Fixed_cost_per_child_kr"] * growth
            merged["Current_cost_per_child_kr"] = merged["Current_cost_per_child_kr"] * growth

        merged["Fixed_Total_Cost_kr"] = merged["Forecast_Students"] * merged["Fixed_cost_per_child_kr"]
        merged["Current_Total_Cost_kr"] = merged["Forecast_Students"] * merged["Current_cost_per_child_kr"]

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

    out = pd.concat(out_parts, ignore_index=True).sort_values(["Region_Code", "Year", "School_Type"]).reset_index(drop=True)
    return out
