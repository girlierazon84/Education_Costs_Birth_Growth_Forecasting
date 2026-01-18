"""src/eduforecast/costs/total_costs.py"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def _standardize_cost_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept either:
        - Fixed_cost_per_child_kr / Current_cost_per_child_kr
        - Fixed_Cost_Per_Child_SEK / Current_Cost_Per_Child_SEK
    and standardize to:
        - Fixed_cost_per_child_kr / Current_cost_per_child_kr
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    mapping_variants = {
        "Fixed_Cost_Per_Child_SEK": "Fixed_cost_per_child_kr",
        "Current_Cost_Per_Child_SEK": "Current_cost_per_child_kr",
        "Fixed_cost_per_child_kr": "Fixed_cost_per_child_kr",
        "Current_cost_per_child_kr": "Current_cost_per_child_kr",
    }

    rename_map = {c: mapping_variants[c] for c in df.columns if c in mapping_variants}
    df = df.rename(columns=rename_map)

    required = {"Year", "Fixed_cost_per_child_kr", "Current_cost_per_child_kr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Cost table is missing required columns {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    df["Year"] = df["Year"].astype(int)
    for c in ["Fixed_cost_per_child_kr", "Current_cost_per_child_kr"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def load_cost_tables(grund_path: Path, gymn_path: Path, *, anchor_max_year: int | None = None):
    grund = pd.read_csv(grund_path)
    gymn = pd.read_csv(gymn_path)

    grund = _standardize_cost_columns(grund)
    gymn = _standardize_cost_columns(gymn)

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
    pop_forecast columns expected:
        Region_Code, Region_Name, Age, Year, Forecast_Population (floats)

    grund/gymn columns expected (standardized by load_cost_tables):
        Year, Fixed_cost_per_child_kr, Current_cost_per_child_kr

    extrapolation:
        - "carry_forward": use last known cost for future years (default)
        - "growth_rate": grow last known cost forward using annual_growth_rate
    """
    df = pop_forecast.copy()
    df["Region_Code"] = df["Region_Code"].astype(str).str.zfill(2)
    df["Region_Name"] = df["Region_Name"].astype(str)
    df["Year"] = df["Year"].astype(int)
    df["Age"] = df["Age"].astype(int)

    # Age ranges (adjust if you want)
    grund_ages = set(range(7, 17))  # 7–16
    gymn_ages = set(range(17, 20))  # 17–19

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

    # normalize types AFTER students exists
    students["Region_Code"] = students["Region_Code"].astype(str).str.zfill(2)
    students["School_Type"] = students["School_Type"].astype(str).str.strip().str.lower()

    # Prepare costs for as-of merge (and keep matched year as Cost_Year)
    grund_c = grund.sort_values("Year")[["Year", "Fixed_cost_per_child_kr", "Current_cost_per_child_kr"]].copy()
    gymn_c = gymn.sort_values("Year")[["Year", "Fixed_cost_per_child_kr", "Current_cost_per_child_kr"]].copy()
    grund_c["Cost_Year"] = grund_c["Year"]
    gymn_c["Cost_Year"] = gymn_c["Year"]

    out_parts = []

    for school_type, cost_df in [("grundskola", grund_c), ("gymnasieskola", gymn_c)]:
        s = students[students["School_Type"] == school_type].sort_values("Year").copy()

        merged = pd.merge_asof(
            s,
            cost_df,
            on="Year",
            direction="backward",
        )

        # If forecasting earlier than first cost year, fallback to earliest cost
        if merged["Fixed_cost_per_child_kr"].isna().any():
            merged = pd.merge_asof(
                s,
                cost_df,
                on="Year",
                direction="forward",
            )

        if extrapolation == "growth_rate":
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

    out = pd.concat(out_parts, ignore_index=True).sort_values(["Region_Code", "Year", "School_Type"])
    return out
