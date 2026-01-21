"""src/eduforecast/reporting/tables.py"""

from __future__ import annotations

import pandas as pd


def _normalize_region(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "Region_Code" in d.columns:
        d["Region_Code"] = d["Region_Code"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(2)
    if "Region_Name" in d.columns:
        d["Region_Name"] = d["Region_Name"].astype(str).str.strip()
    return d


def make_births_summary_table(births_forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Input canonical births forecast:
        Region_Code, Region_Name, Year, Forecast_Births, (optional: Model)

    Output summary:
        Region_Code, Region_Name, (Model), Years, Min, Max, Mean, Total
    """
    d = _normalize_region(births_forecast)
    if d.empty:
        return pd.DataFrame(
            columns=["Region_Code", "Region_Name", "Model", "Year_Start", "Year_End", "Forecast_Min", "Forecast_Max", "Forecast_Mean", "Forecast_Total"]
        )

    d["Year"] = pd.to_numeric(d["Year"], errors="coerce")
    d["Forecast_Births"] = pd.to_numeric(d.get("Forecast_Births"), errors="coerce")
    d = d.dropna(subset=["Region_Code", "Year", "Forecast_Births"]).copy()
    d["Year"] = d["Year"].astype(int)

    group_cols = ["Region_Code", "Region_Name"]
    if "Model" in d.columns:
        d["Model"] = d["Model"].astype(str).str.strip()
        group_cols.append("Model")
    else:
        d["Model"] = "unknown"
        group_cols.append("Model")

    out = (
        d.groupby(group_cols, as_index=False)["Forecast_Births"]
        .agg(Forecast_Min="min", Forecast_Max="max", Forecast_Mean="mean", Forecast_Total="sum")
        .sort_values(["Region_Code"])
        .reset_index(drop=True)
    )

    years = d.groupby(["Region_Code", "Region_Name", "Model"], as_index=False)["Year"].agg(Year_Start="min", Year_End="max")
    out = out.merge(years, on=["Region_Code", "Region_Name", "Model"], how="left")

    return out[
        ["Region_Code", "Region_Name", "Model", "Year_Start", "Year_End", "Forecast_Min", "Forecast_Max", "Forecast_Mean", "Forecast_Total"]
    ]


def make_population_summary_table(pop_forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Input canonical population forecast:
        Region_Code, Region_Name, Age, Year, Forecast_Population

    Output summary (by Region + Year, total 0-19):
        Region_Code, Region_Name, Year, Total_Pop_0_19
    """
    d = _normalize_region(pop_forecast)
    if d.empty:
        return pd.DataFrame(columns=["Region_Code", "Region_Name", "Year", "Total_Pop_0_19"])

    d["Year"] = pd.to_numeric(d["Year"], errors="coerce")
    d["Age"] = pd.to_numeric(d["Age"], errors="coerce")
    d["Forecast_Population"] = pd.to_numeric(d["Forecast_Population"], errors="coerce")
    d = d.dropna(subset=["Region_Code", "Year", "Age", "Forecast_Population"]).copy()
    d["Year"] = d["Year"].astype(int)
    d["Age"] = d["Age"].astype(int)

    out = (
        d.groupby(["Region_Code", "Region_Name", "Year"], as_index=False)["Forecast_Population"]
        .sum()
        .rename(columns={"Forecast_Population": "Total_Pop_0_19"})
        .sort_values(["Region_Code", "Year"])
        .reset_index(drop=True)
    )
    return out


def make_costs_summary_table(costs_forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Input canonical education costs forecast:
        Region_Code, Region_Name, Year, School_Type, Forecast_Students,
        Fixed_Total_Cost_kr, Current_Total_Cost_kr

    Output summary (by Region + Year, total across school types):
        Region_Code, Region_Name, Year, Students_Total, Fixed_Cost_Total_kr, Current_Cost_Total_kr
    """
    d = _normalize_region(costs_forecast)
    if d.empty:
        return pd.DataFrame(
            columns=["Region_Code", "Region_Name", "Year", "Students_Total", "Fixed_Cost_Total_kr", "Current_Cost_Total_kr"]
        )

    d["Year"] = pd.to_numeric(d["Year"], errors="coerce")
    d["Forecast_Students"] = pd.to_numeric(d["Forecast_Students"], errors="coerce")
    d["Fixed_Total_Cost_kr"] = pd.to_numeric(d["Fixed_Total_Cost_kr"], errors="coerce")
    d["Current_Total_Cost_kr"] = pd.to_numeric(d["Current_Total_Cost_kr"], errors="coerce")
    d = d.dropna(subset=["Region_Code", "Year"]).copy()
    d["Year"] = d["Year"].astype(int)

    out = (
        d.groupby(["Region_Code", "Region_Name", "Year"], as_index=False)
        .agg(
            Students_Total=("Forecast_Students", "sum"),
            Fixed_Cost_Total_kr=("Fixed_Total_Cost_kr", "sum"),
            Current_Cost_Total_kr=("Current_Total_Cost_kr", "sum"),
        )
        .sort_values(["Region_Code", "Year"])
        .reset_index(drop=True)
    )
    return out
