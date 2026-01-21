"""src/eduforecast/preprocessing/clean_population.py"""

from __future__ import annotations

import pandas as pd


def clean_population(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize population (0–19) to:

        Region_Code, Region_Name, Age, Year, Number
    """
    d = df.copy()
    d.columns = [c.strip() for c in d.columns]

    rename_map = {
        "Region": "Region_Code",
        "region": "Region_Code",
        "Länskod": "Region_Code",
        "LanKod": "Region_Code",
        "År": "Year",
        "Ar": "Year",
        "year": "Year",
        "Total_Population": "Number",
        "Population": "Number",
        "Antal": "Number",
        "Value": "Number",
        "value": "Number",
    }
    d = d.rename(columns={c: rename_map[c] for c in d.columns if c in rename_map})

    required = {"Region_Code", "Age", "Year", "Number"}
    missing = required - set(d.columns)
    if missing:
        raise KeyError(f"Population missing columns {sorted(missing)}. Found: {list(d.columns)}")

    d["Region_Code"] = d["Region_Code"].astype("string").str.strip().str.zfill(2)
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
