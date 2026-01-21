"""src/eduforecast/preprocessing/clean_births.py"""

from __future__ import annotations

from typing import Mapping

import pandas as pd


DEFAULT_REGION_CODE_TO_NAME: dict[str, str] = {
    "01": "Stockholms län",
    "03": "Uppsala län",
    "04": "Södermanlands län",
    "05": "Östergötlands län",
    "06": "Jönköpings län",
    "07": "Kronobergs län",
    "08": "Kalmar län",
    "09": "Gotlands län",
    "10": "Blekinge län",
    "12": "Skåne län",
    "13": "Hallands län",
    "14": "Västra Götalands län",
    "17": "Värmlands län",
    "18": "Örebro län",
    "19": "Västmanlands län",
    "20": "Dalarnas län",
    "21": "Gävleborgs län",
    "22": "Västernorrlands län",
    "23": "Jämtlands län",
    "24": "Västerbottens län",
    "25": "Norrbottens län",
}


def clean_births(
    df: pd.DataFrame,
    *,
    region_code_to_name: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """
    Standardize births to canonical schema:

        Region_Code, Region_Name, Year, Number

    Accepts common raw variants like:
        Region / Länskod / Region_Code
        Total_Births / Births / Antal / Number
    """
    d = df.copy()
    d.columns = [c.strip() for c in d.columns]

    rename_map = {
        "Region": "Region_Code",
        "region": "Region_Code",
        "Länskod": "Region_Code",
        "LanKod": "Region_Code",
        "region_code": "Region_Code",
        "Län": "Region_Name",
        "Lan": "Region_Name",
        "RegionNamn": "Region_Name",
        "region_name": "Region_Name",
        "År": "Year",
        "Ar": "Year",
        "year": "Year",
        "Total_Births": "Number",
        "total_births": "Number",
        "Births": "Number",
        "births": "Number",
        "Antal": "Number",
        "Value": "Number",
        "value": "Number",
    }
    d = d.rename(columns={c: rename_map[c] for c in d.columns if c in rename_map})

    required = {"Region_Code", "Year", "Number"}
    missing = required - set(d.columns)
    if missing:
        raise KeyError(f"Births missing columns {sorted(missing)}. Found: {list(d.columns)}")

    d["Region_Code"] = (
        d["Region_Code"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(2)
    )

    d["Year"] = pd.to_numeric(d["Year"], errors="coerce").astype("Int64")
    d["Number"] = pd.to_numeric(d["Number"], errors="coerce").astype(float)

    d = d.dropna(subset=["Year", "Number"]).copy()
    d["Year"] = d["Year"].astype(int)

    if "Region_Name" not in d.columns:
        d["Region_Name"] = pd.NA
    d["Region_Name"] = d["Region_Name"].astype("string").str.strip()

    mapping = dict(region_code_to_name or DEFAULT_REGION_CODE_TO_NAME)
    d["Region_Name"] = d["Region_Name"].mask(
        d["Region_Name"].isna() | (d["Region_Name"] == "") | (d["Region_Name"] == d["Region_Code"]),
        d["Region_Code"].map(mapping),
    )
    d["Region_Name"] = d["Region_Name"].fillna(d["Region_Code"]).astype(str).str.strip()

    return (
        d[["Region_Code", "Region_Name", "Year", "Number"]]
        .sort_values(["Region_Code", "Year"])
        .reset_index(drop=True)
    )
