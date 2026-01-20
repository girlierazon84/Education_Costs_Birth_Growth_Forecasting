"""src/eduforecast/io/readers.py"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


# ---------- generic helpers ----------

def read_csv(path: Path, *, dtype: dict[str, Any] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file:\n{path}")
    return pd.read_csv(path, dtype=dtype)


def normalize_region_code(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.zfill(2)


def ensure_region_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure Region_Code and Region_Name exist and are normalized.
    """
    out = df.copy()

    # Region_Code may appear as "Region"
    if "Region_Code" not in out.columns and "Region" in out.columns:
        out = out.rename(columns={"Region": "Region_Code"})

    if "Region_Code" in out.columns:
        out["Region_Code"] = normalize_region_code(out["Region_Code"])
    else:
        out["Region_Code"] = "NA"

    if "Region_Name" in out.columns:
        out["Region_Name"] = out["Region_Name"].astype(str).str.strip()
    else:
        # If we don't have names, keep code as display
        out["Region_Name"] = out["Region_Code"].astype(str)

    return out


# ---------- domain-specific reads ----------

def read_births_raw(path: Path) -> pd.DataFrame:
    """
    Standardize births into:
        Region_Code, Region_Name, Year, Number

    Supports common column variants:
      - Region or Region_Code
      - Number / Total_Births / Antal / Value
    """
    df = read_csv(path)
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # value column -> Number
    if "Number" not in df.columns:
        for candidate in ["Total_Births", "total_births", "Antal", "Value", "value", "Births", "births"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "Number"})
                break

    # Year must exist (maybe År)
    if "Year" not in df.columns:
        for candidate in ["År", "Ar", "year"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "Year"})
                break

    df = ensure_region_cols(df)

    if "Year" not in df.columns:
        raise KeyError(f"Births missing 'Year'. Found columns: {list(df.columns)}")
    if "Number" not in df.columns:
        raise KeyError(f"Births missing 'Number'. Found columns: {list(df.columns)}")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Number"] = pd.to_numeric(df["Number"], errors="coerce")

    df = df.dropna(subset=["Year", "Number"]).copy()
    df["Year"] = df["Year"].astype(int)
    df["Number"] = df["Number"].astype(float)

    return df[["Region_Code", "Region_Name", "Year", "Number"]].sort_values(
        ["Region_Code", "Year"]
    ).reset_index(drop=True)


def read_costs_per_child(path: Path) -> pd.DataFrame:
    """
    Standardize cost-per-child table into:
        Year, Fixed_cost_per_child_kr, Current_cost_per_child_kr
    """
    df = read_csv(path)
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # normalize possible column variants
    df = df.rename(
        columns={
            "Fixed_Cost_Per_Child_SEK": "Fixed_cost_per_child_kr",
            "Current_Cost_Per_Child_SEK": "Current_cost_per_child_kr",
        }
    )

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    for c in ["Fixed_cost_per_child_kr", "Current_cost_per_child_kr"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)
    return df.sort_values("Year").reset_index(drop=True)
