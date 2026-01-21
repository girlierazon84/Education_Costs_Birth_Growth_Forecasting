"""src/eduforecast/preprocessing/clean_costs.py"""

from __future__ import annotations

import pandas as pd


def clean_costs_per_child(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize cost tables to:

        Year, Fixed_cost_per_child_kr, Current_cost_per_child_kr

    Accepts common variants:
        Fixed_Cost_Per_Child_SEK, Current_Cost_Per_Child_SEK
    """
    d = df.copy()
    d.columns = [c.strip() for c in d.columns]

    d = d.rename(
        columns={
            "Fixed_Cost_Per_Child_SEK": "Fixed_cost_per_child_kr",
            "Current_Cost_Per_Child_SEK": "Current_cost_per_child_kr",
            "fixed_cost_per_child_kr": "Fixed_cost_per_child_kr",
            "current_cost_per_child_kr": "Current_cost_per_child_kr",
        }
    )

    if "Year" not in d.columns:
        raise KeyError(f"Costs missing 'Year'. Found: {list(d.columns)}")

    d["Year"] = pd.to_numeric(d["Year"], errors="coerce").astype("Int64")

    for c in ["Fixed_cost_per_child_kr", "Current_cost_per_child_kr"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=["Year"]).copy()
    d["Year"] = d["Year"].astype(int)

    # Keep only relevant columns if present
    cols = ["Year"] + [c for c in ["Fixed_cost_per_child_kr", "Current_cost_per_child_kr"] if c in d.columns]
    return d[cols].sort_values("Year").reset_index(drop=True)
