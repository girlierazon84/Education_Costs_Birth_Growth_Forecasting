"""src/eduforecast/reporting/plots.py"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class PlotPaths:
    births: Path
    population: Path
    costs: Path


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def plot_births_forecast_by_region(
    births_forecast: pd.DataFrame,
    *,
    region_codes: Iterable[str],
    out_dir: Path,
) -> list[Path]:
    """
    Saves one PNG per region:
        births_forecast_{Region_Code}.png

    Expected columns:
        Region_Code, Region_Name, Year, Forecast_Births (optional: Model)
    """
    _ensure_dir(out_dir)

    d = births_forecast.copy()
    d["Region_Code"] = d["Region_Code"].astype("string").str.strip().str.zfill(2)
    d["Region_Name"] = d.get("Region_Name", d["Region_Code"]).astype(str).str.strip()
    d["Year"] = pd.to_numeric(d["Year"], errors="coerce")
    d["Forecast_Births"] = pd.to_numeric(d["Forecast_Births"], errors="coerce")
    d = d.dropna(subset=["Region_Code", "Year", "Forecast_Births"]).copy()
    d["Year"] = d["Year"].astype(int)

    saved: list[Path] = []
    for rc in [str(x).strip().zfill(2) for x in region_codes]:
        sub = d[d["Region_Code"] == rc].sort_values("Year")
        if sub.empty:
            continue
        rn = str(sub["Region_Name"].iloc[0]) if "Region_Name" in sub.columns else ""
        model = str(sub["Model"].iloc[0]) if "Model" in sub.columns else ""

        plt.figure()
        plt.plot(sub["Year"].to_numpy(), sub["Forecast_Births"].to_numpy())
        title = f"Birth forecast — {rc} {rn}"
        if model:
            title += f" ({model})"
        plt.title(title)
        plt.xlabel("Year")
        plt.ylabel("Forecast births")
        out = out_dir / f"births_forecast_{rc}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(out)

    return saved


def plot_population_forecast_by_region_ageband(
    pop_forecast: pd.DataFrame,
    *,
    region_code: str,
    out_dir: Path,
) -> Path | None:
    """
    Saves one PNG for a region with 3 age bands:
      - 0–6, 7–16, 17–19

    Expected columns:
      Region_Code, Region_Name, Age, Year, Forecast_Population
    """
    _ensure_dir(out_dir)

    d = pop_forecast.copy()
    d["Region_Code"] = d["Region_Code"].astype("string").str.strip().str.zfill(2)
    d["Region_Name"] = d.get("Region_Name", d["Region_Code"]).astype(str).str.strip()
    d["Age"] = pd.to_numeric(d["Age"], errors="coerce")
    d["Year"] = pd.to_numeric(d["Year"], errors="coerce")
    d["Forecast_Population"] = pd.to_numeric(d["Forecast_Population"], errors="coerce")
    d = d.dropna(subset=["Region_Code", "Year", "Age", "Forecast_Population"]).copy()
    d["Age"] = d["Age"].astype(int)
    d["Year"] = d["Year"].astype(int)

    rc = str(region_code).strip().zfill(2)
    sub = d[d["Region_Code"] == rc].copy()
    if sub.empty:
        return None

    rn = str(sub["Region_Name"].iloc[0]) if "Region_Name" in sub.columns else ""

    def band(age: int) -> str:
        if 0 <= age <= 6:
            return "0–6"
        if 7 <= age <= 16:
            return "7–16"
        if 17 <= age <= 19:
            return "17–19"
        return "other"

    sub["AgeBand"] = sub["Age"].map(band)
    sub = sub[sub["AgeBand"] != "other"]

    series = (
        sub.groupby(["Year", "AgeBand"], as_index=False)["Forecast_Population"]
        .sum()
        .sort_values(["AgeBand", "Year"])
    )

    plt.figure()
    for band_name in ["0–6", "7–16", "17–19"]:
        s = series[series["AgeBand"] == band_name]
        if s.empty:
            continue
        plt.plot(s["Year"].to_numpy(), s["Forecast_Population"].to_numpy(), label=band_name)

    plt.title(f"Population forecast (age bands) — {rc} {rn}")
    plt.xlabel("Year")
    plt.ylabel("Forecast population")
    plt.legend()
    out = out_dir / f"population_forecast_agebands_{rc}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def plot_costs_by_region_and_schooltype(
    costs_forecast: pd.DataFrame,
    *,
    region_code: str,
    out_dir: Path,
    basis: str = "fixed",
) -> Path | None:
    """
    Saves one PNG for a region with grundskola vs gymnasieskola.

    basis:
      - "fixed" -> Fixed_Total_Cost_kr
      - "current" -> Current_Total_Cost_kr

    Expected columns:
      Region_Code, Region_Name, Year, School_Type,
      Fixed_Total_Cost_kr, Current_Total_Cost_kr
    """
    _ensure_dir(out_dir)

    b = str(basis).strip().lower()
    col = "Fixed_Total_Cost_kr" if b == "fixed" else "Current_Total_Cost_kr"

    d = costs_forecast.copy()
    d["Region_Code"] = d["Region_Code"].astype("string").str.strip().str.zfill(2)
    d["Region_Name"] = d.get("Region_Name", d["Region_Code"]).astype(str).str.strip()
    d["School_Type"] = d["School_Type"].astype(str).str.strip().str.lower()
    d["Year"] = pd.to_numeric(d["Year"], errors="coerce")
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["Region_Code", "Year", col]).copy()
    d["Year"] = d["Year"].astype(int)

    rc = str(region_code).strip().zfill(2)
    sub = d[d["Region_Code"] == rc].copy()
    if sub.empty:
        return None
    rn = str(sub["Region_Name"].iloc[0]) if "Region_Name" in sub.columns else ""

    plt.figure()
    for stype in ["grundskola", "gymnasieskola"]:
        s = sub[sub["School_Type"] == stype].sort_values("Year")
        if s.empty:
            continue
        plt.plot(s["Year"].to_numpy(), s[col].to_numpy(), label=stype)

    plt.title(f"Education costs ({b}) — {rc} {rn}")
    plt.xlabel("Year")
    plt.ylabel(f"Total cost (kr) — {b}")
    plt.legend()
    out = out_dir / f"education_costs_{b}_{rc}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out
