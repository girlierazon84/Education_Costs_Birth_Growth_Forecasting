"""src/eduforecast/io/writers.py"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, path: Path, *, index: bool = False) -> Path:
    """Write DataFrame to CSV (ensures parent folder exists)."""
    ensure_parent_dir(path)
    df.to_csv(path, index=index)
    return path


def write_forecast_artifact(df: pd.DataFrame, path: Path) -> Path:
    """
    Write a forecast artifact with defensive normalization:
    - keep Region_Code zero-padded
    - strip strings
    """
    out = df.copy()
    if "Region_Code" in out.columns:
        out["Region_Code"] = out["Region_Code"].astype("string").str.strip().str.zfill(2)
    if "Region_Name" in out.columns:
        out["Region_Name"] = out["Region_Name"].astype(str).str.strip()
    if "School_Type" in out.columns:
        out["School_Type"] = out["School_Type"].astype(str).str.strip().str.lower()

    return write_csv(out, path, index=False)
