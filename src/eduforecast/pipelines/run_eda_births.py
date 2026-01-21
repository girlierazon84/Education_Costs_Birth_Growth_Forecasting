"""src/eduforecast/pipelines/run_eda_births.py"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eduforecast.common.config import AppConfig


logger = logging.getLogger(__name__)


def _get(cfg_obj: Any, key: str, default: Any = None) -> Any:
    """Support both dict-style and attribute-style config."""
    if cfg_obj is None:
        return default
    if hasattr(cfg_obj, key):
        v = getattr(cfg_obj, key)
        return default if v is None else v
    if isinstance(cfg_obj, dict):
        return cfg_obj.get(key, default)
    return default


def _resolve(cfg: AppConfig, maybe_path: str | Path) -> Path:
    p = Path(maybe_path)
    return p if p.is_absolute() else (cfg.project_root / p).resolve()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_births_sqlite(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as con:
        return pd.read_sql("SELECT * FROM birth_data_per_region", con)


def run_eda_births(cfg: AppConfig) -> None:
    # --- DB path ---
    sqlite_path = _get(cfg.database, "sqlite_path")
    if not sqlite_path:
        raise ValueError("Missing database.sqlite_path in config")
    db_path = _resolve(cfg, sqlite_path)

    births = _read_births_sqlite(db_path)

    # --- Normalize schema ---
    births = births.copy()
    births["Region_Code"] = births["Region_Code"].astype("string").str.strip().str.zfill(2)
    births["Region_Name"] = births.get("Region_Name", births["Region_Code"]).astype(str).str.strip()
    births["Year"] = pd.to_numeric(births["Year"], errors="coerce").astype("Int64")
    births["Number"] = pd.to_numeric(births["Number"], errors="coerce").astype(float)

    births = births.dropna(subset=["Year", "Number"]).copy()
    births["Year"] = births["Year"].astype(int)

    start_year = int(_get(cfg.modeling, "start_year", 1968))
    births = births[births["Year"] >= start_year].copy()

    include = _get(cfg.regions, "include")
    if include:
        include = [str(x).strip().zfill(2) for x in include]
        births = births[births["Region_Code"].isin(include)].copy()

    # --- Output dirs ---
    metrics_dir = _resolve(cfg, _get(cfg.paths, "metrics_dir"))
    figures_dir = _resolve(cfg, _get(cfg.paths, "figures_dir"))
    eda_metrics_dir = metrics_dir / "eda"
    eda_fig_dir = figures_dir / "eda"
    eda_metrics_dir.mkdir(parents=True, exist_ok=True)
    eda_fig_dir.mkdir(parents=True, exist_ok=True)

    # 1) Basic summary
    summary = (
        births.groupby(["Region_Code", "Region_Name"], dropna=False)
        .agg(
            year_min=("Year", "min"),
            year_max=("Year", "max"),
            n_years=("Year", "nunique"),
            n_rows=("Year", "size"),
            number_min=("Number", "min"),
            number_max=("Number", "max"),
            number_mean=("Number", "mean"),
            number_missing=("Number", lambda s: int(pd.isna(s).sum())),
        )
        .reset_index()
        .sort_values(["Region_Code"])
        .reset_index(drop=True)
    )
    out_summary = eda_metrics_dir / "births_summary.csv"
    _ensure_parent(out_summary)
    summary.to_csv(out_summary, index=False)

    # 2) Duplicate keys check (Region_Code, Year)
    dup = (
        births.groupby(["Region_Code", "Year"])
        .size()
        .reset_index(name="count")
        .query("count > 1")
        .sort_values(["Region_Code", "Year"])
        .reset_index(drop=True)
    )
    out_dup = eda_metrics_dir / "births_duplicates.csv"
    _ensure_parent(out_dup)
    dup.to_csv(out_dup, index=False)

    # 3) Missing years per region between each region's min..max  (FIXED)
    missing_rows: list[dict[str, Any]] = []
    for (rc, rn), g in births.groupby(["Region_Code", "Region_Name"], dropna=False):
        ymin, ymax = int(g["Year"].min()), int(g["Year"].max())
        expected = set(range(ymin, ymax + 1))
        observed = set(g["Year"].unique().tolist())
        missing = sorted(expected - observed)
        for y in missing:
            missing_rows.append({"Region_Code": rc, "Region_Name": rn, "Missing_Year": int(y)})

    missing_years = pd.DataFrame(missing_rows, columns=["Region_Code", "Region_Name", "Missing_Year"])
    if not missing_years.empty:
        missing_years = missing_years.sort_values(["Region_Code", "Missing_Year"]).reset_index(drop=True)

    out_missing = eda_metrics_dir / "births_missing_years.csv"
    _ensure_parent(out_missing)
    missing_years.to_csv(out_missing, index=False)

    # 4) Outliers/spikes based on YoY change (robust z-score)
    outlier_rows: list[dict[str, Any]] = []
    for (rc, rn), g in births.groupby(["Region_Code", "Region_Name"], dropna=False):
        gg = g.sort_values("Year").copy()
        gg["YoY"] = gg["Number"].diff()

        yoy = gg["YoY"].dropna().to_numpy(dtype=float)
        if yoy.size < 8:
            continue

        med = float(np.median(yoy))
        mad = float(np.median(np.abs(yoy - med))) or 1.0
        gg["robust_z"] = (gg["YoY"] - med) / (1.4826 * mad)

        flagged = gg.loc[gg["robust_z"].abs() >= 4, ["Year", "Number", "YoY", "robust_z"]]
        for _, r in flagged.iterrows():
            outlier_rows.append(
                {
                    "Region_Code": rc,
                    "Region_Name": rn,
                    "Year": int(r["Year"]),
                    "Number": float(r["Number"]),
                    "YoY": float(r["YoY"]) if pd.notna(r["YoY"]) else np.nan,
                    "robust_z": float(r["robust_z"]) if pd.notna(r["robust_z"]) else np.nan,
                }
            )

    outliers = pd.DataFrame(outlier_rows, columns=["Region_Code", "Region_Name", "Year", "Number", "YoY", "robust_z"])
    if not outliers.empty:
        outliers = outliers.sort_values(["Region_Code", "Year"]).reset_index(drop=True)

    out_outliers = eda_metrics_dir / "births_outliers.csv"
    _ensure_parent(out_outliers)
    outliers.to_csv(out_outliers, index=False)

    logger.info("EDA checks written to: %s", eda_metrics_dir)
    logger.info(
        "Summary rows: %d | Duplicates rows: %d | Missing rows: %d | Outliers rows: %d",
        len(summary),
        len(dup),
        len(missing_years),
        len(outliers),
    )
