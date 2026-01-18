"""src/eduforecast/pipelines/run_forecast.py"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eduforecast.common.config import AppConfig
from eduforecast.costs.total_costs import compute_education_costs, load_cost_tables
from eduforecast.forecasting.predict_births import predict_births_all_regions
from eduforecast.forecasting.predict_population import predict_population_0_19


logger = logging.getLogger(__name__)


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _resolve(cfg: AppConfig, maybe_path: str | Path) -> Path:
    """
    Resolve a path robustly:
    - if absolute: keep
    - if relative: resolve against project_root
    """
    p = Path(maybe_path)
    return p if p.is_absolute() else (cfg.project_root / p).resolve()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _maybe_plot_births(
    births_forecast: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
    figures_dir: Path,
    region_codes: list[str],
) -> None:
    """
    Save quick sanity plots for a small set of regions if present.
    """
    try:
        figures_dir.mkdir(parents=True, exist_ok=True)
        years = np.arange(start_year, end_year + 1, dtype=int)

        available = set(births_forecast["Region_Code"].astype(str).str.zfill(2).unique())
        region_codes = [rc for rc in region_codes if rc in available]
        if not region_codes:
            return

        for rc in region_codes:
            sub = births_forecast[births_forecast["Region_Code"] == rc].copy()
            sub = sub.sort_values("Year")
            if sub.empty:
                continue

            rn = str(sub["Region_Name"].iloc[0]) if "Region_Name" in sub.columns else ""
            model = str(sub["Model"].iloc[0]) if "Model" in sub.columns else ""
            y = pd.to_numeric(sub["Forecast_Births"], errors="coerce").to_numpy(dtype=float)

            if len(y) != len(years):
                # avoid misleading plots if years missing
                continue

            plt.figure()
            plt.plot(years, y)
            plt.title(f"Birth Forecast {start_year}-{end_year}: {rc} {rn} ({model})")
            plt.xlabel("Year")
            plt.ylabel("Forecast births (float)")
            plt.savefig(figures_dir / f"birth_forecast_{rc}.png", dpi=150, bbox_inches="tight")
            plt.close()

    except Exception:
        logger.exception("Birth sanity plotting failed.")


def run_forecast(cfg: AppConfig) -> None:
    """
    Run the full forecast pipeline:
        1) Births forecast per region (from best saved model per region)
        2) Population 0–19 cohort forecast (features table)
        3) Education costs forecast (Fixed and Current bases; DO NOT add them)
    """
    # --- Core config ---
    # Support both cfg.database["sqlite_path"] and cfg.database.sqlite_path styles
    sqlite_path = getattr(cfg.database, "sqlite_path", None)
    if sqlite_path is None and isinstance(cfg.database, dict):
        sqlite_path = cfg.database.get("sqlite_path")

    if not sqlite_path:
        raise ValueError("Missing database.sqlite_path in config")

    db_path = _resolve(cfg, sqlite_path)

    start_year = _safe_int(getattr(cfg.forecast, "start_year", None) or cfg.forecast.get("start_year", 2024), 2024)
    end_year = _safe_int(getattr(cfg.forecast, "end_year", None) or cfg.forecast.get("end_year", 2030), 2030)
    if end_year < start_year:
        raise ValueError("forecast.end_year must be >= forecast.start_year")

    # --- Output dirs ---
    metrics_dir = _resolve(cfg, getattr(cfg.paths, "metrics_dir", None) or cfg.paths.get("metrics_dir"))
    forecasts_dir = _resolve(cfg, getattr(cfg.paths, "forecasts_dir", None) or cfg.paths.get("forecasts_dir"))
    figures_dir = _resolve(cfg, getattr(cfg.paths, "figures_dir", None) or cfg.paths.get("figures_dir")) / "forecasts" / "births"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # --- Load best models registry ---
    best_models_path = metrics_dir / "best_models_births.csv"
    if not best_models_path.exists():
        raise FileNotFoundError(f"Missing {best_models_path}. Run `eduforecast train` first.")

    best = pd.read_csv(best_models_path, dtype={"Region_Code": "string"})
    best["Region_Code"] = best["Region_Code"].astype("string").str.strip().str.zfill(2)
    best["Region_Name"] = best.get("Region_Name", best["Region_Code"]).astype(str).str.strip()

    # Include filter (optional)
    include = getattr(cfg.regions, "include", None)
    if include is None and isinstance(getattr(cfg, "regions", None), dict):
        include = cfg.regions.get("include")

    if include:
        include = [str(x).strip().zfill(2) for x in include]
        best = best[best["Region_Code"].isin(include)].copy()

    # --- Birth forecasting (NEW: uses forecasting module) ---
    births_forecast = predict_births_all_regions(
        best_models_df=best,
        project_root=cfg.project_root,
        start_year=start_year,
        end_year=end_year,
        interval_level=float(getattr(cfg.forecast, "interval_level", 0.95))
        if hasattr(cfg, "forecast") and hasattr(cfg.forecast, "interval_level")
        else 0.95,
    )

    # Normalize + keep stable schema
    if not births_forecast.empty:
        births_forecast["Region_Code"] = births_forecast["Region_Code"].astype("string").str.strip().str.zfill(2)
        births_forecast["Region_Name"] = births_forecast.get("Region_Name", births_forecast["Region_Code"]).astype(str).str.strip()
        births_forecast["Year"] = pd.to_numeric(births_forecast["Year"], errors="coerce").astype("Int64")
        births_forecast["Forecast_Births"] = pd.to_numeric(births_forecast["Forecast_Births"], errors="coerce").astype(float)
        births_forecast = births_forecast.dropna(subset=["Region_Code", "Year", "Forecast_Births"]).copy()
        births_forecast["Year"] = births_forecast["Year"].astype(int)
        births_forecast = births_forecast.sort_values(["Region_Code", "Year"]).reset_index(drop=True)

    births_out_path = forecasts_dir / f"birth_forecast_{start_year}_{end_year}.csv"
    _ensure_parent(births_out_path)
    births_forecast.to_csv(births_out_path, index=False)
    logger.info("Saved births forecast: %s", births_out_path)

    # Summary (simple, robust)
    if births_forecast.empty:
        summary_df = pd.DataFrame(columns=["Region_Code", "Region_Name", "Model", "Forecast_Min", "Forecast_Max", "Forecast_Mean"])
    else:
        summary_df = (
            births_forecast.groupby(["Region_Code", "Region_Name", "Model"], as_index=False)["Forecast_Births"]
            .agg(Forecast_Min="min", Forecast_Max="max", Forecast_Mean="mean")
        )
        summary_df["Forecast_Year_Start"] = int(start_year)
        summary_df["Forecast_Year_End"] = int(end_year)
        summary_df = summary_df.sort_values(["Region_Code"]).reset_index(drop=True)

    summary_path = metrics_dir / "forecast_summary_births.csv"
    _ensure_parent(summary_path)
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved births forecast summary: %s", summary_path)

    # Optional sanity plots
    _maybe_plot_births(
        births_forecast,
        start_year=start_year,
        end_year=end_year,
        figures_dir=figures_dir,
        region_codes=["01", "12", "14"],
    )
    logger.info("Saved plots (subset): %s", figures_dir)

    # --- Population 0–19 (NEW: wrapper) ---
    pop_forecast = predict_population_0_19(
        births_forecast=births_forecast,
        db_path=db_path,
        start_year=start_year,
        end_year=end_year,
    )

    # Normalize schema
    pop_forecast = pop_forecast.copy()
    pop_forecast["Region_Code"] = pop_forecast["Region_Code"].astype("string").str.strip().str.zfill(2)
    pop_forecast["Region_Name"] = pop_forecast.get("Region_Name", pop_forecast["Region_Code"]).astype(str).str.strip()
    pop_forecast["Age"] = pd.to_numeric(pop_forecast["Age"], errors="coerce").astype("Int64")
    pop_forecast["Year"] = pd.to_numeric(pop_forecast["Year"], errors="coerce").astype("Int64")
    pop_forecast["Forecast_Population"] = pd.to_numeric(pop_forecast["Forecast_Population"], errors="coerce").astype(float)
    pop_forecast = pop_forecast.dropna(subset=["Region_Code", "Year", "Age", "Forecast_Population"]).copy()
    pop_forecast["Age"] = pop_forecast["Age"].astype(int)
    pop_forecast["Year"] = pop_forecast["Year"].astype(int)
    pop_forecast = pop_forecast.sort_values(["Region_Code", "Year", "Age"]).reset_index(drop=True)

    pop_out_path = forecasts_dir / f"population_0_19_forecast_{start_year}_{end_year}.csv"
    _ensure_parent(pop_out_path)
    pop_forecast.to_csv(pop_out_path, index=False)
    logger.info("Saved population forecast: %s", pop_out_path)

    # --- Education costs ---
    # Support both cfg.costs.* and cfg.raw["costs"] styles
    if hasattr(cfg, "costs"):
        grund_rel = getattr(cfg.costs, "grundskola_cost_table", None)
        gymn_rel = getattr(cfg.costs, "gymnasieskola_cost_table", None)
        anchor_max_year = getattr(cfg.costs, "anchor_max_year", None)
        extrapolation = getattr(cfg.costs, "extrapolation", "carry_forward")
        annual_growth_rate = float(getattr(cfg.costs, "annual_growth_rate", 0.0))
    else:
        cost_cfg = (cfg.raw or {}).get("costs", {}) if isinstance(getattr(cfg, "raw", {}), dict) else {}
        grund_rel = cost_cfg.get("grundskola_cost_table")
        gymn_rel = cost_cfg.get("gymnasieskola_cost_table")
        anchor_max_year = cost_cfg.get("anchor_max_year")
        extrapolation = cost_cfg.get("extrapolation", "carry_forward")
        annual_growth_rate = float(cost_cfg.get("annual_growth_rate", 0.0))

    if not grund_rel or not gymn_rel:
        raise ValueError("Missing grundskola_cost_table / gymnasieskola_cost_table in config")

    grund_path = _resolve(cfg, grund_rel)
    gymn_path = _resolve(cfg, gymn_rel)

    grund, gymn = load_cost_tables(
        grund_path,
        gymn_path,
        anchor_max_year=anchor_max_year,
    )

    costs_df = compute_education_costs(
        pop_forecast=pop_forecast,
        grund=grund,
        gymn=gymn,
        extrapolation=str(extrapolation),
        annual_growth_rate=float(annual_growth_rate),
    )

    # Ensure stable saved schema for dashboards
    costs_df = costs_df.copy()
    costs_df["Region_Code"] = costs_df["Region_Code"].astype("string").str.strip().str.zfill(2)
    if "Region_Name" in costs_df.columns:
        costs_df["Region_Name"] = costs_df["Region_Name"].astype(str).str.strip()
    costs_df["School_Type"] = costs_df["School_Type"].astype(str).str.strip().str.lower()

    costs_out_path = forecasts_dir / f"education_costs_forecast_{start_year}_{end_year}.csv"
    _ensure_parent(costs_out_path)
    costs_df.to_csv(costs_out_path, index=False)
    logger.info("Saved education costs forecast: %s", costs_out_path)

    logger.info("Forecast pipeline complete.")
