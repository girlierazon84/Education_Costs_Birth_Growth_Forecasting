"""src/eduforecast/pipelines/run_forecast.py"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eduforecast.common.config import AppConfig
from eduforecast.costs import compute_education_costs, load_cost_tables
from eduforecast.forecasting.predict_births import predict_births_all_regions
from eduforecast.forecasting.predict_population import predict_population_0_19
from eduforecast.io.writers import write_forecast_artifact

logger = logging.getLogger(__name__)


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _get(cfg_obj: Any, key: str, default: Any = None) -> Any:
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


def _maybe_plot_births(
    births_forecast: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
    figures_dir: Path,
    region_codes: list[str],
) -> None:
    """Save quick sanity plots for a small set of regions if present."""
    try:
        figures_dir.mkdir(parents=True, exist_ok=True)
        years = np.arange(start_year, end_year + 1, dtype=int)

        available = set(births_forecast["Region_Code"].astype(str).str.zfill(2).unique())
        region_codes = [rc for rc in region_codes if rc in available]
        if not region_codes:
            return

        for rc in region_codes:
            sub = births_forecast[births_forecast["Region_Code"] == rc].sort_values("Year")
            if sub.empty:
                continue

            rn = str(sub["Region_Name"].iloc[0]) if "Region_Name" in sub.columns else ""
            model = str(sub["Model"].iloc[0]) if "Model" in sub.columns else ""
            y = pd.to_numeric(sub["Forecast_Births"], errors="coerce").to_numpy(dtype=float)

            if len(y) != len(years):
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
    # --- DB path ---
    sqlite_path = _get(cfg.database, "sqlite_path")
    if not sqlite_path:
        raise ValueError("Missing database.sqlite_path in config")
    db_path = _resolve(cfg, sqlite_path)

    # --- Forecast horizon ---
    forecast_cfg = cfg.forecast
    start_year = _safe_int(_get(forecast_cfg, "start_year", 2024), 2024)
    end_year = _safe_int(_get(forecast_cfg, "end_year", 2030), 2030)
    if end_year < start_year:
        raise ValueError("forecast.end_year must be >= forecast.start_year")

    # --- Output dirs ---
    paths_cfg = cfg.paths
    metrics_dir = _resolve(cfg, _get(paths_cfg, "metrics_dir"))
    forecasts_dir = _resolve(cfg, _get(paths_cfg, "forecasts_dir"))
    figures_dir = _resolve(cfg, _get(paths_cfg, "figures_dir")) / "forecasts" / "births"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # --- Best models registry ---
    best_models_path = metrics_dir / "best_models_births.csv"
    if not best_models_path.exists():
        raise FileNotFoundError(f"Missing {best_models_path}. Run `eduforecast train` first.")

    best = pd.read_csv(best_models_path, dtype={"Region_Code": "string"})
    best["Region_Code"] = best["Region_Code"].astype("string").str.strip().str.zfill(2)
    best["Region_Name"] = best.get("Region_Name", best["Region_Code"]).astype(str).str.strip()

    include = _get(cfg.regions, "include")
    if include:
        include = [str(x).strip().zfill(2) for x in include]
        best = best[best["Region_Code"].isin(include)].copy()

    interval_level = float(_get(forecast_cfg, "interval_level", 0.95))

    # --- Birth forecasting ---
    births_forecast = predict_births_all_regions(
        best_models_df=best,
        project_root=cfg.project_root,
        start_year=start_year,
        end_year=end_year,
        interval_level=interval_level,
    )

    births_forecast = births_forecast.copy()
    births_forecast["Region_Code"] = births_forecast["Region_Code"].astype("string").str.strip().str.zfill(2)
    births_forecast["Region_Name"] = births_forecast.get("Region_Name", births_forecast["Region_Code"]).astype(str).str.strip()
    births_forecast["Year"] = pd.to_numeric(births_forecast["Year"], errors="coerce").astype("Int64")
    births_forecast["Forecast_Births"] = pd.to_numeric(births_forecast["Forecast_Births"], errors="coerce").astype(float)
    births_forecast = births_forecast.dropna(subset=["Region_Code", "Year", "Forecast_Births"]).copy()
    births_forecast["Year"] = births_forecast["Year"].astype(int)
    births_forecast = births_forecast.sort_values(["Region_Code", "Year"]).reset_index(drop=True)

    births_out_path = forecasts_dir / f"birth_forecast_{start_year}_{end_year}.csv"
    write_forecast_artifact(births_forecast, births_out_path)
    logger.info("Saved births forecast: %s", births_out_path)

    # --- Summary ---
    if births_forecast.empty:
        summary_df = pd.DataFrame(
            columns=["Region_Code", "Region_Name", "Model", "Forecast_Min", "Forecast_Max", "Forecast_Mean"]
        )
    else:
        summary_df = (
            births_forecast.groupby(["Region_Code", "Region_Name", "Model"], as_index=False)["Forecast_Births"]
            .agg(Forecast_Min="min", Forecast_Max="max", Forecast_Mean="mean")
            .sort_values(["Region_Code"])
            .reset_index(drop=True)
        )
        summary_df["Forecast_Year_Start"] = int(start_year)
        summary_df["Forecast_Year_End"] = int(end_year)

    summary_path = metrics_dir / "forecast_summary_births.csv"
    write_forecast_artifact(summary_df, summary_path)
    logger.info("Saved births forecast summary: %s", summary_path)

    _maybe_plot_births(
        births_forecast,
        start_year=start_year,
        end_year=end_year,
        figures_dir=figures_dir,
        region_codes=["01", "12", "14"],
    )

    # --- Population 0–19 ---
    pop_forecast = predict_population_0_19(
        births_forecast=births_forecast,
        db_path=db_path,
        start_year=start_year,
        end_year=end_year,
    )

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
    write_forecast_artifact(pop_forecast, pop_out_path)
    logger.info("Saved population forecast: %s", pop_out_path)

    # --- Costs ---
    costs_cfg = getattr(cfg, "costs", None)
    if costs_cfg is None:
        costs_cfg = _get(cfg.raw, "costs", {})  # fallback

    grund_rel = _get(costs_cfg, "grundskola_cost_table")
    gymn_rel = _get(costs_cfg, "gymnasieskola_cost_table")
    if not grund_rel or not gymn_rel:
        raise ValueError("Missing grundskola_cost_table / gymnasieskola_cost_table in config")

    grund_path = _resolve(cfg, grund_rel)
    gymn_path = _resolve(cfg, gymn_rel)

    tables = load_cost_tables(
        grund_path,
        gymn_path,
        anchor_max_year=_get(costs_cfg, "anchor_max_year", None),
    )

    costs_df = compute_education_costs(
        pop_forecast=pop_forecast,
        grund=tables.grund,
        gymn=tables.gymn,
        extrapolation=str(_get(costs_cfg, "extrapolation", "carry_forward")),
        annual_growth_rate=float(_get(costs_cfg, "annual_growth_rate", 0.0)),
    )

    costs_df = costs_df.copy()
    costs_df["Region_Code"] = costs_df["Region_Code"].astype("string").str.strip().str.zfill(2)
    if "Region_Name" in costs_df.columns:
        costs_df["Region_Name"] = costs_df["Region_Name"].astype(str).str.strip()
    costs_df["School_Type"] = costs_df["School_Type"].astype(str).str.strip().str.lower()

    costs_out_path = forecasts_dir / f"education_costs_forecast_{start_year}_{end_year}.csv"
    write_forecast_artifact(costs_df, costs_out_path)
    logger.info("Saved education costs forecast: %s", costs_out_path)

    logger.info("Forecast pipeline complete.")
