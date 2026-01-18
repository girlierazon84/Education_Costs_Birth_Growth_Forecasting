"""src/eduforecast/pipelines/run_forecast.py"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eduforecast.common.config import AppConfig
from eduforecast.features.build_features import build_population_forecast_features
from eduforecast.costs.total_costs import load_cost_tables, compute_education_costs

logger = logging.getLogger(__name__)


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _resolve(cfg: AppConfig, maybe_path: str | Path) -> Path:
    """Resolve relative paths against project_root."""
    p = Path(maybe_path)
    return p if p.is_absolute() else (cfg.project_root / p).resolve()


def run_forecast(cfg: AppConfig) -> None:
    """
    Run the full forecast pipeline:
      1) Births forecast per region (from best saved model per region)
      2) Population 0–19 cohort forecast (features table)
      3) Education costs forecast (Fixed and Current bases; DO NOT add them)
    """
    # --- Core config ---
    db_path = _resolve(cfg, cfg.database.sqlite_path)

    start_year = _safe_int(cfg.forecast.start_year, 2024)
    end_year = _safe_int(cfg.forecast.end_year, 2030)
    if end_year < start_year:
        raise ValueError("forecast.end_year must be >= forecast.start_year")

    horizon_years = (end_year - start_year) + 1
    years_future = np.arange(start_year, end_year + 1, dtype=int)

    # --- Output dirs ---
    metrics_dir = _resolve(cfg, cfg.paths.metrics_dir)
    forecasts_dir = _resolve(cfg, cfg.paths.forecasts_dir)
    figures_dir = _resolve(cfg, cfg.paths.figures_dir) / "forecasts" / "births"

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

    include = getattr(cfg.regions, "include", None)
    if include:
        include = [str(x).zfill(2) for x in include]
        best = best[best["Region_Code"].isin(include)].copy()

    # Plot only a few regions for sanity (if present)
    plot_region_codes = ["01", "12", "14"]
    plot_region_codes = [rc for rc in plot_region_codes if rc in set(best["Region_Code"])]

    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    # --- Birth forecasting ---
    for _, r in best.iterrows():
        rc = str(r["Region_Code"]).zfill(2)
        rn = str(r.get("Region_Name", "")).strip()
        model_path = _resolve(cfg, str(r["Model_Path"]))

        if not model_path.exists():
            logger.warning("Model file not found for %s (%s): %s", rc, rn, model_path)
            continue

        payload = joblib.load(model_path)
        model_name = str(payload.get("model_name", r.get("Best_Model", "unknown"))).strip()
        model = payload["model"]

        # Expect a statsmodels-like API: predict(steps=horizon)
        yhat = np.asarray(model.predict(steps=horizon_years), dtype=float)
        yhat = np.squeeze(yhat)

        if yhat.ndim != 1 or len(yhat) != horizon_years:
            raise ValueError(f"Bad forecast shape for {rc}: {yhat.shape} (expected ({horizon_years},))")

        for year, pred in zip(years_future, yhat):
            rows.append(
                {
                    "Region_Code": rc,
                    "Region_Name": rn,
                    "Year": int(year),
                    "Model": model_name,
                    "Forecast_Births": float(pred),
                }
            )

        summary_rows.append(
            {
                "Region_Code": rc,
                "Region_Name": rn,
                "Model": model_name,
                "Forecast_Year_Start": int(start_year),
                "Forecast_Year_End": int(end_year),
                "Forecast_Min": float(np.min(yhat)),
                "Forecast_Max": float(np.max(yhat)),
                "Forecast_Mean": float(np.mean(yhat)),
            }
        )

        if rc in plot_region_codes:
            try:
                plt.figure()
                plt.plot(years_future, yhat)
                plt.title(f"Birth Forecast {start_year}-{end_year}: {rc} {rn} ({model_name})")
                plt.xlabel("Year")
                plt.ylabel("Forecast births (float)")
                plt.savefig(figures_dir / f"birth_forecast_{rc}.png", dpi=150, bbox_inches="tight")
                plt.close()
            except Exception:
                logger.exception("Plotting failed for region %s", rc)

    forecast_df = pd.DataFrame(rows).sort_values(["Region_Code", "Year"]).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(["Region_Code"]).reset_index(drop=True)

    births_out_path = forecasts_dir / f"birth_forecast_{start_year}_{end_year}.csv"
    forecast_df.to_csv(births_out_path, index=False)

    summary_path = metrics_dir / "forecast_summary_births.csv"
    summary_df.to_csv(summary_path, index=False)

    logger.info("Birth forecasting complete.")
    logger.info("Saved forecasts: %s", births_out_path)
    logger.info("Saved forecast summary: %s", summary_path)
    logger.info("Saved plots (subset): %s", figures_dir)

    # --- Population 0–19 via features bridge (cohort aging + survival + migration) ---
    pop_forecast = build_population_forecast_features(
        births_forecast=forecast_df,
        db_path=db_path,
        start_year=start_year,
        end_year=end_year,
    )

    pop_out_path = forecasts_dir / f"population_0_19_forecast_{start_year}_{end_year}.csv"
    pop_forecast.to_csv(pop_out_path, index=False)
    logger.info("Saved population forecast: %s", pop_out_path)

    # --- Education costs ---
    grund_path = _resolve(cfg, cfg.costs.grundskola_cost_table)
    gymn_path = _resolve(cfg, cfg.costs.gymnasieskola_cost_table)

    grund, gymn = load_cost_tables(
        grund_path,
        gymn_path,
        anchor_max_year=getattr(cfg.costs, "anchor_max_year", None),
    )

    costs_df = compute_education_costs(
        pop_forecast=pop_forecast,
        grund=grund,
        gymn=gymn,
        extrapolation=str(cfg.costs.extrapolation),
        annual_growth_rate=float(cfg.costs.annual_growth_rate),
    )

    # Ensure stable saved schema for dashboards
    costs_df["Region_Code"] = costs_df["Region_Code"].astype("string").str.strip().str.zfill(2)
    if "Region_Name" in costs_df.columns:
        costs_df["Region_Name"] = costs_df["Region_Name"].astype(str).str.strip()
    costs_df["School_Type"] = costs_df["School_Type"].astype(str).str.strip().str.lower()

    costs_out_path = forecasts_dir / f"education_costs_forecast_{start_year}_{end_year}.csv"
    costs_df.to_csv(costs_out_path, index=False)
    logger.info("Saved education costs forecast: %s", costs_out_path)
