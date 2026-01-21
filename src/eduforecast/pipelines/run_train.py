"""src/eduforecast/pipelines/run_train.py"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eduforecast.common.config import AppConfig
from eduforecast.modeling.baselines import fit_drift, fit_naive_last
from eduforecast.modeling.evaluation import compute_metrics
from eduforecast.modeling.selection import pick_best_model
from eduforecast.modeling.ts_models import fit_ets


logger = logging.getLogger(__name__)


# ---------------- CV splitting ----------------
def last_n_years_cv(years: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Walk-forward: use the last n_splits years as 1-step test points.
    Each split trains on all years < test_year, tests on test_year.
    """
    years_sorted = np.array(sorted(set(years.tolist())), dtype=int)
    if len(years_sorted) <= n_splits + 3:
        n_splits = max(1, len(years_sorted) - 3)

    test_years = years_sorted[-n_splits:]
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for ty in test_years:
        train_idx = np.where(years < ty)[0]
        test_idx = np.where(years == ty)[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        splits.append((train_idx, test_idx))
    return splits


# ---------------- Helpers ----------------
def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _resolve(cfg: AppConfig, maybe_path: str | Path) -> Path:
    p = Path(maybe_path)
    return p if p.is_absolute() else (cfg.project_root / p).resolve()


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


# ---------------- IO ----------------
def _read_births_sqlite(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as con:
        return pd.read_sql("SELECT * FROM birth_data_per_region", con)


def run_train(cfg: AppConfig) -> None:
    # Resolve db path (supports cfg.database.sqlite_path OR dict)
    sqlite_path = _get(cfg.database, "sqlite_path")
    if not sqlite_path:
        raise ValueError("Missing database.sqlite_path in config")
    db_path = _resolve(cfg, sqlite_path)

    births = _read_births_sqlite(db_path)

    # Normalize schema defensively
    births["Region_Code"] = births["Region_Code"].astype("string").str.strip().str.zfill(2)
    births["Region_Name"] = births.get("Region_Name", births["Region_Code"]).astype(str).str.strip()
    births["Year"] = pd.to_numeric(births["Year"], errors="coerce").astype("Int64")
    births["Number"] = pd.to_numeric(births["Number"], errors="coerce")

    births = births.dropna(subset=["Year", "Number"]).copy()
    births["Year"] = births["Year"].astype(int)
    births["Number"] = births["Number"].astype(float)

    start_year = _safe_int(_get(cfg.modeling, "start_year", 1968), 1968)
    births = births[births["Year"] >= start_year].copy()

    include = _get(cfg.regions, "include")
    if include:
        include = [str(x).strip().zfill(2) for x in include]
        births = births[births["Region_Code"].isin(include)].copy()

    candidates = _get(cfg.modeling, "candidates", ["baseline_naive", "drift", "exp_smoothing"])
    metric_primary = str(_get(cfg.modeling, "metric_primary", "rmse")).lower()
    cv_cfg = _get(cfg.modeling, "cv", {}) or {}
    n_splits = _safe_int(_get(cv_cfg, "n_splits", 5), 5)

    # Output folders
    models_dir = _resolve(cfg, _get(cfg.paths, "models_dir")) / "births"
    metrics_dir = _resolve(cfg, _get(cfg.paths, "metrics_dir"))
    figures_dir = _resolve(cfg, _get(cfg.paths, "figures_dir")) / "models"

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    best_models: dict[str, dict[str, Any]] = {}

    for (rc, rn), g in births.groupby(["Region_Code", "Region_Name"], dropna=False):
        gg = g.sort_values("Year").copy()
        years = gg["Year"].to_numpy(dtype=int)
        y = gg["Number"].to_numpy(dtype=float)

        if gg["Year"].nunique() != len(gg):
            logger.warning("Duplicates found for Region %s (%s). Fix duplicates before training.", rc, rn)

        splits = last_n_years_cv(years, n_splits=n_splits)
        if not splits:
            logger.warning("Not enough data for CV in Region %s (%s). Skipping.", rc, rn)
            continue

        region_rows: list[dict[str, Any]] = []
        region_debug: dict[str, dict[str, Any]] = {}

        for model_name in candidates:
            preds: list[float] = []
            truths: list[float] = []
            split_years: list[int] = []

            for train_idx, test_idx in splits:
                y_train = y[train_idx]
                y_test = y[test_idx]

                try:
                    if model_name == "baseline_naive":
                        m = fit_naive_last(y_train)
                        y_pred = m.predict(steps=len(y_test))
                    elif model_name == "drift":
                        m = fit_drift(y_train)
                        y_pred = m.predict(steps=len(y_test))
                    elif model_name == "exp_smoothing":
                        m = fit_ets(y_train)
                        y_pred = m.predict(steps=len(y_test))
                    else:
                        continue
                except Exception as e:
                    logger.exception("Model %s failed for region %s: %s", model_name, rc, e)
                    y_pred = np.full_like(y_test, np.nan, dtype=float)

                preds.extend(np.asarray(y_pred, dtype=float).tolist())
                truths.extend(np.asarray(y_test, dtype=float).tolist())
                split_years.extend(years[test_idx].tolist())

            y_true = np.asarray(truths, dtype=float)
            y_pred = np.asarray(preds, dtype=float)

            mpack = compute_metrics(y_true, y_pred)
            if not np.isfinite(mpack.rmse):
                logger.warning("No valid CV points for %s %s model=%s", rc, rn, model_name)
                continue

            row = {
                "Region_Code": rc,
                "Region_Name": rn,
                "Model": model_name,
                "Start_Year": int(years.min()),
                "End_Year": int(years.max()),
                "CV_Points": int(np.isfinite(y_true).sum()),
                **mpack.as_dict(),
            }
            all_rows.append(row)
            region_rows.append(row)

            # store for plotting / residuals
            region_debug[model_name] = {
                "split_years": split_years,
                "y_true": y_true,
                "y_pred": y_pred,
            }

        if not region_rows:
            continue

        sel = pick_best_model(region_rows, primary=metric_primary if metric_primary in {"rmse", "mae", "smape"} else "rmse")
        best_name = sel.best_model

        # Fit best on full history
        if best_name == "baseline_naive":
            best_model = fit_naive_last(y)
        elif best_name == "drift":
            best_model = fit_drift(y)
        else:
            best_model = fit_ets(y)

        # Residuals for PI (use the CV residuals of best model)
        dbg = region_debug.get(best_name, {})
        y_true_cv = np.asarray(dbg.get("y_true", []), dtype=float)
        y_pred_cv = np.asarray(dbg.get("y_pred", []), dtype=float)
        resid = (y_true_cv - y_pred_cv)
        resid = resid[np.isfinite(resid)]
        sigma = float(np.std(resid)) if resid.size > 5 else None

        payload: dict[str, Any] = {
            "model_name": best_name,
            "region_code": rc,
            "region_name": rn,
            "train_year_min": int(years.min()),
            "train_year_max": int(years.max()),
            "model": best_model,
        }
        if resid.size > 10:
            payload["residuals"] = resid.astype(float)
        if sigma is not None and np.isfinite(sigma) and sigma > 0:
            payload["sigma"] = float(sigma)

        model_path = models_dir / f"births_{rc}_{best_name}.joblib"
        joblib.dump(payload, model_path)

        rel_model_path = model_path.relative_to(cfg.project_root).as_posix()
        best_models[rc] = {"Region_Name": rn, "Best_Model": best_name, "Model_Path": rel_model_path}

        # Diagnostic plot: actual + CV preds of best model
        try:
            dbg = region_debug.get(best_name, {})
            split_years = dbg.get("split_years", [])
            y_pred = np.asarray(dbg.get("y_pred", []), dtype=float)

            plt.figure()
            plt.plot(years, y)  # actual
            if len(split_years) == len(y_pred) and len(split_years) > 0:
                plt.scatter(split_years, y_pred)  # CV preds
            plt.title(f"Births: {rc} {rn} | Best: {best_name}")
            plt.xlabel("Year")
            plt.ylabel("Births (Number)")
            fig_path = figures_dir / f"births_{rc}_cv.png"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close()
        except Exception:
            logger.exception("Failed plotting for region %s", rc)

    metrics_df = pd.DataFrame(all_rows).sort_values(["Region_Code", "RMSE"]).reset_index(drop=True)
    metrics_path = metrics_dir / "model_comparison_births.csv"
    metrics_df.to_csv(metrics_path, index=False)

    best_df = (
        pd.DataFrame.from_dict(best_models, orient="index")
        .reset_index()
        .rename(columns={"index": "Region_Code"})
        .sort_values("Region_Code")
        .reset_index(drop=True)
    )
    best_path = metrics_dir / "best_models_births.csv"
    best_df.to_csv(best_path, index=False)

    logger.info("Training complete.")
    logger.info("Saved model comparison: %s", metrics_path)
    logger.info("Saved best model list: %s", best_path)
    logger.info("Saved models to: %s", models_dir)
