"""src/eduforecast/pipelines/run_train.py"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from eduforecast.common.config import AppConfig
from eduforecast.forecasting.ets_models import ETSModel, NaiveLastModel

logger = logging.getLogger(__name__)


# ---------------- Metrics ----------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


# -------------- CV splitting --------------
def last_n_years_cv(years: np.ndarray, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Walk-forward: use the last n_splits years as 1-step test points.
    Each split trains on all years < test_year, tests on test_year.
    """
    years_sorted = np.array(sorted(set(years.tolist())))
    if len(years_sorted) <= n_splits + 3:
        n_splits = max(1, len(years_sorted) - 3)

    test_years = years_sorted[-n_splits:]
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
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
    """
    Support both dict-style and attribute-style config.
    """
    if cfg_obj is None:
        return default
    if hasattr(cfg_obj, key):
        v = getattr(cfg_obj, key)
        return default if v is None else v
    if isinstance(cfg_obj, dict):
        return cfg_obj.get(key, default)
    return default


# ---------------- Models ----------------
def fit_naive_last(y_train: np.ndarray) -> NaiveLastModel:
    return NaiveLastModel(last_value=float(y_train[-1]))


def fit_ets(y_train: np.ndarray) -> ETSModel:
    try:
        model = ExponentialSmoothing(
            y_train,
            trend="add",
            seasonal=None,
            damped_trend=True,
            initialization_method="estimated",
        )
        fitted = model.fit(optimized=True)
        return ETSModel(fitted=fitted)
    except Exception:
        model = ExponentialSmoothing(
            y_train,
            trend=None,
            seasonal=None,
            initialization_method="estimated",
        )
        fitted = model.fit(optimized=True)
        return ETSModel(fitted=fitted)


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

    births["Region_Code"] = births["Region_Code"].astype(str).str.strip().str.zfill(2)
    births["Region_Name"] = births.get("Region_Name", births["Region_Code"]).astype(str).str.strip()
    births["Year"] = pd.to_numeric(births["Year"], errors="coerce").astype("Int64")
    births["Number"] = pd.to_numeric(births["Number"], errors="coerce")
    births = births.dropna(subset=["Year", "Number"]).copy()
    births["Year"] = births["Year"].astype(int)

    start_year = _safe_int(_get(cfg.modeling, "start_year", 1968), 1968)
    births = births[births["Year"] >= start_year].copy()

    include = _get(cfg.regions, "include")
    if include:
        include = [str(x).strip().zfill(2) for x in include]
        births = births[births["Region_Code"].isin(include)].copy()

    candidates = _get(cfg.modeling, "candidates", ["baseline_naive", "exp_smoothing"])
    metric_primary = str(_get(cfg.modeling, "metric_primary", "rmse")).lower()
    cv_cfg = _get(cfg.modeling, "cv", {}) or {}
    n_splits = _safe_int(_get(cv_cfg, "n_splits", 5), 5)

    # Output folders (supports cfg.paths.* OR dict)
    models_dir = _resolve(cfg, _get(cfg.paths, "models_dir")) / "births"
    metrics_dir = _resolve(cfg, _get(cfg.paths, "metrics_dir"))
    figures_dir = _resolve(cfg, _get(cfg.paths, "figures_dir")) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    best_models: Dict[str, Dict[str, Any]] = {}

    for (rc, rn), g in births.groupby(["Region_Code", "Region_Name"], dropna=False):
        gg = g.sort_values("Year").copy()
        years = gg["Year"].to_numpy(dtype=int)
        y = gg["Number"].to_numpy(dtype=float)

        if gg["Year"].nunique() != len(gg):
            logger.warning("Duplicates found for Region %s (%s). Fix duplicates before training.", rc, rn)

        splits = last_n_years_cv(years, n_splits=n_splits)
        if len(splits) == 0:
            logger.warning("Not enough data for CV in Region %s (%s). Skipping.", rc, rn)
            continue

        region_results: list[tuple[str, Dict[str, Any], list[int], np.ndarray, np.ndarray]] = []

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
            valid = np.isfinite(y_true) & np.isfinite(y_pred)

            if valid.sum() == 0:
                logger.warning("No valid CV points for %s %s model=%s", rc, rn, model_name)
                continue

            y_true_v = y_true[valid]
            y_pred_v = y_pred[valid]

            row = {
                "Region_Code": rc,
                "Region_Name": rn,
                "Model": model_name,
                "Start_Year": int(years.min()),
                "End_Year": int(years.max()),
                "CV_Points": int(len(y_true_v)),
                "RMSE": rmse(y_true_v, y_pred_v),
                "MAE": mae(y_true_v, y_pred_v),
                "SMAPE": smape(y_true_v, y_pred_v),
            }
            all_rows.append(row)

            region_results.append((model_name, row, split_years, y_true, y_pred))

        if not region_results:
            continue

        def key_fn(item):
            _, row, *_ = item
            return row["RMSE"] if metric_primary == "rmse" else row["MAE"]

        region_results.sort(key=key_fn)
        best_name, best_row, best_years, best_true, best_pred = region_results[0]

        # fit best on full history and save
        if best_name == "baseline_naive":
            best_model = fit_naive_last(y)
        else:
            best_model = fit_ets(y)

        payload = {
            "model_name": best_name,
            "region_code": rc,
            "region_name": rn,
            "train_year_min": int(years.min()),
            "train_year_max": int(years.max()),
            "model": best_model,
        }

        model_path = models_dir / f"births_{rc}_{best_name}.joblib"
        joblib.dump(payload, model_path)

        # Store relative path for portability
        rel_model_path = model_path.relative_to(cfg.project_root).as_posix()
        best_models[rc] = {"Region_Name": rn, "Best_Model": best_name, "Model_Path": rel_model_path}

        # diagnostic plot
        try:
            plt.figure()
            plt.plot(years, y)
            plt.scatter(best_years, best_pred)
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
