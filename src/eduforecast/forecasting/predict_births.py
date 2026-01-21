"""src/eduforecast/forecasting/predict_births.py"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from eduforecast.forecasting.intervals import IntervalResult, empirical_pi, normal_pi
from eduforecast.forecasting.ets_models import ETSNoSeason  # legacy compat shim only


logger = logging.getLogger(__name__)


def _install_pickle_compat_shims() -> None:
    """
    Allow loading legacy joblib artifacts saved when classes were recorded as __main__.ClassName.
    We only shim the legacy names that appear in old pickles.
    """
    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        return
    # Old pickle referenced __main__.ETSNoSeason
    setattr(main_mod, "ETSNoSeason", ETSNoSeason)


@dataclass(frozen=True)
class BirthForecastOutput:
    region_code: str
    region_name: str
    model_name: str
    years: np.ndarray
    yhat: np.ndarray
    intervals: IntervalResult | None

    def to_frame(self) -> pd.DataFrame:
        base = pd.DataFrame(
            {
                "Region_Code": self.region_code,
                "Region_Name": self.region_name,
                "Year": self.years.astype(int),
                "Model": self.model_name,
                "Forecast_Births": self.yhat.astype(float),
            }
        )
        if self.intervals is None:
            return base
        iv = self.intervals.to_frame(self.years, "Forecast_Births")
        return base.merge(iv, on="Year", how="left")


def _safe_str(x: Any, default: str = "") -> str:
    try:
        return str(x).strip()
    except Exception:
        return default


def _predict_steps(model: Any, steps: int) -> np.ndarray:
    """
    Support common forecast interfaces:
      - wrapper models: model.predict(steps=steps) or model.predict(steps)
      - statsmodels results: model.forecast(steps)
    """
    steps = int(steps)

    if hasattr(model, "predict"):
        try:
            yhat = model.predict(steps=steps)
            return np.squeeze(np.asarray(yhat, dtype=float))
        except TypeError:
            # maybe predict(steps) without kw
            yhat = model.predict(steps)
            return np.squeeze(np.asarray(yhat, dtype=float))

    if hasattr(model, "forecast"):
        yhat = model.forecast(steps)
        return np.squeeze(np.asarray(yhat, dtype=float))

    raise TypeError("Loaded model object does not support predict/forecast steps interface.")


def _extract_residuals(payload: dict[str, Any]) -> np.ndarray | None:
    res = payload.get("residuals", None)
    if res is None:
        return None
    arr = np.asarray(res, dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr if arr.size > 10 else None


def _extract_sigma(payload: dict[str, Any]) -> float | None:
    sigma = payload.get("sigma", None)
    if sigma is None:
        return None
    try:
        s = float(sigma)
        return s if np.isfinite(s) and s > 0 else None
    except Exception:
        return None


def predict_births_for_region(
    *,
    region_code: str,
    region_name: str,
    model_path: Path,
    start_year: int,
    end_year: int,
    interval_level: float = 0.95,
) -> BirthForecastOutput:
    rc = str(region_code).strip().zfill(2)
    rn = _safe_str(region_name, rc)

    _install_pickle_compat_shims()

    try:
        payload = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model for {rc} ({rn}) at {model_path}: {e}") from e

    model = payload["model"]
    model_name = _safe_str(payload.get("model_name", payload.get("best_model", "unknown")))

    steps = (int(end_year) - int(start_year)) + 1
    years = np.arange(int(start_year), int(end_year) + 1, dtype=int)

    yhat = _predict_steps(model, steps=steps)
    if yhat.ndim != 1 or len(yhat) != steps:
        raise ValueError(f"Bad forecast shape for {rc}: {yhat.shape} expected ({steps},)")

    intervals: IntervalResult | None = None
    residuals = _extract_residuals(payload)
    if residuals is not None:
        intervals = empirical_pi(yhat, residuals=residuals, level=interval_level)
    else:
        sigma = _extract_sigma(payload)
        if sigma is not None:
            intervals = normal_pi(yhat, sigma=sigma, level=interval_level)

    return BirthForecastOutput(
        region_code=rc,
        region_name=rn,
        model_name=model_name,
        years=years,
        yhat=yhat.astype(float),
        intervals=intervals,
    )


def predict_births_all_regions(
    best_models_df: pd.DataFrame,
    *,
    project_root: Path,
    start_year: int,
    end_year: int,
    interval_level: float = 0.95,
) -> pd.DataFrame:
    required = {"Region_Code", "Model_Path"}
    missing = required - set(best_models_df.columns)
    if missing:
        raise KeyError(f"best_models_df missing columns: {sorted(missing)}")

    out_frames: list[pd.DataFrame] = []
    failures: list[str] = []

    for r in best_models_df.itertuples(index=False):
        rc = str(getattr(r, "Region_Code")).strip().zfill(2)
        rn = _safe_str(getattr(r, "Region_Name", rc), rc)

        mp = Path(str(getattr(r, "Model_Path")))
        mp = mp if mp.is_absolute() else (project_root / mp).resolve()

        if not mp.exists():
            failures.append(f"{rc} (missing file)")
            continue

        try:
            pred = predict_births_for_region(
                region_code=rc,
                region_name=rn,
                model_path=mp,
                start_year=start_year,
                end_year=end_year,
                interval_level=interval_level,
            )
            out_frames.append(pred.to_frame())
        except Exception as e:
            failures.append(f"{rc} ({type(e).__name__})")
            logger.warning("Skipping region %s due to forecast error: %s", rc, e)

    if failures:
        logger.warning("Birth forecast skipped %d region(s): %s", len(failures), ", ".join(failures))

    if not out_frames:
        return pd.DataFrame(columns=["Region_Code", "Region_Name", "Year", "Model", "Forecast_Births"])

    return pd.concat(out_frames, ignore_index=True).sort_values(["Region_Code", "Year"]).reset_index(drop=True)
