"""src/eduforecast/forecasting/intervals.py"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IntervalResult:
    """
    Standard output for forecasts with uncertainty.

    forecast: point forecast
    lower: lower prediction interval
    upper: upper prediction interval
    level: e.g. 0.8, 0.95
    kind: "pi" (prediction interval) or "ci" (confidence interval)
    """
    forecast: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    level: float
    kind: str = "pi"

    def to_frame(self, years: Iterable[int], value_col: str) -> pd.DataFrame:
        years = list(years)
        return pd.DataFrame(
            {
                "Year": years,
                value_col: self.forecast.astype(float),
                f"{value_col}_Lower_{int(self.level*100)}": self.lower.astype(float),
                f"{value_col}_Upper_{int(self.level*100)}": self.upper.astype(float),
                "Interval_Kind": self.kind,
                "Interval_Level": float(self.level),
            }
        )


def _z_from_level(level: float) -> float:
    """
    Convert central interval level to a normal z value.
    0.80 -> ~1.2816
    0.90 -> ~1.6449
    0.95 -> ~1.96
    """
    # Avoid scipy dependency; approximate using numpy via inverse error function.
    # z = sqrt(2) * erfinv(level)
    # because P(|Z| <= z) = level  => 2*Phi(z)-1 = level
    # => Phi(z) = (1+level)/2
    # z = sqrt(2) * erfinv(2*Phi(z)-1) = sqrt(2)*erfinv(level)
    # numpy has erf but not erfinv; implement stable approximation:
    # Use rational approximation for erfinv.
    x = float(level)
    x = max(min(x, 0.999999), 1e-6)

    # Winitzki approximation for erfinv
    a = 0.147
    ln = np.log(1.0 - x * x)
    first = 2.0 / (np.pi * a) + ln / 2.0
    second = ln / a
    return float(np.sign(x) * np.sqrt(np.sqrt(first * first - second) - first))


def normal_pi(yhat: np.ndarray, sigma: float, level: float = 0.95) -> IntervalResult:
    """
    Basic prediction interval assuming Normal errors with std = sigma.
    """
    yhat = np.asarray(yhat, dtype=float)
    z = _z_from_level(level)
    lower = yhat - z * sigma
    upper = yhat + z * sigma
    return IntervalResult(forecast=yhat, lower=lower, upper=upper, level=level, kind="pi")


def empirical_pi(yhat: np.ndarray, residuals: np.ndarray, level: float = 0.95) -> IntervalResult:
    """
    Empirical PI using residual quantiles, robust if residuals are non-normal.

    lower = yhat + q((1-level)/2)
    upper = yhat + q(1-(1-level)/2)
    """
    yhat = np.asarray(yhat, dtype=float)
    res = np.asarray(residuals, dtype=float)

    lo_q = np.quantile(res, (1.0 - level) / 2.0)
    hi_q = np.quantile(res, 1.0 - (1.0 - level) / 2.0)
    lower = yhat + lo_q
    upper = yhat + hi_q
    return IntervalResult(forecast=yhat, lower=lower, upper=upper, level=level, kind="pi")
