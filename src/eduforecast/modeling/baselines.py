"""src/eduforecast/modeling/baselines.py"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ForecastModel(Protocol):
    """Minimal protocol expected by the training + forecasting code."""
    def predict(self, steps: int) -> np.ndarray: ...


@dataclass(frozen=True)
class NaiveLastModel:
    """Forecast = last observed value repeated."""
    last_value: float

    def predict(self, steps: int) -> np.ndarray:
        return np.full(shape=(int(steps),), fill_value=float(self.last_value), dtype=float)


@dataclass(frozen=True)
class DriftModel:
    """
    Forecast with a simple drift (trend) estimated from the series:
      drift = (y_last - y_first) / (n-1)
      yhat[t] = y_last + drift*(t+1)
    """
    last_value: float
    drift_per_step: float

    def predict(self, steps: int) -> np.ndarray:
        steps = int(steps)
        horizon = np.arange(1, steps + 1, dtype=float)
        return (self.last_value + self.drift_per_step * horizon).astype(float)


def fit_naive_last(y_train: np.ndarray) -> NaiveLastModel:
    y = np.asarray(y_train, dtype=float)
    return NaiveLastModel(last_value=float(y[-1]))


def fit_drift(y_train: np.ndarray) -> DriftModel:
    y = np.asarray(y_train, dtype=float)
    if len(y) < 2:
        return DriftModel(last_value=float(y[-1]), drift_per_step=0.0)
    drift = float((y[-1] - y[0]) / max(len(y) - 1, 1))
    return DriftModel(last_value=float(y[-1]), drift_per_step=drift)
