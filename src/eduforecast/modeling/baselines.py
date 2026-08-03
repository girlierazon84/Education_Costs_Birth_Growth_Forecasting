"""src/eduforecast/modeling/baselines.py"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ForecastModel(Protocol):
    """Minimal interface expected by training + forecasting code."""
    def predict(self, steps: int) -> np.ndarray: ...


@dataclass(frozen=True)
class NaiveLastModel:
    """Forecast = last observed value repeated."""
    last_value: float

    def predict(self, steps: int) -> np.ndarray:
        steps = int(steps)
        if steps < 0:
            raise ValueError("steps must be >= 0")
        return np.full(shape=(steps,), fill_value=float(self.last_value), dtype=float)


@dataclass(frozen=True)
class DriftModel:
    """
    Forecast with a simple drift estimated from the series:
        drift = (y_last - y_first) / (n-1)
        yhat[t] = y_last + drift*(t+1)
    """
    last_value: float
    drift_per_step: float

    def predict(self, steps: int) -> np.ndarray:
        steps = int(steps)
        if steps < 0:
            raise ValueError("steps must be >= 0")
        horizon = np.arange(1, steps + 1, dtype=float)
        return (self.last_value + self.drift_per_step * horizon).astype(float)

@dataclass(frozen=True)
class DampedDriftModel:
    """
    Production-ready forecast wrapper that structurally dampens linear drift.
    Prevents long-horizon birth models from projecting into impossibility.
    """
    last_value: float
    drift_per_step: float
    phi:float = 0.85   # Geometric decay modifier

    def predict(self, steps:int) -> np.ndarray:
        steps = int(steps)
        if steps < 0:
            raise ValueError("steps must be >= 0")

        horizon = np.arange(1, steps + 1, dtype=float)

        # Converts regular linear timeline [1, 2, 3] into cumulative decay multiplier:
        # Step 1: phi^1
        # Step 2: phi^1 + phi^2
        # Step 3: phi^1 + phi^2 + phi^3
        decay_steps = np.array([np.sum(self.phi ** np.arange(1, int(step) + 1)) for step in horizon], dtype=float)

        return (self.last_value + self.drift_per_step * decay_steps).astype(float)

def fit_naive_last(y_train: np.ndarray) -> NaiveLastModel:
    y = np.asarray(y_train, dtype=float)
    if y.size == 0:
        raise ValueError("Cannot fit NaiveLastModel on empty series.")
    return NaiveLastModel(last_value=float(y[-1]))

def fit_drift(y_train: np.ndarray) -> DriftModel:
    y = np.asarray(y_train, dtype=float)
    if y.size == 0:
        raise ValueError("Cannot fit DriftModel on empty series.")
    if y.size < 2:
        return DriftModel(last_value=float(y[-1]), drift_per_step=0.0)
    drift = float((y[-1] - y[0]) / max(y.size - 1, 1))
    return DriftModel(last_value=float(y[-1]), drift_per_step=drift)

def fit_damped_drift(y_train: np.ndarray, phi: float = 0.85) -> DampedDriftModel:
    """Production instantiation function to fit the new damped structure."""
    y = np.asarray(y_train, dtype=float)
    if y.size == 0:
        raise ValueError("Cannot fit DampedDriftModel on empty series.")
    if y.size < 2:
        return DampedDriftModel(last_value=float(y[-1]), drift_per_step=0.0, phi=phi)
    drift = float((y[-1] - y[0]) / max(y.size - 1, 1))
    return DampedDriftModel(last_value=float(y[-1]), drift_per_step=drift, phi=phi)