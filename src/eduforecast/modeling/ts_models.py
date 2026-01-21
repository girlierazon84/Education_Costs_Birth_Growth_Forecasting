"""src/eduforecast/modeling/ts_models.py"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing


@dataclass(frozen=True)
class ETSModel:
    """
    Wrapper around a statsmodels Holt-Winters fitted model.
    Stored as a normal importable class so joblib can unpickle it.
    """
    fitted: Any

    def predict(self, steps: int) -> np.ndarray:
        return np.asarray(self.fitted.forecast(int(steps)), dtype=float)


def fit_ets(y_train: np.ndarray) -> ETSModel:
    y = np.asarray(y_train, dtype=float)

    # Preferred: damped additive trend, no seasonality
    try:
        model = ExponentialSmoothing(
            y,
            trend="add",
            seasonal=None,
            damped_trend=True,
            initialization_method="estimated",
        )
        fitted = model.fit(optimized=True)
        return ETSModel(fitted=fitted)
    except Exception:
        # Fallback: simpler model if the above fails
        model = ExponentialSmoothing(
            y,
            trend=None,
            seasonal=None,
            initialization_method="estimated",
        )
        fitted = model.fit(optimized=True)
        return ETSModel(fitted=fitted)
