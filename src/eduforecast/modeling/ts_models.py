"""src/eduforecast/modeling/ts_models.py"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing


@dataclass(frozen=True)
class ETSModel:
    """
    Wrapper around statsmodels Holt-Winters fitted result.
    Importable path ensures joblib can unpickle it.
    """
    fitted: Any

    def predict(self, steps: int) -> np.ndarray:
        steps = int(steps)
        if steps < 0:
            raise ValueError("steps must be >= 0")
        if not hasattr(self.fitted, "forecast"):
            raise TypeError("ETS fitted object missing .forecast()")
        return np.asarray(self.fitted.forecast(steps), dtype=float)


def fit_ets(y_train: np.ndarray) -> ETSModel:
    y = np.asarray(y_train, dtype=float)
    if y.size == 0:
        raise ValueError("Cannot fit ETS on empty series.")

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
