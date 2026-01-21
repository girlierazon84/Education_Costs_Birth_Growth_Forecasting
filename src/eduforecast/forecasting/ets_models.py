"""src/eduforecast/forecasting/ets_models.py"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class NaiveLastModel:
    """
    Naive baseline: repeats last observed value.

    Pickle-safe because it lives in eduforecast.forecasting.ets_models
    (not __main__).
    """
    last_value: float

    def predict(self, steps: int) -> np.ndarray:
        return np.full(shape=(int(steps),), fill_value=float(self.last_value), dtype=float)


@dataclass
class ETSModel:
    """
    Wrapper around a statsmodels Holt-Winters fitted result.
    """
    fitted: Any  # statsmodels result object

    def predict(self, steps: int) -> np.ndarray:
        # statsmodels HW results typically support .forecast(steps)
        return np.asarray(self.fitted.forecast(int(steps)), dtype=float)


@dataclass
class ETSNoSeason:
    """
    Compatibility shim for legacy joblib artifacts that were saved with
    __main__.ETSNoSeason.

    It returns a constant forecast (level-only).
    """
    level_: float

    def predict(self, steps: int) -> np.ndarray:
        return np.full(shape=(int(steps),), fill_value=float(self.level_), dtype=float)
