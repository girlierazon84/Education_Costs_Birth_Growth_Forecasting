"""src/eduforecast/modeling/sklearn_models.py"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SklearnWrapper:
    """
    Optional adapter if you later use sklearn models.
    For pure time-series forecasts you usually need feature engineering (X).
    """
    model: Any

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=float)
