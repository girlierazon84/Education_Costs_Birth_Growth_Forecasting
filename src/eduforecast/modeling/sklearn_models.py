"""src/eduforecast/modeling/sklearn_models.py"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SklearnWrapper:
    """
    Optional adapter for sklearn regressors.

    NOTE:
    This is NOT compatible with the pure time-series interface (predict(steps=...))
    unless you implement feature engineering to build X for future steps.
    """
    model: Any

    def predict_X(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=float)
