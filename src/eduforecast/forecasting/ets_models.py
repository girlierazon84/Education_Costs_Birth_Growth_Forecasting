"""src/eduforecast/forecasting/ets_models.py"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ETSNoSeason:
    """
    Legacy compatibility shim for old joblib artifacts that referenced:
        __main__.ETSNoSeason

    It returns a constant forecast. This is ONLY for loading legacy models.
    New training should never save this class.
    """
    level_: float

    def predict(self, steps: int) -> np.ndarray:
        return np.full(shape=(int(steps),), fill_value=float(self.level_), dtype=float)
