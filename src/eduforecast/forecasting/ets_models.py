"""src/eduforecast/forecasting/ets_models.py"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ETSNoSeason:
    """
    Legacy compatibility shim.

    Some older joblib artifacts were saved from notebooks/scripts where the class
    was recorded as __main__.ETSNoSeason. When loading those artifacts inside the
    CLI/package, pickle needs a real importable class with the same name.

    This model returns a constant forecast (level-only). It should NOT be used
    for new training.
    """
    level_: float

    def predict(self, steps: int) -> np.ndarray:
        steps = int(steps)
        if steps < 0:
            raise ValueError("steps must be >= 0")
        return np.full(shape=(steps,), fill_value=float(self.level_), dtype=float)
