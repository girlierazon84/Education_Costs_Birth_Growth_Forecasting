"""src/eduforecast/forecasting/ets_models.py"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ETSNoSeason:
    """
    Simple fallback ETS-like wrapper that can be pickled safely because it lives in a module.

    It mimics a "statsmodels-ish" interface:
        model.predict(steps=horizon)

    NOTE:
    - This is only meant to support legacy saved artifacts that referenced ETSNoSeason.
    - Prefer saving real statsmodels objects going forward.
    """
    level_: float

    def predict(self, steps: int) -> np.ndarray:
        return np.full(shape=(int(steps),), fill_value=float(self.level_), dtype=float)
