"""src/eduforecast/modeling/selection.py"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


PrimaryMetric = Literal["rmse", "mae", "smape"]


@dataclass(frozen=True)
class SelectionResult:
    best_model: str
    best_row: dict

    def to_dict(self) -> dict:
        return {"Best_Model": self.best_model, **self.best_row}


def pick_best_model(rows: list[dict], *, primary: PrimaryMetric = "rmse") -> SelectionResult:
    """
    rows: list of dicts containing:
      - "Model"
      - "RMSE"/"MAE"/"SMAPE" columns (uppercase)
    """
    if not rows:
        raise ValueError("No model rows to select from.")

    metric_key = {"rmse": "RMSE", "mae": "MAE", "smape": "SMAPE"}[primary]

    best_row = None
    best_val = float("inf")

    for r in rows:
        v = r.get(metric_key, np.nan)
        try:
            v = float(v)
        except Exception:
            v = np.nan

        if not np.isfinite(v):
            continue

        if v < best_val:
            best_val = v
            best_row = r

    if best_row is None:
        raise ValueError("All metric values are NaN; cannot select a best model.")

    best_model = str(best_row.get("Model", "unknown"))
    return SelectionResult(best_model=best_model, best_row=dict(best_row))
