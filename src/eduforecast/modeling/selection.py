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


def pick_best_model(rows: list[dict], *, primary: PrimaryMetric | str = "rmse") -> SelectionResult:
    """
    rows: list of dicts containing:
      - "Model"
      - "RMSE"/"MAE"/"SMAPE" columns (uppercase)

    primary: "rmse"|"mae"|"smape" (preferred), but also tolerates "RMSE"/"MAE"/"SMAPE".
    """
    if not rows:
        raise ValueError("No model rows to select from.")

    p = str(primary).strip().lower()
    # tolerate callers accidentally passing uppercase metric keys
    if p in {"rmse", "mae", "smape"}:
        metric_key = {"rmse": "RMSE", "mae": "MAE", "smape": "SMAPE"}[p]
    elif p in {"rmse".upper(), "mae".upper(), "smape".upper()}:
        metric_key = p.upper()
    else:
        metric_key = "RMSE"  # safe fallback

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

    best_model = str(best_row.get("Model", "unknown")).strip()
    return SelectionResult(best_model=best_model, best_row=dict(best_row))
