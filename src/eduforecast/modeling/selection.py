"""src/eduforecast/modeling/selection.py"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


PrimaryMetric = Literal["rmse", "mae", "smape"]


@dataclass(frozen=True)
class SelectionResult:
    best_model: str
    best_row: dict

    def to_dict(self) -> dict:
        return {"Best_Model": self.best_model, **self.best_row}


def pick_best_model(
    rows: list[dict],
    *,
    primary: PrimaryMetric = "rmse",
) -> SelectionResult:
    """
    rows: list of dicts with keys: Model + RMSE/MAE/SMAPE
    """
    if not rows:
        raise ValueError("No model rows to select from.")

    metric_key = {"rmse": "RMSE", "mae": "MAE", "smape": "SMAPE"}[primary]

    df = pd.DataFrame(rows).copy()
    if metric_key not in df.columns:
        raise KeyError(f"Missing metric column {metric_key} in selection rows.")

    df[metric_key] = pd.to_numeric(df[metric_key], errors="coerce")
    df = df.dropna(subset=[metric_key]).copy()
    if df.empty:
        raise ValueError("All metric values are NaN; cannot select a best model.")

    best_idx = int(df[metric_key].idxmin())
    best = df.loc[best_idx].to_dict()

    best_model = str(best.get("Model", "unknown"))
    return SelectionResult(best_model=best_model, best_row=best)
