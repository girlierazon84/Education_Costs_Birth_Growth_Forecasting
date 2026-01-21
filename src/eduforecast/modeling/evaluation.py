"""src/eduforecast/modeling/evaluation.py"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


@dataclass(frozen=True)
class MetricPack:
    rmse: float
    mae: float
    smape: float

    def as_dict(self) -> dict[str, float]:
        return {"RMSE": self.rmse, "MAE": self.mae, "SMAPE": self.smape}


def compute_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> MetricPack:
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_pred), dtype=float)

    valid = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[valid]
    yp = yp[valid]

    if yt.size == 0:
        return MetricPack(rmse=float("nan"), mae=float("nan"), smape=float("nan"))

    return MetricPack(
        rmse=rmse(yt, yp),
        mae=mae(yt, yp),
        smape=smape(yt, yp),
    )
