"""tests/unit/test_evaluation.py"""

from __future__ import annotations

import math

import numpy as np
import pytest

from eduforecast.modeling.evaluation import compute_metrics, mae, rmse, smape


def test_rmse_basic() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])  # error: [0,0,1]
    # MSE = (0 + 0 + 1)/3 = 1/3, RMSE = sqrt(1/3)
    expected = math.sqrt(1.0 / 3.0)
    assert abs(rmse(y_true, y_pred) - expected) < 1e-12


def test_mae_basic() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 1.0])  # abs err: [1,0,2]
    expected = (1.0 + 0.0 + 2.0) / 3.0
    assert abs(mae(y_true, y_pred) - expected) < 1e-12


def test_smape_basic() -> None:
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 190.0])

    # SMAPE per-point:
    # denom1=(|100|+|110|)/2=105 => |100-110|/105=10/105
    # denom2=(|200|+|190|)/2=195 => |200-190|/195=10/195
    expected = ((10.0 / 105.0) + (10.0 / 195.0)) / 2.0 * 100.0
    assert abs(smape(y_true, y_pred) - expected) < 1e-12


def test_smape_zero_zero_safe() -> None:
    # Both zero -> denom becomes 0, implementation replaces denom=1.0
    y_true = np.array([0.0])
    y_pred = np.array([0.0])
    assert smape(y_true, y_pred) == 0.0


def test_metrics_filter_nans_and_infs() -> None:
    # _to_valid_arrays should drop invalid pairs
    y_true = [1.0, float("nan"), 3.0, float("inf")]
    y_pred = [1.0, 2.0, float("nan"), 4.0]

    pack = compute_metrics(y_true, y_pred)

    # Only valid pair is (1.0, 1.0)
    assert pack.rmse == 0.0
    assert pack.mae == 0.0
    assert pack.smape == 0.0


def test_compute_metrics_empty_after_filtering_returns_nan() -> None:
    # All invalid -> empty arrays -> rmse/mae/smape return nan
    y_true = [float("nan"), float("inf")]
    y_pred = [float("nan"), 1.0]

    pack = compute_metrics(y_true, y_pred)

    assert math.isnan(pack.rmse)
    assert math.isnan(pack.mae)
    assert math.isnan(pack.smape)


def test_metric_pack_as_dict_stable_keys() -> None:
    pack = compute_metrics([1.0, 2.0], [1.0, 3.0])
    d = pack.as_dict()
    assert set(d.keys()) == {"RMSE", "MAE", "SMAPE"}
    assert all(isinstance(v, float) for v in d.values())
