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
    is_demographic_override: bool = False
    original_cv_winner: str = ""

    def to_dict(self) -> dict:
        return {
            "Best_Model": self.best_model,
            "Is_Demographic_Override": self.is_demographic_override,
            "Original_CV_Winner": self.original_cv_winner or self.best_model,
            **self.best_row
        }

def pick_best_model(
    rows: list[dict],
    *,
    primary: PrimaryMetric | str = "rmse",
    region_code: str | None = None
) -> SelectionResult:
    """
    Selects the best forecasting model based on the lowest historical cross-validation metric,
    then automatically evaluates and applies Swedish regional demographic safety constraints.

    rows: list of dicts containing:
        - "Model" ("baseline_naive", "drift", "exp_smoothing")
        - "RMSE"/"MAE"/"SMAPE" columns (uppercase)
    primary: "rmse"|"mae"|"smape" (preferred), but also tolerates uppercase variants.
    region_code: Optional 2-digit SCB code (e.g., "01" for Stockholm, "03" for Uppsala) to apply safety overrides.
    """
    if not rows:
        raise ValueError("No model rows to select from.")

    p = str(primary).strip().lower()
    if p in {"rmse", "mae", "smape"}:
        metric_key = {"rmse": "RMSE", "mae": "MAE", "smape": "SMAPE"}[p]
    elif p in {"rmse".upper(), "mae".upper(), "smape".upper()}:
        metric_key = p.upper()
    else:
        metric_key = "RMSE"

    best_row = None
    best_val = float("inf")

    # 1. Standard Mathematical Optimization Loop
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

    # 2. Swedish Demographic Safety Boundary Verification
    is_overridden = False
    original_winner = best_model

    if region_code is not None:
        clean_rc = str(region_code).strip().zfill(2)

        # ✅ FIXED: Expanded the guardrail to cover Stockholm ("01") along with Uppsala ("03") and Halland ("13")
        # Prevents high-growth / metropolitan expansion zones from being locked into flat Naïve lines.
        if clean_rc in {"01", "03", "13"} and best_model == "baseline_naive":
            # Locate the row corresponding to Exponential Smoothing to carry over its metrics
            ets_row = next((r for r in rows if str(r.get("Model")).strip() == "exp_smoothing"), None)
            if ets_row is not None:
                best_model = "exp_smoothing"
                best_row = ets_row
                is_overridden = True

    return SelectionResult(
        best_model=best_model,
        best_row=dict(best_row),
        is_demographic_override=is_overridden,
        original_cv_winner=original_winner
    )
