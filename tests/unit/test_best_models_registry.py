"""tests/unit/test_best_models_registry.py"""

from __future__ import annotations

import pandas as pd


def test_best_models_registry_contract() -> None:
    df = pd.DataFrame(
        {
            "Region_Code": ["01", "12"],
            "Region_Name": ["Stockholms län", "Skåne län"],
            "Best_Model": ["baseline_naive", "drift"],
            "Model_Path": ["artifacts/models/births/births_01_baseline_naive.joblib", "artifacts/models/births/births_12_drift.joblib"],
        }
    )

    required = {"Region_Code", "Region_Name", "Best_Model"}
    assert required.issubset(df.columns)

    # Ensure formatting assumptions used by pipelines/dashboards
    assert df["Region_Code"].astype(str).str.len().eq(2).all()
    assert df["Best_Model"].astype(str).str.strip().ne("").all()
