"""tests/unit/test_evaluation.py"""

from __future__ import annotations

import pandas as pd


def test_best_models_registry_contract() -> None:
    df = pd.DataFrame(
        {
            "Region_Code": ["1", "12"],
            "Region_Name": ["Stockholms län", "Skåne län"],
            "Best_Model": ["ARIMA", "XGBRegressor"],
        }
    )
    # mimic run_forecast normalization
    df["Region_Code"] = df["Region_Code"].astype("string").str.strip().str.zfill(2)
    df["Region_Name"] = df.get("Region_Name", df["Region_Code"]).astype(str).str.strip()

    assert set(["Region_Code", "Region_Name", "Best_Model"]).issubset(df.columns)
    assert df["Region_Code"].tolist() == ["01", "12"]
    assert df["Best_Model"].iloc[0] == "ARIMA"
