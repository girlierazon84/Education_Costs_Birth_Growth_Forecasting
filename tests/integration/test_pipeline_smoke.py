"""tests/integration/test_pipeline_smoke.py"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from eduforecast.pipelines.run_etl import run_etl
from eduforecast.pipelines.run_forecast import run_forecast


def test_pipeline_smoke_etl_and_forecast(
    minimal_data,  # fixture from conftest.py
    monkeypatch,
) -> None:
    cfg = minimal_data

    # --- Create minimal best_models_births.csv required by run_forecast ---
    metrics_dir = (cfg.project_root / "artifacts" / "metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    best_models_path = metrics_dir / "best_models_births.csv"
    pd.DataFrame(
        {
            "Region_Code": ["01", "12"],
            "Region_Name": ["Stockholms län", "Skåne län"],
            "Best_Model": ["DummyModel", "DummyModel"],
        }
    ).to_csv(best_models_path, index=False)

    # --- Monkeypatch forecasting to keep this a FAST smoke test ---
    def fake_predict_births_all_regions(*, best_models_df, project_root, start_year, end_year, interval_level):
        years = list(range(int(start_year), int(end_year) + 1))
        rows = []
        for r in best_models_df.itertuples(index=False):
            for y in years:
                rows.append(
                    {
                        "Region_Code": str(r.Region_Code).zfill(2),
                        "Region_Name": getattr(r, "Region_Name", str(r.Region_Code).zfill(2)),
                        "Year": y,
                        "Forecast_Births": 100.0,
                        "Model": "DummyModel",
                    }
                )
        return pd.DataFrame(rows)

    def fake_predict_population_0_19(*, births_forecast, db_path, start_year, end_year):
        years = list(range(int(start_year), int(end_year) + 1))
        regions = births_forecast[["Region_Code", "Region_Name"]].drop_duplicates()
        rows = []
        for r in regions.itertuples(index=False):
            for y in years:
                for age in range(0, 20):
                    # give a bit of population in school ages too
                    base = 10.0 if 7 <= age <= 19 else 1.0
                    rows.append(
                        {
                            "Region_Code": str(r.Region_Code).zfill(2),
                            "Region_Name": str(r.Region_Name),
                            "Age": age,
                            "Year": y,
                            "Forecast_Population": base,
                        }
                    )
        return pd.DataFrame(rows)

    import eduforecast.forecasting.predict_births as mod_births
    import eduforecast.forecasting.predict_population as mod_pop

    monkeypatch.setattr(mod_births, "predict_births_all_regions", fake_predict_births_all_regions)
    monkeypatch.setattr(mod_pop, "predict_population_0_19", fake_predict_population_0_19)

    # --- Run ETL then Forecast ---
    run_etl(cfg)
    run_forecast(cfg)

    # --- Assert forecast artifacts exist ---
    forecasts_dir = cfg.project_root / "artifacts" / "forecasts"
    births_csv = forecasts_dir / "birth_forecast_2024_2025.csv"
    pop_csv = forecasts_dir / "population_0_19_forecast_2024_2025.csv"
    costs_csv = forecasts_dir / "education_costs_forecast_2024_2025.csv"

    assert births_csv.exists(), f"missing {births_csv}"
    assert pop_csv.exists(), f"missing {pop_csv}"
    assert costs_csv.exists(), f"missing {costs_csv}"

    # --- Basic sanity: non-empty outputs ---
    b = pd.read_csv(births_csv)
    p = pd.read_csv(pop_csv)
    c = pd.read_csv(costs_csv)

    assert len(b) > 0
    assert len(p) > 0
    assert len(c) > 0
