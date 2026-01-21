"""tests/unit/test_costs.py"""

from __future__ import annotations

import pandas as pd
import pytest

from eduforecast.costs.cost_per_child import cost_schedule_for_years
from eduforecast.costs.total_costs import compute_education_costs


def test_cost_schedule_carry_forward() -> None:
    costs = pd.DataFrame(
        {
            "Year": [2023, 2024],
            "Fixed_cost_per_child_kr": [100.0, 110.0],
            "Current_cost_per_child_kr": [200.0, 220.0],
        }
    )

    sched = cost_schedule_for_years(costs, start_year=2024, end_year=2026, method="carry_forward")

    assert sched["Year"].tolist() == [2024, 2025, 2026]
    assert set(["Fixed_cost_per_child_kr", "Current_cost_per_child_kr", "Cost_Year"]).issubset(sched.columns)

    # carry-forward should keep 2024 values for 2025/2026
    assert sched.loc[sched["Year"] == 2025, "Fixed_cost_per_child_kr"].iloc[0] == 110.0
    assert sched.loc[sched["Year"] == 2026, "Current_cost_per_child_kr"].iloc[0] == 220.0

    # and the referenced cost-year should remain 2024
    assert sched.loc[sched["Year"] == 2025, "Cost_Year"].iloc[0] == 2024
    assert sched.loc[sched["Year"] == 2026, "Cost_Year"].iloc[0] == 2024


def test_cost_schedule_growth_rate() -> None:
    costs = pd.DataFrame(
        {
            "Year": [2024],
            "Fixed_cost_per_child_kr": [100.0],
            "Current_cost_per_child_kr": [200.0],
        }
    )

    sched = cost_schedule_for_years(
        costs,
        start_year=2024,
        end_year=2026,
        method="growth_rate",
        annual_growth_rate=0.1,
    )

    assert sched["Year"].tolist() == [2024, 2025, 2026]
    assert set(["Fixed_cost_per_child_kr", "Current_cost_per_child_kr", "Cost_Year"]).issubset(sched.columns)

    # 2024 -> base
    assert sched.loc[sched["Year"] == 2024, "Fixed_cost_per_child_kr"].iloc[0] == 100.0

    # 2025 -> *1.1
    assert sched.loc[sched["Year"] == 2025, "Fixed_cost_per_child_kr"].iloc[0] == pytest.approx(110.0)

    # 2026 -> *1.1^2
    assert sched.loc[sched["Year"] == 2026, "Fixed_cost_per_child_kr"].iloc[0] == pytest.approx(121.0)

    # referenced cost year should still be 2024
    assert sched.loc[sched["Year"] == 2025, "Cost_Year"].iloc[0] == 2024
    assert sched.loc[sched["Year"] == 2026, "Cost_Year"].iloc[0] == 2024


def test_compute_education_costs_basic_math() -> None:
    # Population forecast includes ages 7-16 (grund) and 17-19 (gymn)
    pop = pd.DataFrame(
        {
            "Region_Code": ["01", "01", "01", "01"],
            "Region_Name": ["Stockholm"] * 4,
            "Age": [7, 16, 17, 19],
            "Year": [2024, 2024, 2024, 2024],
            "Forecast_Population": [10, 20, 5, 5],
        }
    )

    grund = pd.DataFrame(
        {"Year": [2024], "Fixed_cost_per_child_kr": [100.0], "Current_cost_per_child_kr": [200.0]}
    )
    gymn = pd.DataFrame(
        {"Year": [2024], "Fixed_cost_per_child_kr": [300.0], "Current_cost_per_child_kr": [400.0]}
    )

    out = compute_education_costs(pop_forecast=pop, grund=grund, gymn=gymn, extrapolation="carry_forward")

    assert set(out["School_Type"].unique()) == {"grundskola", "gymnasieskola"}
    assert set(["Forecast_Students", "Fixed_Total_Cost_kr", "Current_Total_Cost_kr"]).issubset(out.columns)

    grund_row = out[(out["Region_Code"] == "01") & (out["Year"] == 2024) & (out["School_Type"] == "grundskola")].iloc[0]
    gymn_row = out[(out["Region_Code"] == "01") & (out["Year"] == 2024) & (out["School_Type"] == "gymnasieskola")].iloc[0]

    # grund students = 10 + 20 = 30
    assert grund_row["Forecast_Students"] == pytest.approx(30.0)
    assert grund_row["Fixed_Total_Cost_kr"] == pytest.approx(30.0 * 100.0)
    assert grund_row["Current_Total_Cost_kr"] == pytest.approx(30.0 * 200.0)

    # gymn students = 5 + 5 = 10
    assert gymn_row["Forecast_Students"] == pytest.approx(10.0)
    assert gymn_row["Fixed_Total_Cost_kr"] == pytest.approx(10.0 * 300.0)
    assert gymn_row["Current_Total_Cost_kr"] == pytest.approx(10.0 * 400.0)
