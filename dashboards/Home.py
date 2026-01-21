"""dashboards/Home.py"""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def project_root() -> Path:
    # dashboards/Home.py -> dashboards -> project root
    return Path(__file__).resolve().parents[1]


def _exists(rel: str) -> bool:
    return (project_root() / rel).exists()


def _page_link(path: str, label: str) -> None:
    """Prefer st.page_link (Streamlit multipage). Fallback to st.switch_page for older versions."""
    if hasattr(st, "page_link"):
        st.page_link(path, label=label)
    else:
        if st.button(label, use_container_width=True):
            st.switch_page(path)


def main() -> None:
    st.set_page_config(page_title="EduForecast • Home", layout="wide")

    st.title("EduForecast Dashboard")
    st.caption("Navigate: EDA → Model Comparison → Forecasts & Costs")

    st.subheader("Project status")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.write("**Forecast artifacts**")
        st.write("✅" if _exists("artifacts/forecasts/birth_forecast_2024_2030.csv") else "❌", "birth_forecast_2024_2030.csv")
        st.write("✅" if _exists("artifacts/forecasts/population_0_19_forecast_2024_2030.csv") else "❌", "population_0_19_forecast_2024_2030.csv")
        st.write("✅" if _exists("artifacts/forecasts/education_costs_forecast_2024_2030.csv") else "❌", "education_costs_forecast_2024_2030.csv")

        st.write("")
        st.write("**Report pack (optional)**")
        st.write("✅" if _exists("artifacts/forecasts/report_pack/2024_2030/tables/education_costs_summary.csv") else "❌", "report_pack/2024_2030/...")

    with c2:
        st.write("**Metrics artifacts**")
        st.write("✅" if _exists("artifacts/metrics/best_models_births.csv") else "❌", "best_models_births.csv")
        st.write("✅" if _exists("artifacts/metrics/forecast_summary_births.csv") else "❌", "forecast_summary_births.csv")

    with c3:
        st.write("**Raw / external data**")
        st.write("✅" if _exists("data/raw/birth_data_per_region.csv") else "❌", "data/raw/birth_data_per_region.csv")
        st.write("✅" if _exists("data/external/grundskola_costs_per_child.csv") else "❌", "grundskola_costs_per_child.csv")
        st.write("✅" if _exists("data/external/gymnasieskola_costs_per_child.csv") else "❌", "gymnasieskola_costs_per_child.csv")

    st.divider()

    st.subheader("Navigate")
    st.markdown(
        """
- **EDA**: sanity-check births trends + cost table coverage.
- **Model Comparison**: best model per region (+ optional model scores).
- **Forecasts & Costs**: explore costs (choose ONE basis).
"""
    )

    b1, b2, b3 = st.columns(3)
    with b1:
        _page_link("pages/1_EDA.py", "📊 Open EDA")
    with b2:
        _page_link("pages/2_Model_Comparison.py", "🧠 Model Comparison")
    with b3:
        _page_link("pages/3_Forecast_and_Costs.py", "📈 Forecasts & Costs")

    st.divider()

    st.subheader("Run pipeline")
    st.code("eduforecast forecast", language="bash")
    st.caption("Rebuilds artifacts in artifacts/forecasts used by these pages.")

    st.info(
        "SCB totals (e.g., ~55B for gymnasiet 2024) should be compared to **Current (nominal)**. "
        "Fixed vs Current are different price bases — do not add them."
    )


if __name__ == "__main__":
    main()
