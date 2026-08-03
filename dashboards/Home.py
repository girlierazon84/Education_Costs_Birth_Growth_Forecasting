"""
dashboards/Home.py
Home page for the EduForecast Streamlit dashboard.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


FORECAST_FILES = {
    "Birth forecast": "artifacts/forecasts/birth_forecast_2024_2030.csv",
    "Population forecast": "artifacts/forecasts/population_0_19_forecast_2024_2030.csv",
    "Education cost forecast": "artifacts/forecasts/education_costs_forecast_2024_2030.csv",
}

METRIC_FILES = {
    "Best birth models": "artifacts/metrics/best_models_births.csv",
    "Birth forecast summary": "artifacts/metrics/forecast_summary_births.csv",
}

SOURCE_FILES = {
    "Birth data": "data/raw/birth_data_per_region.csv",
    "Compulsory school costs": "data/external/grundskola_costs_per_child.csv",
    "Upper-secondary school costs": "data/external/gymnasieskola_costs_per_child.csv",
}

EDUCATION_COST_FILE = (
    "artifacts/forecasts/education_costs_forecast_2024_2030.csv"
)


def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


def resolve_path(relative_path: str) -> Path:
    """Return an absolute path inside the project."""
    return project_root() / relative_path


def file_exists(relative_path: str) -> bool:
    """Check whether a project file exists."""
    return resolve_path(relative_path).exists()


def format_number(value: float, decimals: int = 0) -> str:
    """Format a number using spaces as thousands separators."""
    if pd.isna(value):
        return "—"

    formatted = f"{value:,.{decimals}f}"
    return formatted.replace(",", " ")


def format_currency(value: float) -> str:
    """Format a SEK amount using readable units."""
    if pd.isna(value):
        return "—"

    absolute_value = abs(value)

    if absolute_value >= 1_000_000_000:
        return f"SEK {value / 1_000_000_000:,.1f} bn"

    if absolute_value >= 1_000_000:
        return f"SEK {value / 1_000_000:,.1f} m"

    if absolute_value >= 1_000:
        return f"SEK {value / 1_000:,.1f} k"

    return f"SEK {format_number(value)}"


def render_page_link(path: str, label: str, icon: str) -> None:
    """Render a dashboard navigation link."""
    full_label = f"{icon} {label}"

    if hasattr(st, "page_link"):
        st.page_link(path, label=full_label, use_container_width=True)
    else:
        if st.button(full_label, use_container_width=True):
            st.switch_page(path)


@st.cache_data(show_spinner=False)
def load_forecast_overview() -> dict[str, float | int] | None:
    """Load headline values from the education cost forecast."""
    file_path = resolve_path(EDUCATION_COST_FILE)

    if not file_path.exists():
        return None

    try:
        df = pd.read_csv(file_path)
    except (OSError, pd.errors.ParserError):
        return None

    required_columns = {
        "Year",
        "Forecast_Students",
        "Current_Total_Cost_kr",
        "Fixed_Total_Cost_kr",
    }

    if not required_columns.issubset(df.columns):
        return None

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Forecast_Students"] = pd.to_numeric(
        df["Forecast_Students"],
        errors="coerce",
    )
    df["Current_Total_Cost_kr"] = pd.to_numeric(
        df["Current_Total_Cost_kr"],
        errors="coerce",
    )
    df["Fixed_Total_Cost_kr"] = pd.to_numeric(
        df["Fixed_Total_Cost_kr"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "Year",
            "Forecast_Students",
            "Current_Total_Cost_kr",
            "Fixed_Total_Cost_kr",
        ]
    ).copy()

    if df.empty:
        return None

    first_year = int(df["Year"].min())
    final_year = int(df["Year"].max())

    final_year_data = df[df["Year"] == final_year]

    total_student_years = float(df["Forecast_Students"].sum())
    total_current_cost = float(df["Current_Total_Cost_kr"].sum())
    total_fixed_cost = float(df["Fixed_Total_Cost_kr"].sum())

    final_year_students = float(
        final_year_data["Forecast_Students"].sum()
    )
    final_year_current_cost = float(
        final_year_data["Current_Total_Cost_kr"].sum()
    )

    region_count = (
        int(df["Region_Code"].nunique())
        if "Region_Code" in df.columns
        else 0
    )

    return {
        "first_year": first_year,
        "final_year": final_year,
        "region_count": region_count,
        "total_student_years": total_student_years,
        "total_current_cost": total_current_cost,
        "total_fixed_cost": total_fixed_cost,
        "final_year_students": final_year_students,
        "final_year_current_cost": final_year_current_cost,
    }


def render_status_group(
    title: str,
    files: dict[str, str],
) -> None:
    """Display the availability of a group of project files."""
    st.markdown(f"**{title}**")

    for label, relative_path in files.items():
        available = file_exists(relative_path)
        status = "Available" if available else "Missing"
        icon = "✅" if available else "⚠️"

        st.markdown(
            f"{icon} **{label}**  \n"
            f"<span style='color: #6b7280; font-size: 0.88rem;'>"
            f"{status} · `{relative_path}`"
            f"</span>",
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="EduForecast",
        page_icon="📊",
        layout="wide",
    )

    st.title("EduForecast")
    st.markdown(
        "Explore regional birth trends, model performance, population forecasts, "
        "and projected education costs for Sweden."
    )

    forecast_overview = load_forecast_overview()

    if forecast_overview is not None:
        period = (
            f"{forecast_overview['first_year']}–"
            f"{forecast_overview['final_year']}"
        )

        st.subheader("Forecast overview")
        st.caption(
            f"Headline figures from the current forecast period, {period}."
        )

        metric_1, metric_2, metric_3, metric_4 = st.columns(4)

        metric_1.metric(
            "Forecast period",
            period,
        )

        metric_2.metric(
            "Regions covered",
            format_number(forecast_overview["region_count"]),
        )

        metric_3.metric(
            "Student-years",
            format_number(
                forecast_overview["total_student_years"]
            ),
            help=(
                "The total forecast student population summed across "
                "all years in the forecast period."
            ),
        )

        metric_4.metric(
            "Total cost · current prices",
            format_currency(
                forecast_overview["total_current_cost"]
            ),
            help=(
                "Combined forecast education expenditure using current "
                "prices across the full period."
            ),
        )

        final_1, final_2, final_3 = st.columns(3)

        final_1.metric(
            f"Students in {forecast_overview['final_year']}",
            format_number(
                forecast_overview["final_year_students"]
            ),
        )

        final_2.metric(
            f"Cost in {forecast_overview['final_year']}",
            format_currency(
                forecast_overview["final_year_current_cost"]
            ),
        )

        final_3.metric(
            "Total cost · fixed prices",
            format_currency(
                forecast_overview["total_fixed_cost"]
            ),
            help=(
                "Combined forecast expenditure expressed using a "
                "constant price basis."
            ),
        )

    else:
        st.info(
            "The forecast overview will appear here after the education "
            "cost forecast has been generated."
        )

    st.divider()

    st.subheader("Explore the dashboard")
    st.caption(
        "Start with the data overview, review model performance, "
        "then examine the final forecasts and cost estimates."
    )

    page_1, page_2, page_3 = st.columns(3)

    with page_1:
        st.markdown("### Data overview")
        st.write(
            "Review birth trends, missing years, unusual values, "
            "and the coverage of the education cost tables."
        )
        render_page_link(
            "pages/1_EDA.py",
            "Open data overview",
            "📊",
        )

    with page_2:
        st.markdown("### Model comparison")
        st.write(
            "Compare forecasting models by region and review the "
            "model selected for each regional birth series."
        )
        render_page_link(
            "pages/2_Model_Comparison.py",
            "Open model comparison",
            "🧠",
        )

    with page_3:
        st.markdown("### Forecasts and costs")
        st.write(
            "Explore projected student numbers and education costs "
            "by year, region, school type, and price basis."
        )
        render_page_link(
            "pages/3_Forecast_and_Costs.py",
            "Open forecasts and costs",
            "📈",
        )

    st.divider()

    with st.expander("Data and forecast availability", expanded=False):
        st.caption(
            "This section shows whether the files required by the "
            "dashboard are available in the deployed project."
        )

        status_1, status_2, status_3 = st.columns(3)

        with status_1:
            render_status_group(
                "Forecast files",
                FORECAST_FILES,
            )

        with status_2:
            render_status_group(
                "Model outputs",
                METRIC_FILES,
            )

        with status_3:
            render_status_group(
                "Source data",
                SOURCE_FILES,
            )

        report_pack_path = (
            "artifacts/forecasts/report_pack/"
            "2024_2030/tables/education_costs_summary.csv"
        )

        st.divider()

        report_pack_available = file_exists(report_pack_path)
        report_icon = "✅" if report_pack_available else "ℹ️"
        report_status = (
            "Available"
            if report_pack_available
            else "Not generated"
        )

        st.markdown(
            f"{report_icon} **Optional report pack** — {report_status}"
        )

    with st.expander("Price basis and interpretation", expanded=False):
        st.markdown(
            """
            **Current prices** include projected changes in prices over time
            and are the appropriate basis when comparing the results with
            nominal published expenditure figures.

            **Fixed prices** hold the price basis constant. They are useful
            for understanding changes caused mainly by student numbers and
            education demand.

            Current-price and fixed-price totals are alternative views of
            the same forecast. They should not be added together.
            """
        )

    with st.expander("Rebuild the forecast locally", expanded=False):
        st.write(
            "Run the following command from the project root:"
        )
        st.code(
            "python -m eduforecast.cli forecast "
            "--config-path configs/config.yaml",
            language="bash",
        )
        st.caption(
            "This regenerates the forecast files used by the dashboard."
        )


if __name__ == "__main__":
    main()