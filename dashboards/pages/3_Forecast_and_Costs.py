"""
dashboards/pages/3_Forecast_and_Costs.py

Interactive dashboard for exploring forecast student populations and
education costs by region, school type, year, and price basis.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from eduforecast.io.readers import read_births_raw
from eduforecast.preprocessing.clean_births import clean_births


CURRENT_COST_COLUMN = "Current_Total_Cost_kr"
FIXED_COST_COLUMN = "Fixed_Total_Cost_kr"

SCHOOL_TYPE_LABELS = {
    "grundskola": "Compulsory school",
    "gymnasieskola": "Upper-secondary school",
}


def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[2]


def render_plot(fig: go.Figure) -> None:
    """Render a Plotly chart across the available page width."""
    fig.update_layout(
        margin=dict(l=20, r=20, t=70, b=20),
        hovermode="x unified",
        legend_title_text="",
    )

    try:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except TypeError:
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )


def render_dataframe(df: pd.DataFrame) -> None:
    """Render a dataframe across the available page width."""
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a dataframe to UTF-8 encoded CSV bytes."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def format_number(value: float, decimals: int = 0) -> str:
    """Format a numeric value using spaces as thousands separators."""
    if pd.isna(value):
        return "—"

    formatted = f"{value:,.{decimals}f}"
    return formatted.replace(",", " ")


def format_currency(value: float, *, compact: bool = True) -> str:
    """Format a SEK amount in a readable form."""
    if pd.isna(value):
        return "—"

    absolute_value = abs(value)

    if compact:
        if absolute_value >= 1_000_000_000:
            return f"SEK {value / 1_000_000_000:,.1f} bn"
        if absolute_value >= 1_000_000:
            return f"SEK {value / 1_000_000:,.1f} m"
        if absolute_value >= 1_000:
            return f"SEK {value / 1_000:,.1f} k"

    return f"SEK {format_number(value)}"


def format_percent(value: float) -> str:
    """Format a percentage change."""
    if pd.isna(value):
        return "—"

    return f"{value:+.1f}%"


def calculate_change(first_value: float, last_value: float) -> float | None:
    """Calculate percentage change between two values."""
    if pd.isna(first_value) or pd.isna(last_value) or first_value == 0:
        return None

    return ((last_value - first_value) / first_value) * 100.0


def looks_like_bad_region_name(name: str, code: str) -> bool:
    """Identify missing or numeric region names."""
    normalized_name = str(name).strip()
    normalized_code = str(code).strip()

    return (
        not normalized_name
        or normalized_name.lower() in {"nan", "none", "<na>"}
        or normalized_name == normalized_code
        or normalized_name.isdigit()
    )


def school_type_label(value: str) -> str:
    """Return a readable school-type label."""
    normalized = str(value).strip().lower()
    return SCHOOL_TYPE_LABELS.get(normalized, normalized.replace("_", " ").title())


@st.cache_data(show_spinner=False)
def load_region_lookup_from_births() -> dict[str, str]:
    """Build a reliable Region_Code-to-Region_Name lookup."""
    file_path = (
        project_root()
        / "data"
        / "raw"
        / "birth_data_per_region.csv"
    )

    if not file_path.exists():
        return {}

    raw_births = read_births_raw(file_path)
    births = clean_births(raw_births)

    regions = births[["Region_Code", "Region_Name"]].copy()
    regions["Region_Code"] = (
        regions["Region_Code"]
        .astype("string")
        .str.strip()
        .str.zfill(2)
    )
    regions["Region_Name"] = (
        regions["Region_Name"]
        .astype("string")
        .str.strip()
    )

    regions = (
        regions
        .dropna(subset=["Region_Code", "Region_Name"])
        .sort_values(["Region_Code", "Region_Name"])
        .drop_duplicates(subset=["Region_Code"])
    )

    return dict(
        zip(
            regions["Region_Code"].astype(str),
            regions["Region_Name"].astype(str),
        )
    )


@st.cache_data(show_spinner=False)
def load_costs_csv() -> pd.DataFrame:
    """Load and prepare the education cost forecast dataset."""
    file_path = (
        project_root()
        / "artifacts"
        / "forecasts"
        / "education_costs_forecast_2024_2030.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            "The education cost forecast file could not be found.\n\n"
            f"Expected location:\n{file_path}\n\n"
            "Run the forecast pipeline before opening this page."
        )

    df = pd.read_csv(
        file_path,
        dtype={"Region_Code": "string"},
    )

    required_columns = {
        "Region_Code",
        "School_Type",
        "Year",
        "Forecast_Students",
        FIXED_COST_COLUMN,
        CURRENT_COST_COLUMN,
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "The forecast file is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    df["Region_Code"] = (
        df["Region_Code"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(2)
    )

    if "Region_Name" not in df.columns:
        df["Region_Name"] = df["Region_Code"]

    df["Region_Name"] = (
        df["Region_Name"]
        .astype("string")
        .str.strip()
    )

    region_lookup = load_region_lookup_from_births()

    if region_lookup:
        bad_name_mask = df.apply(
            lambda row: looks_like_bad_region_name(
                row["Region_Name"],
                row["Region_Code"],
            ),
            axis=1,
        )

        df.loc[bad_name_mask, "Region_Name"] = (
            df.loc[bad_name_mask, "Region_Code"]
            .map(region_lookup)
            .fillna(df.loc[bad_name_mask, "Region_Name"])
        )

    df["School_Type"] = (
        df["School_Type"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    df["School_Type_Label"] = df["School_Type"].map(school_type_label)

    df["Year"] = pd.to_numeric(
        df["Year"],
        errors="coerce",
    ).astype("Int64")

    numeric_columns = [
        "Forecast_Students",
        FIXED_COST_COLUMN,
        CURRENT_COST_COLUMN,
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=[
            "Region_Code",
            "Year",
            "Forecast_Students",
            FIXED_COST_COLUMN,
            CURRENT_COST_COLUMN,
        ]
    ).copy()

    df["Year"] = df["Year"].astype(int)

    student_denominator = df["Forecast_Students"].replace(0, pd.NA)

    df["Fixed_Per_Student_kr"] = (
        df[FIXED_COST_COLUMN] / student_denominator
    )
    df["Current_Per_Student_kr"] = (
        df[CURRENT_COST_COLUMN] / student_denominator
    )
    df["Fixed_Minus_Current_kr"] = (
        df[FIXED_COST_COLUMN] - df[CURRENT_COST_COLUMN]
    )

    return df.sort_values(
        ["Region_Code", "Year", "School_Type"]
    ).reset_index(drop=True)


def build_yearly_summary(
    df: pd.DataFrame,
    selected_cost_column: str,
) -> pd.DataFrame:
    """Aggregate students and costs by year."""
    summary = (
        df.groupby("Year", as_index=False)
        .agg(
            Forecast_Students=("Forecast_Students", "sum"),
            Fixed_Total_Cost_kr=(FIXED_COST_COLUMN, "sum"),
            Current_Total_Cost_kr=(CURRENT_COST_COLUMN, "sum"),
        )
        .sort_values("Year")
    )

    summary["Selected_Total_Cost_kr"] = summary[selected_cost_column]

    denominator = summary["Forecast_Students"].replace(0, pd.NA)
    summary["Selected_Cost_Per_Student_kr"] = (
        summary["Selected_Total_Cost_kr"] / denominator
    )

    return summary


def create_cost_chart(
    yearly_summary: pd.DataFrame,
    basis_label: str,
) -> go.Figure:
    """Create the main yearly cost forecast chart."""
    chart_data = yearly_summary.copy()
    chart_data["Cost_Billion_SEK"] = (
        chart_data["Selected_Total_Cost_kr"] / 1_000_000_000
    )

    fig = px.area(
        chart_data,
        x="Year",
        y="Cost_Billion_SEK",
        markers=True,
        title=f"Annual education cost forecast — {basis_label}",
        labels={
            "Year": "Year",
            "Cost_Billion_SEK": "Cost, SEK billion",
        },
    )

    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Total cost: SEK %{y:,.2f} bn"
            "<extra></extra>"
        ),
    )

    fig.update_yaxes(
        tickprefix="SEK ",
        ticksuffix=" bn",
        rangemode="tozero",
    )

    return fig


def create_students_chart(yearly_summary: pd.DataFrame) -> go.Figure:
    """Create the annual student forecast chart."""
    fig = px.line(
        yearly_summary,
        x="Year",
        y="Forecast_Students",
        markers=True,
        title="Forecast student population",
        labels={
            "Year": "Year",
            "Forecast_Students": "Students",
        },
    )

    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Students: %{y:,.0f}"
            "<extra></extra>"
        ),
    )

    fig.update_yaxes(
        tickformat=",",
        rangemode="tozero",
    )

    return fig


def create_cost_basis_comparison(
    yearly_summary: pd.DataFrame,
) -> go.Figure:
    """Compare fixed and current cost forecasts."""
    comparison = yearly_summary[
        [
            "Year",
            FIXED_COST_COLUMN,
            CURRENT_COST_COLUMN,
        ]
    ].melt(
        id_vars="Year",
        var_name="Cost_Basis",
        value_name="Total_Cost_kr",
    )

    comparison["Cost_Basis"] = comparison["Cost_Basis"].map(
        {
            FIXED_COST_COLUMN: "Fixed prices",
            CURRENT_COST_COLUMN: "Current prices",
        }
    )

    comparison["Total_Cost_Billion_SEK"] = (
        comparison["Total_Cost_kr"] / 1_000_000_000
    )

    fig = px.line(
        comparison,
        x="Year",
        y="Total_Cost_Billion_SEK",
        color="Cost_Basis",
        markers=True,
        title="Fixed-price and current-price comparison",
        labels={
            "Year": "Year",
            "Total_Cost_Billion_SEK": "Cost, SEK billion",
            "Cost_Basis": "",
        },
    )

    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=7),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "%{fullData.name}: SEK %{y:,.2f} bn"
            "<extra></extra>"
        ),
    )

    fig.update_yaxes(
        tickprefix="SEK ",
        ticksuffix=" bn",
        rangemode="tozero",
    )

    return fig


def create_school_breakdown_chart(
    breakdown: pd.DataFrame,
    year: int,
    basis_label: str,
) -> go.Figure:
    """Create the school-type cost breakdown chart."""
    chart_data = breakdown.copy()
    chart_data["Total_Cost_Million_SEK"] = (
        chart_data["Selected_Total_Cost_kr"] / 1_000_000
    )

    fig = px.bar(
        chart_data,
        x="School_Type_Label",
        y="Total_Cost_Million_SEK",
        text="Total_Cost_Million_SEK",
        title=f"Cost by school type in {year}",
        labels={
            "School_Type_Label": "",
            "Total_Cost_Million_SEK": "Cost, SEK million",
        },
    )

    fig.update_traces(
        texttemplate="SEK %{text:,.0f} m",
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            f"Basis: {basis_label}<br>"
            "Total cost: SEK %{y:,.1f} m"
            "<extra></extra>"
        ),
    )

    fig.update_yaxes(
        tickprefix="SEK ",
        ticksuffix=" m",
        rangemode="tozero",
    )

    return fig


def prepare_display_table(
    df: pd.DataFrame,
    selected_cost_column: str,
) -> pd.DataFrame:
    """Prepare a readable summary table for display."""
    display_df = df.copy()

    display_df["Total cost"] = display_df[selected_cost_column].map(
        lambda value: format_currency(value, compact=False)
    )
    display_df["Students"] = display_df["Forecast_Students"].map(
        lambda value: format_number(value)
    )

    denominator = display_df["Forecast_Students"].replace(0, pd.NA)
    display_df["Cost per student"] = (
        display_df[selected_cost_column] / denominator
    ).map(lambda value: format_currency(value, compact=False))

    return display_df[
        [
            "Region_Code",
            "Region_Name",
            "Year",
            "Students",
            "Total cost",
            "Cost per student",
        ]
    ].rename(
        columns={
            "Region_Code": "Region code",
            "Region_Name": "Region",
        }
    )


def main() -> None:
    st.set_page_config(
        page_title="Forecast & Education Costs",
        page_icon="📊",
        layout="wide",
    )

    st.title("Forecast & Education Costs")
    st.caption(
        "Projected student numbers and education expenditure for Swedish regions, "
        "2024–2030."
    )

    try:
        df = load_costs_csv()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.stop()

    with st.sidebar:
        st.header("View settings")

        basis = st.radio(
            "Price basis",
            options=["Current prices", "Fixed prices"],
            index=0,
            help=(
                "Current prices include projected price changes. "
                "Fixed prices show costs in constant purchasing-power terms."
            ),
        )

        selected_cost_column = (
            CURRENT_COST_COLUMN
            if basis == "Current prices"
            else FIXED_COST_COLUMN
        )

        years = sorted(df["Year"].unique().tolist())
        year_min = min(years)
        year_max = max(years)

        year_range = st.slider(
            "Forecast period",
            min_value=year_min,
            max_value=year_max,
            value=(year_min, year_max),
            step=1,
        )

        regions = (
            df[["Region_Code", "Region_Name"]]
            .drop_duplicates()
            .sort_values("Region_Code")
        )

        region_options = {
            "(All regions)": None,
            **{
                f"{row.Region_Code} - {row.Region_Name}": row.Region_Code
                for row in regions.itertuples(index=False)
            },
        }

        selected_region_label = st.selectbox(
            "Region",
            options=list(region_options.keys()),
            index=0,
        )
        selected_region_code = region_options[selected_region_label]

        school_type_options = {
            "Both school types": None,
            **{
                school_type_label(value): value
                for value in sorted(df["School_Type"].dropna().unique())
            },
        }

        selected_school_label = st.selectbox(
            "School type",
            options=list(school_type_options.keys()),
            index=0,
        )
        selected_school_type = school_type_options[selected_school_label]

        st.divider()

        show_detailed_rows = st.checkbox(
            "Show detailed forecast rows",
            value=False,
        )

    filtered_df = df[
        (df["Year"] >= year_range[0])
        & (df["Year"] <= year_range[1])
    ].copy()

    if selected_region_code is not None:
        filtered_df = filtered_df[
            filtered_df["Region_Code"] == selected_region_code
        ].copy()

    if selected_school_type is not None:
        filtered_df = filtered_df[
            filtered_df["School_Type"] == selected_school_type
        ].copy()

    if filtered_df.empty:
        st.warning(
            "No forecast data matches the selected filters. "
            "Adjust the region, school type, or forecast period."
        )
        st.stop()

    yearly_summary = build_yearly_summary(
        filtered_df,
        selected_cost_column,
    )

    total_students = float(filtered_df["Forecast_Students"].sum())
    selected_total_cost = float(filtered_df[selected_cost_column].sum())
    selected_cost_per_student = (
        selected_total_cost / total_students
        if total_students > 0
        else float("nan")
    )

    first_year_row = yearly_summary.iloc[0]
    last_year_row = yearly_summary.iloc[-1]

    student_change = calculate_change(
        float(first_year_row["Forecast_Students"]),
        float(last_year_row["Forecast_Students"]),
    )
    cost_change = calculate_change(
        float(first_year_row["Selected_Total_Cost_kr"]),
        float(last_year_row["Selected_Total_Cost_kr"]),
    )
    cost_per_student_change = calculate_change(
        float(first_year_row["Selected_Cost_Per_Student_kr"]),
        float(last_year_row["Selected_Cost_Per_Student_kr"]),
    )

    scope_parts = [
        selected_region_label,
        selected_school_label,
        f"{year_range[0]}–{year_range[1]}",
    ]

    st.subheader("Forecast overview")
    st.caption(" · ".join(scope_parts))

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    metric_1.metric(
        "Forecast student-years",
        format_number(total_students),
        help=(
            "The sum of forecast students across all selected years. "
            "This is not a count of unique individuals."
        ),
    )

    metric_2.metric(
        f"Total cost · {basis.lower()}",
        format_currency(selected_total_cost),
        help="The combined education cost across the selected forecast period.",
    )

    metric_3.metric(
        "Average cost per student-year",
        format_currency(selected_cost_per_student),
        help=(
            "Total selected cost divided by total forecast student-years "
            "for the selected period."
        ),
    )

    metric_4.metric(
        f"Cost change · {year_range[0]} to {year_range[1]}",
        format_percent(cost_change),
        help="Change in annual education cost from the first to the final selected year.",
    )

    with st.expander("How to read these figures", expanded=False):
        st.markdown(
            """
            **Current prices** include projected changes in prices over time.

            **Fixed prices** express future costs using a constant price basis, which
            makes it easier to compare underlying changes in student numbers and
            education demand.

            **Forecast student-years** are summed across years. For example, one
            forecast student included in each of seven years contributes seven
            student-years to the period total.
            """
        )

    st.divider()

    st.subheader("Forecast trends")

    left_chart, right_chart = st.columns([1.35, 1])

    with left_chart:
        render_plot(
            create_cost_chart(
                yearly_summary,
                basis,
            )
        )

    with right_chart:
        render_plot(create_students_chart(yearly_summary))

    trend_1, trend_2, trend_3 = st.columns(3)

    trend_1.metric(
        "Student change",
        format_percent(student_change),
        help="Change in annual forecast students from the first to the final selected year.",
    )
    trend_2.metric(
        "Total cost change",
        format_percent(cost_change),
        help="Change in annual total cost from the first to the final selected year.",
    )
    trend_3.metric(
        "Cost per student change",
        format_percent(cost_per_student_change),
        help="Change in annual cost per student from the first to the final selected year.",
    )

    with st.expander("Compare current and fixed prices", expanded=False):
        st.caption(
            "This chart is provided for comparison only. "
            "The dashboard totals above use the selected price basis."
        )
        render_plot(create_cost_basis_comparison(yearly_summary))

    st.divider()

    st.subheader("School-type breakdown")

    available_breakdown_years = sorted(filtered_df["Year"].unique().tolist())

    selected_breakdown_year = st.selectbox(
        "Breakdown year",
        options=available_breakdown_years,
        index=len(available_breakdown_years) - 1,
    )

    school_breakdown = (
        filtered_df[
            filtered_df["Year"] == int(selected_breakdown_year)
        ]
        .groupby(
            ["School_Type", "School_Type_Label"],
            as_index=False,
        )
        .agg(
            Forecast_Students=("Forecast_Students", "sum"),
            Fixed_Total_Cost_kr=(FIXED_COST_COLUMN, "sum"),
            Current_Total_Cost_kr=(CURRENT_COST_COLUMN, "sum"),
        )
        .sort_values("School_Type_Label")
    )

    school_breakdown["Selected_Total_Cost_kr"] = (
        school_breakdown[selected_cost_column]
    )

    render_plot(
        create_school_breakdown_chart(
            school_breakdown,
            int(selected_breakdown_year),
            basis,
        )
    )

    breakdown_display = school_breakdown.copy()
    breakdown_display["Students"] = breakdown_display[
        "Forecast_Students"
    ].map(format_number)
    breakdown_display["Total cost"] = breakdown_display[
        "Selected_Total_Cost_kr"
    ].map(format_currency)

    breakdown_display = breakdown_display[
        [
            "School_Type_Label",
            "Students",
            "Total cost",
        ]
    ].rename(
        columns={
            "School_Type_Label": "School type",
        }
    )

    render_dataframe(breakdown_display)

    st.divider()

    st.subheader("Regional and yearly totals")

    totals_by_region_year = (
        filtered_df.groupby(
            ["Region_Code", "Region_Name", "Year"],
            as_index=False,
        )
        .agg(
            Forecast_Students=("Forecast_Students", "sum"),
            Fixed_Total_Cost_kr=(FIXED_COST_COLUMN, "sum"),
            Current_Total_Cost_kr=(CURRENT_COST_COLUMN, "sum"),
        )
        .sort_values(["Year", "Region_Code"])
    )

    totals_by_region_year["Selected_Total_Cost_kr"] = (
        totals_by_region_year[selected_cost_column]
    )

    display_totals = prepare_display_table(
        totals_by_region_year,
        selected_cost_column,
    )

    render_dataframe(display_totals)

    st.divider()

    st.subheader("Download data")
    st.caption(
        "Download either the filtered source rows or the summarized regional totals."
    )

    download_1, download_2 = st.columns(2)

    with download_1:
        st.download_button(
            label="Download filtered forecast rows",
            data=to_csv_bytes(filtered_df),
            file_name="education_costs_filtered.csv",
            mime="text/csv",
            width="stretch",
        )

    with download_2:
        st.download_button(
            label="Download totals by region and year",
            data=to_csv_bytes(totals_by_region_year),
            file_name="education_costs_totals_by_region_year.csv",
            mime="text/csv",
            width="stretch",
        )

    if show_detailed_rows:
        st.divider()
        st.subheader("Detailed forecast rows")

        detailed_display = filtered_df.copy()
        detailed_display["School_Type"] = detailed_display[
            "School_Type"
        ].map(school_type_label)

        detailed_display = detailed_display.rename(
            columns={
                "Region_Code": "Region code",
                "Region_Name": "Region",
                "School_Type": "School type",
                "Forecast_Students": "Forecast students",
                FIXED_COST_COLUMN: "Fixed total cost (SEK)",
                CURRENT_COST_COLUMN: "Current total cost (SEK)",
                "Fixed_Per_Student_kr": "Fixed cost per student (SEK)",
                "Current_Per_Student_kr": "Current cost per student (SEK)",
            }
        )

        visible_columns = [
            "Region code",
            "Region",
            "Year",
            "School type",
            "Forecast students",
            "Fixed total cost (SEK)",
            "Current total cost (SEK)",
            "Fixed cost per student (SEK)",
            "Current cost per student (SEK)",
        ]

        render_dataframe(
            detailed_display[
                [
                    column
                    for column in visible_columns
                    if column in detailed_display.columns
                ]
            ]
        )


if __name__ == "__main__":
    main()