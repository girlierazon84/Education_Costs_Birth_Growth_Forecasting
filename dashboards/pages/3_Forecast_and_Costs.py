"""dashboards/pages/3_Forecast_and_Costs.py"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from eduforecast.io.readers import read_births_raw
from eduforecast.preprocessing.clean_births import clean_births


def project_root() -> Path:
    # dashboards/pages/3_Forecast_and_Costs.py -> dashboards/pages -> dashboards -> project root
    return Path(__file__).resolve().parents[2]


def _plot(fig) -> None:
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def _df(df: pd.DataFrame) -> None:
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        st.dataframe(df, use_container_width=True)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _looks_like_bad_region_name(name: str, code: str) -> bool:
    """
    Catch common bad values that cause UI like '01 - 01' or '01 - 1'.
    """
    n = str(name).strip()
    c = str(code).strip()
    if n == "" or n.lower() == "nan":
        return True
    if n == c:
        return True
    # numeric-only labels like "1" or "01"
    if n.isdigit():
        return True
    return False


@st.cache_data(show_spinner=False)
def load_region_lookup_from_births() -> dict[str, str]:
    """
    Build Region_Code -> Region_Name mapping using the SAME cleaning pipeline as 1_EDA.py.
    This ensures names like 'Stockholms län' instead of '1'.
    """
    root = project_root()
    f = root / "data" / "raw" / "birth_data_per_region.csv"
    if not f.exists():
        return {}

    raw = read_births_raw(f)
    births = clean_births(raw)  # <- critical: ensures canonical Region_Name

    regs = births[["Region_Code", "Region_Name"]].drop_duplicates().sort_values("Region_Code")
    # Prefer longest / most descriptive name if duplicates exist
    regs["Region_Name"] = regs["Region_Name"].astype(str).str.strip()
    regs = regs.sort_values(["Region_Code", "Region_Name"], ascending=[True, False]).drop_duplicates("Region_Code")

    return dict(zip(regs["Region_Code"].tolist(), regs["Region_Name"].tolist()))


@st.cache_data(show_spinner=False)
def load_costs_csv() -> pd.DataFrame:
    root = project_root()
    f = root / "artifacts" / "forecasts" / "education_costs_forecast_2024_2030.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing forecast file:\n{f}\n\nRun: eduforecast forecast")

    df = pd.read_csv(f, dtype={"Region_Code": "string"})
    df["Region_Code"] = df["Region_Code"].astype("string").str.strip().str.zfill(2)

    # Normalize Region_Name (may be missing or wrong in artifacts)
    df["Region_Name"] = df.get("Region_Name", df["Region_Code"]).astype(str).str.strip()

    lookup = load_region_lookup_from_births()
    if lookup:
        bad = df.apply(lambda r: _looks_like_bad_region_name(r["Region_Name"], r["Region_Code"]), axis=1)
        if bad.any():
            df.loc[bad, "Region_Name"] = df.loc[bad, "Region_Code"].map(lookup).fillna(df.loc[bad, "Region_Name"])

    df["School_Type"] = df["School_Type"].astype(str).str.strip().str.lower()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    for c in ["Forecast_Students", "Fixed_Total_Cost_kr", "Current_Total_Cost_kr"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Year", "Forecast_Students", "Fixed_Total_Cost_kr", "Current_Total_Cost_kr"]).copy()
    df["Year"] = df["Year"].astype(int)

    denom = df["Forecast_Students"].replace(0, pd.NA)
    df["Fixed_Per_Student_kr"] = df["Fixed_Total_Cost_kr"] / denom
    df["Current_Per_Student_kr"] = df["Current_Total_Cost_kr"] / denom
    df["Fixed_Minus_Current_kr"] = df["Fixed_Total_Cost_kr"] - df["Current_Total_Cost_kr"]

    return df


def main() -> None:
    st.set_page_config(page_title="Education Costs", layout="wide")

    st.title("Education Costs Forecast (2024–2030)")
    st.caption("Charts and KPIs are based on ONE chosen basis (Current OR Fixed). No double counting.")

    df = load_costs_csv()

    with st.sidebar:
        st.header("Filters")

        basis = st.radio(
            "Cost basis (use ONE basis for totals)",
            ["Current (nominal)", "Fixed (real)"],
            index=0,
        )
        cost_col = "Current_Total_Cost_kr" if basis.startswith("Current") else "Fixed_Total_Cost_kr"

        years = sorted(df["Year"].unique().tolist())
        year_min, year_max = min(years), max(years)
        year_range = st.slider("Year range", year_min, year_max, (year_min, year_max), step=1)

        regions = df[["Region_Code", "Region_Name"]].drop_duplicates().sort_values("Region_Code")

        # ✅ Display: "01 - Stockholm" (or "01 - Stockholms län" depending on your cleaned names)
        region_options = ["(All regions)"] + [f"{r.Region_Code} - {r.Region_Name}" for r in regions.itertuples(index=False)]
        region_sel = st.selectbox("Region", region_options, index=0)

        school_options = ["(Both)"] + sorted(df["School_Type"].unique().tolist())
        school_sel = st.selectbox("School type", school_options, index=0)

        st.divider()
        show_raw = st.checkbox("Show raw rows table (large)", value=False)

    fdf = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])].copy()

    if region_sel != "(All regions)":
        rc = region_sel.split("-")[0].strip()
        fdf = fdf[fdf["Region_Code"] == rc].copy()

    if school_sel != "(Both)":
        fdf = fdf[fdf["School_Type"] == school_sel].copy()

    kpi_students = float(fdf["Forecast_Students"].sum())
    denom = max(kpi_students, 1e-9)

    kpi_fixed = float(fdf["Fixed_Total_Cost_kr"].sum())
    kpi_current = float(fdf["Current_Total_Cost_kr"].sum())
    kpi_selected = float(fdf[cost_col].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast students", f"{kpi_students:,.0f}")
    c2.metric("Fixed total (SEK)", f"{kpi_fixed:,.0f}")
    c3.metric("Current total (SEK)", f"{kpi_current:,.0f}")
    c4.metric(f"Total ({basis})", f"{kpi_selected:,.0f}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Fixed / student (SEK)", f"{(kpi_fixed / denom):,.0f}")
    c6.metric("Current / student (SEK)", f"{(kpi_current / denom):,.0f}")
    c7.metric(f"/ student ({basis})", f"{(kpi_selected / denom):,.0f}")

    st.divider()

    by_year = (
        fdf.groupby("Year", as_index=False)
        .agg(
            Forecast_Students=("Forecast_Students", "sum"),
            Fixed_Total_Cost_kr=("Fixed_Total_Cost_kr", "sum"),
            Current_Total_Cost_kr=("Current_Total_Cost_kr", "sum"),
        )
        .sort_values("Year")
    )
    by_year["Total_SelectedBasis_kr"] = by_year[cost_col]

    left, right = st.columns(2)
    with left:
        _plot(px.line(by_year, x="Year", y="Total_SelectedBasis_kr", markers=True, title=f"Total cost over time ({basis})"))
    with right:
        _plot(px.line(by_year, x="Year", y="Forecast_Students", markers=True, title="Forecast students over time"))

    _plot(
        px.line(
            by_year,
            x="Year",
            y=["Fixed_Total_Cost_kr", "Current_Total_Cost_kr"],
            markers=True,
            title="Fixed vs Current totals over time (comparison only)",
        )
    )

    year_pick = st.selectbox("Breakdown year", sorted(fdf["Year"].unique().tolist()), index=0)

    by_year_school = (
        fdf[fdf["Year"] == int(year_pick)]
        .groupby("School_Type", as_index=False)
        .agg(
            Forecast_Students=("Forecast_Students", "sum"),
            Fixed_Total_Cost_kr=("Fixed_Total_Cost_kr", "sum"),
            Current_Total_Cost_kr=("Current_Total_Cost_kr", "sum"),
        )
        .sort_values("School_Type")
    )
    by_year_school["Total_SelectedBasis_kr"] = by_year_school[cost_col]

    _plot(
        px.bar(
            by_year_school,
            x="School_Type",
            y="Total_SelectedBasis_kr",
            title=f"Total cost by school type ({year_pick}) — {basis}",
            text_auto=".2s",
        )
    )

    st.subheader("Download")
    st.download_button(
        "Download filtered rows (CSV)",
        data=to_csv_bytes(fdf),
        file_name="education_costs_filtered.csv",
        mime="text/csv",
    )

    totals_region_year = (
        fdf.groupby(["Region_Code", "Region_Name", "Year"], as_index=False)[
            ["Forecast_Students", "Fixed_Total_Cost_kr", "Current_Total_Cost_kr"]
        ]
        .sum()
        .sort_values(["Year", "Region_Code"])
    )
    totals_region_year["Total_SelectedBasis_kr"] = totals_region_year[cost_col]

    st.download_button(
        "Download totals by region & year (CSV)",
        data=to_csv_bytes(totals_region_year),
        file_name="education_costs_totals_by_region_year.csv",
        mime="text/csv",
    )

    st.subheader("Totals by Region & Year")
    _df(totals_region_year)

    if show_raw:
        st.subheader("Raw rows")
        _df(fdf)


if __name__ == "__main__":
    main()
