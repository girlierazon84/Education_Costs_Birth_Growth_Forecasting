"""dashboards/pages/1_EDA.py"""

from __future__ import annotations

from pathlib import Path
import io

import pandas as pd
import streamlit as st
import plotly.express as px

from eduforecast.io.readers import read_births_raw, read_costs_per_child
from eduforecast.preprocessing.clean_births import clean_births
from eduforecast.preprocessing.clean_costs import clean_costs_per_child


def project_root() -> Path:
    # dashboards/pages/1_EDA.py -> dashboards/pages -> dashboards -> project root
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


@st.cache_data(show_spinner=False)
def load_births() -> pd.DataFrame:
    raw = read_births_raw(project_root() / "data" / "raw" / "birth_data_per_region.csv")
    return clean_births(raw)


@st.cache_data(show_spinner=False)
def load_costs_external() -> tuple[pd.DataFrame, pd.DataFrame]:
    grund_raw = read_costs_per_child(project_root() / "data" / "external" / "grundskola_costs_per_child.csv")
    gymn_raw = read_costs_per_child(project_root() / "data" / "external" / "gymnasieskola_costs_per_child.csv")
    return clean_costs_per_child(grund_raw), clean_costs_per_child(gymn_raw)


def main() -> None:
    st.set_page_config(page_title="EDA • EduForecast", layout="wide")
    st.title("EDA — Births & Cost Tables")
    st.caption("Sanity-check births trends + cost table coverage.")

    births = load_births()

    with st.sidebar:
        st.header("EDA Filters")
        years = sorted(births["Year"].unique().tolist())
        yr_min, yr_max = min(years), max(years)
        default_start = 1968 if 1968 in years else yr_min
        year_range = st.slider("Year range", yr_min, yr_max, (default_start, yr_max), step=1)

        regs = births[["Region_Code", "Region_Name"]].drop_duplicates().sort_values("Region_Code")
        reg_opts = ["(National total)"] + [f"{r.Region_Code} — {r.Region_Name}" for r in regs.itertuples(index=False)]
        reg_sel = st.selectbox("Region", reg_opts, index=0)

        show_table = st.checkbox("Show raw births table", value=False)

    bdf = births[(births["Year"] >= year_range[0]) & (births["Year"] <= year_range[1])].copy()

    if reg_sel == "(National total)":
        series = bdf.groupby("Year", as_index=False)["Number"].sum()
        title = "Total births — National"
    else:
        rc = reg_sel.split("—")[0].strip()
        series = bdf[bdf["Region_Code"] == rc].groupby("Year", as_index=False)["Number"].sum()
        title = f"Total births — {reg_sel}"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(bdf):,}")
    c2.metric("Regions", f"{bdf['Region_Code'].nunique():,}")
    c3.metric("Min year", f"{bdf['Year'].min()}")
    c4.metric("Max year", f"{bdf['Year'].max()}")

    fig = px.line(series, x="Year", y="Number", markers=True, title=title)
    _plot(fig)

    yoy = series.sort_values("Year").copy()
    yoy["YoY_change_pct"] = yoy["Number"].pct_change() * 100.0
    fig_yoy = px.histogram(yoy.dropna(), x="YoY_change_pct", nbins=40, title="YoY % change distribution")
    _plot(fig_yoy)

    st.subheader("Missing years report")
    if reg_sel != "(National total)":
        rc = reg_sel.split("—")[0].strip()
        ryears = sorted(births[births["Region_Code"] == rc]["Year"].unique().tolist())
        expected = list(range(min(ryears), max(ryears) + 1))
        missing = sorted(set(expected) - set(ryears))
        st.write(f"Missing years for {rc}: {missing if missing else 'None'}")
    else:
        st.info("Select a region to run missing-years report at region level.")

    if show_table:
        _df(bdf.sort_values(["Region_Code", "Year"]))

    st.download_button(
        "Download filtered births (CSV)",
        data=to_csv_bytes(bdf),
        file_name="eda_births_filtered.csv",
        mime="text/csv",
    )

    st.divider()

    st.subheader("Cost tables (external)")
    grund, gymn = load_costs_external()

    left, right = st.columns(2)
    with left:
        fig_g = px.line(
            grund,
            x="Year",
            y=[c for c in ["Fixed_cost_per_child_kr", "Current_cost_per_child_kr"] if c in grund.columns],
            markers=True,
            title="Grundskola cost per child (fixed vs current)",
        )
        _plot(fig_g)

    with right:
        fig_y = px.line(
            gymn,
            x="Year",
            y=[c for c in ["Fixed_cost_per_child_kr", "Current_cost_per_child_kr"] if c in gymn.columns],
            markers=True,
            title="Gymnasieskola cost per child (fixed vs current)",
        )
        _plot(fig_y)

    st.write(
        f"Grundskola years: {grund['Year'].min()}–{grund['Year'].max()} • "
        f"Gymnasieskola years: {gymn['Year'].min()}–{gymn['Year'].max()}"
    )


if __name__ == "__main__":
    main()
