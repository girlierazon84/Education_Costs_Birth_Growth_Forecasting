"""dashboards/pages/1_EDA.py"""

from __future__ import annotations

import io
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

from eduforecast.io.readers import read_births_raw, read_costs_per_child_raw
from eduforecast.preprocessing.clean_births import clean_births
from eduforecast.preprocessing.clean_costs import clean_costs_per_child
from eduforecast.validation.checks import validate_births_canonical, validate_df
from eduforecast.validation.schemas import COSTS_PER_CHILD_CANONICAL


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _apply_professional_theme(fig, y_title: str = "") -> None:
    """Applies a clean, non-AI corporate styling layer with standardized typography."""
    fig.update_layout(
        margin=dict(l=50, r=30, t=40, b=40),
        hovermode="x unified",
        title_font=dict(size=14, color="#0f172a", family="Arial, sans-serif"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text=""
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#f1f5f9",
            title_text="",
            tickfont=dict(color="#475569")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#f1f5f9",
            title_text=y_title,
            tickfont=dict(color="#475569")
        ),
    )

    # ✅ FIX: Kontrollera graftyp (type) innan linjebredd appliceras för att förhindra krasch på Histogram
    if hasattr(fig, "update_traces"):
        fig.update_traces(
            line=dict(width=2),
            selector=dict(type="scatter")  # Gäller endast för px.line (vilket Plotly internt mappar som scatter)
        )


def _plot(fig, y_title: str = "") -> None:
    _apply_professional_theme(fig, y_title=y_title)
    st.plotly_chart(fig, use_container_width=True)


def _df(df: pd.DataFrame) -> None:
    st.dataframe(df, use_container_width=True)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


@st.cache_data(show_spinner=False)
def load_births() -> pd.DataFrame:
    raw = read_births_raw(project_root() / "data" / "raw" / "birth_data_per_region.csv")
    df = clean_births(raw)

    # ✅ FIX FÖR MOLNET: Tvinga textkolumnerna till explicit 'string'-typ 
    # Detta kringgår hur Linux/Windows tolkar råa str-objekt och rensar cachen permanent
    df["Region_Code"] = df["Region_Code"].astype("string")
    df["Region_Name"] = df["Region_Name"].astype("string")
    return df


@st.cache_data(show_spinner=False)
def load_costs_external() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = project_root()
    grund_raw = read_costs_per_child_raw(root / "data" / "external" / "grundskola_costs_per_child.csv")
    gymn_raw = read_costs_per_child_raw(root / "data" / "external" / "gymnasieskola_costs_per_child.csv")

    grund = clean_costs_per_child(grund_raw)
    gymn = clean_costs_per_child(gymn_raw)

    # ✅ FIX: Tvinga kostnadskolumnerna till flyttal (float) för att matcha schemavalideringen perfekt
    target_cols = ["Fixed_cost_per_child_kr", "Current_cost_per_child_kr"]
    for col in target_cols:
        if col in grund.columns:
            grund[col] = grund[col].astype(float)
        if col in gymn.columns:
            gymn[col] = gymn[col].astype(float)

    return grund, gymn


def main() -> None:
    st.set_page_config(page_title="EDA • EduForecast", layout="wide")

    st.title("Exploratory Data Analysis (EDA)")
    st.markdown(
        "Historical baseline analysis of regional birth indicators and verification of external cost registers."
    )
    st.write("")

    births = load_births()
    grund, gymn = load_costs_external()

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("Filters")

        years = sorted(births["Year"].unique().tolist())
        yr_min, yr_max = min(years), max(years)
        default_start = 1968 if 1968 in years else yr_min
        year_range = st.slider("Historical window", yr_min, yr_max, (default_start, yr_max), step=1)

        regs = births[["Region_Code", "Region_Name"]].drop_duplicates().sort_values("Region_Code")
        reg_opts = ["(National total)"] + [f"{r.Region_Code} - {r.Region_Name}" for r in regs.itertuples(index=False)]
        reg_sel = st.selectbox("Geographic region", reg_opts, index=0)

        st.divider()
        show_table = st.checkbox("Display raw data rows", value=False)

    # Apply data transformations
    bdf = births[(births["Year"] >= year_range[0]) & (births["Year"] <= year_range[1])].copy()

    if reg_sel == "(National total)":
        series = bdf.groupby("Year", as_index=False)["Number"].sum()
        chart_title = "Historical Birth Trends - Sweden Nationwide"
    else:
        rc = reg_sel.split("-")[0].strip()
        series = bdf[bdf["Region_Code"] == rc].groupby("Year", as_index=False)["Number"].sum()
        chart_title = f"Historical Birth Trends - {reg_sel}"

    # --- Segment 1: Birth Demographics ---
    st.subheader("Demographic Baselines")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total observed records", f"{len(bdf):,}")
    with c2:
        st.metric("Monitored regions", f"{bdf['Region_Code'].nunique():,}")
    with c3:
        st.metric("Data baseline year", f"{bdf['Year'].min()}")
    with c4:
        latest_val = int(series.iloc[-1]["Number"]) if not series.empty else 0
        prev_val = int(series.iloc[-2]["Number"]) if len(series) > 1 else latest_val
        delta = latest_val - prev_val
        st.metric("Latest annual births", f"{latest_val:,}", delta=f"{delta:+,}" if delta != 0 else None, delta_color="inverse" if delta < 0 else "normal")

    # Main trends grid
    left, right = st.columns([2.5, 1])
    with left:
        with st.container(border=True):
            fig_line = px.line(series, x="Year", y="Number", markers=True, title=chart_title)
            fig_line.update_traces(line_color="#1e3a8a")
            _plot(fig_line, y_title="Number of Births")

    with right:
        with st.container(border=True):
            st.markdown("**Validation Registry Check**")
            try:
                validate_births_canonical(births, start_year=1968).raise_if_failed()
                st.caption("✅ Passed canonical schema verification.")
            except Exception as e:
                st.error(f"❌ Schema Mismatch:\n\n{e}")

            st.divider()
            st.markdown("**Sequence Integrity Audit**")
            if reg_sel != "(National total)":
                rc = reg_sel.split("-")[0].strip()
                ryears = sorted(births[births["Region_Code"] == rc]["Year"].unique().tolist())
                expected = list(range(min(ryears), max(ryears) + 1))
                missing = sorted(set(expected) - set(ryears))
                if missing:
                    st.warning(f"Missing years for code {rc}: {missing}")
                else:
                    st.caption("✅ No temporal holes found in this region.")
            else:
                st.caption("Filter by an explicit region to check local temporal integrity holes.")

    # Volatility histogram wrapper
    st.write("")
    with st.container(border=True):
        yoy = series.sort_values("Year").copy()
        yoy["YoY_change_pct"] = yoy["Number"].pct_change() * 100.0
        fig_hist = px.histogram(yoy.dropna(), x="YoY_change_pct", nbins=30, title="Year-over-Year Percentage Change Distribution")
        fig_hist.update_traces(marker_color="#475569", marker_line=dict(width=0.5, color="#ffffff"))
        _plot(fig_hist, y_title="Frequency Count")

    if show_table:
        st.write("")
        st.subheader("Filtered Ingestion Rows")
        _df(bdf.sort_values(["Region_Code", "Year"]))
        st.download_button(
            "Download filtered births extract (CSV)",
            data=to_csv_bytes(bdf),
            file_name="eda_births_filtered.csv",
            mime="text/csv",
        )

    # --- Segment 2: Cost Matrix Verification ---
    st.write("")
    st.divider()
    st.subheader("External Unit Cost Tables")

    cost_left, cost_right = st.columns([2.5, 1])

    with cost_right:
        with st.container(border=True):
            st.markdown("**Cost Schema Validation**")
            try:
                validate_df(grund, schema=COSTS_PER_CHILD_CANONICAL, year_col="Year").raise_if_failed()
                validate_df(gymn, schema=COSTS_PER_CHILD_CANONICAL, year_col="Year").raise_if_failed()
                st.caption("✅ Passed canonical cost specifications.")
            except Exception as e:
                st.error(f"❌ Cost Schema Mismatch:\n\n{e}")

            st.divider()
            st.markdown("**Database History Range**")
            st.caption(f"**Grundskola table bounds**: {grund['Year'].min()} - {grund['Year'].max()}")
            st.caption(f"**Gymnasieskola table bounds**: {gymn['Year'].min()} - {gymn['Year'].max()}")

    with cost_left:
        c_left, c_right = st.columns(2)
        cols_target = [c for c in ["Fixed_cost_per_child_kr", "Current_cost_per_child_kr"] if c in grund.columns]
        labels_map = {"Fixed_cost_per_child_kr": "Fixed (Real)", "Current_cost_per_child_kr": "Current (Nominal)"}

        with c_left:
            with st.container(border=True):
                fig_grund = px.line(grund, x="Year", y=cols_target, markers=True, title="Grundskola - Unit Cost Over Time", labels=labels_map)
                _plot(fig_grund, y_title="Cost per Child (SEK)")

        with c_right:
            with st.container(border=True):
                fig_gymn = px.line(gymn, x="Year", y=cols_target, markers=True, title="Gymnasieskola - Unit Cost Over Time", labels=labels_map)
                _plot(fig_gymn, y_title="Cost per Child (SEK)")


if __name__ == "__main__":
    main()

