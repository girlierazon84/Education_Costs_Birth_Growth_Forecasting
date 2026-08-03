"""dashboards/pages/2_Model_Comparison.py"""

from __future__ import annotations

import io
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _apply_chart_theme(fig, y_title: str = "") -> None:
    fig.update_layout(
        margin=dict(l=55, r=30, t=35, b=40),
        hovermode="x unified",
        title_font=dict(size=13, color="#0f172a", family="Arial, sans-serif"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1,
            title_text="",
            font=dict(size=11, color="#334155", family="Arial, sans-serif")
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
    if hasattr(fig, "update_traces"):
        fig.update_traces(
            line=dict(width=2),
            selector=dict(type="scatter")
        )


def _plot(fig, y_title: str = "") -> None:
    _apply_chart_theme(fig, y_title=y_title)
    st.plotly_chart(fig, use_container_width=True)


def _df(df: pd.DataFrame) -> None:
    st.dataframe(df, use_container_width=True, hide_index=True)


def _read_csv(path: Path, *, dtype: dict | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file path target: {path}")
    return pd.read_csv(path, dtype=dtype)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


@st.cache_data(show_spinner=False)
def load_best_models() -> pd.DataFrame:
    root = project_root()
    f = root / "artifacts" / "metrics" / "best_models_births.csv"
    df = _read_csv(f, dtype={"Region_Code": "string"})
    df["Region_Code"] = df["Region_Code"].astype("string").str.strip().str.zfill(2)
    df["Region_Name"] = df.get("Region_Name", df["Region_Code"]).astype(str).str.strip()
    df["Best_Model"] = df.get("Best_Model", "unknown").astype(str).str.strip()
    return df


@st.cache_data(show_spinner=False)
def try_load_model_scores() -> pd.DataFrame | None:
    root = project_root()
    f = root / "artifacts" / "metrics" / "model_scores_births.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, dtype={"Region_Code": "string"})
    if "Region_Code" in df.columns:
        df["Region_Code"] = df["Region_Code"].astype("string").str.strip().str.zfill(2)
    if "Model" in df.columns:
        df["Model"] = df["Model"].astype(str).str.strip()
    return df


def main() -> None:
    st.set_page_config(page_title="Model Evaluation • EduForecast", layout="wide")
    st.title("Model Optimization & Selection Matrix")
    st.markdown(
        "Analysis of cross-validation error profiles and production time-series model assignments across Swedish counties."
    )
    st.write("")

    best = load_best_models()
    scores = try_load_model_scores()

    if "is_demographic_override" in best.columns:
        overridden_count = int(best["is_demographic_override"].sum())
        if overridden_count > 0:
            st.info(
                f"Demographic Rule Layer: Automated framework constraints are active. Selection parameters for {overridden_count} "
                f"metropolitan/commuter counties (Stockholm, Uppsala, Halland) have been assigned to exp_smoothing "
                f"to stabilize long-term educational infrastructure and cost trends."
            )

    tab_assignments, tab_metrics = st.tabs(["Regional Assignments", "Cross-Validation Error Analysis"])

    # --- TAB 1: REGIONAL ASSIGNMENTS ---
    with tab_assignments:
        st.write("")
        col_table, col_summary = st.columns([1.3, 1])

        with col_table:
            st.markdown("**Production Model Assignments**")
            # Filter layout columns cleanly for human readability
            display_cols = [c for c in ["Region_Code", "Region_Name", "Best_Model", "Statistical_CV_Winner"] if c in best.columns]
            if not display_cols:
                display_cols = list(best.columns)
            _df(best[display_cols].sort_values("Region_Code"))

        with col_summary:
            st.markdown("**Selection Frequency Analysis**")
            with st.container(border=True):
                counts = best.groupby("Best_Model").size().reset_index(name="Count").sort_values("Regions" if "Regions" in best.columns else "Count", ascending=False)
                fig_bar = px.bar(counts, x="Best_Model", y="Count", title="Model Selection Frequency Analysis", text_auto=True)
                fig_bar.update_traces(marker_color="#1e3a8a")
                _plot(fig_bar, y_title="Number of Regions")

            st.write("")
            st.download_button(
                "Export Assignment Registry (CSV)",
                data=to_csv_bytes(best),
                file_name="best_models_births.csv",
                mime="text/csv",
                use_container_width=True
            )

    # --- TAB 2: DETAILED CV PERFORMANCE ---
    with tab_metrics:
        st.write("")
        if scores is None:
            st.info(
                "Granular performance evaluation logs not found. "
                "Run the model training pipeline to generate backtest tables."
            )
            return

        control_col, download_col = st.columns([1.5, 2])
        with control_col:
            metric_options = [c for c in scores.columns if c.lower() in {"rmse", "mae", "smape", "mape"}]
            if metric_options and {"Region_Code", "Model"}.issubset(scores.columns):
                metric = st.selectbox("Select target validation metric", metric_options, index=0)
            else:
                metric = None

        if metric:
            tmp = scores.copy()
            tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
            tmp = tmp.dropna(subset=[metric])

            st.write("")
            split_left, split_right = st.columns([1.2, 1])

            with split_left:
                st.markdown(f"**Optimal Historical Performers (Ranked by Minimum {metric})**")
                top_performers = tmp.sort_values(metric).groupby("Region_Code", as_index=False).head(1)
                _df(top_performers.sort_values("Region_Code"))

            with split_right:
                st.markdown(f"**Error Density Profiles ({metric})**")
                with st.container(border=True):
                    fig_hist = px.histogram(
                        tmp,
                        x=metric,
                        color="Model",
                        nbins=30,
                        title=f"Error Density Profiles ({metric})",
                        color_discrete_sequence=["#1e3a8a", "#475569", "#0284c7"]
                    )
                    fig_hist.update_traces(marker_line=dict(width=0.5, color="#ffffff"))
                    _plot(fig_hist, y_title="Frequency Count")

            with download_col:
                st.write("")
                st.download_button(
                    "Export Full Error Matrix (CSV)",
                    data=to_csv_bytes(scores),
                    file_name="model_scores_births.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            _df(scores)


if __name__ == "__main__":
    main()
