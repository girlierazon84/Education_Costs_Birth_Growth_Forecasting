"""dashboards/pages/2_Model_Comparison.py"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


def project_root() -> Path:
    # dashboards/pages/2_Model_Comparison.py -> dashboards/pages -> dashboards -> project root
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


def _read_csv(path: Path, *, dtype: dict | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file:\n{path}")
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
    st.set_page_config(page_title="Model Comparison • EduForecast", layout="wide")
    st.title("Model Comparison — Birth Forecast Models")
    st.caption("Best model per region + optional score table if you export one.")

    best = load_best_models()
    scores = try_load_model_scores()

    st.subheader("Best model per region")
    _df(best.sort_values("Region_Code"))

    counts = best.groupby("Best_Model").size().reset_index(name="Regions").sort_values("Regions", ascending=False)
    _plot(px.bar(counts, x="Best_Model", y="Regions", text_auto=True, title="How many regions chose each model"))

    st.download_button(
        "Download best_models_births.csv",
        data=to_csv_bytes(best),
        file_name="best_models_births.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Model score table (optional)")

    if scores is None:
        st.info(
            "No artifacts/metrics/model_scores_births.csv found.\n"
            "If you export your training evaluation table, it will appear here."
        )
        return

    _df(scores)

    possible_metric_cols = [c for c in scores.columns if c.lower() in {"rmse", "mae", "smape", "mape"}]
    if possible_metric_cols and {"Region_Code", "Model"}.issubset(scores.columns):
        metric = st.selectbox("Primary metric column", possible_metric_cols, index=0)

        tmp = scores.copy()
        tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
        tmp = tmp.dropna(subset=[metric])

        st.subheader("Top model per region (by selected metric)")
        top = tmp.sort_values(metric).groupby("Region_Code", as_index=False).head(1)
        _df(top.sort_values("Region_Code"))

        _plot(px.histogram(tmp, x=metric, color="Model", nbins=40, title=f"Metric distribution: {metric}"))

    st.download_button(
        "Download model scores (CSV)",
        data=to_csv_bytes(scores),
        file_name="model_scores_births.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
