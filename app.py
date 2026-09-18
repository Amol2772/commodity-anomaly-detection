"""
Commodity Anomaly Detection Dashboard
Interactive dashboard for the commodity anomaly-detection portfolio project
(Commerzbank Commodity Trading Desk prep).

Run locally with:  streamlit run app.py
Deploy the same way as EDA Copilot (push to GitHub, connect on Streamlit Cloud).

Expects, in the same folder:
- project_config.py
- data/gold.csv, data/silver.csv, data/wti_crude.csv  (from download_commodity_data.py)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ruptures as rpt
import streamlit as st
from sklearn.ensemble import IsolationForest

from project_config import (
    ANNUALISATION,
    IF_CONTAMINATION,
    MAD_SCALE,
    ROLL_WINDOW,
    TICKERS,
    Z_THRESHOLD,
    rolling_mad,
)

st.set_page_config(page_title="Commodity Anomaly Dashboard", page_icon="🕯️", layout="wide")

DATA_DIR = Path("data")

# --- Visual identity: each commodity's accent reflects the real material ----
# (gold -> warm gold, silver -> cool platinum, WTI -> the dark petrol-teal sheen
# of crude oil), rather than one arbitrary brand color for everything.
ACCENT = {
    "gold": "#D4AF37",
    "silver": "#C9CDD1",
    "wti_crude": "#2F6F6A",
}
BG = "#0E1117"
PANEL = "#161B22"
BORDER = "#242B36"
TEXT = "#E8E8E8"
MUTED = "#9CA3AF"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"], .stMarkdown, p, div {{
    font-family: 'Inter', -apple-system, sans-serif;
}}
h1, h2, h3 {{
    font-family: 'Inter', sans-serif;
    letter-spacing: -0.3px;
}}
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: {PANEL};
    border-radius: 6px 6px 0 0;
    padding: 8px 18px;
    color: {MUTED};
}}
.stTabs [aria-selected="true"] {{
    color: {TEXT} !important;
}}
.metric-card {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 16px 20px;
    height: 100%;
}}
.metric-label {{
    color: {MUTED};
    font-size: 13px;
    margin-bottom: 6px;
}}
.metric-value {{
    color: {TEXT};
    font-family: 'JetBrains Mono', monospace;
    font-size: 26px;
    font-weight: 600;
}}
</style>
""", unsafe_allow_html=True)


def metric_card(label, value, accent):
    st.markdown(
        f"""<div class="metric-card" style="border-left: 3px solid {accent};">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def plotly_base_layout(title):
    return dict(
        title=title,
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(family="Inter, sans-serif", color=TEXT, size=12),
        margin=dict(t=50, l=10, r=10, b=10),
    )


# --- Cached data / feature pipeline -----------------------------------------
# Underscore-prefixed params tell Streamlit not to hash them (they're large
# DataFrames/dicts derived from already-cached upstream calls, not user input).

@st.cache_data
def load_data():
    dfs = {}
    for key in TICKERS:
        path = DATA_DIR / f"{key}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        dfs[key] = pd.read_csv(path, index_col="date", parse_dates=True)
    return dfs


@st.cache_data
def compute_features(_dfs):
    returns, vol = {}, {}
    for key in TICKERS:
        r = _dfs[key]["Close"].pct_change().dropna()
        returns[key] = r
        vol[key] = pd.DataFrame({
            "vol_10d": r.rolling(10).std() * ANNUALISATION,
            "vol_30d": r.rolling(30).std() * ANNUALISATION,
        })
    return returns, vol


@st.cache_data
def compute_z_flags(_returns):
    z_flags = {}
    for key in TICKERS:
        r = _returns[key]
        roll_median = r.rolling(ROLL_WINDOW).median()
        roll_mad = rolling_mad(r, ROLL_WINDOW) * MAD_SCALE
        z_robust = (r - roll_median) / roll_mad.replace(0, np.nan)
        z_flags[key] = r[z_robust.abs() > Z_THRESHOLD]
    return z_flags


@st.cache_data
def compute_isolation_forest(_returns, _vol):
    if_flags = {}
    for key in TICKERS:
        feat_df = pd.DataFrame({
            "return": _returns[key],
            "vol_10d": _vol[key]["vol_10d"],
            "vol_30d": _vol[key]["vol_30d"],
        }).dropna()
        iso = IsolationForest(n_estimators=200, contamination=IF_CONTAMINATION, random_state=42)
        preds = iso.fit_predict(feat_df)
        if_flags[key] = feat_df.index[preds == -1]
    return if_flags


@st.cache_data
def compute_changepoints(_returns, pen=15):
    changepoints = {}
    for key in TICKERS:
        robust_vol = (rolling_mad(_returns[key], 30) * MAD_SCALE * ANNUALISATION).dropna()
        signal = robust_vol.values.reshape(-1, 1)
        algo = rpt.Pelt(model="rbf", min_size=30, jump=5).fit(signal)
        bkps = algo.predict(pen=pen)
        changepoints[key] = [robust_vol.index[i - 1] for i in bkps[:-1]]
    return changepoints


@st.cache_data
def compute_correlations(_returns):
    returns_df = pd.DataFrame({key: _returns[key] for key in TICKERS}).dropna()
    pearson = returns_df.corr(method="pearson")
    spearman = returns_df.corr(method="spearman")
    pearson.columns = pearson.index = list(TICKERS.values())
    spearman.columns = spearman.index = list(TICKERS.values())
    return pearson, spearman


def filter_by_date(series_or_df, start, end):
    """Filter against the object's OWN index — vol/returns have one fewer row
    than price (pct_change drops the first day), so a mask from one can't be
    reused on the other."""
    idx_dates = series_or_df.index.date
    return series_or_df[(idx_dates >= start) & (idx_dates <= end)]


# --- Load everything ---------------------------------------------------------

try:
    dfs = load_data()
except FileNotFoundError as e:
    st.error(
        f"Couldn't find {e}. Run `download_commodity_data.py` first, and make sure "
        "this app sits in the same folder as the resulting `data/` directory."
    )
    st.stop()

returns, vol = compute_features(dfs)
z_flags = compute_z_flags(returns)

# --- Sidebar controls -----------------------------------------------------

st.sidebar.header("Controls")
commodity_key = st.sidebar.selectbox(
    "Commodity", options=list(TICKERS.keys()), format_func=lambda k: TICKERS[k]
)
accent = ACCENT[commodity_key]

show_if = st.sidebar.checkbox("Show Isolation Forest flags", value=False)
show_regimes = st.sidebar.checkbox("Show volatility regime shading", value=True)

date_min = dfs[commodity_key].index.min().date()
date_max = dfs[commodity_key].index.max().date()
date_range = st.sidebar.slider(
    "Date range", min_value=date_min, max_value=date_max, value=(date_min, date_max)
)

with st.sidebar.expander("ℹ️ About the anomaly baseline"):
    st.write(
        f"Anomalies are flagged using a rolling z-score that's resistant to being thrown off "
        f"by a single extreme day — it looks back {ROLL_WINDOW} days and flags anything "
        f"unusually far from the recent typical range. The sensitivity can be tuned in the "
        f"project's settings file if you want more or fewer flags."
    )

# --- Header -------------------------------------------------------------------

st.title("Commodity Anomaly Detection Dashboard")
st.markdown(f'<div style="height:4px; background:{accent}; border-radius:2px; margin: 4px 0 20px 0; width:120px;"></div>', unsafe_allow_html=True)
st.caption(
    "Portfolio project — commodity price anomaly & volatility detection, "
    "built for Commerzbank Commodity Trading Desk prep."
)

# --- Filter by date range -----------------------------------------------------

price = filter_by_date(dfs[commodity_key]["Close"], date_range[0], date_range[1])
flagged = z_flags[commodity_key]
flagged_in_range = flagged[(flagged.index.date >= date_range[0]) & (flagged.index.date <= date_range[1])]

# --- Metrics row ---------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Latest close", f"${dfs[commodity_key]['Close'].iloc[-1]:,.2f}", accent)
with c2:
    metric_card("30-day volatility (ann.)", f"{vol[commodity_key]['vol_30d'].iloc[-1]:.1%}", accent)
with c3:
    last_flag = flagged.index.max() if len(flagged) else None
    metric_card("Last flagged anomaly", str(last_flag.date()) if last_flag is not None else "None", accent)
with c4:
    metric_card("Total flagged (full history)", f"{len(flagged)}", accent)

st.markdown("<br>", unsafe_allow_html=True)

# --- Tabs ---------------------------------------------------------------------

tab_overview, tab_vol, tab_cross, tab_quality = st.tabs(
    ["📈 Overview", "📊 Volatility & Regimes", "🔗 Cross-Commodity", "🧪 Data Quality"]
)

# ===== Tab 1: Overview =====
with tab_overview:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price.index, y=price, mode="lines", name="Close price",
        line=dict(width=1.3, color=accent),
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=flagged_in_range.index, y=price.loc[flagged_in_range.index],
        mode="markers", name="Z-score flagged",
        marker=dict(color="#E85D5D", size=8, line=dict(width=1, color=BG)),
        hovertemplate="%{x|%Y-%m-%d}<br>Flagged, return %{customdata:.1%}<extra></extra>",
        customdata=flagged_in_range.values,
    ))

    if show_if:
        if_flags = compute_isolation_forest(returns, vol)
        if_dates = if_flags[commodity_key]
        if_dates_in_range = if_dates[(if_dates.date >= date_range[0]) & (if_dates.date <= date_range[1])]
        fig.add_trace(go.Scatter(
            x=if_dates_in_range, y=price.reindex(if_dates_in_range),
            mode="markers", name="Isolation Forest flagged",
            marker=dict(color="#F5A623", size=7, symbol="x"),
        ))

    fig.update_layout(**plotly_base_layout(f"{TICKERS[commodity_key]} — price with flagged anomalies"))
    fig.update_layout(height=480, xaxis_title="Date", yaxis_title="Price (USD)", legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Flagged anomaly days")
    if len(flagged_in_range):
        table = pd.DataFrame({
            "Date": flagged_in_range.index.date,
            "Return": [f"{val:.1%}" for val in flagged_in_range.values],
        }).sort_values("Date", ascending=False)
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.write("No flagged anomalies in the selected date range.")

# ===== Tab 2: Volatility & Regimes =====
with tab_vol:
    v = filter_by_date(vol[commodity_key], date_range[0], date_range[1])

    vol_fig = go.Figure()
    vol_fig.add_trace(go.Scatter(x=v.index, y=v["vol_10d"], name="10-day vol", line=dict(width=1, color=MUTED)))
    vol_fig.add_trace(go.Scatter(x=v.index, y=v["vol_30d"], name="30-day vol", line=dict(width=1.8, color=accent)))

    if show_regimes:
        changepoints = compute_changepoints(returns)
        cps_in_range = sorted([cp for cp in changepoints[commodity_key] if date_range[0] <= cp.date() <= date_range[1]])
        boundaries = [pd.Timestamp(date_range[0])] + cps_in_range + [pd.Timestamp(date_range[1])]
        for i in range(len(boundaries) - 1):
            if i % 2 == 0:
                vol_fig.add_vrect(x0=boundaries[i], x1=boundaries[i + 1], fillcolor="white", opacity=0.04, line_width=0)

    vol_fig.update_yaxes(type="log")
    vol_fig.update_layout(**plotly_base_layout(f"{TICKERS[commodity_key]} — rolling annualised volatility (log scale)"))
    vol_fig.update_layout(height=420, legend=dict(orientation="h", y=1.08))
    st.plotly_chart(vol_fig, use_container_width=True)

    if show_regimes:
        st.caption(
            "Shaded bands mark distinct volatility regimes (detected via changepoint analysis on a "
            "robust volatility signal) — each band is a period the market behaved consistently before shifting."
        )

# ===== Tab 3: Cross-Commodity =====
with tab_cross:
    st.subheader("How the three commodities compare")

    summary_rows = []
    for key, label in TICKERS.items():
        summary_rows.append({
            "Commodity": label,
            "Latest close": f"${dfs[key]['Close'].iloc[-1]:,.2f}",
            "30-day vol (ann.)": f"{vol[key]['vol_30d'].iloc[-1]:.1%}",
            "Total flagged": len(z_flags[key]),
            "Flag rate": f"{len(z_flags[key]) / len(returns[key]) * 100:.1f}%",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Return correlation")
    st.caption(
        "Pearson (sensitive to single extreme days) vs. Spearman (rank-based, more robust) — "
        "shown side by side since a large gap between the two flags a correlation being distorted by one outlier."
    )

    pearson, spearman = compute_correlations(returns)
    col_a, col_b = st.columns(2)
    for col, corr, title in [(col_a, pearson, "Pearson"), (col_b, spearman, "Spearman (rank-based)")]:
        with col:
            heat = go.Figure(data=go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index,
                zmin=-1, zmax=1, colorscale="RdBu", zmid=0,
                text=corr.round(2).values, texttemplate="%{text}",
                colorbar=dict(thickness=12),
            ))
            heat.update_layout(**plotly_base_layout(title))
            heat.update_layout(height=340)
            st.plotly_chart(heat, use_container_width=True)

# ===== Tab 4: Data Quality =====
with tab_quality:
    st.subheader("Data quality summary")
    quality_summary = []
    for key, label in TICKERS.items():
        df = dfs[key]
        quality_summary.append({
            "Commodity": label,
            "Rows": len(df),
            "Date range": f"{df.index.min().date()} to {df.index.max().date()}",
            "Zero-volume days": int((df["Volume"] == 0).sum()),
            "% zero-volume": round((df["Volume"] == 0).mean() * 100, 2),
        })
    st.dataframe(pd.DataFrame(quality_summary), use_container_width=True, hide_index=True)

    st.caption(
        "Zero-volume days are expected around holidays and don't necessarily indicate a data problem — "
        "see the EDA notebook for the run-length check confirming these aren't extended gaps."
    )
