"""
streamlit_app/kpi_dashboard.py

KPI Intelligence Dashboard
===========================
Standalone Streamlit app powered by kpi_engine.py.

Accepts the same CSV format as the main app (post-segmentation output),
or raw RFM data and runs the full KPI suite.

Tabs
----
1. 📈 Revenue      — Total revenue, ARPU, AOV, segment revenue share
2. 👥 Customers    — Base health, active rate, churn rate, tenure
3. 🎯 Segments     — Per-segment KPI table + charts
4. ⚡ Engagement   — Frequency distribution, recency buckets
5. 🔥 Risk         — Revenue at risk, churn risk breakdown
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# ── Path fix ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH     = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_PATH)

from kpi_engine import (
    compute_all_kpis,
    revenue_kpis,
    customer_kpis,
    segment_kpis,
    engagement_kpis,
    risk_kpis,
)
from churn.churn_analysis import add_churn_analysis

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KPI Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme colours ─────────────────────────────────────────────────────────────
PALETTE = {
    "primary":   "#0A84FF",
    "success":   "#30D158",
    "warning":   "#FFD60A",
    "danger":    "#FF453A",
    "muted":     "#636366",
    "surface":   "#1C1C1E",
    "segments":  ["#0A84FF", "#30D158", "#FFD60A", "#FF453A", "#BF5AF2", "#FF9F0A"],
}

RISK_COLORS = {
    "High Risk":   PALETTE["danger"],
    "Medium Risk": PALETTE["warning"],
    "Low Risk":    PALETTE["success"],
}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 0.78rem; color: #8E8E93; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-card {
        background: #1C1C1E;
        border: 1px solid #2C2C2E;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #F2F2F7;
        letter-spacing: -0.01em;
        margin: 1.2rem 0 0.6rem 0;
    }
    div[data-testid="stTabs"] button { font-size: 0.88rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR — DATA UPLOAD
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 KPI Dashboard")
    st.caption("Revenue · Customers · Segments · Risk")
    st.divider()

    st.markdown("### Load Data")
    uploaded = st.file_uploader(
        "Upload CSV (RFM + optional Segment_Name, ChurnRisk)",
        type=["csv"],
        help="Columns: Recency, Frequency, Monetary — plus optionally Tenure, AvgOrderValue, Segment_Name, ChurnRisk",
    )

    st.divider()
    st.markdown("**Required columns**")
    st.code("Recency, Frequency, Monetary", language=None)
    st.markdown("**Recommended additions**")
    st.code("Tenure, AvgOrderValue\nSegment_Name, ChurnRisk", language=None)
    st.caption("Run the main app first to get a fully-enriched CSV, or upload raw RFM data.")

    auto_churn = st.checkbox(
        "Auto-compute Churn Risk if missing",
        value=True,
        help="Uses the same RFM-weighted scoring from the churn module.",
    )


# ═════════════════════════════════════════════════════════════════════════════
# LOAD & PREPARE DATA
# ═════════════════════════════════════════════════════════════════════════════
st.title("KPI Intelligence Dashboard 📊")
st.caption("Powered by `src/kpi_engine.py` · Revenue · Customer Health · Segment Performance · Risk")

if not uploaded:
    st.info("👈 Upload a CSV in the sidebar to get started.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

# Auto-add churn if missing
if auto_churn and "ChurnRisk" not in df.columns:
    try:
        df = add_churn_analysis(df)
        st.sidebar.success("✅ Churn scores computed automatically.")
    except Exception:
        st.sidebar.warning("Could not compute churn — ensure Recency, Frequency, Monetary are present.")

# Compute all KPIs
try:
    kpis = compute_all_kpis(df)
except Exception as e:
    st.error(f"KPI computation failed: {e}")
    import traceback
    st.code(traceback.format_exc())
    st.stop()

rev  = kpis.get("revenue", {})
cust = kpis.get("customer", {})
eng  = kpis.get("engagement", {})
seg  = kpis.get("segment", pd.DataFrame())
risk = kpis.get("risk", {})

has_segment = not seg.empty
has_risk    = bool(risk)


# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL SUMMARY ROW
# ═════════════════════════════════════════════════════════════════════════════
g1, g2, g3, g4, g5 = st.columns(5)
g1.metric("Total Revenue",    f"£{rev.get('total_revenue', 0):,.0f}")
g2.metric("Total Customers",  f"{cust.get('total_customers', 0):,}")
g3.metric("ARPU",             f"£{rev.get('arpu', 0):,.2f}")
g4.metric("Active Rate",      f"{cust.get('active_rate_pct', 0):.1f}%")
g5.metric("Revenue at Risk",  f"{risk.get('revenue_at_risk_pct', 0):.1f}%" if has_risk else "N/A")

st.divider()


# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["📈 Revenue", "👥 Customers", "🎯 Segments", "⚡ Engagement", "🔥 Risk"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — REVENUE
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<p class="section-header">Revenue Overview</p>', unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Total Revenue",     f"£{rev.get('total_revenue', 0):,.2f}")
    r2.metric("ARPU",              f"£{rev.get('arpu', 0):,.2f}")
    r3.metric("Avg Order Value",   f"£{rev.get('avg_order_value', 0):,.2f}")
    r4.metric("Top 10% Rev Share", f"{rev.get('top10_pct_revenue_share', 0):.1f}%")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        # Revenue distribution histogram
        fig_rev_hist = px.histogram(
            df, x="Monetary",
            nbins=40,
            title="Revenue Distribution per Customer",
            color_discrete_sequence=[PALETTE["primary"]],
            labels={"Monetary": "Customer Revenue (£)"},
        )
        fig_rev_hist.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#F2F2F7",
        )
        st.plotly_chart(fig_rev_hist, use_container_width=True)

    with col_b:
        if has_segment:
            fig_rev_seg = px.pie(
                seg, values="TotalRevenue", names="Segment",
                title="Revenue Share by Segment",
                color_discrete_sequence=PALETTE["segments"],
                hole=0.45,
            )
            fig_rev_seg.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#F2F2F7",
            )
            st.plotly_chart(fig_rev_seg, use_container_width=True)
        else:
            st.info("Add `Segment_Name` to your CSV to see revenue by segment.")

    # Revenue Pareto — top customers cumulative
    st.markdown('<p class="section-header">Revenue Concentration (Pareto)</p>', unsafe_allow_html=True)
    sorted_mon = df["Monetary"].sort_values(ascending=False).reset_index(drop=True)
    cumulative = sorted_mon.cumsum() / sorted_mon.sum() * 100
    pareto_df  = pd.DataFrame({
        "Customer Rank": range(1, len(sorted_mon) + 1),
        "Cumulative Revenue %": cumulative,
    })
    fig_pareto = px.line(
        pareto_df, x="Customer Rank", y="Cumulative Revenue %",
        title="Cumulative Revenue % by Customer Rank",
        color_discrete_sequence=[PALETTE["primary"]],
    )
    fig_pareto.add_hline(y=80, line_dash="dot", line_color=PALETTE["warning"],
                         annotation_text="80% revenue threshold", annotation_position="right")
    fig_pareto.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#F2F2F7",
    )
    st.plotly_chart(fig_pareto, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — CUSTOMERS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<p class="section-header">Customer Base Health</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers",  f"{cust.get('total_customers', 0):,}")
    c2.metric("Active Customers", f"{cust.get('active_customers', 0):,}")
    c3.metric("Active Rate",      f"{cust.get('active_rate_pct', 0):.1f}%")
    c4.metric("Churn Rate",
              f"{cust.get('churn_rate_pct', 0):.1f}%" if cust.get('churn_rate_pct') is not None else "N/A")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        # Recency distribution
        fig_rec = px.histogram(
            df, x="Recency", nbins=35,
            title="Recency Distribution (days since last purchase)",
            color_discrete_sequence=[PALETTE["success"]],
            labels={"Recency": "Recency (days)"},
        )
        fig_rec.add_vline(x=90, line_dash="dot", line_color=PALETTE["warning"],
                          annotation_text="Active threshold (90d)", annotation_position="top right")
        fig_rec.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#F2F2F7",
        )
        st.plotly_chart(fig_rec, use_container_width=True)

    with col_b:
        # Recency buckets: active / lapsing / inactive
        buckets = pd.cut(
            df["Recency"],
            bins=[0, 30, 90, 180, 365, float("inf")],
            labels=["0–30d", "31–90d", "91–180d", "181–365d", "365d+"],
            right=True,
        ).value_counts().sort_index().reset_index()
        buckets.columns = ["Recency Bucket", "Customers"]
        fig_buckets = px.bar(
            buckets, x="Recency Bucket", y="Customers",
            title="Customer Count by Recency Bucket",
            color="Customers",
            color_continuous_scale=["#30D158", "#FFD60A", "#FF6961", "#FF453A", "#8B0000"],
            text="Customers",
        )
        fig_buckets.update_traces(textposition="outside")
        fig_buckets.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#F2F2F7",
        )
        st.plotly_chart(fig_buckets, use_container_width=True)

    # Tenure if available
    if "Tenure" in df.columns and cust.get("avg_tenure_days") is not None:
        st.markdown('<p class="section-header">Customer Tenure</p>', unsafe_allow_html=True)
        t1, t2 = st.columns(2)
        t1.metric("Avg Tenure (days)", f"{cust['avg_tenure_days']:,.0f}")
        t2.metric("Median Recency",    f"{cust.get('median_recency_days', 0):.0f} days")

        fig_tenure = px.histogram(
            df, x="Tenure", nbins=35,
            title="Customer Tenure Distribution (days since first purchase)",
            color_discrete_sequence=[PALETTE["primary"]],
            labels={"Tenure": "Tenure (days)"},
        )
        fig_tenure.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#F2F2F7",
        )
        st.plotly_chart(fig_tenure, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — SEGMENTS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    if not has_segment:
        st.warning("No `Segment_Name` column found. Run segmentation in the main app first, then re-upload.")
    else:
        st.markdown('<p class="section-header">Segment Performance</p>', unsafe_allow_html=True)

        st.dataframe(
            seg.style.format({
                "TotalRevenue":     "£{:,.2f}",
                "ARPU":             "£{:,.2f}",
                "AvgMonetary":      "£{:,.2f}",
                "AvgOrderValue":    "£{:,.2f}",
                "RevenueSharePct":  "{:.1f}%",
                "CustomerSharePct": "{:.1f}%",
                "HighRiskPct":      "{:.1f}%" if "HighRiskPct" in seg.columns else None,
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()
        s1, s2 = st.columns(2)

        with s1:
            fig_seg_rev = px.bar(
                seg, x="Segment", y="TotalRevenue",
                title="Total Revenue by Segment",
                color="Segment",
                color_discrete_sequence=PALETTE["segments"],
                text="RevenueSharePct",
                labels={"TotalRevenue": "Revenue (£)", "RevenueSharePct": "Revenue Share %"},
            )
            fig_seg_rev.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_seg_rev.update_layout(
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#F2F2F7",
            )
            st.plotly_chart(fig_seg_rev, use_container_width=True)

        with s2:
            fig_seg_cust = px.bar(
                seg, x="Segment", y="CustomerCount",
                title="Customer Count by Segment",
                color="Segment",
                color_discrete_sequence=PALETTE["segments"],
                text="CustomerSharePct",
                labels={"CustomerCount": "Customers", "CustomerSharePct": "Customer Share %"},
            )
            fig_seg_cust.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_seg_cust.update_layout(
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#F2F2F7",
            )
            st.plotly_chart(fig_seg_cust, use_container_width=True)

        # Radar — avg RFM per segment
        st.markdown('<p class="section-header">Segment RFM Radar</p>', unsafe_allow_html=True)
        radar_cols = ["AvgRecency", "AvgFrequency", "AvgMonetary"]
        if all(c in seg.columns for c in radar_cols):
            radar_df = seg[["Segment"] + radar_cols].copy()
            for col in radar_cols:
                col_max = radar_df[col].max()
                radar_df[col] = radar_df[col] / col_max if col_max > 0 else radar_df[col]

            fig_radar = go.Figure()
            categories = ["Avg Recency", "Avg Frequency", "Avg Monetary"]
            for i, row in radar_df.iterrows():
                vals = [row["AvgRecency"], row["AvgFrequency"], row["AvgMonetary"]]
                vals += [vals[0]]  # close polygon
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=row["Segment"],
                    line_color=PALETTE["segments"][i % len(PALETTE["segments"])],
                    opacity=0.7,
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Normalised RFM Profile per Segment",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#F2F2F7",
            )
            st.plotly_chart(fig_radar, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — ENGAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<p class="section-header">Engagement KPIs</p>', unsafe_allow_html=True)

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Avg Recency",       f"{eng.get('avg_recency', 0):.1f} days")
    e2.metric("Avg Frequency",     f"{eng.get('avg_frequency', 0):.2f}")
    e3.metric("Single-purchase %", f"{eng.get('pct_single_purchase', 0):.1f}%")
    e4.metric("High-freq % (>10)", f"{eng.get('pct_high_frequency', 0):.1f}%")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        fig_freq = px.histogram(
            df, x="Frequency", nbins=40,
            title="Purchase Frequency Distribution",
            color_discrete_sequence=[PALETTE["primary"]],
            labels={"Frequency": "Number of Purchases"},
        )
        fig_freq.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#F2F2F7",
        )
        st.plotly_chart(fig_freq, use_container_width=True)

    with col_b:
        # Recency vs Frequency scatter
        hover = [c for c in ["Monetary", "Segment_Name", "ChurnRisk"] if c in df.columns]
        color_col = "Segment_Name" if "Segment_Name" in df.columns else None
        fig_rfm_scatter = px.scatter(
            df.sample(min(3000, len(df)), random_state=42),
            x="Recency", y="Frequency",
            color=color_col,
            color_discrete_sequence=PALETTE["segments"],
            hover_data=hover,
            opacity=0.55,
            title="Recency vs Frequency (sample up to 3,000)",
            labels={"Recency": "Recency (days)", "Frequency": "Frequency"},
        )
        fig_rfm_scatter.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#F2F2F7",
        )
        st.plotly_chart(fig_rfm_scatter, use_container_width=True)

    # AOV distribution if available
    if "AvgOrderValue" in df.columns:
        st.markdown('<p class="section-header">Average Order Value</p>', unsafe_allow_html=True)
        fig_aov = px.histogram(
            df, x="AvgOrderValue", nbins=35,
            title="Average Order Value Distribution",
            color_discrete_sequence=[PALETTE["warning"]],
            labels={"AvgOrderValue": "AOV (£)"},
        )
        fig_aov.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#F2F2F7",
        )
        st.plotly_chart(fig_aov, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — RISK
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    if not has_risk:
        st.warning("No `ChurnRisk` column found. Enable 'Auto-compute Churn Risk' in the sidebar or add the column to your CSV.")
    else:
        st.markdown('<p class="section-header">Revenue & Churn Risk</p>', unsafe_allow_html=True)

        rk1, rk2, rk3, rk4 = st.columns(4)
        rk1.metric("🔴 High Risk",        f"{risk['n_high_risk']:,}  ({risk['high_risk_pct']:.1f}%)")
        rk2.metric("🟡 Medium Risk",      f"{risk['n_medium_risk']:,}  ({risk['medium_risk_pct']:.1f}%)")
        rk3.metric("🟢 Low Risk",         f"{risk['n_low_risk']:,}  ({risk['low_risk_pct']:.1f}%)")
        rk4.metric("💸 Revenue at Risk",  f"£{risk['revenue_at_risk']:,.0f}  ({risk['revenue_at_risk_pct']:.1f}%)")

        st.divider()
        col_a, col_b = st.columns(2)

        with col_a:
            risk_pie = pd.DataFrame({
                "Risk Level": ["High Risk", "Medium Risk", "Low Risk"],
                "Count": [risk["n_high_risk"], risk["n_medium_risk"], risk["n_low_risk"]],
            })
            fig_risk_pie = px.pie(
                risk_pie, values="Count", names="Risk Level",
                title="Customer Distribution by Risk Level",
                color="Risk Level",
                color_discrete_map=RISK_COLORS,
                hole=0.45,
            )
            fig_risk_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#F2F2F7",
            )
            st.plotly_chart(fig_risk_pie, use_container_width=True)

        with col_b:
            if "ChurnScore" in df.columns:
                fig_score_dist = px.histogram(
                    df, x="ChurnScore",
                    color="ChurnRisk",
                    color_discrete_map=RISK_COLORS,
                    nbins=30,
                    title="Churn Score Distribution by Risk Level",
                    labels={"ChurnScore": "Churn Score (0 = safe → 1 = at risk)"},
                    barmode="overlay",
                    opacity=0.75,
                )
                fig_score_dist.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#F2F2F7",
                )
                st.plotly_chart(fig_score_dist, use_container_width=True)

        # Revenue at risk by segment
        if has_segment and "HighRiskPct" in seg.columns:
            st.markdown('<p class="section-header">Risk Exposure by Segment</p>', unsafe_allow_html=True)
            fig_seg_risk = px.bar(
                seg, x="Segment", y="HighRiskPct",
                title="High-Risk Customer % per Segment",
                color="HighRiskPct",
                color_continuous_scale=["#30D158", "#FFD60A", "#FF453A"],
                text="HighRiskPct",
                labels={"HighRiskPct": "High Risk %"},
            )
            fig_seg_risk.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_seg_risk.update_layout(
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#F2F2F7",
            )
            st.plotly_chart(fig_seg_risk, use_container_width=True)

        # Recency vs Monetary scatter coloured by risk
        st.markdown('<p class="section-header">Risk Landscape</p>', unsafe_allow_html=True)
        fig_risk_scatter = px.scatter(
            df.sample(min(3000, len(df)), random_state=42),
            x="Recency", y="Monetary",
            color="ChurnRisk",
            color_discrete_map=RISK_COLORS,
            opacity=0.6,
            title="Recency vs Monetary — coloured by Churn Risk (sample up to 3,000)",
            labels={"Recency": "Recency (days)", "Monetary": "Monetary (£)"},
            hover_data=[c for c in ["Frequency", "ChurnScore", "Segment_Name"] if c in df.columns],
        )
        fig_risk_scatter.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#F2F2F7",
        )
        st.plotly_chart(fig_risk_scatter, use_container_width=True)

        # Download enriched CSV
        st.divider()
        st.download_button(
            label="⬇️ Download Enriched KPI CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="kpi_enriched.csv",
            mime="text/csv",
        )
