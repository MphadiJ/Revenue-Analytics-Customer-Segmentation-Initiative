import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Ensure src is on the path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from churn.churn_analysis import add_churn_analysis, segment_churn_summary

# ── Design tokens 
RISK_COLORS = {
    "High Risk":   "#e74c3c",
    "Medium Risk": "#f1c40f",
    "Low Risk":    "#2ecc71",
}

def churn_decision(score: float) -> str:
    """Binary + nuanced churn verdict per customer."""
    if score >= 0.65:   return "🔴 Will Churn"
    elif score >= 0.35: return "🟡 At Risk"
    else:               return "🟢 Retained"

def churn_action(risk: str) -> str:
    """Recommended retention action per risk tier."""
    return {
        "High Risk":   "Immediate retention offer — discount or loyalty reward",
        "Medium Risk": "Re-engagement campaign — personalised email or SMS",
        "Low Risk":    "Standard loyalty programme — no urgent action needed",
    }.get(risk, "—")


def churn_tab():
    st.subheader("⚠️ Churn Risk Analysis")
    st.caption(
        "Identify customers most likely to disengage — "
        "with per-customer churn scores, decisions, and recommended actions."
    )

    # ── Guard: need segmented data 
    if st.session_state.get("segmented_df") is None:
        st.warning("Run Customer Segmentation first (Tab 1) to unlock churn analysis.")
        return

    df = st.session_state["segmented_df"].copy()

    # ── Auto-score on load (no button needed) ─────────────────────────────────
    if "ChurnScore" not in df.columns:
        df = add_churn_analysis(df)                        # weighted RFM 0–1 score
        df["ChurnDecision"]      = df["ChurnScore"].apply(churn_decision)
        df["RecommendedAction"]  = df["ChurnRisk"].apply(churn_action)
        st.session_state["analysis_df"]  = df
        st.session_state["segmented_df"] = df
    else:
        # Already scored — just ensure decision columns exist
        if "ChurnDecision" not in df.columns:
            df["ChurnDecision"]     = df["ChurnScore"].apply(churn_decision)
            df["RecommendedAction"] = df["ChurnRisk"].apply(churn_action)
            st.session_state["analysis_df"]  = df
            st.session_state["segmented_df"] = df

    # ── KPI cards ─────────────────────────────────────────────────────────────
    total          = len(df)
    high_df        = df[df["ChurnRisk"] == "High Risk"]
    medium_df      = df[df["ChurnRisk"] == "Medium Risk"]
    low_df         = df[df["ChurnRisk"] == "Low Risk"]
    revenue_at_risk = high_df["Monetary"].sum()
    avg_score       = df["ChurnScore"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers",      f"{total:,}")
    c2.metric("🔴 Will Churn",        f"{len(high_df):,}",
              f"{len(high_df)/total*100:.1f}% of base")
    c3.metric("🟡 At Risk",           f"{len(medium_df):,}",
              f"{len(medium_df)/total*100:.1f}% of base")
    c4.metric("🟢 Retained",          f"{len(low_df):,}",
              f"{len(low_df)/total*100:.1f}% of base")
    c5.metric("💰 Revenue at Risk",   f"£{revenue_at_risk:,.0f}",
              "Sum of High Risk monetary value")

    st.divider()

    # ── Charts row 1 ──────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Risk Distribution")
        risk_counts = (
            df["ChurnRisk"]
            .value_counts()
            .reindex(["High Risk", "Medium Risk", "Low Risk"])
            .reset_index()
        )
        risk_counts.columns = ["Risk Tier", "Count"]
        fig_pie = px.pie(
            risk_counts, values="Count", names="Risk Tier",
            hole=0.5,
            color="Risk Tier",
            color_discrete_map=RISK_COLORS,
        )
        fig_pie.update_traces(textinfo="percent+label", textfont_size=12)
        fig_pie.update_layout(showlegend=False, margin=dict(t=20, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown("#### Churn Score Distribution")
        fig_hist = px.histogram(
            df, x="ChurnScore", nbins=30,
            color_discrete_sequence=["#58A6FF"],
            labels={"ChurnScore": "Churn Score (0 = safe, 1 = churning)"},
        )
        fig_hist.add_vline(x=0.65, line_dash="dash", line_color="#e74c3c",
                           annotation_text="High Risk threshold")
        fig_hist.add_vline(x=0.35, line_dash="dash", line_color="#f1c40f",
                           annotation_text="Medium Risk threshold")
        fig_hist.update_layout(margin=dict(t=20, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Charts row 2 ──────────────────────────────────────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("#### Churn Risk by Segment")
        if "Segment_Name" in df.columns:
            cross_tab = (
                df.groupby(["Segment_Name", "ChurnRisk"])
                .size()
                .unstack(fill_value=0)
                .reset_index()
            )
            melt_cols = [c for c in ["Low Risk","Medium Risk","High Risk"] if c in cross_tab.columns]
            fig_bar = px.bar(
                cross_tab, x="Segment_Name", y=melt_cols,
                barmode="stack",
                color_discrete_map=RISK_COLORS,
                labels={"value": "Customers", "variable": "Risk Tier"},
            )
            fig_bar.update_layout(margin=dict(t=20, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

    with col_d:
        st.markdown("#### Avg Churn Score per Segment")
        if "Segment_Name" in df.columns:
            seg_churn = (
                df.groupby("Segment_Name")["ChurnScore"]
                .mean()
                .reset_index()
                .sort_values("ChurnScore", ascending=False)
            )
            fig_seg = px.bar(
                seg_churn, x="Segment_Name", y="ChurnScore",
                color="ChurnScore",
                color_continuous_scale=[[0,"#2ecc71"],[0.5,"#f1c40f"],[1,"#e74c3c"]],
                text="ChurnScore",
            )
            fig_seg.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_seg.update_layout(
                margin=dict(t=20, b=10),
                coloraxis_showscale=False,
                yaxis_range=[0, 1],
            )
            st.plotly_chart(fig_seg, use_container_width=True)

    # ── Scatter: Recency vs Monetary ──────────────────────────────────────────
    st.markdown("#### Churn Risk Landscape — Recency vs Monetary")
    hover_cols = [c for c in ["CustomerID","Frequency","ChurnScore","Segment_Name"] if c in df.columns]
    fig_scatter = px.scatter(
        df, x="Recency", y="Monetary",
        color="ChurnRisk",
        color_discrete_map=RISK_COLORS,
        hover_data=hover_cols,
        opacity=0.65,
        labels={"Recency": "Recency (days)", "Monetary": "Monetary Value (£)"},
    )
    fig_scatter.update_layout(margin=dict(t=20, b=10))
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    # ── Top at-risk customers ─────────────────────────────────────────────────
    st.markdown("#### 🔴 Top 10 Customers Most Likely to Churn")
    st.caption("Sorted by churn score — highest risk first. These customers need immediate action.")

    top_risk_cols = [
        c for c in
        ["CustomerID","Segment_Name","ChurnScore","ChurnDecision",
         "ChurnRisk","RecommendedAction","Recency","Frequency","Monetary"]
        if c in df.columns
    ]
    top10 = (
        df[top_risk_cols]
        .sort_values("ChurnScore", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    top10.index = range(1, len(top10) + 1)
    st.dataframe(top10, use_container_width=True)

    st.divider()

    # ── Full customer churn table ─────────────────────────────────────────────
    st.markdown("#### Full Churn Intelligence Table")

    # Filter controls
    filter_col1, filter_col2 = st.columns([1, 3])
    with filter_col1:
        risk_filter = st.selectbox(
            "Filter by Risk",
            options=["All", "High Risk", "Medium Risk", "Low Risk"],
        )

    display_df = df.copy() if risk_filter == "All" else df[df["ChurnRisk"] == risk_filter].copy()
    display_df = display_df[top_risk_cols].sort_values("ChurnScore", ascending=False)
    display_df.index = range(1, len(display_df) + 1)
    st.dataframe(display_df, use_container_width=True)

    # ── Download ──────────────────────────────────────────────────────────────
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Full Churn Intelligence Report",
        data=csv_data,
        file_name="customer_churn_intelligence.csv",
        mime="text/csv",
        use_container_width=True,
    )
