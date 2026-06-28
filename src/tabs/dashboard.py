import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def dashboard_tab():
    st.subheader("📈 Executive KPI Dashboard")

    if st.session_state["analysis_df"] is None:
        st.warning("⚠️ The Dashboard requires historical metrics. Please complete Churn Risk Analysis first.")
        return

    df = st.session_state["analysis_df"]

    # 1. Macro Metrics Row
    st.markdown("### Operational Vital Signs")
    c1, c2, c3, c4 = st.columns(4)

    total_revenue = df["Monetary"].sum()
    avg_ltv = df["Monetary"].mean()
    high_risk_pct = (df["ChurnRisk"] == "High Risk").sum() / len(df) * 100
    avg_tenure = df["Tenure"].mean()

    c1.metric("Total Platform Revenue", f"£{total_revenue:,.2f}")
    c2.metric("Average Customer LTV", f"£{avg_ltv:,.2f}")
    c3.metric("Platform Churn Rate Exposure", f"{high_risk_pct:.1f}%")
    c4.metric("Mean Account Age (Tenure)", f"{avg_tenure:.1f} Months")

    st.divider()

    # 2. Visual Deep-Dives
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Financial Value Matrix: LTV vs. Order Frequency")
        # Scatter layout tracking value dynamics
        fig_scatter = px.scatter(
            df,
            x="Frequency",
            y="Monetary",
            color="Segment_Name",
            size="AvgOrderValue",
            hover_name="Segment_Name",
            log_x=True,
            title="Revenue Concentration vs Engagement Frequency",
            labels={"Monetary": "Lifetime Value (£)", "Frequency": "Purchase Count (Log Scale)"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_right:
        st.markdown("#### Recency Distributions by Customer Cohort")
        fig_box = px.box(
            df,
            x="Segment_Name",
            y="Recency",
            color="Segment_Name",
            title="Days Since Last Activity by Segment",
            labels={"Segment_Name": "Cohort", "Recency": "Days Inactive"}
        )
        st.plotly_chart(fig_box, use_container_width=True)
