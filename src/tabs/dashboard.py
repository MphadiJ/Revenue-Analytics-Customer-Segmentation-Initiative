import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def dashboard_tab():
    st.subheader("📈 Executive Command Center")
    
    if st.session_state["analysis_df"] is None:
        st.warning("⚠️ The Executive Dashboard requires active metrics. Please complete Churn Risk Analysis first.")
        return

    df = st.session_state["analysis_df"]

    # --- 1. THE CEO'S CORE STRATEGIC METRICS ---
    total_revenue = df["Monetary"].sum()
    total_customers = len(df)
    
    high_risk_df = df[df["ChurnRisk"] == "High Risk"]
    revenue_at_risk = high_risk_df["Monetary"].sum()
    revenue_at_risk_pct = (revenue_at_risk / total_revenue) * 100 if total_revenue > 0 else 0

    # Top-tier high-value customer count (Champions + Loyalists)
    premium_segments = ["Champions", "Loyalists", "Potential Loyalists"]
    premium_df = df[df["Segment_Name"].isin(premium_segments)] if "Segment_Name" in df.columns else df
    premium_revenue_pct = (premium_df["Monetary"].sum() / total_revenue) * 100 if total_revenue > 0 else 0

    # Clean, high-impact semantic tiles
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gross Platform Revenue", f"£{total_revenue:,.0f}")
    c2.metric("Active Customer Base", f"{total_customers:,} accounts")
    c3.metric("Revenue Protection Exposure", f"£{revenue_at_risk:,.0f}", f"{revenue_at_risk_pct:.1f}% At Risk", delta_color="inverse")
    c4.metric("Core Segment Concentration", f"{premium_revenue_pct:.1f}%", "Top Core Cohorts")

    st.divider()

    # --- 2. SIMPLE, MACRO VISUALS FOR THE BOARDROOM ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 💰 Revenue Concentration Leaderboard")
        st.caption("Which customer segments are contributing the most to our gross financial runway?")
        
        # Aggregate financial breakdown per segment
        rev_leaderboard = df.groupby("Segment_Name")["Monetary"].sum().reset_index()
        rev_leaderboard = rev_leaderboard.sort_values(by="Monetary", ascending=True) # Ascending for horizontal layout orientation
        
        fig_leaderboard = px.bar(
            rev_leaderboard,
            y="Segment_Name",
            x="Monetary",
            orientation="h",
            text="Monetary",
            labels={"Monetary": "Total Value (£)", "Segment_Name": "Customer Cohort"},
            color="Monetary",
            color_continuous_scale="Blues"
        )
        # Format layout text labels inside bars cleanly
        fig_leaderboard.update_traces(texttemplate='£%{text:,.0f}', textposition='outside')
        fig_leaderboard.update_layout(showlegend=False, height=320, margin=dict(l=10, r=40, t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_leaderboard, use_container_width=True)

    with col_right:
        st.markdown("#### 🛡️ Financial Portfolio Health Matrix")
        st.caption("What is our net financial risk split across the system operational tiers?")
        
        # Aggregate financial asset health allocations
        risk_revenue = df.groupby("ChurnRisk")["Monetary"].sum().reset_index()
        
        fig_donut = px.pie(
            risk_revenue,
            values="Monetary",
            names="ChurnRisk",
            hole=0.55,
            color="ChurnRisk",
            color_discrete_map={
                "Low Risk": "#2ecc71",     # Solid operational green
                "Medium Risk": "#f1c40f",  # Warning gold
                "High Risk": "#e74c3c"     # Critical defense red
            }
        )
        fig_donut.update_traces(textposition='inside', textinfo='percent+label')
        fig_donut.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_donut, use_container_width=True)

    st.divider()

    # --- 3. DYNAMIC STRATEGIC SUMMARY GRID ---
    st.markdown("#### 🏁 Immediate C-Suite Action Matrix")
    
    # Let's generate dynamic high-level numbers for the text grid
    highest_rev_segment = rev_leaderboard.iloc[-1]["Segment_Name"] if not rev_leaderboard.empty else "N/A"
    highest_rev_value = rev_leaderboard.iloc[-1]["Monetary"] if not rev_leaderboard.empty else 0
    
    g1, g2 = st.columns(2)
    with g1:
        st.info(f"🏆 **Growth Opportunity:** The **{highest_rev_segment}** cohort remains your single largest financial engine, holding **£{highest_rev_value:,.0f}** in enterprise value. Ensure client success teams have zero friction points here.")
    with g2:
        if revenue_at_risk_pct > 10:
            st.error(f"🚨 **Risk Exposure Warning:** **{revenue_at_risk_pct:.1f}%** of our transactional assets are flagged with high churn indicators. This accounts for **£{revenue_at_risk:,.0f}** in exposed capital that requires defensive account validation.")
        else:
            st.success("✨ **Portfolio Stability:** Churn metrics are currently pacing well within nominal historical baselines across all high-value corporate vectors.")
