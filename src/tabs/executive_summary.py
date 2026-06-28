import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def executive_summary_tab():
    st.subheader("🧠 Automated Business Insight Engine")

    if st.session_state["analysis_df"] is None:
        st.warning("⚠️ Summary engine requires data. Complete the Churn Risk Analysis tab first.")
        return

    df = st.session_state["analysis_df"]

    # --- CORE METRIC CALCULATIONS ---
    total_customers = len(df)
    total_revenue = df["Monetary"].sum()
    
    high_risk_df = df[df["ChurnRisk"] == "High Risk"]
    high_risk_count = len(high_risk_df)
    revenue_at_risk = high_risk_df["Monetary"].sum()
    revenue_at_risk_pct = (revenue_at_risk / total_revenue) * 100 if total_revenue > 0 else 0

    # Business Health Logic Score (Out of 10)
    base_score = 10.0
    deductions = (high_risk_count / total_customers * 5) + (revenue_at_risk_pct / 100 * 5)
    health_score = max(1.0, min(10.0, base_score - deductions))

    if health_score >= 7.5:
        score_color = "green"
        status_text = "STRONG"
    elif health_score >= 5.0:
        score_color = "orange"
        status_text = "STABLE WITH RISK"
    else:
        score_color = "red"
        status_text = "CRITICAL ACTION REQUIRED"

    # --- REPORT HEADER BLOCK ---
    st.markdown(f"""
    <div style="padding:20px; border-radius:10px; background-color:rgba(128,128,128,0.1); border-left: 6px solid {score_color}; margin-bottom: 25px;">
        <h3 style='margin-top:0;'>EXECUTIVE DOSSIER REPORT</h3>
        <p><strong>Overall Core Health Index:</strong> <span style='color:{score_color}; font-size:20px; font-weight:bold;'>{health_score:.1f} / 10</span> ({status_text})</p>
        <hr style='border: 0.5px solid rgba(128,128,128,0.2);'>
        <strong>Financial & Operational Breakdown:</strong>
        <ul>
            <li><b>Gross Revenue Base Assessed:</b> £{total_revenue:,.2f}</li>
            <li><b>Active Customer Database Volume:</b> {total_customers:,} active entities</li>
            <li><b>High Vulnerability Accounts:</b> {high_risk_count:,} accounts ({ (high_risk_count/total_customers)*100:.1f}% base volume)</li>
            <li><b>Total Financial Runway Exposure:</b> £{revenue_at_risk:,.2f} (<span style='color:#e74c3c;'><b>{revenue_at_risk_pct:.1f}%</b></span> of global portfolio value)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # --- INTERACTIVE WIN-BACK SIMULATOR ---
    st.markdown("### 🎛️ Revenue Recovery 'What-If' Simulator")
    st.write("Simulate targeted strategic campaigns to recover vulnerable accounts and calculate projected reclaimed pipeline value.")

    # Visual container for the widget layout
    col_input, col_viz = st.columns([2, 3])

    with col_input:
        st.markdown("#### Simulation Levers")
        # Let users select the target retention rate
        target_conversion = st.slider(
            "Target Win-Back Conversion Rate (%)",
            min_value=0,
            max_value=100,
            value=25,
            step=5,
            help="What percentage of high-risk customers do you expect to save via targeted incentives or customer success outreach?"
        )
        
        # Calculate simulation outputs
        sim_conversion_decimal = target_conversion / 100.0
        projected_saved_customers = int(high_risk_count * sim_conversion_decimal)
        projected_revenue_reclaimed = revenue_at_risk * sim_conversion_decimal
        net_revenue_lost = revenue_at_risk - projected_revenue_reclaimed
        
        # Recalculate speculative Health Score post-simulation
        remaining_high_risk = high_risk_count - projected_saved_customers
        sim_deductions = (remaining_high_risk / total_customers * 5) + (net_revenue_lost / total_revenue * 5)
        sim_health_score = max(1.0, min(10.0, base_score - sim_deductions))

        # Dynamic metric tiles within the simulation box
        st.metric(
            "Projected Revenue Reclaimed", 
            f"£{projected_revenue_reclaimed:,.2f}",
            delta=f"+{target_conversion}% recovery efficacy"
        )
        st.metric(
            "Accounts Rescued", 
            f"{projected_saved_customers:,} Clients",
            delta=f"{high_risk_count - projected_saved_customers:,} remaining at risk",
            delta_color="inverse"
        )
        st.metric(
            "Simulated Health Score Output",
            f"{sim_health_score:.1f} / 10",
            delta=f"{sim_health_score - health_score:+.1f} Improvement"
        )

    with col_viz:
        # Generate a waterfall or stacked gauge layout visualizing exposure allocation
        st.markdown("#### Portfolio Exposure Realignment")
        
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Bar(
            name='Net Revenue at Risk (Lost)',
            y=['Revenue Run-Rate'],
            x=[net_revenue_lost],
            orientation='h',
            marker=dict(color='#e74c3c')
        ))
        fig_sim.add_trace(go.Bar(
            name='Projected Reclaimed Revenue',
            y=['Revenue Run-Rate'],
            x=[projected_revenue_reclaimed],
            orientation='h',
            marker=dict(color='#2ecc71')
        ))
        fig_sim.add_trace(go.Bar(
            name='Stable Portfolio Revenue',
            y=['Revenue Run-Rate'],
            x=[total_revenue - revenue_at_risk],
            orientation='h',
            marker=dict(color='#34495e')
        ))

        fig_sim.update_layout(
            barmode='stack',
            height=260,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_sim, use_container_width=True)

    st.divider()

    # --- TARGETED STRATEGIC RECOMMENDATIONS ---
    st.markdown("### 📋 Targeted Strategic Action Playbook")
    
    recs = []
    if revenue_at_risk_pct > 15:
        recs.append(f"🔴 **High-Exposure Account Defense:** Over 15% of total revenue is concentrated in high-risk churn tiers. If your team hits the **{target_conversion}% win-back goal**, you will secure **£{projected_revenue_reclaimed:,.2f}** that would otherwise walk out the door.")
    else:
        recs.append("🟢 **Revenue Retention:** Portfolio churn exposure remains within nominal variances. Continue standard lifecycle management tracks.")

    if "Segment_Name" in df.columns:
        segments = df["Segment_Name"].unique()
        if "Champions" in segments:
            recs.append("⭐ **Value Optimization:** Protect your 'Champions' cohort. Roll out exclusivity benefits or early-access structures to ensure competitor lock-out.")
        if "At Risk" in segments or "Hibernating" in segments:
            recs.append(f"🔄 **Targeted Campaign Budgeting:** Allocate defensive marketing budgets against the 'At Risk' segment. Based on your simulation, spending up to **£{projected_revenue_reclaimed * 0.10:,.2f}** (10% of reclaimed value) yields a highly profitable ROI framework.")

    for item in recs:
        st.write(item)

    st.divider()

    # --- SUMMARY EXPORT CAPABILITIES ---
    st.markdown("### Report Generation and Exporting")
    summary_text = f"""EXECUTIVE ACCOUNT SUMMARY WITH SIMULATION VARIANCE
------------------------------------------------------
Baseline Health Score: {health_score:.1f} / 10
Target Win-Back Rate: {target_conversion}%
Simulated Post-Campaign Health Score: {sim_health_score:.1f} / 10

Total Revenue Base Assessed: £{total_revenue:,.2f}
Gross Financial Exposure: £{revenue_at_risk:,.2f}
Projected Revenue Reclaimed: £{projected_revenue_reclaimed:,.2f}
Net Residual Revenue Risked: £{net_revenue_lost:,.2f}
"""
    st.download_button(
        label="📄 Export Executive Briefing & Simulation Model (.txt)",
        data=summary_text,
        file_name="Executive_Revenue_Simulation_Report.txt",
        mime="text/plain",
        use_container_width=True
    )
