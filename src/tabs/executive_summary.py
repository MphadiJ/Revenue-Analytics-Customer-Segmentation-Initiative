import streamlit as st
import pandas as pd


def executive_summary_tab():
    st.subheader("🧠 Automated Business Insight Engine")

    if st.session_state["analysis_df"] is None:
        st.warning("⚠️ Summary engine requires data. Complete the Churn Risk Analysis tab first.")
        return

    df = st.session_state["analysis_df"]

    # Calculate dynamic core variables
    total_customers = len(df)
    total_revenue = df["Monetary"].sum()

    high_risk_df = df[df["ChurnRisk"] == "High Risk"]
    high_risk_count = len(high_risk_df)
    revenue_at_risk = high_risk_df["Monetary"].sum()
    revenue_at_risk_pct = (revenue_at_risk / total_revenue) * 100 if total_revenue > 0 else 0

    # Business Health Logic Score (Out of 10)
    # Deducts score points based on systemic risk exposures
    base_score = 10.0
    deductions = (high_risk_count / total_customers * 5) + (revenue_at_risk_pct / 100 * 5)
    health_score = max(1.0, min(10.0, base_score - deductions))

    # Determine status color
    if health_score >= 7.5:
        score_color = "green"
        status_text = "STRONG"
    elif health_score >= 5.0:
        score_color = "orange"
        status_text = "STABLE WITH RISK"
    else:
        score_color = "red"
        status_text = "CRITICAL ACTION REQUIRED"

    # 1. Executive Summary Output Display Block
    st.markdown(f"""
    <div style="padding:20px; border-radius:10px; background-color:rgba(128,128,128,0.1); border-left: 6px solid {score_color};">
        <h3 style='margin-top:0;'>EXECUTIVE DOSSIER REPORT</h3>
        <p><strong>Overall Core Health Index:</strong> <span style='color:{score_color}; font-size:20px; font-weight:bold;'>{health_score:.1f} / 10</span> ({status_text})</p>
        <hr style='border: 0.5px solid rgba(128,128,128,0.2);'>
        <strong>Financial & Operational Breakdown:</strong>
        <ul>
            <li><b>Gross Revenue Base Assessed:</b> £{total_revenue:,.2f}</li>
            <li><b>Active Customer Database Volume:</b> {total_customers:,} active entities</li>
            <li><b>High Vulnerability Accounts:</b> {high_risk_count:,} accounts ({(high_risk_count / total_customers) * 100:.1f}% base volume)</li>
            <li><b>Total Financial Runway Exposure:</b> £{revenue_at_risk:,.2f} (<span style='color:#e74c3c;'><b>{revenue_at_risk_pct:.1f}%</b></span> of global portfolio value)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 Targeted Strategic Action Playbook")

    # 2. Programmatic Strategic Recommendations Generation
    recs = []
    if revenue_at_risk_pct > 15:
        recs.append(
            "🔴 **High-Exposure Account Defense:** Over 15% of total recurring revenue is concentrated within high-risk churn tiers. Instruct the client success group to initiate immediate programmatic discovery interviews with top-tier accounts in this cohort.")
    else:
        recs.append(
            "🟢 **Revenue Retention:** Portfolio churn exposure remains within nominal variances. Continue standard lifecycle management tracks.")

    # Segment specific recommendations
    if "Segment_Name" in df.columns:
        segments = df["Segment_Name"].unique()
        if "Champions" in segments:
            recs.append(
                "⭐ **Value Optimization:** Protect your 'Champions' cohort. Roll out exclusivity benefits, beta programs, or early-access loyalty structures to prevent poaching by competitors.")
        if "At Risk" in segments or "Hibernating" in segments:
            recs.append(
                "🔄 **Win-Back Campaign Infrastructure:** Design specific email sequences and automated win-back pricing bundles targeting dormant accounts to recapture sliding volume.")

    recs.append(
        "📊 **Data-Driven Triggers:** Set up automated Slack or email alerts for the customer success team. These should trigger whenever an account's transactional inactivity passes the 75th percentile mark.")

    for item in recs:
        st.write(item)

    st.divider()

    # 3. Export Capabilities
    st.markdown("### Report Generation and Exporting")
    summary_text = f"""EXECUTIVE ACCOUNT SUMMARY
-------------------------------------------
Health Score: {health_score:.1f} / 10 ({status_text})
Total Revenue Base: £{total_revenue:,.2f}
Total Exposure: £{revenue_at_risk:,.2f} ({revenue_at_risk_pct:.1f}%)
Total Flagged Customers: {high_risk_count} accounts
"""
    st.download_button(
        label="📄 Export Executive Briefing Document Summary (.txt)",
        data=summary_text,
        file_name="Executive_Revenue_Summary.txt",
        mime="text/plain",
        use_container_width=True
    )
