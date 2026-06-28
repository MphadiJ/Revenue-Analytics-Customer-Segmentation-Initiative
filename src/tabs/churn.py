import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def churn_tab():
    st.subheader("⚠️ Churn Risk Analysis")

    # 1. Guard rail check: Ensure segmentation has run first
    if st.session_state["segmented_df"] is None:
        st.warning("⚠️ Please upload a dataset and run Customer Segmentation first.")
        return

    df = st.session_state["segmented_df"].copy()

    st.write(
        """
        Analyze customer churn vulnerabilities. This module evaluates engagement patterns, 
        flags accounts with high flight risk, and estimates revenue exposure.
        """
    )

    # 2. Churn Logic & Scoring Calculations
    # (Adjust thresholds or point allocations to match your exact model rules)
    if st.button("🚀 Calculate Churn Risk Scores", use_container_width=True):
        with st.spinner("Analyzing customer behavioral risk metrics..."):

            # Example heuristic-based scoring if using a rule engine,
            # or replace with your model inference call: pipeline.churn_model.predict(df)
            recency_hi = df["Recency"].quantile(0.75)
            frequency_lo = df["Frequency"].quantile(0.25)
            tenure_lo = df["Tenure"].quantile(0.25)

            risk_scores = []
            for _, row in df.iterrows():
                score = 0
                if row["Recency"] >= recency_hi: score += 40
                if row["Frequency"] <= frequency_lo: score += 30
                if row["Tenure"] <= tenure_lo: score += 30
                risk_scores.append(min(score, 100))

            df["ChurnScore"] = risk_scores

            # Map into operational risk buckets
            def assign_risk_tier(score):
                if score >= 70: return "High Risk"
                if score >= 40: return "Medium Risk"
                return "Low Risk"

            df["ChurnRisk"] = df["ChurnScore"].apply(assign_risk_tier)

            # Commit the enriched dataframe to state to unlock remaining tabs
            st.session_state["analysis_df"] = df
            st.success("Churn risk modeling completed successfully!")

    # 3. Render analytical views if calculation state exists
    if st.session_state["analysis_df"] is not None:
        analysis_df = st.session_state["analysis_df"]

        # High-level Risk Summary Metrics
        c1, c2, c3 = st.columns(3)
        total_cust = len(analysis_df)
        high_risk_df = analysis_df[analysis_df["ChurnRisk"] == "High Risk"]
        high_risk_count = len(high_risk_df)
        revenue_at_risk = high_risk_df["Monetary"].sum()

        c1.metric("High Churn Risk Customers", f"{high_risk_count} accounts",
                  f"{(high_risk_count / total_cust) * 100:.1f}% of total")
        c2.metric("Total Revenue Exposure", f"£{revenue_at_risk:,.2f}")
        c3.metric("Average High-Risk Churn Score", f"{high_risk_df['ChurnScore'].mean():.1f} / 100")

        st.divider()

        # Risk Tier Splits vs Segment Cross-Tabulation
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### Risk Distribution Breakdown")
            risk_counts = analysis_df["ChurnRisk"].value_counts().reset_index()
            risk_counts.columns = ["Risk Tier", "Count"]

            fig_pie = px.pie(
                risk_counts,
                values="Count",
                names="Risk Tier",
                hole=0.4,
                color="Risk Tier",
                color_discrete_map={"Low Risk": "#2ecc71", "Medium Risk": "#f1c40f", "High Risk": "#e74c3c"}
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_right:
            st.markdown("### Where is Churn Concentration Highest?")
            cross_tab = analysis_df.groupby(["Segment_Name", "ChurnRisk"]).size().unstack(fill_value=0).reset_index()

            fig_bar = px.bar(
                cross_tab,
                x="Segment_Name",
                y=["Low Risk", "Medium Risk", "High Risk"],
                title="Risk Vectors by Customer Segment",
                barmode="stack",
                color_discrete_map={"Low Risk": "#2ecc71", "Medium Risk": "#f1c40f", "High Risk": "#e74c3c"}
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        # Download updated analytics data
        csv_data = analysis_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Enriched Churn & Segmentation Matrix",
            data=csv_data,
            file_name="customer_churn_intelligence.csv",
            mime="text/csv",
            use_container_width=True
        )
