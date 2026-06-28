import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def single_customer_tab(pipeline):
    st.subheader("👤 Single Customer Lookup Tool")

    if st.session_state["analysis_df"] is None:
        st.warning("⚠️ High-resolution customer profiles require completing Churn Risk Analysis first.")
        return

    df = st.session_state["analysis_df"]

    # Provide search input by index/ID
    st.markdown("### Locate Individual Customer Profile")
    customer_id = st.selectbox(
        "Select Customer Reference Record Number:",
        options=df.index.tolist(),
        help="Select a database row item to parse their current behavioral health profile."
    )

    if customer_id is not None:
        cust_profile = df.loc[customer_id]

        # Profile Summary Cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Assigned Cohort Segment", str(cust_profile["Segment_Name"]))

        risk_tier = cust_profile["ChurnRisk"]
        m2.metric("Churn Risk Category", str(risk_tier))
        m3.metric("Lifetime Monetary Value", f"£{cust_profile['Monetary']:,.2f}")
        m4.metric("Days Since Last Purchase", f"{int(cust_profile['Recency'])} days")

        st.divider()

        # Gauge Chart for Churn Score
        col_g, col_d = st.columns([1, 1])

        with col_g:
            st.markdown("### Behavioral Churn Index Value")
            score_val = float(cust_profile["ChurnScore"])

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score_val,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#2c3e50"},
                    'steps': [
                        {'range': [0, 40], 'color': "#2ecc71"},
                        {'range': [40, 70], 'color': "#f1c40f"},
                        {'range': [70, 100], 'color': "#e74c3c"}
                    ]
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_d:
            st.markdown("### Underlying Metrics Grid")
            metrics_display = pd.DataFrame(cust_profile[
                                               ["Recency", "Tenure", "Frequency", "Monetary", "AvgOrderValue"]
                                           ]).rename(columns={customer_id: "Value"})
            st.table(metrics_display)
