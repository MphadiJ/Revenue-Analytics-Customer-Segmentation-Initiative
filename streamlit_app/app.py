
"""
streamlit_app/app.py

Retail Customer Segmentation + Churn Analysis App
Tabs:
    1. Segmentation      — existing K-Means pipeline (unchanged)
    2. Churn Analysis    — NEW: RFM-based churn risk scoring per customer & segment
    3. Single Customer   — existing manual input + churn risk added
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# ── Path fix ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH     = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_PATH)

from inference.inference import InferencePipeline
from churn.churn_analysis import (
    add_churn_analysis,
    segment_churn_summary,
    classify_churn_risk,
    compute_single_customer_churn_score,
    explain_churn_score,
)

# ── Load models once ──────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return InferencePipeline(
        preprocessor_path=os.path.join(PROJECT_ROOT, "models", "preprocessor.pkl"),
        model_path=os.path.join(PROJECT_ROOT, "models", "kmeans_best.pkl"),
    )

pipeline = load_pipeline()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Intelligence App",
    page_icon="🛒",
    layout="wide",
)

st.title("Retail Customer Intelligence App 🛒")
st.caption("Segmentation · Churn Risk · Single Customer Prediction")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Segmentation", "⚠️ Churn Analysis", "👤 Single Customer"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — SEGMENTATION  (original logic preserved)
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Customer Segmentation")
    st.write("Upload your RFM data to predict customer segments.")

    uploaded_file = st.file_uploader(
        "Upload CSV — must have: Recency, Tenure, Frequency, Monetary, AvgOrderValue",
        type=["csv"],
        key="seg_upload",
    )

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("**Uploaded data preview:**")
            st.dataframe(df.head())

            df_features  = df[["Recency", "Tenure", "Frequency", "Monetary", "AvgOrderValue"]].copy()
            processed_df = pipeline._preprocess(df_features)
            clusters     = pipeline.model.predict(processed_df)

            result = df.copy()
            result["Segment"]      = clusters
            segment_map            = pipeline._name_segments(result)
            result["Segment_Name"] = result["Segment"].map(segment_map)

            st.success("Segments predicted successfully.")
            st.dataframe(result)

            # Pass to churn tab via session state
            st.session_state["segmented_df"] = result

            st.info("✅ Data is ready — open the **Churn Analysis** tab to see risk scores.")

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHURN ANALYSIS  (new)
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Churn Risk Analysis ⚠️")

    RISK_COLORS = {
        "High Risk":   "#e74c3c",
        "Medium Risk": "#f39c12",
        "Low Risk":    "#2ecc71",
    }
    RISK_EMOJI = {
        "High Risk":   "🔴",
        "Medium Risk": "🟡",
        "Low Risk":    "🟢",
    }

    # ── Data source ───────────────────────────────────────────────────────────
    has_session_data = "segmented_df" in st.session_state

    if has_session_data:
        st.info(
            "Segmented data detected from Tab 1. "
            "Tick the box below to use it, or upload a separate file."
        )

    use_session = False
    if has_session_data:
        use_session = st.checkbox("Use segmented data from Segmentation tab", value=True)

    churn_file = st.file_uploader(
        "Or upload a CSV (must have: Recency, Frequency, Monetary, Segment_Name)",
        type=["csv"],
        key="churn_upload",
    )

    # Resolve the DataFrame to analyse
    churn_df = None
    if use_session and has_session_data:
        churn_df = st.session_state["segmented_df"].copy()
    elif churn_file:
        churn_df = pd.read_csv(churn_file)

    # ── Analysis ──────────────────────────────────────────────────────────────
    if churn_df is not None:
        try:
            churn_df = add_churn_analysis(churn_df)
            total     = len(churn_df)
            n_high    = (churn_df["ChurnRisk"] == "High Risk").sum()
            n_medium  = (churn_df["ChurnRisk"] == "Medium Risk").sum()
            n_low     = (churn_df["ChurnRisk"] == "Low Risk").sum()

            # ── KPI cards ─────────────────────────────────────────────────────
            st.markdown("### Overview")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Customers",  total)
            c2.metric("🔴 High Risk",     f"{n_high}  ({n_high  / total * 100:.1f}%)")
            c3.metric("🟡 Medium Risk",   f"{n_medium} ({n_medium / total * 100:.1f}%)")
            c4.metric("🟢 Low Risk",      f"{n_low}   ({n_low   / total * 100:.1f}%)")

            st.divider()

            # ── Charts row 1 ──────────────────────────────────────────────────
            col_a, col_b = st.columns(2)

            with col_a:
                risk_counts = (
                    churn_df["ChurnRisk"]
                    .value_counts()
                    .reindex(["High Risk", "Medium Risk", "Low Risk"])
                    .reset_index()
                )
                risk_counts.columns = ["ChurnRisk", "Count"]
                fig1 = px.bar(
                    risk_counts,
                    x="ChurnRisk", y="Count",
                    color="ChurnRisk",
                    color_discrete_map=RISK_COLORS,
                    title="Customer Count by Churn Risk Level",
                    text="Count",
                )
                fig1.update_traces(textposition="outside")
                fig1.update_layout(showlegend=False)
                st.plotly_chart(fig1, use_container_width=True)

            with col_b:
                fig2 = px.histogram(
                    churn_df, x="ChurnScore",
                    nbins=30,
                    color_discrete_sequence=["#3498db"],
                    title="Churn Score Distribution",
                    labels={"ChurnScore": "Churn Score (0 = safe, 1 = at risk)"},
                )
                st.plotly_chart(fig2, use_container_width=True)

            # ── Segment breakdown ─────────────────────────────────────────────
            if "Segment_Name" in churn_df.columns:
                st.markdown("### Churn Risk by Segment")
                summary = segment_churn_summary(churn_df)
                st.dataframe(summary, use_container_width=True)

                col_c, col_d = st.columns(2)

                with col_c:
                    fig3 = px.bar(
                        summary,
                        x="Segment_Name", y="AvgChurnScore",
                        color="AvgChurnScore",
                        color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
                        title="Average Churn Score per Segment",
                        labels={"AvgChurnScore": "Avg Churn Score"},
                        text="AvgChurnScore",
                    )
                    fig3.update_traces(textposition="outside")
                    st.plotly_chart(fig3, use_container_width=True)

                with col_d:
                    fig4 = px.bar(
                        summary,
                        x="Segment_Name",
                        y=["HighRiskCount", "MediumRiskCount", "LowRiskCount"],
                        title="Risk Breakdown per Segment",
                        labels={"value": "Customer Count", "variable": "Risk Level"},
                        color_discrete_map={
                            "HighRiskCount":   "#e74c3c",
                            "MediumRiskCount": "#f39c12",
                            "LowRiskCount":    "#2ecc71",
                        },
                        barmode="stack",
                    )
                    st.plotly_chart(fig4, use_container_width=True)

            # ── Scatter: Recency vs Monetary coloured by risk ─────────────────
            st.markdown("### Churn Risk Landscape")
            hover_cols = [c for c in ["Frequency", "ChurnScore", "Segment_Name"] if c in churn_df.columns]
            fig5 = px.scatter(
                churn_df,
                x="Recency", y="Monetary",
                color="ChurnRisk",
                color_discrete_map=RISK_COLORS,
                hover_data=hover_cols,
                title="Recency vs Monetary — coloured by Churn Risk",
                opacity=0.65,
            )
            st.plotly_chart(fig5, use_container_width=True)

            # ── Full table + download ─────────────────────────────────────────
            st.markdown("### Full Customer Churn Table")
            display_cols = [
                c for c in
                ["CustomerID", "Recency", "Frequency", "Monetary",
                 "Segment_Name", "ChurnScore", "ChurnRisk"]
                if c in churn_df.columns
            ]
            st.dataframe(churn_df[display_cols], use_container_width=True)

            csv_bytes = churn_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Churn Results CSV",
                data=csv_bytes,
                file_name="churn_results.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Error running churn analysis: {e}")
            import traceback
            st.code(traceback.format_exc())

    else:
        st.warning(
            "No data loaded yet. "
            "Either run segmentation in Tab 1 first, or upload a CSV above."
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — SINGLE CUSTOMER  (original logic + churn risk)
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Predict Segment & Churn Risk for a Single Customer")

    recency       = st.number_input("Recency (days since last purchase)",  min_value=0,   value=0)
    tenure        = st.number_input("Tenure (days since first purchase)",  min_value=0,   value=0)
    frequency     = st.number_input("Frequency (number of purchases)",     min_value=0,   value=0)
    monetary      = st.number_input("Monetary (total spend £)",            min_value=0.0, value=0.0)
    avg_order_val = st.number_input("AvgOrderValue (Monetary / Frequency)",min_value=0.0, value=0.0)

    if st.button("Predict Segment & Churn Risk"):
        manual_df = pd.DataFrame([{
            "Recency":       recency,
            "Tenure":        tenure,
            "Frequency":     frequency,
            "Monetary":      monetary,
            "AvgOrderValue": avg_order_val,
        }])

        try:
            # ── Segment prediction ────────────────────────────────────────────
            df_features  = manual_df[["Recency", "Tenure", "Frequency", "Monetary", "AvgOrderValue"]].copy()
            processed_df = pipeline._preprocess(df_features)
            clusters     = pipeline.model.predict(processed_df)
            manual_df["Segment"]      = clusters
            segment_map               = pipeline._name_segments(manual_df)
            manual_df["Segment_Name"] = manual_df["Segment"].map(segment_map)

            # ── Churn prediction ──────────────────────────────────────────────
            churn_score = compute_single_customer_churn_score(recency, frequency, monetary)
            churn_risk  = classify_churn_risk(churn_score)
            explanation = explain_churn_score(recency, frequency, monetary)

            manual_df["ChurnScore"] = churn_score
            manual_df["ChurnRisk"]  = churn_risk

            # ── Results display ───────────────────────────────────────────────
            seg_name   = manual_df["Segment_Name"].iloc[0]
            risk_emoji = {"High Risk": "🔴", "Medium Risk": "🟡", "Low Risk": "🟢"}

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.success(f"**Segment:** {seg_name}")
            with res_col2:
                st.info(f"**Churn Risk:** {risk_emoji.get(churn_risk, '')} {churn_risk}  (score: {churn_score:.4f})")

            # ── Score breakdown ───────────────────────────────────────────────
            st.markdown("#### Churn Score Breakdown")
            breakdown = pd.DataFrame([
                {
                    "Driver":       "Recency",
                    "Raw Value":    f"{recency} days",
                    "Weight":       "50%",
                    "Contribution": explanation["recency_component"],
                },
                {
                    "Driver":       "Frequency",
                    "Raw Value":    f"{frequency} purchases",
                    "Weight":       "30%",
                    "Contribution": explanation["frequency_component"],
                },
                {
                    "Driver":       "Monetary",
                    "Raw Value":    f"£{monetary:,.2f}",
                    "Weight":       "20%",
                    "Contribution": explanation["monetary_component"],
                },
            ])
            st.dataframe(breakdown, use_container_width=True, hide_index=True)

            # ── Gauge bar 
            fig_gauge = px.bar(
                x=[churn_score], y=["Churn Score"],
                orientation="h",
                range_x=[0, 1],
                color_discrete_sequence=[
                    "#e74c3c" if churn_score >= 0.65
                    else "#f39c12" if churn_score >= 0.35
                    else "#2ecc71"
                ],
                title=f"Churn Score: {churn_score:.4f}  |  {churn_risk}",
            )
            fig_gauge.update_layout(showlegend=False, height=200)
            st.plotly_chart(fig_gauge, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())
