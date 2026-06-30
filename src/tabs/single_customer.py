import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from churn.churn_analysis import (
    compute_single_customer_churn_score,
    classify_churn_risk,
    explain_churn_score,
)

# ── Design tokens ─────────────────────────────────────────────────────────────
GREEN  = "#2ecc71"
AMBER  = "#f1c40f"
RED    = "#e74c3c"
BLUE   = "#58A6FF"
DARK   = "rgba(0,0,0,0)"
BORDER = "#30363D"
TXT    = "#E6EDF3"
MUTED  = "#8B949E"
CARD   = "rgba(22,27,34,0.9)"

RISK_COLORS = {"High Risk": RED, "Medium Risk": AMBER, "Low Risk": GREEN}


def _risk_color(risk: str) -> str:
    return RISK_COLORS.get(risk, BLUE)


def _churn_decision(score: float) -> str:
    if score >= 0.65:   return "🔴 Will Churn"
    elif score >= 0.35: return "🟡 At Risk"
    else:               return "🟢 Retained"


def _churn_gauge(score: float, title: str = "Churn Risk Score") -> go.Figure:
    """Gauge chart — axis correctly scaled 0 to 1."""
    color = RED if score >= 0.65 else AMBER if score >= 0.35 else GREEN
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number=dict(valueformat=".4f", font=dict(color=color, size=28)),
        title=dict(text=title, font=dict(color=TXT, size=13)),
        gauge=dict(
            axis=dict(range=[0, 1], tickfont=dict(color=MUTED, size=10)),
            bar=dict(color=color, thickness=0.25),
            bgcolor="rgba(0,0,0,0)",
            bordercolor=BORDER,
            steps=[
                dict(range=[0.00, 0.35], color="rgba(46,204,113,0.15)"),
                dict(range=[0.35, 0.65], color="rgba(241,196,15,0.15)"),
                dict(range=[0.65, 1.00], color="rgba(231,76,60,0.15)"),
            ],
            threshold=dict(
                line=dict(color=color, width=3),
                thickness=0.8,
                value=score,
            ),
        ),
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor=DARK,
        font=dict(color=TXT),
    )
    return fig


def single_customer_tab(pipeline):
    st.subheader("👤 Single Customer Intelligence")
    st.caption(
        "Look up an existing customer by ID — or enter RFM values manually "
        "to predict segment and churn risk for any new customer."
    )

    # ── Guard ─────────────────────────────────────────────────────────────────
    if st.session_state.get("analysis_df") is None:
        st.warning("Data loads automatically on startup. If you see this, please refresh.")
        return

    df = st.session_state["analysis_df"]

    # ── Mode toggle ───────────────────────────────────────────────────────────
    mode = st.radio(
        "Mode",
        ["🔍 Look Up Existing Customer", "✏️ Manual Prediction"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # MODE 1 — LOOK UP EXISTING CUSTOMER
    # ══════════════════════════════════════════════════════════════════════════
    if mode == "🔍 Look Up Existing Customer":

        # Search by CustomerID if available, otherwise by row index
        if "CustomerID" in df.columns:
            customer_options = df["CustomerID"].astype(str).tolist()
            selected_id      = st.selectbox(
                "Select Customer ID",
                options=customer_options,
                help="Search by Customer ID",
            )
            cust_profile = df[df["CustomerID"].astype(str) == selected_id].iloc[0]
        else:
            selected_idx = st.selectbox(
                "Select Customer (Row)",
                options=df.index.tolist(),
            )
            cust_profile = df.loc[selected_idx]

        seg_name    = str(cust_profile.get("Segment_Name", "—"))
        risk_tier   = str(cust_profile.get("ChurnRisk",    "—"))
        churn_score = float(cust_profile.get("ChurnScore",  0.0))
        decision    = str(cust_profile.get("ChurnDecision", _churn_decision(churn_score)))
        action      = str(cust_profile.get("RecommendedAction", "—"))
        risk_color  = _risk_color(risk_tier)

        # ── Profile banner ────────────────────────────────────────────────────
        cust_label = cust_profile.get("CustomerID", selected_idx if "CustomerID" not in df.columns else "—")
        st.markdown(f"""
        <div style="padding:16px 20px;border-radius:10px;background:{CARD};
                    border-left:6px solid {risk_color};margin-bottom:16px;">
            <p style="margin:0;font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:{MUTED};">
                CUSTOMER PROFILE
            </p>
            <p style="margin:4px 0 0;font-size:22px;font-weight:700;color:{TXT};">
                {cust_label}
            </p>
            <p style="margin:4px 0 0;font-size:13px;color:{risk_color};font-weight:600;">
                {decision} &nbsp;·&nbsp; {seg_name}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── KPI cards ─────────────────────────────────────────────────────────
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Segment",           seg_name)
        m2.metric("Churn Decision",    decision)
        m3.metric("Churn Risk Tier",   risk_tier)
        m4.metric("Lifetime Value",    f"£{cust_profile['Monetary']:,.2f}")
        m5.metric("Last Purchase",     f"{int(cust_profile['Recency'])} days ago")

        st.divider()

        # ── Gauge + metrics ───────────────────────────────────────────────────
        col_g, col_m, col_a = st.columns([2, 1, 2])

        with col_g:
            st.plotly_chart(_churn_gauge(churn_score), use_container_width=True)

        with col_m:
            st.markdown("<br>", unsafe_allow_html=True)
            rfm_cols = [c for c in ["Recency","Tenure","Frequency","Monetary","AvgOrderValue"] if c in cust_profile.index]
            for col in rfm_cols:
                val = cust_profile[col]
                fmt = f"£{val:,.2f}" if col in ["Monetary","AvgOrderValue"] else f"{val:,.0f}"
                st.metric(col, fmt)

        with col_a:
            st.markdown("#### 💡 Recommended Action")
            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {risk_color};border-radius:8px;
                        padding:16px;margin-top:8px;">
                <p style="margin:0;font-size:13px;color:{TXT};line-height:1.7;">{action}</p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── Comparison vs segment average ─────────────────────────────────────
        if "Segment_Name" in df.columns and seg_name in df["Segment_Name"].values:
            st.markdown(f"#### 📊 {seg_name} — Customer vs Segment Average")

            seg_avg = df[df["Segment_Name"] == seg_name][
                [c for c in ["Recency","Frequency","Monetary","AvgOrderValue","ChurnScore"] if c in df.columns]
            ].mean()

            compare_cols = [c for c in ["Recency","Frequency","Monetary","AvgOrderValue","ChurnScore"] if c in cust_profile.index]
            compare_df   = pd.DataFrame({
                "Metric":          compare_cols,
                "This Customer":   [cust_profile[c] for c in compare_cols],
                "Segment Average": [seg_avg[c]       for c in compare_cols],
            }).round(2)

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                name="This Customer",
                x=compare_df["Metric"],
                y=compare_df["This Customer"],
                marker_color=BLUE,
                text=compare_df["This Customer"].round(1),
                textposition="outside",
            ))
            fig_comp.add_trace(go.Bar(
                name="Segment Average",
                x=compare_df["Metric"],
                y=compare_df["Segment Average"],
                marker_color=f"rgba(88,166,255,0.3)",
                text=compare_df["Segment Average"].round(1),
                textposition="outside",
            ))
            fig_comp.update_layout(
                barmode="group",
                height=320,
                margin=dict(t=20, b=10, l=10, r=10),
                paper_bgcolor=DARK,
                plot_bgcolor=DARK,
                xaxis=dict(showgrid=False, color=MUTED),
                yaxis=dict(gridcolor=BORDER, color=MUTED),
                font=dict(color=TXT),
                legend=dict(font=dict(color=TXT)),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # MODE 2 — MANUAL PREDICTION
    # ══════════════════════════════════════════════════════════════════════════
    else:
        st.markdown("#### Enter Customer RFM Values")
        st.caption("The pipeline will predict segment and churn risk in real time.")

        col_a, col_b = st.columns(2)
        with col_a:
            recency       = st.number_input("Recency (days since last purchase)",   min_value=0,   value=30)
            tenure        = st.number_input("Tenure (days since first purchase)",   min_value=0,   value=365)
            frequency     = st.number_input("Frequency (number of purchases)",      min_value=1,   value=5)
        with col_b:
            monetary      = st.number_input("Monetary (total spend £)",             min_value=0.0, value=500.0)
            avg_order_val = st.number_input("AvgOrderValue (Monetary / Frequency)", min_value=0.0, value=100.0)

        if st.button("🚀 Predict Segment & Churn Risk", type="primary", use_container_width=True):
            try:
                # Segment prediction
                manual_df = pd.DataFrame([{
                    "Recency": recency, "Tenure": tenure, "Frequency": frequency,
                    "Monetary": monetary, "AvgOrderValue": avg_order_val,
                }])
                features        = manual_df[["Recency","Tenure","Frequency","Monetary","AvgOrderValue"]].copy()
                processed       = pipeline._preprocess(features)
                clusters        = pipeline.model.predict(processed)
                manual_df["Segment"]      = clusters
                seg_map                   = pipeline._name_segments(manual_df)
                manual_df["Segment_Name"] = manual_df["Segment"].map(seg_map)
                seg_name    = manual_df["Segment_Name"].iloc[0]

                # Churn scoring
                churn_score = compute_single_customer_churn_score(recency, frequency, monetary)
                churn_risk  = classify_churn_risk(churn_score)
                decision    = _churn_decision(churn_score)
                explanation = explain_churn_score(recency, frequency, monetary)
                risk_color  = _risk_color(churn_risk)

                # ── Results banner ────────────────────────────────────────────
                st.markdown(f"""
                <div style="padding:16px 20px;border-radius:10px;background:{CARD};
                            border-left:6px solid {risk_color};margin:16px 0;">
                    <p style="margin:0;font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:{MUTED};">
                        PREDICTION RESULT
                    </p>
                    <p style="margin:4px 0 0;font-size:22px;font-weight:700;color:{TXT};">
                        {seg_name} &nbsp;·&nbsp; <span style="color:{risk_color};">{decision}</span>
                    </p>
                    <p style="margin:4px 0 0;font-size:13px;color:{MUTED};">
                        Churn Score: <b style="color:{risk_color};font-family:monospace;">{churn_score:.4f}</b>
                        &nbsp;·&nbsp; Risk Tier: <b style="color:{risk_color};">{churn_risk}</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                col_g, col_b2 = st.columns([1, 1])

                with col_g:
                    st.plotly_chart(_churn_gauge(churn_score, "Churn Risk Score"), use_container_width=True)

                with col_b2:
                    st.markdown("#### Score Breakdown")
                    breakdown = pd.DataFrame([
                        {"Driver":"Recency",   "Raw Value":f"{recency} days",        "Weight":"50%", "Contribution": explanation["recency_component"]},
                        {"Driver":"Frequency", "Raw Value":f"{frequency} purchases",  "Weight":"30%", "Contribution": explanation["frequency_component"]},
                        {"Driver":"Monetary",  "Raw Value":f"£{monetary:,.2f}",       "Weight":"20%", "Contribution": explanation["monetary_component"]},
                    ])
                    st.dataframe(breakdown, use_container_width=True, hide_index=True)

                    # Progress bar per driver
                    for _, row in breakdown.iterrows():
                        pct = min(row["Contribution"] / churn_score, 1.0) if churn_score > 0 else 0
                        st.markdown(f"<small style='color:{MUTED};'>{row['Driver']} ({row['Weight']})</small>", unsafe_allow_html=True)
                        st.progress(pct)

            except Exception as e:
                st.error(f"Prediction error: {e}")
                import traceback
                st.code(traceback.format_exc())
