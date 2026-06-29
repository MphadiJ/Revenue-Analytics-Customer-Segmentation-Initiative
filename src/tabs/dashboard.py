import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from kpi_engine import (
    monthly_revenue_trend,
    revenue_by_country,
    top_products_by_revenue,
    orders_by_day_of_week,
)

RISK_COLORS = {
    "Low Risk":    "#2ecc71",
    "Medium Risk": "#f1c40f",
    "High Risk":   "#e74c3c",
}

# Actual segment names from the trained K-Means model
PREMIUM_SEGMENTS = ["High-Value Customers", "Loyal Customers"]


def dashboard_tab():
    st.subheader("📈 Executive Command Center")

    # ── Guard ─────────────────────────────────────────────────────────────────
    if st.session_state.get("analysis_df") is None:
        st.warning("Complete Customer Segmentation first to unlock this dashboard.")
        return

    df     = st.session_state["analysis_df"]
    raw_df = st.session_state.get("raw_df")          # transaction-level data

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    total_revenue        = df["Monetary"].sum()
    total_customers      = len(df)
    high_risk_df         = df[df["ChurnRisk"] == "High Risk"] if "ChurnRisk" in df.columns else df.iloc[0:0]
    revenue_at_risk      = high_risk_df["Monetary"].sum()
    revenue_at_risk_pct  = (revenue_at_risk / total_revenue * 100) if total_revenue > 0 else 0

    # Premium segment revenue — using ACTUAL model segment names
    premium_df          = df[df["Segment_Name"].isin(PREMIUM_SEGMENTS)] if "Segment_Name" in df.columns else df.iloc[0:0]
    premium_revenue_pct = (premium_df["Monetary"].sum() / total_revenue * 100) if total_revenue > 0 else 0

    # Revenue leaderboard — defined BEFORE columns so it's available everywhere
    rev_leaderboard = (
        df.groupby("Segment_Name")["Monetary"]
        .sum()
        .reset_index()
        .sort_values("Monetary", ascending=True)
    )
    highest_seg   = rev_leaderboard.iloc[-1]["Segment_Name"] if not rev_leaderboard.empty else "N/A"
    highest_val   = rev_leaderboard.iloc[-1]["Monetary"]     if not rev_leaderboard.empty else 0

    # ── KPI cards ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gross Revenue",          f"£{total_revenue:,.0f}")
    c2.metric("Active Customers",       f"{total_customers:,}")
    c3.metric("Revenue at Risk",        f"£{revenue_at_risk:,.0f}",
              f"{revenue_at_risk_pct:.1f}% of base",
              delta_color="inverse")
    c4.metric("Premium Segment Share",  f"{premium_revenue_pct:.1f}%",
              "High-Value + Loyal")

    st.divider()

    # ── Row 1: Revenue trend + Portfolio health ────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 📅 Monthly Revenue Trend")
        if raw_df is not None:
            monthly = monthly_revenue_trend(raw_df)
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=monthly["YearMonth"], y=monthly["Revenue"],
                mode="lines+markers",
                line=dict(color="#58A6FF", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(88,166,255,0.08)",
                hovertemplate="<b>%{x}</b><br>£%{y:,.0f}<extra></extra>",
            ))
            fig_trend.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor ="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#30363D"),
                font=dict(color="#E6EDF3"),
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Raw transaction data not loaded — upload data to see revenue trend.")

    with col_b:
        st.markdown("#### 🛡️ Revenue by Risk Tier")
        if "ChurnRisk" in df.columns:
            risk_revenue = df.groupby("ChurnRisk")["Monetary"].sum().reset_index()
            fig_donut = px.pie(
                risk_revenue,
                values="Monetary", names="ChurnRisk",
                hole=0.55,
                color="ChurnRisk",
                color_discrete_map=RISK_COLORS,
            )
            fig_donut.update_traces(textposition="inside", textinfo="percent+label")
            fig_donut.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("Run Churn Analysis first to see portfolio health.")

    st.divider()

    # ── Row 2: Segment leaderboard + Top products ─────────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("#### 💰 Revenue by Customer Segment")
        fig_seg = px.bar(
            rev_leaderboard,
            y="Segment_Name", x="Monetary",
            orientation="h",
            text="Monetary",
            color="Monetary",
            color_continuous_scale="Blues",
            labels={"Monetary": "Total Value (£)", "Segment_Name": "Segment"},
        )
        fig_seg.update_traces(texttemplate="£%{text:,.0f}", textposition="outside")
        fig_seg.update_layout(
            height=300,
            margin=dict(l=10, r=60, t=10, b=10),
            showlegend=False,
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor ="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            font=dict(color="#E6EDF3"),
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    with col_d:
        st.markdown("#### 🛍️ Top 5 Products by Revenue")
        if raw_df is not None:
            top5 = top_products_by_revenue(raw_df, 5)
            fig_prod = px.bar(
                top5.sort_values("Revenue"),
                y="Description", x="Revenue",
                orientation="h",
                text="Revenue",
                color="Revenue",
                color_continuous_scale="Teal",
                labels={"Revenue": "Revenue (£)", "Description": "Product"},
            )
            fig_prod.update_traces(texttemplate="£%{text:,.0f}", textposition="outside")
            fig_prod.update_layout(
                height=300,
                margin=dict(l=10, r=60, t=10, b=10),
                showlegend=False,
                coloraxis_showscale=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor ="rgba(0,0,0,0)",
                font=dict(color="#E6EDF3"),
            )
            st.plotly_chart(fig_prod, use_container_width=True)
        else:
            st.info("Raw transaction data not loaded — upload data to see product performance.")

    st.divider()

    # ── Row 3: Geography + Day of week ────────────────────────────────────────
    if raw_df is not None:
        col_e, col_f = st.columns(2)

        with col_e:
            st.markdown("#### 🌍 Revenue by Country (Top 5)")
            top_countries = revenue_by_country(raw_df, 5)
            fig_geo = px.bar(
                top_countries.sort_values("Revenue"),
                y="Country", x="Revenue",
                orientation="h",
                text="Revenue",
                color="Revenue",
                color_continuous_scale="Blues",
            )
            fig_geo.update_traces(texttemplate="£%{text:,.0f}", textposition="outside")
            fig_geo.update_layout(
                height=280,
                margin=dict(l=10, r=60, t=10, b=10),
                showlegend=False,
                coloraxis_showscale=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor ="rgba(0,0,0,0)",
                font=dict(color="#E6EDF3"),
            )
            st.plotly_chart(fig_geo, use_container_width=True)

        with col_f:
            st.markdown("#### 📅 Orders by Day of Week")
            dow = orders_by_day_of_week(raw_df)
            fig_dow = px.bar(
                dow, x="DayOfWeek", y="Orders",
                color_discrete_sequence=["#58A6FF"],
                text="Orders",
            )
            fig_dow.update_traces(textposition="outside")
            fig_dow.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor ="rgba(0,0,0,0)",
                font=dict(color="#E6EDF3"),
            )
            st.plotly_chart(fig_dow, use_container_width=True)

        st.divider()

    # ── Dynamic C-Suite action matrix ─────────────────────────────────────────
    st.markdown("#### 🏁 Strategic Action Matrix")

    g1, g2 = st.columns(2)

    with g1:
        st.info(
            f"🏆 **Top Revenue Driver:** The **{highest_seg}** segment "
            f"is your single largest revenue engine at **£{highest_val:,.0f}**. "
            f"Prioritise retention and upsell programmes within this cohort."
        )
        if premium_revenue_pct > 0:
            st.success(
                f"✅ **Premium Concentration:** High-Value and Loyal customers "
                f"account for **{premium_revenue_pct:.1f}%** of total revenue. "
                f"This is a healthy concentration in your best segments."
            )

    with g2:
        if revenue_at_risk_pct > 20:
            st.error(
                f"🚨 **Critical Exposure:** **{revenue_at_risk_pct:.1f}%** of revenue "
                f"(**£{revenue_at_risk:,.0f}**) is held by High Risk customers. "
                f"Immediate retention intervention is required."
            )
        elif revenue_at_risk_pct > 10:
            st.warning(
                f"⚠️ **Elevated Risk:** **{revenue_at_risk_pct:.1f}%** of revenue "
                f"(**£{revenue_at_risk:,.0f}**) is at risk. "
                f"Deploy targeted re-engagement campaigns to Medium and High Risk segments."
            )
        else:
            st.success(
                f"✅ **Portfolio Stable:** Only **{revenue_at_risk_pct:.1f}%** of revenue "
                f"is flagged at risk. Churn metrics are within healthy thresholds. "
                f"Maintain current retention strategy."
            )
        if len(high_risk_df) > 0:
            avg_high_recency = high_risk_df["Recency"].mean()
            st.info(
                f"📋 **At-Risk Profile:** High Risk customers have an average recency "
                f"of **{avg_high_recency:.0f} days** since last purchase. "
                f"Customers inactive beyond 90 days should be prioritised for win-back campaigns."
            )
