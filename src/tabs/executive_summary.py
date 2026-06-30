import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from kpi_engine import monthly_revenue_trend

# ── Design tokens ─────────────────────────────────────────────────────────────
GREEN  = "#2ecc71"
AMBER  = "#f1c40f"
RED    = "#e74c3c"
BLUE   = "#58A6FF"
PURPLE = "#BC8CFF"
DARK   = "rgba(0,0,0,0)"
CARD   = "rgba(22,27,34,0.9)"
BORDER = "#30363D"
TXT    = "#E6EDF3"
MUTED  = "#8B949E"

RISK_COLORS = {"Low Risk": GREEN, "Medium Risk": AMBER, "High Risk": RED}

# Actual segment names from the trained K-Means model
PREMIUM_SEGMENTS  = ["High-Value Customers", "Loyal Customers"]
AT_RISK_SEGMENTS  = ["At-Risk Customers", "Occasional Buyers"]


def _health_color(score: float) -> str:
    if score >= 7.5: return GREEN
    if score >= 5.0: return AMBER
    return RED


def _health_label(score: float) -> str:
    if score >= 7.5: return "STRONG"
    if score >= 5.0: return "STABLE — MONITOR"
    return "CRITICAL — ACTION REQUIRED"


def executive_summary_tab():
    st.subheader("🧠 Executive Intelligence Summary")
    st.caption(
        "Automated business health scoring, revenue exposure analysis, "
        "win-back simulation, and strategic action playbook."
    )

    # ── Guard ─────────────────────────────────────────────────────────────────
    if st.session_state.get("analysis_df") is None:
        st.warning("Data loads automatically on startup. If you see this, please refresh the page.")
        return

    df     = st.session_state["analysis_df"]
    raw_df = st.session_state.get("raw_df")

    # ── Core metric calculations ───────────────────────────────────────────────
    total_customers     = len(df)
    total_revenue       = df["Monetary"].sum()

    high_risk_df        = df[df["ChurnRisk"] == "High Risk"]  if "ChurnRisk"    in df.columns else df.iloc[0:0]
    medium_risk_df      = df[df["ChurnRisk"] == "Medium Risk"] if "ChurnRisk"   in df.columns else df.iloc[0:0]
    low_risk_df         = df[df["ChurnRisk"] == "Low Risk"]    if "ChurnRisk"   in df.columns else df.iloc[0:0]

    high_risk_count     = len(high_risk_df)
    medium_risk_count   = len(medium_risk_df)
    revenue_at_risk     = high_risk_df["Monetary"].sum()     if "Monetary" in high_risk_df.columns    else 0
    revenue_at_risk_pct = (revenue_at_risk / total_revenue * 100) if total_revenue > 0 else 0

    avg_churn_score     = df["ChurnScore"].mean() if "ChurnScore" in df.columns else 0

    premium_df          = df[df["Segment_Name"].isin(PREMIUM_SEGMENTS)] if "Segment_Name" in df.columns else df.iloc[0:0]
    premium_revenue     = premium_df["Monetary"].sum() if "Monetary" in premium_df.columns else 0
    premium_revenue_pct = (premium_revenue / total_revenue * 100) if total_revenue > 0 else 0

    # Business health score (0–10)
    base_score    = 10.0
    deductions    = (high_risk_count / total_customers * 5) + (revenue_at_risk_pct / 100 * 5)
    health_score  = round(max(1.0, min(10.0, base_score - deductions)), 1)
    h_color       = _health_color(health_score)
    h_label       = _health_label(health_score)

    # ── Health score banner ────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="
        padding: 20px 24px;
        border-radius: 10px;
        background: {CARD};
        border-left: 6px solid {h_color};
        margin-bottom: 24px;
    ">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
            <div>
                <p style="margin:0;font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:{MUTED};">
                    BUSINESS HEALTH INDEX
                </p>
                <p style="margin:4px 0 0;font-size:36px;font-weight:700;color:{h_color};font-family:monospace;">
                    {health_score} <span style="font-size:18px;color:{MUTED};">/ 10</span>
                </p>
                <p style="margin:4px 0 0;font-size:13px;font-weight:600;color:{h_color};">{h_label}</p>
            </div>
            <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:16px 32px;">
                <div>
                    <p style="margin:0;font-size:11px;color:{MUTED};text-transform:uppercase;letter-spacing:.08em;">Gross Revenue</p>
                    <p style="margin:2px 0 0;font-size:18px;font-weight:600;color:{TXT};font-family:monospace;">£{total_revenue:,.0f}</p>
                </div>
                <div>
                    <p style="margin:0;font-size:11px;color:{MUTED};text-transform:uppercase;letter-spacing:.08em;">Active Customers</p>
                    <p style="margin:2px 0 0;font-size:18px;font-weight:600;color:{TXT};font-family:monospace;">{total_customers:,}</p>
                </div>
                <div>
                    <p style="margin:0;font-size:11px;color:{MUTED};text-transform:uppercase;letter-spacing:.08em;">Revenue at Risk</p>
                    <p style="margin:2px 0 0;font-size:18px;font-weight:600;color:{RED};font-family:monospace;">£{revenue_at_risk:,.0f}</p>
                </div>
                <div>
                    <p style="margin:0;font-size:11px;color:{MUTED};text-transform:uppercase;letter-spacing:.08em;">Avg Churn Score</p>
                    <p style="margin:2px 0 0;font-size:18px;font-weight:600;color:{AMBER};font-family:monospace;">{avg_churn_score:.3f}</p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI cards row ──────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("🔴 High Risk",        f"{high_risk_count:,}",     f"{high_risk_count/total_customers*100:.1f}% of base")
    k2.metric("🟡 Medium Risk",      f"{medium_risk_count:,}",   f"{medium_risk_count/total_customers*100:.1f}% of base")
    k3.metric("🟢 Premium Segments", f"{len(premium_df):,}",     f"{premium_revenue_pct:.1f}% of revenue")
    k4.metric("💰 Revenue Exposed",  f"£{revenue_at_risk:,.0f}", f"{revenue_at_risk_pct:.1f}% at risk", delta_color="inverse")
    k5.metric("📊 Health Score",     f"{health_score} / 10",     f"{h_label}")

    st.divider()

    # ── Row 1: Revenue trend + Risk revenue split ──────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 📅 Revenue Trend")
        if raw_df is not None and not raw_df.empty:
            monthly = monthly_revenue_trend(raw_df)
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=monthly["YearMonth"], y=monthly["Revenue"],
                mode="lines+markers",
                line=dict(color=BLUE, width=2.5),
                fill="tozeroy",
                fillcolor="rgba(88,166,255,0.08)",
                hovertemplate="<b>%{x}</b><br>£%{y:,.0f}<extra></extra>",
            ))
            fig_trend.update_layout(
                height=280, margin=dict(t=10,b=10,l=10,r=10),
                paper_bgcolor=DARK, plot_bgcolor=DARK,
                xaxis=dict(showgrid=False, color=MUTED),
                yaxis=dict(gridcolor=BORDER, color=MUTED),
                font=dict(color=TXT),
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Raw transaction data unavailable.")

    with col_b:
        st.markdown("#### 🛡️ Revenue Split by Risk Tier")
        if "ChurnRisk" in df.columns:
            risk_rev = df.groupby("ChurnRisk")["Monetary"].sum().reset_index()
            fig_donut = px.pie(
                risk_rev, values="Monetary", names="ChurnRisk",
                hole=0.55,
                color="ChurnRisk",
                color_discrete_map=RISK_COLORS,
            )
            fig_donut.update_traces(textinfo="percent+label", textfont_size=12)
            fig_donut.update_layout(
                height=280, margin=dict(t=10,b=10,l=10,r=10),
                showlegend=False, paper_bgcolor=DARK,
            )
            st.plotly_chart(fig_donut, use_container_width=True)

    st.divider()

    # ── Row 2: Segment revenue + Churn score per segment ──────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("#### 💰 Revenue by Segment")
        if "Segment_Name" in df.columns:
            seg_rev = (
                df.groupby("Segment_Name")["Monetary"]
                .sum().reset_index()
                .sort_values("Monetary", ascending=True)
            )
            fig_seg = px.bar(
                seg_rev, y="Segment_Name", x="Monetary",
                orientation="h", text="Monetary",
                color="Monetary",
                color_continuous_scale=[[0,"rgba(88,166,255,0.3)"],[1,BLUE]],
                labels={"Monetary":"Revenue (£)","Segment_Name":""},
            )
            fig_seg.update_traces(texttemplate="£%{text:,.0f}", textposition="outside")
            fig_seg.update_layout(
                height=280, margin=dict(t=10,b=10,l=10,r=80),
                showlegend=False, coloraxis_showscale=False,
                paper_bgcolor=DARK, plot_bgcolor=DARK,
                xaxis=dict(showgrid=False, color=MUTED),
                yaxis=dict(showgrid=False, color=TXT),
                font=dict(color=TXT),
            )
            st.plotly_chart(fig_seg, use_container_width=True)

    with col_d:
        st.markdown("#### ⚠️ Avg Churn Score per Segment")
        if "Segment_Name" in df.columns and "ChurnScore" in df.columns:
            seg_churn = (
                df.groupby("Segment_Name")["ChurnScore"]
                .mean().reset_index()
                .sort_values("ChurnScore", ascending=False)
            )
            fig_churn = px.bar(
                seg_churn, x="Segment_Name", y="ChurnScore",
                color="ChurnScore",
                color_continuous_scale=[[0,GREEN],[0.5,AMBER],[1,RED]],
                text="ChurnScore",
            )
            fig_churn.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_churn.update_layout(
                height=280, margin=dict(t=10,b=10,l=10,r=10),
                showlegend=False, coloraxis_showscale=False,
                yaxis_range=[0, 1],
                paper_bgcolor=DARK, plot_bgcolor=DARK,
                xaxis=dict(showgrid=False, color=MUTED),
                yaxis=dict(gridcolor=BORDER, color=MUTED),
                font=dict(color=TXT),
            )
            st.plotly_chart(fig_churn, use_container_width=True)

    st.divider()

    # ── Win-back simulator ────────────────────────────────────────────────────
    st.markdown("#### 🎛️ Revenue Recovery Win-Back Simulator")
    st.caption(
        "Adjust the win-back conversion rate to simulate how much revenue "
        "your team can recover from High Risk customers."
    )

    sim_col, viz_col = st.columns([2, 3])

    with sim_col:
        target_conversion = st.slider(
            "Win-Back Conversion Rate (%)",
            min_value=0, max_value=100, value=25, step=5,
            help="% of High Risk customers retained through targeted campaigns",
        )

        sim_decimal              = target_conversion / 100
        projected_saved          = int(high_risk_count * sim_decimal)
        projected_reclaimed      = revenue_at_risk * sim_decimal
        net_lost                 = revenue_at_risk - projected_reclaimed
        remaining_high_risk      = high_risk_count - projected_saved
        sim_deductions           = (remaining_high_risk / total_customers * 5) + (net_lost / total_revenue * 5)
        sim_health_score         = round(max(1.0, min(10.0, base_score - sim_deductions)), 1)
        sim_color                = _health_color(sim_health_score)

        st.metric("Revenue Reclaimed",    f"£{projected_reclaimed:,.0f}", f"+{target_conversion}% recovery rate")
        st.metric("Customers Rescued",    f"{projected_saved:,}",         f"{remaining_high_risk:,} still at risk", delta_color="inverse")
        st.metric("Simulated Health",     f"{sim_health_score} / 10",      f"{sim_health_score - health_score:+.1f} vs current")

    with viz_col:
        st.markdown("#### Portfolio Revenue Realignment")
        fig_waterfall = go.Figure()
        fig_waterfall.add_trace(go.Bar(
            name="Stable Revenue",
            y=["Revenue Split"],
            x=[total_revenue - revenue_at_risk],
            orientation="h",
            marker=dict(color=BLUE),
        ))
        fig_waterfall.add_trace(go.Bar(
            name="Reclaimed Revenue",
            y=["Revenue Split"],
            x=[projected_reclaimed],
            orientation="h",
            marker=dict(color=GREEN),
        ))
        fig_waterfall.add_trace(go.Bar(
            name="Net Revenue Lost",
            y=["Revenue Split"],
            x=[net_lost],
            orientation="h",
            marker=dict(color=RED),
        ))
        fig_waterfall.update_layout(
            barmode="stack",
            height=220,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(color=TXT, size=11)),
            paper_bgcolor=DARK,
            plot_bgcolor=DARK,
            xaxis=dict(showgrid=False, color=MUTED),
            yaxis=dict(showgrid=False, color=MUTED),
            font=dict(color=TXT),
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

    st.divider()

    # ── Strategic action playbook ──────────────────────────────────────────────
    st.markdown("#### 📋 Strategic Action Playbook")

    segments = df["Segment_Name"].unique().tolist() if "Segment_Name" in df.columns else []

    play_col1, play_col2 = st.columns(2)

    with play_col1:
        if revenue_at_risk_pct > 20:
            st.error(
                f"🚨 **Critical Exposure:** {revenue_at_risk_pct:.1f}% of revenue "
                f"(**£{revenue_at_risk:,.0f}**) is in High Risk accounts. "
                f"At a {target_conversion}% win-back rate your team recovers "
                f"**£{projected_reclaimed:,.0f}**. Immediate intervention is required."
            )
        elif revenue_at_risk_pct > 10:
            st.warning(
                f"⚠️ **Elevated Risk:** {revenue_at_risk_pct:.1f}% of revenue "
                f"(**£{revenue_at_risk:,.0f}**) is flagged at risk. "
                f"Deploy targeted re-engagement campaigns — "
                f"at {target_conversion}% conversion you recover **£{projected_reclaimed:,.0f}**."
            )
        else:
            st.success(
                f"✅ **Portfolio Stable:** Only {revenue_at_risk_pct:.1f}% of revenue is at risk. "
                f"Churn metrics are within healthy thresholds. "
                f"Maintain current retention strategy."
            )

        if "At-Risk Customers" in segments:
            campaign_budget = projected_reclaimed * 0.10
            st.info(
                f"🔄 **Re-Engagement Budget:** Allocate up to **£{campaign_budget:,.0f}** "
                f"(10% of recoverable revenue) to targeted win-back campaigns "
                f"against At-Risk Customers — this yields a 10× ROI framework."
            )

    with play_col2:
        if "High-Value Customers" in segments:
            hv_revenue = df[df["Segment_Name"]=="High-Value Customers"]["Monetary"].sum()
            hv_pct     = hv_revenue / total_revenue * 100
            st.success(
                f"⭐ **Protect Your Engine:** High-Value Customers hold "
                f"**£{hv_revenue:,.0f}** ({hv_pct:.1f}% of total revenue). "
                f"Prioritise retention, upsell, and loyalty structures in this cohort above all others."
            )

        if "Loyal Customers" in segments:
            loyal_df  = df[df["Segment_Name"]=="Loyal Customers"]
            loyal_avg = loyal_df["Monetary"].mean() if not loyal_df.empty else 0
            st.info(
                f"🔵 **Loyalty Upsell Opportunity:** Loyal Customers average "
                f"**£{loyal_avg:,.0f}** spend. "
                f"Introduce tiered reward schemes to migrate this cohort into High-Value status."
            )

        if avg_churn_score > 0.5:
            st.error(
                f"📉 **Portfolio-Wide Churn Alert:** Average churn score of "
                f"**{avg_churn_score:.3f}** is above 0.5 — over half your customer base "
                f"shows moderate to high disengagement signals. "
                f"A systemic retention programme is recommended."
            )

    st.divider()

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("#### 📄 Export Executive Report")

    summary_text = f"""EXECUTIVE INTELLIGENCE SUMMARY
════════════════════════════════════════════════════

BUSINESS HEALTH INDEX
  Score            : {health_score} / 10
  Status           : {h_label}

PORTFOLIO OVERVIEW
  Gross Revenue    : £{total_revenue:,.2f}
  Active Customers : {total_customers:,}
  Premium Revenue  : £{premium_revenue:,.2f} ({premium_revenue_pct:.1f}% of total)

CHURN RISK BREAKDOWN
  High Risk        : {high_risk_count:,} customers ({high_risk_count/total_customers*100:.1f}%)
  Medium Risk      : {medium_risk_count:,} customers ({medium_risk_count/total_customers*100:.1f}%)
  Revenue at Risk  : £{revenue_at_risk:,.2f} ({revenue_at_risk_pct:.1f}% of portfolio)
  Avg Churn Score  : {avg_churn_score:.4f}

WIN-BACK SIMULATION
  Target Rate      : {target_conversion}%
  Customers Saved  : {projected_saved:,}
  Revenue Reclaimed: £{projected_reclaimed:,.2f}
  Net Revenue Lost : £{net_lost:,.2f}
  Post-Campaign Health Score: {sim_health_score} / 10

════════════════════════════════════════════════════
"""
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        st.download_button(
            "📄 Download Executive Report (.txt)",
            data=summary_text,
            file_name="executive_intelligence_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col_dl2:
        if "ChurnRisk" in df.columns:
            high_risk_export = high_risk_df[[
                c for c in
                ["CustomerID","Segment_Name","ChurnScore","ChurnRisk",
                 "RecommendedAction","Recency","Frequency","Monetary"]
                if c in high_risk_df.columns
            ]].sort_values("ChurnScore", ascending=False)

            st.download_button(
                "⬇️ Download High Risk Customer List (.csv)",
                data=high_risk_export.to_csv(index=False).encode("utf-8"),
                file_name="high_risk_customers.csv",
                mime="text/csv",
                use_container_width=True,
            )
