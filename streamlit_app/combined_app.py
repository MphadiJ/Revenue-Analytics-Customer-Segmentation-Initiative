import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from datetime import timedelta
 
# ── Path fix ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH     = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_PATH)
 
from inference.inference import InferencePipeline
from churn.churn_analysis import (
    add_churn_analysis, segment_churn_summary,
    classify_churn_risk, compute_single_customer_churn_score, explain_churn_score,
)
from kpi_engine import (
    load_raw_data, apply_filters, compute_rfm, compute_top_kpis,
    monthly_revenue_trend, revenue_by_country, revenue_heatmap,
    orders_by_day_of_week, top_products_by_revenue, top_products_by_volume,
    top_customers_by_revenue, customer_order_frequency_distribution,
    revenue_by_new_vs_returning,
)
 
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ── Design tokens ─────────────────────────────────────────────────────────────
NAVY     = "#0D1117"
CARD_BG  = "#161B22"
BORDER   = "#30363D"
BLUE     = "#58A6FF"
GREEN    = "#3FB950"
AMBER    = "#E3B341"
RED      = "#F85149"
PURPLE   = "#BC8CFF"
TXT_PRI  = "#E6EDF3"
TXT_SEC  = "#8B949E"
 
LAYOUT   = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font         =dict(color=TXT_PRI, family="monospace", size=12),
    xaxis        =dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TXT_SEC, size=11)),
    yaxis        =dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TXT_SEC, size=11)),
    legend       =dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TXT_SEC)),
    margin       =dict(l=40, r=20, t=40, b=40),
    colorway     =[BLUE, GREEN, AMBER, RED, PURPLE],
)
 
RISK_COLORS = {"High Risk": RED, "Medium Risk": AMBER, "Low Risk": GREEN}
 
st.markdown(f"""
<style>
  html, body, [class*="css"] {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background-color: {NAVY}; color: {TXT_PRI};
  }}
  .stApp {{ background-color: {NAVY}; }}
  [data-testid="stSidebar"] {{ background-color: {CARD_BG}; border-right: 1px solid {BORDER}; }}
  [data-testid="stSidebar"] * {{ color: {TXT_PRI} !important; }}
  [data-testid="stMetric"] {{
    background: {CARD_BG}; border: 1px solid {BORDER};
    border-radius: 10px; padding: 1rem 1.25rem;
  }}
  [data-testid="stMetricLabel"] {{ color:{TXT_SEC}!important; font-size:11px!important; letter-spacing:.08em; text-transform:uppercase; }}
  [data-testid="stMetricValue"] {{ color:{TXT_PRI}!important; font-family:monospace!important; font-size:24px!important; }}
  .stTabs [data-baseweb="tab-list"] {{ background:{CARD_BG}; border-bottom:1px solid {BORDER}; border-radius:8px 8px 0 0; gap:0; }}
  .stTabs [data-baseweb="tab"] {{ color:{TXT_SEC}!important; font-size:13px; padding:.6rem 1rem; }}
  .stTabs [aria-selected="true"] {{ color:{BLUE}!important; border-bottom:2px solid {BLUE}!important; background:transparent!important; }}
  [data-testid="stDataFrame"] {{ border:1px solid {BORDER}; border-radius:8px; }}
  h1,h2,h3 {{ color:{TXT_PRI}; }}
  .slabel {{ font-size:11px; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:{TXT_SEC}; margin-bottom:.5rem; margin-top:1.25rem; display:block; }}
  .sdivider {{ border:none; border-top:1px solid {BORDER}; margin:1.5rem 0; }}
  .seg-card {{
    background:{CARD_BG}; border:1px solid {BORDER}; border-radius:10px;
    padding:1rem 1.25rem; margin-bottom:.75rem;
  }}
  .seg-card h4 {{ margin:0 0 .3rem; color:{TXT_PRI}; font-size:14px; }}
  .seg-card p  {{ margin:0; color:{TXT_SEC}; font-size:12px; line-height:1.6; }}
</style>
""", unsafe_allow_html=True)
 
 
# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return InferencePipeline(
        preprocessor_path=os.path.join(PROJECT_ROOT, "models", "preprocessor.pkl"),
        model_path        =os.path.join(PROJECT_ROOT, "models", "kmeans_best.pkl"),
    )
 
# ── Load & process data ───────────────────────────────────────────────────────
@st.cache_data
def load_and_segment():
    """
    Load raw transactions, compute RFM, run segmentation + churn.
    Returns: (df_raw, df_rfm_segmented)
    """
    data_path = os.path.join(PROJECT_ROOT, "raw data", "rt_data.csv")
    df_raw    = load_raw_data(data_path)
 
    # Compute RFM from raw transactions
    rfm = compute_rfm(df_raw)
 
    # Run segmentation pipeline
    pipeline = load_pipeline()
    features  = rfm[["Recency","Tenure","Frequency","Monetary","AvgOrderValue"]].copy()
    processed = pipeline._preprocess(features)
    clusters  = pipeline.model.predict(processed)
 
    rfm["Segment"]      = clusters
    seg_map             = pipeline._name_segments(rfm)
    rfm["Segment_Name"] = rfm["Segment"].map(seg_map)
 
    # Embed churn analysis
    rfm = add_churn_analysis(rfm)
 
    return df_raw, rfm
 
 
with st.spinner("Loading data and running segmentation pipeline..."):
    try:
        df_raw, df_seg = load_and_segment()
        pipeline       = load_pipeline()
        data_ok        = True
    except Exception as e:
        st.error(f"Failed to load data or models: {e}")
        data_ok = False
        st.stop()
 
 
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div style='font-size:20px;font-weight:700;margin-bottom:1rem;'>📊 Retail Intelligence</div>", unsafe_allow_html=True)
    st.divider()
 
    st.markdown(f"<span class='slabel'>DATE RANGE</span>", unsafe_allow_html=True)
    min_d = df_raw["InvoiceDate"].dt.date.min()
    max_d = df_raw["InvoiceDate"].dt.date.max()
    date_range = st.date_input("Range", value=(min_d, max_d), min_value=min_d, max_value=max_d, label_visibility="collapsed")
    start_date = date_range[0] if len(date_range) > 0 else min_d
    end_date   = date_range[1] if len(date_range) > 1 else max_d
 
    st.divider()
 
    st.markdown(f"<span class='slabel'>COUNTRY</span>", unsafe_allow_html=True)
    all_countries = sorted(df_raw["Country"].unique().tolist())
    sel_countries = st.multiselect("Countries", options=all_countries, default=[], placeholder="All countries", label_visibility="collapsed")
    countries_filter = sel_countries if sel_countries else None
 
    st.divider()
 
    # Filtered views
    df_filt = apply_filters(df_raw, start_date, end_date, countries_filter)
 
    total_days = (end_date - start_date).days
    prev_end   = start_date - timedelta(days=1)
    prev_start = prev_end   - timedelta(days=total_days)
    df_prev    = apply_filters(df_raw, prev_start, prev_end, countries_filter)
 
    st.markdown(f"<div style='font-size:12px;color:{TXT_SEC};'><b style='color:{TXT_PRI};'>{len(df_filt):,}</b> transactions</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:12px;color:{TXT_SEC};margin-top:4px;'><b style='color:{TXT_PRI};'>{df_filt['CustomerID'].nunique():,}</b> customers</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:12px;color:{TXT_SEC};margin-top:4px;'><b style='color:{TXT_PRI};'>{len(df_seg):,}</b> customers segmented</div>", unsafe_allow_html=True)
 
 
# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<h1 style='font-size:26px;margin-bottom:2px;'>Retail Customer Intelligence Platform</h1>
<div style='color:{TXT_SEC};font-size:13px;margin-bottom:1.25rem;'>
  Revenue · Segmentation · Churn Risk · Products · Geography &nbsp;·&nbsp;
  <span style='color:{BLUE};'>{start_date} → {end_date}</span>
</div>
""", unsafe_allow_html=True)
 
 
# ── KPI cards (always visible) ────────────────────────────────────────────────
kpis = compute_top_kpis(df_filt, df_prev)
 
def fmt_d(d): return f"{d:+.1f}%" if d is not None else None
 
c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
c1.metric("Total Revenue",       f"£{kpis['total_revenue']['value']:,.0f}",          fmt_d(kpis['total_revenue']['delta']))
c2.metric("Total Orders",        f"{kpis['total_orders']['value']:,}",               fmt_d(kpis['total_orders']['delta']))
c3.metric("Unique Customers",    f"{kpis['unique_customers']['value']:,}",           fmt_d(kpis['unique_customers']['delta']))
c4.metric("Avg Order Value",     f"£{kpis['avg_order_value']['value']:,.0f}",        fmt_d(kpis['avg_order_value']['delta']))
c5.metric("Rev / Customer",      f"£{kpis['avg_revenue_per_customer']['value']:,.0f}",None)
c6.metric("Units Sold",          f"{kpis['units_sold']['value']:,}",                 None)
c7.metric("Repeat Rate",         f"{kpis['repeat_customer_rate']['value']:.1f}%",    None)
 
st.markdown("<hr class='sdivider'>", unsafe_allow_html=True)

# TABS
tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
    "📊 Overview",
    "💰 Revenue",
    "🎯 Segmentation",
    "⚠️ Churn Analysis",
    "🛍️ Products",
    "🌍 Geography",
    "👤 Single Customer",
])
 
 
# TAB 1 — OVERVIEW
with tab1:
    monthly = monthly_revenue_trend(df_filt)
 
    col_a, col_b = st.columns([3,1])
 
    with col_a:
        st.markdown("<span class='slabel'>Monthly Revenue Trend</span>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["YearMonth"], y=monthly["Revenue"],
            mode="lines+markers",
            line=dict(color=BLUE, width=2.5),
            marker=dict(size=5),
            fill="tozeroy", fillcolor=f"rgba(88,166,255,0.07)",
            hovertemplate="<b>%{x}</b><br>£%{y:,.0f}<extra></extra>",
        ))
        fig.update_layout(**LAYOUT, height=300)
        st.plotly_chart(fig, use_container_width=True)
 
    with col_b:
        st.markdown("<span class='slabel'>Segment Distribution</span>", unsafe_allow_html=True)
        seg_counts = df_seg["Segment_Name"].value_counts().reset_index()
        seg_counts.columns = ["Segment","Count"]
        fig2 = px.pie(seg_counts, values="Count", names="Segment", hole=0.55,
                      color_discrete_sequence=[BLUE,GREEN,AMBER,RED])
        fig2.update_traces(textinfo="percent+label", textfont_size=11)
        fig2.update_layout(**LAYOUT, height=300, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
 
    # Churn + segment summary side by side
    col_c, col_d = st.columns(2)
 
    with col_c:
        st.markdown("<span class='slabel'>Churn Risk Overview</span>", unsafe_allow_html=True)
        risk_counts = df_seg["ChurnRisk"].value_counts().reindex(["High Risk","Medium Risk","Low Risk"]).reset_index()
        risk_counts.columns = ["Risk","Count"]
        fig3 = px.bar(risk_counts, x="Risk", y="Count",
                      color="Risk", color_discrete_map=RISK_COLORS, text="Count")
        fig3.update_traces(textposition="outside")
        fig3.update_layout(**LAYOUT, height=280, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
 
    with col_d:
        st.markdown("<span class='slabel'>Top 5 Countries by Revenue</span>", unsafe_allow_html=True)
        top5 = revenue_by_country(df_filt, 5)
        fig4 = px.bar(top5.sort_values("Revenue"), x="Revenue", y="Country",
                      orientation="h", color="Revenue",
                      color_continuous_scale=[[0,f"rgba(88,166,255,0.3)"],[1,BLUE]],
                      text="Revenue")
        fig4.update_traces(texttemplate="£%{text:,.0f}", textposition="outside")
        fig4.update_layout(**LAYOUT, height=280, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)
 
# TAB 2 — REVENUE
with tab2:
    col_a, col_b = st.columns([3,1])
 
    with col_a:
        st.markdown("<span class='slabel'>Monthly Revenue + MoM Growth</span>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly["YearMonth"], y=monthly["Revenue"],
            name="Revenue", marker_color=BLUE, opacity=0.7,
            hovertemplate="<b>%{x}</b><br>£%{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=monthly["YearMonth"], y=monthly["MoM_Growth"],
            name="MoM %", mode="lines+markers",
            line=dict(color=AMBER, width=2), yaxis="y2",
            hovertemplate="<b>%{x}</b><br>MoM: %{y:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            **LAYOUT, height=340,
            yaxis2=dict(overlaying="y", side="right", showgrid=False,
                        tickfont=dict(color=AMBER, size=11), ticksuffix="%"),
            barmode="overlay",
        )
        st.plotly_chart(fig, use_container_width=True)
 
    with col_b:
        st.markdown("<span class='slabel'>Orders by Day of Week</span>", unsafe_allow_html=True)
        dow = orders_by_day_of_week(df_filt)
        fig2 = px.bar(dow, x="Orders", y="DayOfWeek", orientation="h",
                      color_discrete_sequence=[GREEN], text="Orders")
        fig2.update_traces(textposition="outside")
        fig2.update_layout(**LAYOUT, height=340, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
 
    st.markdown("<span class='slabel'>Revenue Heatmap — Year × Month</span>", unsafe_allow_html=True)
    hm = revenue_heatmap(df_filt)
    month_cols = [c for c in hm.columns if c != "Year"]
    fig3 = go.Figure(go.Heatmap(
        z=hm[month_cols].values.tolist(),
        x=month_cols, y=hm["Year"].astype(str).tolist(),
        colorscale=[[0,CARD_BG],[0.5,f"rgba(88,166,255,0.5)"],[1,BLUE]],
        hovertemplate="<b>%{y} %{x}</b><br>£%{z:,.0f}<extra></extra>",
    ))
    fig3.update_layout(**LAYOUT, height=220)
    st.plotly_chart(fig3, use_container_width=True)
 

# TAB 3 — SEGMENTATION
with tab3:
    st.markdown("<span class='slabel'>Segment Profiles</span>", unsafe_allow_html=True)
 
    seg_summary = (
        df_seg.groupby("Segment_Name")
        .agg(
            Customers  =("CustomerID","count"),
            AvgRecency =("Recency",   "mean"),
            AvgFreq    =("Frequency", "mean"),
            AvgMonetary=("Monetary",  "mean"),
            AvgChurn   =("ChurnScore","mean"),
        )
        .reset_index()
        .round(2)
    )
 
    seg_icons = {
        "High-Value Customers": ("🟢", "High spend, high frequency, recent buyers"),
        "Loyal Customers":      ("🔵", "Consistent purchasers with long tenure"),
        "At-Risk Customers":    ("🟡", "Previously active, now disengaging"),
        "Occasional Buyers":    ("⚪", "Low frequency, low spend, infrequent visits"),
    }
 
    cols = st.columns(len(seg_summary))
    for i, (_, row) in enumerate(seg_summary.iterrows()):
        icon, desc = seg_icons.get(row["Segment_Name"], ("⚫",""))
        with cols[i]:
            st.markdown(f"""
            <div class='seg-card'>
              <h4>{icon} {row['Segment_Name']}</h4>
              <p>{desc}</p>
              <p style='margin-top:.5rem;'>
                <b style='color:{TXT_PRI};'>{int(row['Customers']):,}</b> customers<br>
                Avg Recency: <b style='color:{TXT_PRI};'>{row['AvgRecency']:.0f}d</b><br>
                Avg Orders: <b style='color:{TXT_PRI};'>{row['AvgFreq']:.1f}</b><br>
                Avg Spend: <b style='color:{TXT_PRI};'>£{row['AvgMonetary']:,.0f}</b><br>
                Avg Churn: <b style='color:{RED if row["AvgChurn"]>=0.65 else AMBER if row["AvgChurn"]>=0.35 else GREEN};'>{row['AvgChurn']:.2f}</b>
              </p>
            </div>
            """, unsafe_allow_html=True)
 
    col_a, col_b = st.columns(2)
 
    with col_a:
        st.markdown("<span class='slabel'>Customer Count per Segment</span>", unsafe_allow_html=True)
        fig = px.bar(seg_summary, x="Segment_Name", y="Customers",
                     color="Segment_Name",
                     color_discrete_sequence=[BLUE,GREEN,AMBER,RED],
                     text="Customers")
        fig.update_traces(textposition="outside")
        fig.update_layout(**LAYOUT, height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
 
    with col_b:
        st.markdown("<span class='slabel'>Avg Monetary Value per Segment</span>", unsafe_allow_html=True)
        fig2 = px.bar(seg_summary, x="Segment_Name", y="AvgMonetary",
                      color="AvgMonetary",
                      color_continuous_scale=[[0,f"rgba(88,166,255,0.3)"],[1,BLUE]],
                      text="AvgMonetary")
        fig2.update_traces(texttemplate="£%{text:,.0f}", textposition="outside")
        fig2.update_layout(**LAYOUT, height=320, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)
 
    # Scatter: Recency vs Monetary coloured by segment
    st.markdown("<span class='slabel'>RFM Landscape — Recency vs Monetary by Segment</span>", unsafe_allow_html=True)
    fig3 = px.scatter(
        df_seg, x="Recency", y="Monetary",
        color="Segment_Name",
        color_discrete_sequence=[BLUE,GREEN,AMBER,RED],
        hover_data=["CustomerID","Frequency","ChurnRisk"],
        opacity=0.6,
    )
    fig3.update_layout(**LAYOUT, height=380)
    st.plotly_chart(fig3, use_container_width=True)
 
 
# TAB 4 — CHURN ANALYSIS
with tab4:
    total      = len(df_seg)
    n_high     = (df_seg["ChurnRisk"]=="High Risk").sum()
    n_medium   = (df_seg["ChurnRisk"]=="Medium Risk").sum()
    n_low      = (df_seg["ChurnRisk"]=="Low Risk").sum()
 
    ch1,ch2,ch3,ch4 = st.columns(4)
    ch1.metric("Total Customers",   f"{total:,}")
    ch2.metric("🔴 High Risk",      f"{n_high:,} ({n_high/total*100:.1f}%)")
    ch3.metric("🟡 Medium Risk",    f"{n_medium:,} ({n_medium/total*100:.1f}%)")
    ch4.metric("🟢 Low Risk",       f"{n_low:,} ({n_low/total*100:.1f}%)")
 
    st.markdown("<hr class='sdivider'>", unsafe_allow_html=True)
 
    col_a, col_b = st.columns(2)
 
    with col_a:
        st.markdown("<span class='slabel'>Churn Score Distribution</span>", unsafe_allow_html=True)
        fig = px.histogram(df_seg, x="ChurnScore", nbins=30,
                           color_discrete_sequence=[BLUE],
                           labels={"ChurnScore":"Churn Score (0 = safe, 1 = at risk)"})
        fig.update_layout(**LAYOUT, height=300)
        st.plotly_chart(fig, use_container_width=True)
 
    with col_b:
        st.markdown("<span class='slabel'>Churn Risk by Segment</span>", unsafe_allow_html=True)
        churn_seg = segment_churn_summary(df_seg)
        fig2 = px.bar(
            churn_seg, x="Segment_Name",
            y=["HighRiskCount","MediumRiskCount","LowRiskCount"],
            color_discrete_map={
                "HighRiskCount":RED, "MediumRiskCount":AMBER, "LowRiskCount":GREEN
            },
            barmode="stack",
        )
        fig2.update_layout(**LAYOUT, height=300)
        st.plotly_chart(fig2, use_container_width=True)
 
    st.markdown("<span class='slabel'>Churn Risk Landscape — Recency vs Monetary</span>", unsafe_allow_html=True)
    fig3 = px.scatter(
        df_seg, x="Recency", y="Monetary",
        color="ChurnRisk", color_discrete_map=RISK_COLORS,
        hover_data=["CustomerID","Frequency","ChurnScore","Segment_Name"],
        opacity=0.6,
    )
    fig3.update_layout(**LAYOUT, height=380)
    st.plotly_chart(fig3, use_container_width=True)
 
    st.markdown("<span class='slabel'>Full Churn Table</span>", unsafe_allow_html=True)
    display_cols = ["CustomerID","Recency","Frequency","Monetary","Segment_Name","ChurnScore","ChurnRisk"]
    st.dataframe(df_seg[display_cols], use_container_width=True)
 
    csv = df_seg.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Churn Results CSV", csv, "churn_results.csv", "text/csv")
 
 
# TAB 5 — PRODUCTS
with tab5:
    col_a, col_b = st.columns(2)
 
    with col_a:
        st.markdown("<span class='slabel'>Top 10 Products by Revenue</span>", unsafe_allow_html=True)
        pr = top_products_by_revenue(df_filt, 10)
        fig = px.bar(pr.sort_values("Revenue"), x="Revenue", y="Description",
                     orientation="h",
                     color="Revenue",
                     color_continuous_scale=[[0,f"rgba(88,166,255,0.3)"],[1,BLUE]],
                     text="Revenue")
        fig.update_traces(texttemplate="£%{text:,.0f}", textposition="outside")
        fig.update_layout(**LAYOUT, height=400, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
 
    with col_b:
        st.markdown("<span class='slabel'>Top 10 Products by Units Sold</span>", unsafe_allow_html=True)
        pv = top_products_by_volume(df_filt, 10)
        fig2 = px.bar(pv.sort_values("UnitsSold"), x="UnitsSold", y="Description",
                      orientation="h",
                      color="UnitsSold",
                      color_continuous_scale=[[0,f"rgba(63,185,80,0.3)"],[1,GREEN]],
                      text="UnitsSold")
        fig2.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig2.update_layout(**LAYOUT, height=400, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)
 
    st.markdown("<span class='slabel'>Revenue vs Units Sold — Top 30 Products</span>", unsafe_allow_html=True)
    top30 = top_products_by_revenue(df_filt, 30)
    fig3  = px.scatter(top30, x="UnitsSold", y="Revenue",
                       text="Description", size="Revenue",
                       color="Revenue",
                       color_continuous_scale=[[0,AMBER],[1,BLUE]])
    fig3.update_traces(textposition="top center", textfont_size=9)
    fig3.update_layout(**LAYOUT, height=420, coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)
 

# TAB 6 — GEOGRAPHY
with tab6:
    country_rev = revenue_by_country(df_filt, 10)
 
    col_a, col_b = st.columns([2,1])
 
    with col_a:
        st.markdown("<span class='slabel'>Top 10 Countries by Revenue</span>", unsafe_allow_html=True)
        fig = px.bar(country_rev.sort_values("Revenue"),
                     x="Revenue", y="Country", orientation="h",
                     color="Revenue",
                     color_continuous_scale=[[0,f"rgba(88,166,255,0.3)"],[1,BLUE]],
                     text="Revenue")
        fig.update_traces(texttemplate="£%{text:,.0f}", textposition="outside")
        fig.update_layout(**LAYOUT, height=400, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
 
    with col_b:
        st.markdown("<span class='slabel'>Revenue Share</span>", unsafe_allow_html=True)
        fig2 = px.pie(country_rev, values="Revenue", names="Country", hole=0.5,
                      color_discrete_sequence=[BLUE,GREEN,AMBER,RED,PURPLE,"#79C0FF","#56D364","#FFA657","#FF7B72","#D2A8FF"])
        fig2.update_traces(textinfo="percent", textfont_size=11)
        fig2.update_layout(**LAYOUT, height=400)
        st.plotly_chart(fig2, use_container_width=True)
 
    country_rev["AOV"] = (country_rev["Revenue"] / country_rev["Orders"]).round(2)
    country_rev["Revenue"] = country_rev["Revenue"].apply(lambda x: f"£{x:,.0f}")
    country_rev["AOV"]     = country_rev["AOV"].apply(lambda x: f"£{x:,.2f}")
    country_rev.index = range(1, len(country_rev)+1)
    st.dataframe(country_rev[["Country","Revenue","Orders","AOV"]], use_container_width=True)
 
# TAB 7 — SINGLE CUSTOMER
with tab7:
    st.subheader("Predict Segment & Churn Risk for a Single Customer")
    st.caption("Enter RFM values manually — the pipeline will predict segment and churn risk in real time.")
 
    col_a, col_b = st.columns(2)
    with col_a:
        recency       = st.number_input("Recency (days since last purchase)",   min_value=0,   value=30)
        tenure        = st.number_input("Tenure (days since first purchase)",   min_value=0,   value=365)
        frequency     = st.number_input("Frequency (number of purchases)",      min_value=1,   value=5)
    with col_b:
        monetary      = st.number_input("Monetary (total spend £)",             min_value=0.0, value=500.0)
        avg_order_val = st.number_input("AvgOrderValue (Monetary / Frequency)", min_value=0.0, value=100.0)
 
    if st.button("Predict", type="primary"):
        manual_df = pd.DataFrame([{
            "Recency":recency,"Tenure":tenure,"Frequency":frequency,
            "Monetary":monetary,"AvgOrderValue":avg_order_val,
        }])
        try:
            features    = manual_df[["Recency","Tenure","Frequency","Monetary","AvgOrderValue"]].copy()
            processed   = pipeline._preprocess(features)
            clusters    = pipeline.model.predict(processed)
            manual_df["Segment"]      = clusters
            seg_map                   = pipeline._name_segments(manual_df)
            manual_df["Segment_Name"] = manual_df["Segment"].map(seg_map)
 
            churn_score = compute_single_customer_churn_score(recency, frequency, monetary)
            churn_risk  = classify_churn_risk(churn_score)
            explanation = explain_churn_score(recency, frequency, monetary)
 
            seg_name   = manual_df["Segment_Name"].iloc[0]
            risk_emoji = {"High Risk":"🔴","Medium Risk":"🟡","Low Risk":"🟢"}
            risk_color = {RED:"High Risk", AMBER:"Medium Risk", GREEN:"Low Risk"}
 
            r1, r2 = st.columns(2)
            with r1:
                st.success(f"**Segment:** {seg_name}")
            with r2:
                color = RED if churn_risk=="High Risk" else AMBER if churn_risk=="Medium Risk" else GREEN
                st.markdown(
                    f"<div style='background:{color}22;border:1px solid {color};border-radius:8px;"
                    f"padding:.75rem 1rem;font-size:15px;font-weight:500;color:{color};'>"
                    f"{risk_emoji.get(churn_risk,'')} Churn Risk: {churn_risk} &nbsp;·&nbsp; Score: {churn_score:.4f}</div>",
                    unsafe_allow_html=True,
                )
 
            st.markdown("<span class='slabel'>Score Breakdown</span>", unsafe_allow_html=True)
            breakdown = pd.DataFrame([
                {"Driver":"Recency",   "Raw Value":f"{recency} days",       "Weight":"50%", "Contribution":explanation["recency_component"]},
                {"Driver":"Frequency", "Raw Value":f"{frequency} purchases", "Weight":"30%", "Contribution":explanation["frequency_component"]},
                {"Driver":"Monetary",  "Raw Value":f"£{monetary:,.2f}",      "Weight":"20%", "Contribution":explanation["monetary_component"]},
            ])
            st.dataframe(breakdown, use_container_width=True, hide_index=True)
 
            fig_g = px.bar(
                x=[churn_score], y=["Churn Score"], orientation="h", range_x=[0,1],
                color_discrete_sequence=[RED if churn_score>=0.65 else AMBER if churn_score>=0.35 else GREEN],
                title=f"Churn Score: {churn_score:.4f}  |  {churn_risk}",
            )
            fig_g.update_layout(**LAYOUT, height=180, showlegend=False)
            st.plotly_chart(fig_g, use_container_width=True)
 
        except Exception as e:
            st.error(f"Prediction error: {e}")
            import traceback
            st.code(traceback.format_exc())
