import os
import sys
import streamlit as st

# Get the absolute path of the directory containing this app file
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

# Move up one level to reach the true project root path
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

# Insert both paths into sys.path so Python can see all modules
for path in [PROJECT_ROOT, SRC_PATH]:
    if path not in sys.path:
        sys.path.insert(0, path)

# IMPORT PIPELINE & TABS
from inference.inference import InferencePipeline
from tabs.segmentation import segmentation_tab
from tabs.churn import churn_tab
from tabs.single_customer import single_customer_tab
from tabs.dashboard import dashboard_tab
from tabs.executive_summary import executive_summary_tab

# --- SESSION STATE INITIALIZATION (Must be done first to avoid KeyErrors) ---
DEFAULT_STATE = {
    "raw_df": None,
    "segmented_df": None,
    "analysis_df": None,
    "kpis": None
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

# PAGE CONFIG
st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CACHE MODEL PIPELINE
@st.cache_resource
def load_pipeline():
    return InferencePipeline(
        preprocessor_path=os.path.join(PROJECT_ROOT, "models", "preprocessor.pkl"),
        model_path=os.path.join(PROJECT_ROOT, "models", "kmeans_best.pkl")
    )

pipeline = load_pipeline()

# CACHE DATA INGESTION PIPELINE
@st.cache_data
def auto_load_data():
    from kpi_engine import load_raw_data, compute_rfm
    from churn.churn_analysis import add_churn_analysis

    data_path = os.path.join(PROJECT_ROOT, "raw data", "rt_data.csv")
    df_raw = load_raw_data(data_path)
    rfm = compute_rfm(df_raw)

    features = rfm[["Recency", "Tenure", "Frequency", "Monetary", "AvgOrderValue"]].copy()
    processed = pipeline._preprocess(features)
    clusters = pipeline.model.predict(processed)
    rfm["Segment"] = clusters
    seg_map = pipeline._name_segments(rfm)
    rfm["Segment_Name"] = rfm["Segment"].map(seg_map)
    rfm = add_churn_analysis(rfm)
    return df_raw, rfm

# Lazy evaluation data trigger
if st.session_state["analysis_df"] is None:
    with st.spinner("Loading analytical engines..."):
        df_raw, df_seg = auto_load_data()
        st.session_state["raw_df"] = df_raw
        st.session_state["segmented_df"] = df_seg
        st.session_state["analysis_df"] = df_seg

# --- SIDEBAR WORKFLOW ---
with st.sidebar:
    st.title("📊 Customer Intelligence")
    st.caption("Revenue Analytics Platform")
    st.divider()
    
    st.success("Workflow State")
    data_loaded = st.session_state["analysis_df"] is not None
    seg_done = data_loaded and "Segment_Name" in st.session_state["analysis_df"].columns
    churn_done = data_loaded and "ChurnRisk" in st.session_state["analysis_df"].columns

    st.markdown(f"""
    {'✅' if data_loaded else '⬜'} Data Loaded  
    {'✅' if seg_done    else '⬜'} Segmentation  
    {'✅' if churn_done  else '⬜'} Churn Analysis  
    """)

    st.divider()

    if st.session_state["analysis_df"] is not None:
        st.success("Dataset Active")
        st.metric("Total Customer Rows", len(st.session_state["analysis_df"]))
    else:
        st.warning("No active dataset loaded")

# --- MAIN APP HEADER ---
st.title("Customer Intelligence & Revenue Analytics Platform")
st.caption("Customer Segmentation • Churn Prediction • Business KPIs • Executive Insights")

# --- HOME EXECUTIVE METRICS GRID ---
if st.session_state["analysis_df"] is not None:
    df = st.session_state["analysis_df"]
    
    # Grid Row 1: High-Level Portfolio Performance
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df):,}")
    
    if "Segment_Name" in df.columns:
        c2.metric("Identified Segments", df["Segment_Name"].nunique())
        
    if "ChurnRisk" in df.columns:
        high_risk_count = (df["ChurnRisk"] == "High Risk").sum()
        c3.metric("High Churn Risk Cohort", f"{high_risk_count:,}")
        
    if "Monetary" in df.columns:
        c4.metric("Total Revenue Pool", f"£{df['Monetary'].sum():,.0f}")
        
    st.write("") # Spacing row
    
    # Grid Row 2: Analytical Financial Risk Metrics
    c5, c6 = st.columns(2)
    
    if "ChurnScore" in df.columns:
        avg_churn = df["ChurnScore"].mean()
        c5.metric("Average Portfolio Churn Risk Score", f"{avg_churn:.1%}" if avg_churn <= 1.0 else f"{avg_churn:.2f}")
        
    if "Monetary" in df.columns and "ChurnRisk" in df.columns:
        revenue_at_risk = df[df["ChurnRisk"] == "High Risk"]["Monetary"].sum()
        c6.metric("Financial Revenue Exposure (At Risk)", f"£{revenue_at_risk:,.0f}")

st.divider()

# --- ANALYTICAL WORKSPACE TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Segmentation",
    "⚠️ Churn Tracking",
    "👤 Single Customer",
    "📈 Revenue KPIs",
    "🧠 Executive Summary"
])

with tab1:
    segmentation_tab(pipeline)

with tab2:
    churn_tab()

with tab3:
    single_customer_tab(pipeline)

with tab4:
    dashboard_tab()

with tab5:
    executive_summary_tab()
