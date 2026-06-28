import streamlit as st
import os
import sys

# PATHS & ENVIRONMENT ALIGNMENT
import os
import sys

# Get the absolute path of the directory containing app2.py (streamlit_app/)
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

# Move up one level to reach the true project root path
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

# Insert both paths into sys.path so Python can see all modules
for path in [PROJECT_ROOT, SRC_PATH]:
    if path not in sys.path:
        sys.path.insert(0, path)

# IMPORT PIPELINE

from inference.inference import InferencePipeline

# Import Tabs
from tabs.segmentation import segmentation_tab
from tabs.churn import churn_tab
from tabs.single_customer import single_customer_tab
from tabs.dashboard import dashboard_tab
from tabs.executive_summary import executive_summary_tab


# PAGE CONFIG


st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CACHE MODEL

@st.cache_resource
def load_pipeline():

    return InferencePipeline(
        preprocessor_path=os.path.join(
            PROJECT_ROOT,
            "models",
            "preprocessor.pkl"
        ),

        model_path=os.path.join(
            PROJECT_ROOT,
            "models",
            "kmeans_best.pkl"
        )
    )

pipeline = load_pipeline()

# SESSION STATE

DEFAULT_STATE = {

    "raw_df": None,

    "segmented_df": None,

    "analysis_df": None,

    "kpis": None
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value

# SIDEBAR

with st.sidebar:

    st.title("📊 Customer Intelligence")

    st.caption(
        "Revenue Analytics Platform"
    )

    st.divider()

    st.success("Workflow")

    st.markdown("""
1. Upload Data

2. Segment Customers

3. Predict Churn

4. Review KPIs

5. Executive Summary
""")

    st.divider()

    if st.session_state["analysis_df"] is not None:

        st.success("Dataset Ready")

        st.metric(
            "Customers",
            len(st.session_state["analysis_df"])
        )

    else:

        st.warning(
            "No dataset loaded"
        )
# HEADER

st.title(
    "Customer Intelligence & Revenue Analytics Platform"
)

st.caption(
    """
Customer Segmentation • Churn Prediction •
Business KPIs • Executive Insights
"""
)

# HOME METRICS

if st.session_state["analysis_df"] is not None:

    df = st.session_state["analysis_df"]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Customers",
        len(df)
    )

    if "Segment_Name" in df.columns:

        c2.metric(
            "Segments",
            df["Segment_Name"].nunique()
        )

    if "ChurnRisk" in df.columns:

        high = (df["ChurnRisk"]=="High Risk").sum()

        c3.metric(
            "High Risk",
            high
        )

    if "Monetary" in df.columns:

        c4.metric(
            "Revenue",
            f"£{df['Monetary'].sum():,.0f}"
        )

st.divider()

# TABS

tab1, tab2, tab3, tab4, tab5 = st.tabs(

    [

        "📊 Customer Segmentation",

        "⚠️ Churn Analysis",

        "👤 Single Customer",

        "📈 KPI Dashboard",

        "🧠 Executive Summary"

    ]

)

# TABS

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
