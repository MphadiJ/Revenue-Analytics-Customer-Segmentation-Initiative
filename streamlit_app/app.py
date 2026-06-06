import streamlit as st
import pandas as pd
import sys
import os

# Path fix FIRST - before any custom imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_PATH)

from inference.inference import InferencePipeline

# Load models
pipeline = InferencePipeline(
    preprocessor_path=os.path.join(PROJECT_ROOT, "models", "preprocessor.pkl"),
    model_path=os.path.join(PROJECT_ROOT, "models", "kmeans_best.pkl")
)

st.title("Retail Customer Segmentation App 🛒")
st.write("Upload your RFM data or enter manually to see customer segments.")

uploaded_file = st.file_uploader(
    "Upload CSV (must have Recency, Tenure, Frequency, Monetary, AvgOrderValue)",
    type=["csv"]
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.dataframe(df.head())

        # Step by step debug
        df_features = df[['Recency', 'Tenure', 'Frequency', 'Monetary', 'AvgOrderValue']].copy()
        processed_df = pipeline._preprocess(df_features)
        clusters = pipeline.model.predict(processed_df)

        result = df.copy()
        result['Segment'] = clusters

        st.write("Raw cluster numbers:", result['Segment'].value_counts().to_dict())

        segment_map = pipeline._name_segments(result)
        st.write("Segment map:", segment_map)

        result['Segment_Name'] = result['Segment'].map(segment_map)

        st.success("Predicted Segments:")
        st.dataframe(result)

    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())

st.subheader("Or enter a single customer manually")
recency = st.number_input("Recency (days since last purchase)", min_value=0)
tenure = st.number_input("Tenure (days since first purchase)", min_value=0)
frequency = st.number_input("Frequency (number of purchases)", min_value=0)
monetary = st.number_input("Monetary (total spend)", min_value=0.0)
avg_order_value = st.number_input("AvgOrderValue (Monetary / Frequency)", min_value=0.0)

if st.button("Predict Segment for this customer"):
    manual_df = pd.DataFrame([{
        "Recency": recency,
        "Tenure": tenure,
        "Frequency": frequency,
        "Monetary": monetary,
        "AvgOrderValue": avg_order_value
    }])
    try:
        df_features = manual_df[['Recency', 'Tenure', 'Frequency', 'Monetary', 'AvgOrderValue']].copy()
        processed_df = pipeline._preprocess(df_features)
        clusters = pipeline.model.predict(processed_df)
        manual_df['Segment'] = clusters

        segment_map = pipeline._name_segments(manual_df)
        manual_df['Segment_Name'] = manual_df['Segment'].map(segment_map)

        st.success("Prediction:")
        st.dataframe(manual_df)
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())
