import streamlit as st
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join('/home/selowa-mphadi/PycharmProjects/pythonProject/kmeans clustering')))
from src.inference.inference import InferencePipeline

# Load models
pipeline = InferencePipeline(
    preprocessor_path="/home/selowa-mphadi/PycharmProjects/pythonProject/kmeans clustering/models/preprocessor.pkl",
    model_path="/home/selowa-mphadi/PycharmProjects/pythonProject/kmeans clustering/models/kmeans_best.pkl"
)

# App UI
st.title("Retail Customer Segmentation App 🛒")
st.write("Upload your RFM data or enter manually to see customer segments.")

# Option: Upload CSV
uploaded_file = st.file_uploader("Upload CSV (must have Recency, Tenure, Frequency, Monetary, AvgOrderValue)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.dataframe(df.head())

        # Predict
        predictions = pipeline.predict(df)
        st.write("Predicted Segments:")
        st.dataframe(predictions)

    except Exception as e:
        st.error(f"Error: {e}")

# Option: Manual input
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
    pred = pipeline.predict(manual_df)
    st.write(pred)
