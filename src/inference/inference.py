import sys
import os
import pandas as pd
import numpy as np
import joblib

# Path fix - add repo root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_PATH)

from features.build_features import FeatureEngineer
from Transformer.Preprocessing import DataPreprocessor


class InferencePipeline:
    def __init__(self, preprocessor_path, model_path):
        self.preprocessor = joblib.load(preprocessor_path)
        self.model = joblib.load(model_path)
        self.expected_features = list(self.preprocessor.scaler.feature_names_in_)

    def _preprocess(self, rfm_df):
        df = rfm_df.copy()
        df = self.preprocessor.handle_outliers(df, ['Monetary', 'Frequency', 'Recency', 'Tenure', 'AvgOrderValue'])
        df = self.preprocessor.transform_skewed_data(df)
        df = df[[col for col in self.expected_features if col in df.columns]]
        for col in self.expected_features:
            if col not in df.columns:
                df[col] = 0
        df = df[self.expected_features]
        scaled = self.preprocessor.scaler.transform(df)
        return pd.DataFrame(scaled, columns=self.expected_features, index=df.index)

    def _name_segments(self, df_with_clusters):
        cluster_summary = df_with_clusters.groupby('Segment').mean(numeric_only=True)

        # Rank clusters by Monetary value descending
        sorted_clusters = cluster_summary['Monetary'].sort_values(ascending=False).index.tolist()

        labels = ['High Value', 'Frequent Buyer', 'Regular', 'At Risk']
        segment_map = {}
        for i, cluster in enumerate(sorted_clusters):
            segment_map[cluster] = labels[i] if i < len(labels) else 'Regular'

        return segment_map

    def predict(self, rfm_df):
        if not isinstance(rfm_df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        required_cols = ['Recency', 'Tenure', 'Frequency', 'Monetary', 'AvgOrderValue']
        missing_cols = [col for col in required_cols if col not in rfm_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df_features = rfm_df[required_cols].copy()
        processed_df = self._preprocess(df_features)
        clusters = self.model.predict(processed_df)

        result = rfm_df.copy()
        result['Segment'] = clusters
        result['Segment_Name'] = result['Segment'].map(self._name_segments(result))

        return result
