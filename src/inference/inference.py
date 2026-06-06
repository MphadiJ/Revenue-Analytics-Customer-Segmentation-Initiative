import sys
import os
import pandas as pd
import numpy as np
import joblib

# Path fix - add repo root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_PATH)

# Import WITHOUT src. prefix
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
        cluster_summary = df_with_clusters.groupby('Segment').mean()
        cluster_summary['R_rank'] = cluster_summary['Recency'].rank(ascending=False)
        cluster_summary['F_rank'] = cluster_summary['Frequency'].rank(ascending=True)
        cluster_summary['M_rank'] = cluster_summary['Monetary'].rank(ascending=True)
        segment_map = {}
        for cluster in cluster_summary.index:
            r = cluster_summary.loc[cluster, 'R_rank']
            f = cluster_summary.loc[cluster, 'F_rank']
            m = cluster_summary.loc[cluster, 'M_rank']
            if f >= cluster_summary['F_rank'].quantile(0.75) and \
               m >= cluster_summary['M_rank'].quantile(0.75) and \
               r <= cluster_summary['R_rank'].quantile(0.25):
                segment_map[cluster] = "High Value"
            elif f >= cluster_summary['F_rank'].quantile(0.75):
                segment_map[cluster] = "Frequent Buyer"
            elif r >= cluster_summary['R_rank'].quantile(0.75):
                segment_map[cluster] = "At Risk"
            else:
                segment_map[cluster] = "Regular"
        return segment_map

    def predict(self, rfm_df):
        if not isinstance(rfm_df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        required_cols = ['Recency', 'Tenure', 'Frequency', 'Monetary', 'AvgOrderValue']
        missing_cols = [col for col in required_cols if col not in rfm_df.columns]
        if missing_cols:
            raise ValueError
