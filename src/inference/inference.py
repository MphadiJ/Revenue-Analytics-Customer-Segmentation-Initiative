import sys
import os
import pandas as pd
import numpy as np
import joblib
sys.path.append(os.path.abspath(os.path.join('/home/selowa-mphadi/PycharmProjects/pythonProject/kmeans clustering')))
from src.features.build_features import FeatureEngineer
from src.Transformer.Preprocessing import DataPreprocessor

class InferencePipeline:
    def __init__(self, preprocessor_path, model_path):
        # Load preprocessor (handles scaling, skew, outliers)
        self.preprocessor = joblib.load(preprocessor_path)
        # Load trained KMeans model
        self.model = joblib.load(model_path)
        # Save expected features
        self.expected_features = list(self.preprocessor.scaler.feature_names_in_)

    # Preprocessing step
    def _preprocess(self, rfm_df):
        df = rfm_df.copy()

        # Handle outliers and skew
        df = self.preprocessor.handle_outliers(df, ['Monetary', 'Frequency', 'Recency', 'Tenure', 'AvgOrderValue'])
        df = self.preprocessor.transform_skewed_data(df)

        # 🔒 Ensure correct features for model
        # Drop unexpected columns
        df = df[[col for col in self.expected_features if col in df.columns]]

        # Add missing columns (fill with 0)
        for col in self.expected_features:
            if col not in df.columns:
                df[col] = 0

        # Ensure correct order
        df = df[self.expected_features]

        # Scale
        scaled = self.preprocessor.scaler.transform(df)
        return pd.DataFrame(scaled, columns=self.expected_features, index=df.index)

    # Segment naming
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

    # Main prediction method
    def predict(self, rfm_df):
        """
        Args:
            rfm_df (pd.DataFrame): Must contain:
                Recency, Tenure, Frequency, Monetary, AvgOrderValue
        Returns:
            pd.DataFrame with Segment and Segment_Name
        """
        if not isinstance(rfm_df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        required_cols = ['Recency', 'Tenure', 'Frequency', 'Monetary', 'AvgOrderValue']
        missing_cols = [col for col in required_cols if col not in rfm_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Use only RFM features
        df_features = rfm_df[required_cols].copy()

        # Preprocess
        processed_df = self._preprocess(df_features)

        # Predict clusters
        clusters = self.model.predict(processed_df)

        # Attach results
        result = rfm_df.copy()
        result['Segment'] = clusters
        result['Segment_Name'] = result['Segment'].map(self._name_segments(result))

        return result
