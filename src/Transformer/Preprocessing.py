import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self, test_ratio=0.2):
        self.test_ratio = test_ratio
        self.scaler = StandardScaler()
        self.skewed_features = ['Monetary', 'Frequency', 'AvgOrderValue', 'Tenure']

    def handle_outliers(self, df, columns):
        """Caps outliers at the 95th percentile (Winsorization)."""
        df_capped = df.copy()
        for col in columns:
            if col in df_capped.columns:
                upper_limit = df_capped[col].quantile(0.95)
                df_capped[col] = np.where(df_capped[col] > upper_limit, upper_limit, df_capped[col])
        return df_capped

    def transform_skewed_data(self, df):
        """Applies Log transformation to handle right-skewed distributions."""
        df_log = df.copy()
        for col in self.skewed_features:
            if col in df_log.columns:
                df_log[col] = np.log1p(df_log[col])
        return df_log

    def time_based_split(self, rfm_df):
        """
        Splits data based on Recency as a proxy for time.
        Lower Recency = more recent customers → goes to test set.
        """

        # Sort by Recency (ascending = most recent first)
        rfm_sorted = rfm_df.sort_values(by='Recency', ascending=True)

        split_index = int(len(rfm_sorted) * self.test_ratio)

        test_df = rfm_sorted.iloc[:split_index]   # Most recent customers
        train_df = rfm_sorted.iloc[split_index:]  # Older customers

        return train_df, test_df

    def run_pipeline(self, rfm_df):
        """
        Complete Pipeline:
        1. Time-based Split
        2. Cap Outliers
        3. Log Transform
        4. Scale
        """

        # 1. Time-based Split
        train_df, test_df = self.time_based_split(rfm_df)

        # 2. Handle Outliers (fit logic on train only)
        cols_to_cap = ['Monetary', 'Frequency', 'Recency']
        train_df = self.handle_outliers(train_df, cols_to_cap)
        test_df = self.handle_outliers(test_df, cols_to_cap)

        # 3. Transform Skewed Features
        train_df = self.transform_skewed_data(train_df)
        test_df = self.transform_skewed_data(test_df)

        # 4. Scale Data (fit only on training set)
        train_scaled = self.scaler.fit_transform(train_df)
        test_scaled = self.scaler.transform(test_df)

        # Convert back to DataFrame
        train_final = pd.DataFrame(train_scaled, columns=train_df.columns, index=train_df.index)
        test_final = pd.DataFrame(test_scaled, columns=test_df.columns, index=test_df.index)

        print(f"Preprocessing Complete.")
        print(f"Train shape: {train_final.shape}, Test shape: {test_final.shape}")

        return train_final, test_final

    def save_preprocessor(self, filepath="models/preprocessor.pkl"):
        """Saves the fitted scaler and preprocessor."""
        joblib.dump(self, filepath)
        print(f"Preprocessor saved successfully to {filepath}")
