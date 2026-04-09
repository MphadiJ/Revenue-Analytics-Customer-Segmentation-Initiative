import pandas as pd
import numpy as np
import joblib


class FeatureEngineer:
    def __init__(self):
        pass

    def create_rfm_features(self, df):
        """
        Engineers advanced RFM features.
        Monetary excludes zero-price transactions.
        """

        df = df.copy()

        # 1. Create TotalPrice
        if 'TotalPrice' not in df.columns:
            df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

        # 2. Flag zero-price transactions (useful for future modeling)
        df['is_free'] = (df['UnitPrice'] == 0).astype(int)

        # 3. Reference date for Recency
        snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

        print("Engineering RFM features...")

        # 🔹 Core RFM Aggregation

        # Recency & Tenure (use full dataset)
        date_agg = df.groupby('CustomerID')['InvoiceDate'].agg(
            Recency=lambda x: (snapshot_date - x.max()).days,
            Tenure=lambda x: (snapshot_date - x.min()).days
        )

        # Frequency (all transactions)
        freq_agg = df.groupby('CustomerID')['InvoiceNo'].nunique().rename('Frequency')

        # Monetary (EXCLUDE zero-price rows)
        monetary_agg = (
            df[df['UnitPrice'] > 0]
            .groupby('CustomerID')['TotalPrice']
            .sum()
            .rename('Monetary')
        )

        # Free transaction ratio (advanced feature)
        free_ratio = df.groupby('CustomerID')['is_free'].mean().rename('FreeTransactionRatio')


        # Combine all features
        rfm = pd.concat([date_agg, freq_agg, monetary_agg, free_ratio], axis=1)

        # Fill missing Monetary (customers with only free items)
        rfm['Monetary'] = rfm['Monetary'].fillna(0)

        # 4. Avg Order Value (safe division)
        rfm['AvgOrderValue'] = np.where(
            rfm['Frequency'] > 0,
            rfm['Monetary'] / rfm['Frequency'],
            0
        )

        # 5. Clean infinities / NaNs
        for col in ['Monetary', 'AvgOrderValue']:
            rfm[col] = rfm[col].replace([np.inf, -np.inf], 0).fillna(0)

        print(f"Feature Engineering complete. Shape: {rfm.shape}")
        return rfm

    def save_engineer(self, filepath="models/feature_engineer.pkl"):
        """Saves the FeatureEngineer instance."""
        joblib.dump(self, filepath)
        print(f"FeatureEngineer saved successfully to {filepath}")
