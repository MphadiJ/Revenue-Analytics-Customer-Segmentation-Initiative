import pandas as pd
import os

class DataLoader:
    def __init__(self, data_path):
        """
        Initializes the loader with the root data directory.
        """
        self.data_path = data_path

    def load_and_inspect(self, filename):
        """
        Loads a CSV file and returns a comprehensive health check:
        1. First 5 rows
        2. Dataframe Info
        3. Missing Value Percentages
        4. Statistical Summary
        """
        full_path = os.path.join(self.data_path, 'raw', filename)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The file {filename} was not found at {full_path}")

        # Load the dataframe
        df = pd.read_csv(full_path)

        print(f"{'=' * 30}\nDATA LOADED: {filename}\n{'=' * 30}")

        # 1. First 5 Rows
        print("\n--- FIRST 5 ROWS ---")
        print(df.head())

        # 2. DataFrame Info (Schema and Memory)
        print("\n--- DATAFRAME INFO ---")
        df.info()

        # 3. Missing Value Percentage
        print("\n--- MISSING VALUE PERCENTAGE ---")
        missing_pct = (df.isnull().sum() / len(df)) * 100
        print(missing_pct[missing_pct > 0].sort_values(ascending=False))

        # 4. Statistical Summary
        print("\n--- STATISTICAL SUMMARY (DESCRIBE) ---")
        print(df.describe())

        return df

    def save_processed(self, df, filename):
        """Saves the output to the processed directory."""
        save_path = os.path.join(self.data_path, 'processed', filename)
        df.to_csv(save_path, index=False)
        print(f"\nSuccessfully saved processed data to: {save_path}")
