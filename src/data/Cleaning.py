import pandas as pd


class DataPreprocessor:
    def __init__(self):
        pass

    def clean_data(self, df):
        """
        Initial cleaning pipeline:
        1. Drops exact row duplicates.
        2. Removes records with missing CustomerIDs.
        3. Filters out non-positive Quantities.
        4. Standardizes datetime formats.
        5. Converts IDs to categorical features for memory efficiency.
        """
        initial_count = len(df)

        # 1. Drop Duplicates
        df = df.drop_duplicates()
        after_dup_count = len(df)

        # 2. Handle Missing CustomerIDs
        df = df.dropna(subset=['CustomerID'])
        after_null_count = len(df)

        # 3. Filter Quantity
        df = df[df['Quantity'] > 0]
        after_quant_count = len(df)

        # 4. Standardize InvoiceDate
        if 'InvoiceDate' in df.columns:
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

        # 5. Convert IDs to Categorical
        # We use .astype('category') to save memory and signify these aren't continuous numbers
        if 'CustomerID' in df.columns:
            df['CustomerID'] = df['CustomerID'].astype(int).astype('category')

        if 'InvoiceNo' in df.columns:
            df['InvoiceNo'] = df['InvoiceNo'].astype('category')

        # Log the Cleaning Results
        print(f"\n{'=' * 30}\nCLEANING REPORT\n{'=' * 30}")
        print(f"Initial Rows: {initial_count}")
        print(f"Duplicates Removed: {initial_count - after_dup_count}")
        print(f"Missing Customers Removed: {after_dup_count - after_null_count}")
        print(f"Invalid Quantities Removed: {after_null_count - after_quant_count}")
        print(f"Final Cleaned Rows: {len(df)}")
        print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
        print(f"{'=' * 30}\n")

        return df
