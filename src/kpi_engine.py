import pandas as pd
import numpy as np
from pathlib import Path

# --- DATA LOADING & CLEANING ---

def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load and clean raw retail transaction CSV.
    Removes cancellations, null customers, and recomputes TotalPrice.
    Ensures YearMonth is built immediately for time-series grouping stability.
    """
    df = pd.read_csv(filepath)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df.dropna(subset=["CustomerID", "InvoiceDate"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    
    # Feature Engineering
    df["TotalPrice"]  = df["Quantity"] * df["UnitPrice"]
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["Year"]        = df["InvoiceDate"].dt.year
    df["Month"]       = df["InvoiceDate"].dt.month
    df["DayOfWeek"]   = df["InvoiceDate"].dt.day_name()
    df["Date"]        = df["InvoiceDate"].dt.date  # Native datetime.date objects
    df["CustomerID"]  = df["CustomerID"].astype(int).astype(str)
    return df


def apply_filters(df: pd.DataFrame, start_date=None, end_date=None, countries=None) -> pd.DataFrame:
    """
    Optimized to utilize pre-calculated 'Date' column to prevent repeated datetime conversion overhead.
    """
    mask = pd.Series(True, index=df.index)
    if start_date:
        mask &= df["Date"] >= start_date
    if end_date:
        mask &= df["Date"] <= end_date
    if countries:
        mask &= df["Country"].isin(countries)
    return df[mask].copy()


# --- RFM FEATURE ENGINEERING ---

def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RFM features from raw transactions.
    Produces: CustomerID, Recency, Frequency, Monetary, Tenure, AvgOrderValue
    """
    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerID", observed=True)
        .agg(
            Recency    =("InvoiceDate", lambda x: (reference_date - x.max()).days),
            Frequency  =("InvoiceNo",   "nunique"),
            Monetary   =("TotalPrice",  "sum"),
            FirstOrder =("InvoiceDate", "min"),
        )
        .reset_index()
    )
    rfm["Tenure"]        = (reference_date - rfm["FirstOrder"]).dt.days
    rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]
    rfm = rfm.drop(columns=["FirstOrder"])
    return rfm


# --- TOP-LINE KPIs ---

def compute_top_kpis(df: pd.DataFrame, df_prev: pd.DataFrame | None = None) -> dict:
    orders   = df.groupby("InvoiceNo")["TotalPrice"].sum()
    customers = df.groupby("CustomerID")["InvoiceNo"].nunique()

    total_revenue        = df["TotalPrice"].sum()
    total_orders         = df["InvoiceNo"].nunique()
    unique_customers     = df["CustomerID"].nunique()
    avg_order_value      = orders.mean() if not orders.empty else 0
    avg_rev_per_cust      = total_revenue / unique_customers if unique_customers else 0
    units_sold           = int(df["Quantity"].sum())
    repeat_rate          = (customers > 1).sum() / unique_customers * 100 if unique_customers else 0

    def delta(current, prev_df, fn):
        if prev_df is None or prev_df.empty:
            return None
        pv = fn(prev_df)
        return ((current - pv) / pv * 100) if pv else None

    return {
        "total_revenue":            {"value": total_revenue,    "delta": delta(total_revenue,    df_prev, lambda d: d["TotalPrice"].sum())},
        "total_orders":             {"value": total_orders,     "delta": delta(total_orders,     df_prev, lambda d: d["InvoiceNo"].nunique())},
        "unique_customers":         {"value": unique_customers, "delta": delta(unique_customers, df_prev, lambda d: d["CustomerID"].nunique())},
        "avg_order_value":          {"value": avg_order_value,  "delta": delta(avg_order_value,  df_prev, lambda d: d.groupby("InvoiceNo")["TotalPrice"].sum().mean())},
        "avg_revenue_per_customer": {"value": avg_rev_per_cust, "delta": None},
        "units_sold":               {"value": units_sold,       "delta": None},
        "repeat_customer_rate":     {"value": repeat_rate,      "delta": None},
    }


# --- REVENUE ANALYTICS ---

def monthly_revenue_trend(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.groupby("YearMonth", observed=True)
        .agg(Revenue=("TotalPrice","sum"), Orders=("InvoiceNo","nunique"))
        .reset_index()
    )
    monthly["MoM_Growth"] = monthly["Revenue"].pct_change() * 100
    return monthly


def revenue_by_country(df: pd.DataFrame, top_n=10) -> pd.DataFrame:
    return (
        df.groupby("Country", observed=True)
        .agg(Revenue=("TotalPrice","sum"), Orders=("InvoiceNo","nunique"))
        .reset_index()
        .sort_values("Revenue", ascending=False)
        .head(top_n)
    )


def revenue_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses logical reindexing and explicit observed checks to guarantee time-series safety.
    """
    pivot = (
        df.groupby(["Year","Month"], observed=True)["TotalPrice"]
        .sum().unstack(fill_value=0)
    )
    
    # Reindex columns to map month names accurately
    month_map = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 
                 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}
    
    pivot = pivot.reindex(columns=range(1, 13), fill_value=0)
    pivot.columns = [month_map[m] for m in pivot.columns]
    return pivot.reset_index()


def orders_by_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    return (
        df.groupby("DayOfWeek", observed=True)
        .agg(Orders=("InvoiceNo","nunique"), Revenue=("TotalPrice","sum"))
        .reset_index()
        .set_index("DayOfWeek")
        .reindex(day_order, fill_value=0)
        .reset_index()
    )


# --- PRODUCT ANALYTICS ---

def top_products_by_revenue(df: pd.DataFrame, top_n=10) -> pd.DataFrame:
    return (
        df.groupby("Description", observed=True)
        .agg(Revenue=("TotalPrice","sum"), UnitsSold=("Quantity","sum"))
        .reset_index().sort_values("Revenue", ascending=False).head(top_n)
    )


def top_products_by_volume(df: pd.DataFrame, top_n=10) -> pd.DataFrame:
    return (
        df.groupby("Description", observed=True)
        .agg(UnitsSold=("Quantity","sum"), Revenue=("TotalPrice","sum"))
        .reset_index().sort_values("UnitsSold", ascending=False).head(top_n)
    )


# --- CUSTOMER ANALYTICS ---

def top_customers_by_revenue(df: pd.DataFrame, top_n=10) -> pd.DataFrame:
    return (
        df.groupby("CustomerID", observed=True)
        .agg(Revenue=("TotalPrice","sum"), Orders=("InvoiceNo","nunique"))
        .reset_index().sort_values("Revenue", ascending=False).head(top_n)
    )


def customer_order_frequency_distribution(df: pd.DataFrame) -> pd.DataFrame:
    freq = df.groupby("CustomerID", observed=True)["InvoiceNo"].nunique().reset_index()
    freq.columns = ["CustomerID","OrderCount"]
    freq["Bucket"] = pd.cut(
        freq["OrderCount"],
        bins=[0,1,2,5,10,20,50,99999],
        labels=["1","2","3–5","6–10","11–20","21–50","50+"],
    )
    res = freq.groupby("Bucket", observed=True)["CustomerID"].count().reset_index(name="CustomerCount")
    res["CustomerCount"] = res["CustomerCount"].astype(int)
    return res


def revenue_by_new_vs_returning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses the exact first InvoiceNo identifier per customer group 
    to robustly segment Cohort classes instead of checking identical timestamps.
    """
    first_orders = df.groupby("CustomerID")["InvoiceNo"].transform("min")
    
    df_cohort = df.copy()
    df_cohort["CustomerType"] = np.where(df_cohort["InvoiceNo"] == first_orders, "New", "Returning")
    
    return (
        df_cohort.groupby("CustomerType", observed=True)
        .agg(Revenue=("TotalPrice","sum"), Customers=("CustomerID","nunique"))
        .reset_index()
    )
