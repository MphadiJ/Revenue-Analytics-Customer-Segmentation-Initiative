 import pandas as pd
import numpy as np
from pathlib import Path
 
 
# DATA LOADING & CLEANING
 
def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load and clean raw retail transaction CSV.
    Removes cancellations, null customers, and recomputes TotalPrice.
    """
    df = pd.read_csv(filepath)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], infer_datetime_format=True)
    df = df.dropna(subset=["CustomerID", "InvoiceDate"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    df["TotalPrice"]  = df["Quantity"] * df["UnitPrice"]
    df["YearMonth"]   = df["InvoiceDate"].dt.to_period("M")
    df["Year"]        = df["InvoiceDate"].dt.year
    df["Month"]       = df["InvoiceDate"].dt.month
    df["DayOfWeek"]   = df["InvoiceDate"].dt.day_name()
    df["Date"]        = df["InvoiceDate"].dt.date
    df["CustomerID"]  = df["CustomerID"].astype(int).astype(str)
    return df
 
 
def apply_filters(df, start_date=None, end_date=None, countries=None):
    mask = pd.Series(True, index=df.index)
    if start_date:
        mask &= df["InvoiceDate"].dt.date >= start_date
    if end_date:
        mask &= df["InvoiceDate"].dt.date <= end_date
    if countries:
        mask &= df["Country"].isin(countries)
    return df[mask].copy()
 

# RFM FEATURE ENGINEERING  (feeds directly into InferencePipeline)
 
def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RFM features from raw transactions.
    Produces: CustomerID, Recency, Frequency, Monetary, Tenure, AvgOrderValue
    — exactly the columns expected by InferencePipeline._preprocess()
    """
    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
 
    rfm = (
        df.groupby("CustomerID")
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
 
 
# TOP-LINE KPIs
 
def compute_top_kpis(df: pd.DataFrame, df_prev: pd.DataFrame | None = None) -> dict:
    orders   = df.groupby("InvoiceNo")["TotalPrice"].sum()
    customers = df.groupby("CustomerID")["InvoiceNo"].nunique()
 
    total_revenue        = df["TotalPrice"].sum()
    total_orders         = df["InvoiceNo"].nunique()
    unique_customers     = df["CustomerID"].nunique()
    avg_order_value      = orders.mean()
    avg_rev_per_cust     = total_revenue / unique_customers if unique_customers else 0
    units_sold           = int(df["Quantity"].sum())
    repeat_rate          = (customers > 1).sum() / unique_customers * 100 if unique_customers else 0
 
    def delta(current, prev_df, fn):
        if prev_df is None or prev_df.empty:
            return None
        pv = fn(prev_df)
        return ((current - pv) / pv * 100) if pv else None
 
    return {
        "total_revenue":           {"value": total_revenue,    "delta": delta(total_revenue,    df_prev, lambda d: d["TotalPrice"].sum())},
        "total_orders":            {"value": total_orders,     "delta": delta(total_orders,     df_prev, lambda d: d["InvoiceNo"].nunique())},
        "unique_customers":        {"value": unique_customers, "delta": delta(unique_customers, df_prev, lambda d: d["CustomerID"].nunique())},
        "avg_order_value":         {"value": avg_order_value,  "delta": delta(avg_order_value,  df_prev, lambda d: d.groupby("InvoiceNo")["TotalPrice"].sum().mean())},
        "avg_revenue_per_customer":{"value": avg_rev_per_cust, "delta": None},
        "units_sold":              {"value": units_sold,       "delta": None},
        "repeat_customer_rate":    {"value": repeat_rate,      "delta": None},
    }
 

# REVENUE ANALYTICS
 
def monthly_revenue_trend(df):
    monthly = (
        df.groupby("YearMonth")
        .agg(Revenue=("TotalPrice","sum"), Orders=("InvoiceNo","nunique"))
        .reset_index()
    )
    monthly["YearMonth"]  = monthly["YearMonth"].astype(str)
    monthly["MoM_Growth"] = monthly["Revenue"].pct_change() * 100
    return monthly
 
 
def revenue_by_country(df, top_n=10):
    return (
        df.groupby("Country")
        .agg(Revenue=("TotalPrice","sum"), Orders=("InvoiceNo","nunique"))
        .reset_index()
        .sort_values("Revenue", ascending=False)
        .head(top_n)
    )
 
 
def revenue_heatmap(df):
    pivot = (
        df.groupby(["Year","Month"])["TotalPrice"]
        .sum().unstack(fill_value=0)
    )
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = month_names[:len(pivot.columns)]
    return pivot.reset_index()
 
 
def orders_by_day_of_week(df):
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    return (
        df.groupby("DayOfWeek")
        .agg(Orders=("InvoiceNo","nunique"), Revenue=("TotalPrice","sum"))
        .reindex(day_order).reset_index()
    )
 

# PRODUCT ANALYTICS
 
def top_products_by_revenue(df, top_n=10):
    return (
        df.groupby("Description")
        .agg(Revenue=("TotalPrice","sum"), UnitsSold=("Quantity","sum"))
        .reset_index().sort_values("Revenue", ascending=False).head(top_n)
    )
 
 
def top_products_by_volume(df, top_n=10):
    return (
        df.groupby("Description")
        .agg(UnitsSold=("Quantity","sum"), Revenue=("TotalPrice","sum"))
        .reset_index().sort_values("UnitsSold", ascending=False).head(top_n)
    )
 
 
# CUSTOMER ANALYTICS
 
def top_customers_by_revenue(df, top_n=10):
    return (
        df.groupby("CustomerID")
        .agg(Revenue=("TotalPrice","sum"), Orders=("InvoiceNo","nunique"))
        .reset_index().sort_values("Revenue", ascending=False).head(top_n)
    )
 
 
def customer_order_frequency_distribution(df):
    freq = df.groupby("CustomerID")["InvoiceNo"].nunique().reset_index()
    freq.columns = ["CustomerID","OrderCount"]
    freq["Bucket"] = pd.cut(
        freq["OrderCount"],
        bins=[0,1,2,5,10,20,50,999],
        labels=["1","2","3–5","6–10","11–20","21–50","50+"],
    )
    return freq.groupby("Bucket", observed=True)["CustomerID"].count().reset_index(name="CustomerCount")
 
 
def revenue_by_new_vs_returning(df):
    first_order = df.groupby("CustomerID")["InvoiceDate"].min().reset_index()
    first_order.columns = ["CustomerID","FirstOrderDate"]
    df2 = df.merge(first_order, on="CustomerID")
    df2["CustomerType"] = np.where(df2["InvoiceDate"] == df2["FirstOrderDate"],"New","Returning")
    return (
        df2.groupby("CustomerType")
        .agg(Revenue=("TotalPrice","sum"), Customers=("CustomerID","nunique"))
        .reset_index()
    )
