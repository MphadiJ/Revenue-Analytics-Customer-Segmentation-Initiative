"""
src/kpi_engine.py

KPI Engine — Revenue & Customer Analytics
==========================================
Computes business-level KPIs from RFM + Segment + Churn data.

KPI Groups
----------
1. Revenue KPIs      — Total Revenue, ARPU, AOV, Revenue per Segment
2. Customer KPIs     — Total Customers, Active Rate, Retention proxy, Churn Rate
3. Segment KPIs      — Size, Revenue share, Avg RFM per segment
4. Engagement KPIs   — Avg Frequency, Avg Recency, Avg Tenure
5. Risk KPIs         — High/Medium/Low risk counts & revenue at risk
"""

import pandas as pd
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

ACTIVE_RECENCY_THRESHOLD = 90   # customers last seen within 90 days = "active"


# ─────────────────────────────────────────────────────────────────────────────
# 1. REVENUE KPIs
# ─────────────────────────────────────────────────────────────────────────────

def revenue_kpis(df: pd.DataFrame) -> dict:
    """
    Compute top-level revenue KPIs.

    Required columns: Monetary
    Optional columns: Frequency, AvgOrderValue

    Returns
    -------
    dict with keys:
        total_revenue, arpu, avg_order_value, median_monetary,
        top10_pct_revenue_share, revenue_std
    """
    _require(df, ["Monetary"])

    total        = df["Monetary"].sum()
    n            = len(df)
    arpu         = total / n if n > 0 else 0

    aov = (
        df["AvgOrderValue"].mean()
        if "AvgOrderValue" in df.columns
        else (df["Monetary"] / df["Frequency"].replace(0, np.nan)).mean()
        if "Frequency" in df.columns
        else arpu
    )

    top10_threshold  = df["Monetary"].quantile(0.90)
    top10_revenue    = df.loc[df["Monetary"] >= top10_threshold, "Monetary"].sum()
    top10_pct_share  = (top10_revenue / total * 100) if total > 0 else 0

    return {
        "total_revenue":          round(float(total), 2),
        "arpu":                   round(float(arpu), 2),
        "avg_order_value":        round(float(aov), 2),
        "median_monetary":        round(float(df["Monetary"].median()), 2),
        "top10_pct_revenue_share":round(float(top10_pct_share), 1),
        "revenue_std":            round(float(df["Monetary"].std()), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. CUSTOMER KPIs
# ─────────────────────────────────────────────────────────────────────────────

def customer_kpis(df: pd.DataFrame) -> dict:
    """
    Compute customer base health KPIs.

    Required columns: Recency
    Optional columns: ChurnRisk, Tenure

    Returns
    -------
    dict with keys:
        total_customers, active_customers, active_rate_pct,
        churn_rate_pct, avg_tenure_days, median_recency_days
    """
    _require(df, ["Recency"])

    n         = len(df)
    n_active  = (df["Recency"] <= ACTIVE_RECENCY_THRESHOLD).sum()
    active_rate = (n_active / n * 100) if n > 0 else 0

    churn_rate = (
        ((df["ChurnRisk"] == "High Risk").sum() / n * 100)
        if "ChurnRisk" in df.columns else None
    )

    avg_tenure = (
        df["Tenure"].mean()
        if "Tenure" in df.columns else None
    )

    return {
        "total_customers":     int(n),
        "active_customers":    int(n_active),
        "active_rate_pct":     round(float(active_rate), 1),
        "churn_rate_pct":      round(float(churn_rate), 1) if churn_rate is not None else None,
        "avg_tenure_days":     round(float(avg_tenure), 1) if avg_tenure is not None else None,
        "median_recency_days": round(float(df["Recency"].median()), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. SEGMENT KPIs
# ─────────────────────────────────────────────────────────────────────────────

def segment_kpis(df: pd.DataFrame, segment_col: str = "Segment_Name") -> pd.DataFrame:
    """
    Per-segment KPI summary.

    Required columns: Monetary, Recency, Frequency, <segment_col>

    Returns
    -------
    DataFrame with one row per segment, columns:
        Segment, CustomerCount, CustomerSharePct,
        TotalRevenue, RevenueSharePct, ARPU,
        AvgRecency, AvgFrequency, AvgMonetary,
        HighRiskPct (if ChurnRisk present)
    """
    _require(df, ["Monetary", "Recency", "Frequency", segment_col])

    total_revenue  = df["Monetary"].sum()
    total_customers = len(df)

    agg_dict = {
        "CustomerCount": ("Monetary", "count"),
        "TotalRevenue":  ("Monetary", "sum"),
        "ARPU":          ("Monetary", "mean"),
        "AvgRecency":    ("Recency",  "mean"),
        "AvgFrequency":  ("Frequency","mean"),
        "AvgMonetary":   ("Monetary", "mean"),
    }

    if "ChurnRisk" in df.columns:
        agg_dict["HighRiskPct"] = (
            "ChurnRisk",
            lambda x: round((x == "High Risk").sum() / len(x) * 100, 1)
        )

    summary = df.groupby(segment_col, as_index=False).agg(**agg_dict)
    summary.rename(columns={segment_col: "Segment"}, inplace=True)

    summary["CustomerSharePct"] = (
        summary["CustomerCount"] / total_customers * 100
    ).round(1)

    summary["RevenueSharePct"] = (
        summary["TotalRevenue"] / total_revenue * 100
        if total_revenue > 0 else 0
    ).round(1)

    for col in ["TotalRevenue", "ARPU", "AvgRecency", "AvgFrequency", "AvgMonetary"]:
        summary[col] = summary[col].round(2)

    return summary.sort_values("TotalRevenue", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENGAGEMENT KPIs
# ─────────────────────────────────────────────────────────────────────────────

def engagement_kpis(df: pd.DataFrame) -> dict:
    """
    Customer engagement / activity KPIs.

    Required columns: Recency, Frequency
    Optional columns: Tenure, AvgOrderValue

    Returns
    -------
    dict with keys:
        avg_recency, avg_frequency, avg_order_value,
        pct_single_purchase, pct_high_frequency (freq > 10)
    """
    _require(df, ["Recency", "Frequency"])

    n = len(df)
    pct_single = (df["Frequency"] == 1).sum() / n * 100 if n > 0 else 0
    pct_high_f = (df["Frequency"] > 10).sum() / n * 100 if n > 0 else 0

    aov = (
        df["AvgOrderValue"].mean()
        if "AvgOrderValue" in df.columns else None
    )

    return {
        "avg_recency":          round(float(df["Recency"].mean()), 1),
        "avg_frequency":        round(float(df["Frequency"].mean()), 2),
        "avg_order_value":      round(float(aov), 2) if aov is not None else None,
        "pct_single_purchase":  round(float(pct_single), 1),
        "pct_high_frequency":   round(float(pct_high_f), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. RISK KPIs
# ─────────────────────────────────────────────────────────────────────────────

def risk_kpis(df: pd.DataFrame) -> dict:
    """
    Revenue-at-risk and churn risk KPIs.

    Required columns: ChurnRisk, Monetary

    Returns
    -------
    dict with keys:
        n_high_risk, n_medium_risk, n_low_risk,
        revenue_at_risk, revenue_at_risk_pct,
        high_risk_pct, medium_risk_pct, low_risk_pct
    """
    _require(df, ["ChurnRisk", "Monetary"])

    n = len(df)

    n_high   = (df["ChurnRisk"] == "High Risk").sum()
    n_medium = (df["ChurnRisk"] == "Medium Risk").sum()
    n_low    = (df["ChurnRisk"] == "Low Risk").sum()

    rev_at_risk     = df.loc[df["ChurnRisk"] == "High Risk", "Monetary"].sum()
    total_revenue   = df["Monetary"].sum()
    rev_at_risk_pct = (rev_at_risk / total_revenue * 100) if total_revenue > 0 else 0

    return {
        "n_high_risk":          int(n_high),
        "n_medium_risk":        int(n_medium),
        "n_low_risk":           int(n_low),
        "revenue_at_risk":      round(float(rev_at_risk), 2),
        "revenue_at_risk_pct":  round(float(rev_at_risk_pct), 1),
        "high_risk_pct":        round(float(n_high  / n * 100), 1) if n > 0 else 0,
        "medium_risk_pct":      round(float(n_medium / n * 100), 1) if n > 0 else 0,
        "low_risk_pct":         round(float(n_low   / n * 100), 1) if n > 0 else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. FULL KPI REPORT  (convenience wrapper)
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_kpis(
    df: pd.DataFrame,
    segment_col: str = "Segment_Name",
) -> dict:
    """
    Run all KPI groups and return a single nested dict.

    Keys: revenue, customer, engagement, segment (DataFrame), risk (if ChurnRisk present)

    Example
    -------
    >>> kpis = compute_all_kpis(df)
    >>> kpis["revenue"]["total_revenue"]
    >>> kpis["segment"]  # DataFrame
    """
    result = {
        "revenue":    revenue_kpis(df),
        "customer":   customer_kpis(df),
        "engagement": engagement_kpis(df),
    }

    if segment_col in df.columns:
        result["segment"] = segment_kpis(df, segment_col)

    if "ChurnRisk" in df.columns and "Monetary" in df.columns:
        result["risk"] = risk_kpis(df)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def _require(df: pd.DataFrame, cols: list) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"KPI engine: missing required columns: {missing}")
