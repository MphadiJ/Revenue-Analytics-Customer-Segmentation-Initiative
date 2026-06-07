"""
src/churn/churn_analysis.py

RFM-based churn analysis module.
Plugs directly into the existing segmentation pipeline — no new model training required.

Logic:
    - High Recency  (not purchased recently)  → high churn signal  (50% weight)
    - Low Frequency (rarely purchases)         → high churn signal  (30% weight)
    - Low Monetary  (low total spend)          → high churn signal  (20% weight)

Churn Risk Bands:
    High Risk   : score >= 0.65
    Medium Risk : score  0.35 – 0.64
    Low Risk    : score <  0.35
"""

import pandas as pd
import numpy as np

# ── Weights ──────────────────────────────────────────────────────────────────
CHURN_WEIGHTS = {
    "Recency":   0.50,   # strongest signal — days since last purchase
    "Frequency": 0.30,   # purchase count
    "Monetary":  0.20,   # total spend
}

# ── Risk thresholds ───────────────────────────────────────────────────────────
HIGH_RISK_THRESHOLD   = 0.65
MEDIUM_RISK_THRESHOLD = 0.35

# ── Single-customer absolute thresholds (used when only 1 row, no population) ─
_RECENCY_BANDS   = [(30, 0.10), (90, 0.30), (180, 0.55), (365, 0.80), (float("inf"), 1.00)]
_FREQUENCY_BANDS = [(1, 1.00), (3, 0.70), (10, 0.45), (25, 0.20), (float("inf"), 0.10)]
_MONETARY_BANDS  = [(100, 1.00), (500, 0.70), (2000, 0.45), (5000, 0.20), (float("inf"), 0.10)]


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _min_max_norm(series: pd.Series) -> pd.Series:
    """Normalise a series to [0, 1]. Returns 0.5 if all values are identical."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def _band_lookup(value: float, bands: list) -> float:
    """Map a scalar value to a score using ordered threshold bands."""
    for threshold, score in bands:
        if value <= threshold:
            return score
    return bands[-1][1]


# ─────────────────────────────────────────────────────────────────────────────
# BATCH SCORING  (for uploaded CSV — population-normalised)
# ─────────────────────────────────────────────────────────────────────────────

def compute_churn_scores(df: pd.DataFrame) -> pd.Series:
    """
    Compute a churn risk score in [0, 1] for every row in df.
    Requires columns: Recency, Frequency, Monetary.
    Higher score = higher churn risk.
    """
    required = ["Recency", "Frequency", "Monetary"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for churn scoring: {missing}")

    rfm = df[required].copy().astype(float)

    recency_norm   = _min_max_norm(rfm["Recency"])            # high = high risk
    frequency_norm = 1 - _min_max_norm(rfm["Frequency"])      # low  = high risk
    monetary_norm  = 1 - _min_max_norm(rfm["Monetary"])       # low  = high risk

    score = (
        CHURN_WEIGHTS["Recency"]   * recency_norm   +
        CHURN_WEIGHTS["Frequency"] * frequency_norm +
        CHURN_WEIGHTS["Monetary"]  * monetary_norm
    )
    return score.round(4)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-CUSTOMER SCORING  (manual input — absolute rule-based thresholds)
# ─────────────────────────────────────────────────────────────────────────────

def compute_single_customer_churn_score(
    recency: float,
    frequency: float,
    monetary: float,
) -> float:
    """
    Compute a churn risk score for a single customer using absolute thresholds.
    Returns a float in [0, 1].
    """
    r = _band_lookup(recency,   _RECENCY_BANDS)
    f = _band_lookup(frequency, _FREQUENCY_BANDS)
    m = _band_lookup(monetary,  _MONETARY_BANDS)

    score = (
        CHURN_WEIGHTS["Recency"]   * r +
        CHURN_WEIGHTS["Frequency"] * f +
        CHURN_WEIGHTS["Monetary"]  * m
    )
    return round(float(score), 4)


# ─────────────────────────────────────────────────────────────────────────────
# RISK CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_churn_risk(score: float) -> str:
    """Map a churn score to a risk label."""
    if score >= HIGH_RISK_THRESHOLD:
        return "High Risk"
    elif score >= MEDIUM_RISK_THRESHOLD:
        return "Medium Risk"
    return "Low Risk"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def add_churn_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ChurnScore and ChurnRisk columns to a customer-level DataFrame.

    Args:
        df: Customer-level DataFrame with at least Recency, Frequency, Monetary.

    Returns:
        df with two new columns:
            ChurnScore  float [0, 1]  — higher = higher churn risk
            ChurnRisk   str           — 'High Risk' | 'Medium Risk' | 'Low Risk'
    """
    df = df.copy()
    df["ChurnScore"] = compute_churn_scores(df)
    df["ChurnRisk"]  = df["ChurnScore"].apply(classify_churn_risk)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SEGMENT-LEVEL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def segment_churn_summary(
    df: pd.DataFrame,
    segment_col: str = "Segment_Name",
) -> pd.DataFrame:
    """
    Aggregate churn metrics per segment.

    Returns a DataFrame with columns:
        Segment_Name, CustomerCount, AvgChurnScore,
        HighRiskCount, MediumRiskCount, LowRiskCount, HighRiskPct
    """
    if segment_col not in df.columns:
        raise ValueError(f"Column '{segment_col}' not found. Run segmentation first.")
    if "ChurnScore" not in df.columns or "ChurnRisk" not in df.columns:
        raise ValueError("Run add_churn_analysis(df) before calling segment_churn_summary().")

    summary = (
        df.groupby(segment_col, as_index=False)
        .agg(
            CustomerCount  = ("ChurnScore", "count"),
            AvgChurnScore  = ("ChurnScore", "mean"),
            HighRiskCount  = ("ChurnRisk",  lambda x: (x == "High Risk").sum()),
            MediumRiskCount= ("ChurnRisk",  lambda x: (x == "Medium Risk").sum()),
            LowRiskCount   = ("ChurnRisk",  lambda x: (x == "Low Risk").sum()),
        )
    )
    summary["AvgChurnScore"] = summary["AvgChurnScore"].round(4)
    summary["HighRiskPct"]   = (
        summary["HighRiskCount"] / summary["CustomerCount"] * 100
    ).round(1)

    return summary.sort_values("AvgChurnScore", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# CHURN SCORE EXPLANATION  (for UI tooltip / detail view)
# ─────────────────────────────────────────────────────────────────────────────

def explain_churn_score(recency: float, frequency: float, monetary: float) -> dict:
    """
    Return a breakdown of what is driving the churn score for a single customer.
    Useful for the Streamlit single-customer detail panel.
    """
    r = _band_lookup(recency,   _RECENCY_BANDS)
    f = _band_lookup(frequency, _FREQUENCY_BANDS)
    m = _band_lookup(monetary,  _MONETARY_BANDS)

    total = round(
        CHURN_WEIGHTS["Recency"]   * r +
        CHURN_WEIGHTS["Frequency"] * f +
        CHURN_WEIGHTS["Monetary"]  * m,
        4,
    )

    return {
        "total_score":        total,
        "risk_label":         classify_churn_risk(total),
        "recency_component":  round(CHURN_WEIGHTS["Recency"]   * r, 4),
        "frequency_component":round(CHURN_WEIGHTS["Frequency"] * f, 4),
        "monetary_component": round(CHURN_WEIGHTS["Monetary"]  * m, 4),
        "recency_raw":        recency,
        "frequency_raw":      frequency,
        "monetary_raw":       monetary,
    }
