"""
Aggregation and analysis functions.

Takes a cleaned earthquake DataFrame and produces analytical outputs:
  - Time-series rollups (daily, weekly, monthly counts and avg magnitudes)
  - Magnitude distribution (counts per tier and per integer bucket)
  - Depth distribution (shallow/intermediate/deep breakdown)
  - Regional rankings (top countries by activity)
  - Historical baseline comparison (current period vs historical norm)

All functions are pure: they take a DataFrame and return a DataFrame.
No side effects, no plotting, no Streamlit dependencies.
"""

import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime, timedelta


# ============================================================
# TIME-SERIES ROLLUPS
# ============================================================

def daily_activity(df: pd.DataFrame) -> pd.DataFrame:
    """Count of events and avg magnitude per day. Useful for line charts."""
    if df.empty:
        return pd.DataFrame(columns=["date", "event_count", "avg_magnitude", "max_magnitude"])

    grouped = df.groupby("date").agg(
        event_count=("id", "count"),
        avg_magnitude=("magnitude", "mean"),
        max_magnitude=("magnitude", "max"),
    ).reset_index()

    grouped["avg_magnitude"] = grouped["avg_magnitude"].round(2)
    grouped["max_magnitude"] = grouped["max_magnitude"].round(2)
    return grouped.sort_values("date")


def weekly_activity(df: pd.DataFrame) -> pd.DataFrame:
    """Count of events and avg magnitude per ISO week. Useful for trend charts."""
    if df.empty:
        return pd.DataFrame(columns=["year_week", "event_count", "avg_magnitude", "max_magnitude"])

    grouped = df.groupby("year_week").agg(
        event_count=("id", "count"),
        avg_magnitude=("magnitude", "mean"),
        max_magnitude=("magnitude", "max"),
    ).reset_index()

    grouped["avg_magnitude"] = grouped["avg_magnitude"].round(2)
    grouped["max_magnitude"] = grouped["max_magnitude"].round(2)
    return grouped.sort_values("year_week")


def rolling_average_magnitude(df: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """
    Daily counts and a 7-day rolling average of magnitude.
    Smooths short-term noise — useful for spotting genuine trends.
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "event_count", "avg_magnitude", "rolling_avg"])

    daily = daily_activity(df)
    daily["rolling_avg"] = (
        daily["avg_magnitude"]
        .rolling(window=window_days, min_periods=1)
        .mean()
        .round(2)
    )
    return daily


# ============================================================
# MAGNITUDE DISTRIBUTION
# ============================================================

def magnitude_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count and percentage of events per risk tier.
    Returns rows in canonical tier order (Minor → Great).
    """
    if df.empty:
        return pd.DataFrame(columns=["risk_tier", "event_count", "percentage"])

    tier_order = ["Minor", "Light", "Moderate", "Strong", "Major", "Great"]
    counts = df["risk_tier"].value_counts().reindex(tier_order, fill_value=0)
    total = counts.sum()

    result = pd.DataFrame({
        "risk_tier": counts.index,
        "event_count": counts.values,
        "percentage": (counts.values / total * 100).round(1) if total > 0 else 0,
    })
    return result


def magnitude_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count of events per integer magnitude bucket (M2-3, M3-4, ..., M8+).
    Useful for histogram-style charts showing the Gutenberg-Richter pattern.
    """
    if df.empty:
        return pd.DataFrame(columns=["magnitude_bucket", "event_count"])

    df = df.copy()
    df["magnitude_bucket"] = df["magnitude"].apply(_bucket_magnitude)
    bucket_order = ["M0-1", "M1-2", "M2-3", "M3-4", "M4-5", "M5-6", "M6-7", "M7-8", "M8+"]
    counts = df["magnitude_bucket"].value_counts().reindex(bucket_order, fill_value=0)

    return pd.DataFrame({
        "magnitude_bucket": counts.index,
        "event_count": counts.values,
    })


def _bucket_magnitude(mag: float) -> str:
    if pd.isna(mag):
        return "Unknown"
    if mag >= 8:
        return "M8+"
    return f"M{int(mag)}-{int(mag) + 1}"


# ============================================================
# DEPTH DISTRIBUTION
# ============================================================

def depth_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of events per depth category."""
    if df.empty:
        return pd.DataFrame(columns=["depth_category", "event_count", "percentage"])

    category_order = ["Shallow", "Intermediate", "Deep"]
    counts = df["depth_category"].value_counts().reindex(category_order, fill_value=0)
    total = counts.sum()

    return pd.DataFrame({
        "depth_category": counts.index,
        "event_count": counts.values,
        "percentage": (counts.values / total * 100).round(1) if total > 0 else 0,
    })


def depth_by_region(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Per-country breakdown of shallow vs intermediate vs deep events.
    Useful for the depth analysis section — answers
    "is this region dominated by shallow or deep quakes?"
    """
    if df.empty:
        return pd.DataFrame()

    top_countries = df["country_name"].value_counts().head(top_n).index
    filtered = df[df["country_name"].isin(top_countries)]

    pivot = pd.crosstab(filtered["country_name"], filtered["depth_category"])
    # Ensure all three columns exist even if missing in the data
    for col in ["Shallow", "Intermediate", "Deep"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["Shallow", "Intermediate", "Deep"]]
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False).reset_index()
    return pivot


# ============================================================
# REGIONAL RANKINGS
# ============================================================

def top_countries(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Top N countries ranked by event count, with magnitude and significance summaries.
    This is the core data behind the Top 10 risk report.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "country_name", "event_count", "avg_magnitude",
            "max_magnitude", "avg_significance", "shallow_pct"
        ])

    grouped = df.groupby("country_name").agg(
        event_count=("id", "count"),
        avg_magnitude=("magnitude", "mean"),
        max_magnitude=("magnitude", "max"),
        avg_significance=("significance", "mean"),
        shallow_count=("depth_category", lambda x: (x == "Shallow").sum()),
    ).reset_index()

    grouped["avg_magnitude"] = grouped["avg_magnitude"].round(2)
    grouped["max_magnitude"] = grouped["max_magnitude"].round(2)
    grouped["avg_significance"] = grouped["avg_significance"].round(0).astype(int)
    grouped["shallow_pct"] = (grouped["shallow_count"] / grouped["event_count"] * 100).round(0).astype(int)
    grouped = grouped.drop(columns=["shallow_count"])

    return grouped.sort_values("event_count", ascending=False).head(top_n).reset_index(drop=True)


def continent_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Continent-level rollup. Useful for high-level KPIs."""
    if df.empty:
        return pd.DataFrame(columns=["continent", "event_count", "avg_magnitude"])

    grouped = df.groupby("continent").agg(
        event_count=("id", "count"),
        avg_magnitude=("magnitude", "mean"),
    ).reset_index()

    grouped["avg_magnitude"] = grouped["avg_magnitude"].round(2)
    return grouped.sort_values("event_count", ascending=False)


# ============================================================
# HISTORICAL BASELINE COMPARISON
# ============================================================

def compare_to_baseline(
    current_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    current_window_days: int,
    baseline_window_days: int,
) -> pd.DataFrame:
    """
    Compare current-period activity to historical baseline, per country.

    Both DataFrames should already be cleaned. The function:
      1. Counts events per country in each window
      2. Normalizes both to "events per day" (so windows of different lengths compare fairly)
      3. Computes the % deviation from baseline
      4. Flags countries with anomalous activity (|deviation| > 50%)

    Parameters
    ----------
    current_df : pd.DataFrame
        Cleaned earthquakes for the current window.
    baseline_df : pd.DataFrame
        Cleaned earthquakes for the historical baseline window.
    current_window_days : int
        Number of days covered by current_df (for rate normalization).
    baseline_window_days : int
        Number of days covered by baseline_df.

    Returns
    -------
    pd.DataFrame with columns:
      country_name, current_count, current_per_day,
      baseline_per_day, deviation_pct, anomaly_flag
    """
    if current_df.empty:
        return pd.DataFrame(columns=[
            "country_name", "current_count", "current_per_day",
            "baseline_per_day", "deviation_pct", "anomaly_flag"
        ])

    current_counts = current_df["country_name"].value_counts()
    baseline_counts = baseline_df["country_name"].value_counts() if not baseline_df.empty else pd.Series(dtype=int)

    rows = []
    for country, curr_count in current_counts.items():
        current_per_day = curr_count / max(current_window_days, 1)
        baseline_count = baseline_counts.get(country, 0)
        baseline_per_day = baseline_count / max(baseline_window_days, 1)

        if baseline_per_day > 0:
            deviation = (current_per_day - baseline_per_day) / baseline_per_day * 100
        else:
            # No historical data for this country — can't compute deviation meaningfully.
            # Mark as None rather than infinity.
            deviation = None

        if deviation is None:
            anomaly = "No baseline"
        elif abs(deviation) >= 50:
            anomaly = "Anomalous (above)" if deviation > 0 else "Anomalous (below)"
        else:
            anomaly = "Normal"

        rows.append({
            "country_name": country,
            "current_count": int(curr_count),
            "current_per_day": round(current_per_day, 2),
            "baseline_per_day": round(baseline_per_day, 2),
            "deviation_pct": round(deviation, 1) if deviation is not None else None,
            "anomaly_flag": anomaly,
        })

    return pd.DataFrame(rows).sort_values("current_count", ascending=False).reset_index(drop=True)


# ============================================================
# NOTABLE EVENTS DETECTION
# ============================================================

def find_largest_event(df: pd.DataFrame) -> Optional[dict]:
    """Return the single highest-magnitude event as a dict, or None if empty."""
    if df.empty:
        return None
    row = df.loc[df["magnitude"].idxmax()]
    return {
        "magnitude": float(row["magnitude"]),
        "place": row["place"],
        "country": row["country_name"],
        "depth_km": float(row["depth_km"]),
        "datetime_utc": row["datetime_utc"],
        "felt_reports": int(row["felt_reports"]) if pd.notna(row["felt_reports"]) else 0,
        "tsunami": int(row["tsunami"]),
        "url": row["url"],
    }


def find_tsunami_events(df: pd.DataFrame) -> pd.DataFrame:
    """All events flagged with a tsunami warning."""
    if df.empty:
        return pd.DataFrame()
    return df[df["tsunami"] == 1].copy()


def find_high_felt_events(df: pd.DataFrame, min_reports: int = 100) -> pd.DataFrame:
    """Events that generated significant public 'did you feel it' reports."""
    if df.empty:
        return pd.DataFrame()
    return df[df["felt_reports"].fillna(0) >= min_reports].copy()