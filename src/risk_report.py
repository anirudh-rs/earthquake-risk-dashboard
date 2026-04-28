"""
Risk scoring and narrative generation.

Two main responsibilities:

1. score_regions(): compute a composite risk score per country and assign
   plain-English risk ratings (Low/Moderate/High/Severe).

2. generate_notable_events_narrative(): auto-write a short bulleted summary
   of the most newsworthy events and patterns in the current data — the
   "human-readable" headline that turns the dashboard into a brief.

These are intentionally rules-based (not ML). For risk reporting, transparency
beats sophistication: a stakeholder needs to see exactly why a region was
flagged, not trust a black-box score.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from src import analytics


# Risk rating thresholds — composite score is on a 0-100 scale.
# Calibrated against typical global activity at M4.5+ filtering.
RISK_THRESHOLDS = [
    (0, 25, "Low"),
    (25, 50, "Moderate"),
    (50, 75, "High"),
    (75, 101, "Severe"),
]


# ============================================================
# RISK SCORING
# ============================================================

def score_regions(
    df: pd.DataFrame,
    baseline_df: Optional[pd.DataFrame] = None,
    current_window_days: int = 30,
    baseline_window_days: int = 90,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Compute a composite risk score per country and rank the top N.

    The score blends four inputs:
      - Event count (weight 30)         — raw activity volume
      - Max magnitude reached (weight 30) — single-event severity
      - Average significance (weight 20)  — USGS's own impact-weighted score
      - Shallow event % (weight 20)       — depth-weighted damage potential

    Each input is normalized to 0-1 across the dataset, weighted, summed,
    and rescaled to 0-100. This is a transparent, defensible composite —
    every input can be explained to a stakeholder in one sentence.

    If baseline_df is provided, an anomaly flag is added.

    Returns one row per region with all components and a final risk_rating.
    """
    if df.empty:
        return _empty_risk_report()

    # Start from the top countries summary
    base = analytics.top_countries(df, top_n=top_n)
    if base.empty:
        return _empty_risk_report()

    # Normalize each component to 0-1
    def normalize(series: pd.Series) -> pd.Series:
        if series.max() == series.min():
            return pd.Series(0.5, index=series.index)  # all equal → middle
        return (series - series.min()) / (series.max() - series.min())

    base["count_norm"] = normalize(base["event_count"])
    base["mag_norm"] = normalize(base["max_magnitude"])
    base["sig_norm"] = normalize(base["avg_significance"])
    base["shallow_norm"] = base["shallow_pct"] / 100.0  # already a percentage

    # Weighted composite
    base["risk_score"] = (
        base["count_norm"] * 30
        + base["mag_norm"] * 30
        + base["sig_norm"] * 20
        + base["shallow_norm"] * 20
    ).round(1)

    base["risk_rating"] = base["risk_score"].apply(_score_to_rating)

    # Drop the intermediate normalized columns from the user-facing output
    base = base.drop(columns=["count_norm", "mag_norm", "sig_norm", "shallow_norm"])

    # Attach baseline anomaly flag if available
    if baseline_df is not None and not baseline_df.empty:
        comparison = analytics.compare_to_baseline(
            current_df=df,
            baseline_df=baseline_df,
            current_window_days=current_window_days,
            baseline_window_days=baseline_window_days,
        )
        flag_lookup = dict(zip(comparison["country_name"], comparison["anomaly_flag"]))
        deviation_lookup = dict(zip(comparison["country_name"], comparison["deviation_pct"]))
        base["anomaly_flag"] = base["country_name"].map(flag_lookup).fillna("No baseline")
        base["deviation_pct"] = base["country_name"].map(deviation_lookup)

    # Add a per-row plain-English summary
    base["summary"] = base.apply(_build_region_summary, axis=1)

    return base.sort_values("risk_score", ascending=False).reset_index(drop=True)


def _score_to_rating(score: float) -> str:
    for low, high, name in RISK_THRESHOLDS:
        if low <= score < high:
            return name
    return "Severe"


def _build_region_summary(row: pd.Series) -> str:
    """Generate a one-sentence plain-English summary for a region."""
    country = row["country_name"]
    count = int(row["event_count"])
    max_mag = row["max_magnitude"]
    shallow_pct = int(row["shallow_pct"])
    rating = row["risk_rating"]

    base = (
        f"{country} recorded {count} events with a peak magnitude of M{max_mag:.1f}; "
        f"{shallow_pct}% were shallow."
    )

    # Append anomaly context if present
    if "anomaly_flag" in row.index and pd.notna(row.get("anomaly_flag")):
        flag = row["anomaly_flag"]
        if flag == "Anomalous (above)":
            dev = row.get("deviation_pct")
            if pd.notna(dev) and dev < 1000:  # skip absurd ratios from near-zero baselines
                base += f" Activity is {int(dev)}% above the 90-day baseline — flagged as anomalous."
            else:
                base += " Activity is well above the 90-day baseline — flagged as anomalous."
        elif flag == "Anomalous (below)":
            dev = row.get("deviation_pct")
            if pd.notna(dev):
                base += f" Activity is {abs(int(dev))}% below the 90-day baseline — unusually quiet."

    base += f" Risk rating: {rating}."
    return base


def _empty_risk_report() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "country_name", "event_count", "avg_magnitude", "max_magnitude",
        "avg_significance", "shallow_pct", "risk_score", "risk_rating", "summary"
    ])


# ============================================================
# NOTABLE EVENTS NARRATIVE
# ============================================================

def generate_notable_events_narrative(
    df: pd.DataFrame,
    baseline_df: Optional[pd.DataFrame] = None,
    current_window_days: int = 30,
    baseline_window_days: int = 90,
) -> List[str]:
    """
    Auto-generate 3-6 plain-English bullet points summarizing the period.

    The narrative is rules-based: each bullet checks for a specific
    condition (largest event, tsunami, high felt reports, anomalies) and
    only appears if that condition is met. This guarantees every bullet
    is factually grounded in the data.
    """
    bullets = []
    if df.empty:
        return ["No earthquake events recorded for this period."]

    # 1) Headline event
    largest = analytics.find_largest_event(df)
    if largest["depth_km"] < 70:
        depth_desc = "a shallow"
    elif largest["depth_km"] < 300:
        depth_desc = "an intermediate"
    else:
        depth_desc = "a deep"

    bullets.append(
        f"The largest event was a M{largest['magnitude']:.1f} earthquake near "
        f"{largest['place']}, at {depth_desc} depth of {largest['depth_km']:.0f} km. "
        f"USGS recorded {largest['felt_reports']} 'did you feel it' reports."
    )

    # 2) Tsunami advisories
    tsunami_events = analytics.find_tsunami_events(df)
    if not tsunami_events.empty:
        bullets.append(
            f"{len(tsunami_events)} event(s) triggered tsunami advisories this period — "
            f"the most significant was a M{tsunami_events['magnitude'].max():.1f} event "
            f"near {tsunami_events.loc[tsunami_events['magnitude'].idxmax(), 'place']}."
        )

    # 3) High-felt events (felt by many people, even at modest magnitudes)
    high_felt = analytics.find_high_felt_events(df, min_reports=100)
    if not high_felt.empty:
        top_felt = high_felt.loc[high_felt["felt_reports"].idxmax()]
        bullets.append(
            f"The most widely-felt event was a M{top_felt['magnitude']:.1f} near "
            f"{top_felt['place']} — {int(top_felt['felt_reports']):,} public reports submitted "
            f"despite its modest magnitude, indicating shallow depth and/or proximity to populated areas."
        )

    # 4) Activity volume context
    total = len(df)
    bullets.append(
        f"A total of {total:,} earthquakes were recorded over the period, "
        f"with {(df['risk_tier'].isin(['Strong', 'Major', 'Great'])).sum()} reaching M5.0 or higher."
    )

    # 5) Anomaly callouts (if baseline provided)
    if baseline_df is not None and not baseline_df.empty:
        comparison = analytics.compare_to_baseline(
            current_df=df,
            baseline_df=baseline_df,
            current_window_days=current_window_days,
            baseline_window_days=baseline_window_days,
        )
        # Filter to anomalies with meaningful baseline (avoid 2000% spikes from near-zero)
        meaningful = comparison[
            (comparison["anomaly_flag"].isin(["Anomalous (above)", "Anomalous (below)"]))
            & (comparison["baseline_per_day"] >= 0.1)  # at least 1 event every 10 days historically
        ]
        if not meaningful.empty:
            above = meaningful[meaningful["anomaly_flag"] == "Anomalous (above)"].head(3)
            below = meaningful[meaningful["anomaly_flag"] == "Anomalous (below)"].head(3)
            if not above.empty:
                names = ", ".join(above["country_name"].head(3).tolist())
                bullets.append(
                    f"Above-baseline activity flagged in: {names} — "
                    f"recording notably more events than the 90-day historical norm."
                )
            if not below.empty:
                names = ", ".join(below["country_name"].head(3).tolist())
                bullets.append(
                    f"Below-baseline (unusually quiet) regions: {names} — "
                    f"recording fewer events than the 90-day historical norm."
                )

    # 6) Dominant region
    top_country = df["country_name"].value_counts().head(1)
    if not top_country.empty:
        country_name = top_country.index[0]
        country_count = top_country.iloc[0]
        country_pct = round(country_count / total * 100, 1)
        bullets.append(
            f"{country_name} was the most active region, accounting for {country_count} events "
            f"({country_pct}% of total activity)."
        )

    return bullets