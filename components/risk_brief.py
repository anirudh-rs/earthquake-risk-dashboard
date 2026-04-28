"""
Risk brief components — the 'intelligence summary' section of the dashboard.

Two main render functions:
  - render_notable_events(): auto-generated narrative bullets
  - render_risk_report(): Top 10 styled table with risk-tier pills and anomaly flags
"""

import streamlit as st
import pandas as pd
from typing import List, Optional


# Maps risk_rating string → CSS pill class
RISK_PILL_CLASSES = {
    "Low":      "risk-pill-low",
    "Moderate": "risk-pill-moderate",
    "High":     "risk-pill-high",
    "Severe":   "risk-pill-severe",
}

# Maps anomaly_flag → display label and CSS color
ANOMALY_DISPLAY = {
    "Anomalous (above)":            {"label": "▲ Above baseline",      "color": "#fb923c"},
    "Anomalous (below)":            {"label": "▼ Below baseline",      "color": "#06d6f7"},
    "Normal":                       {"label": "— Normal",               "color": "#5c6577"},
    "No baseline":                  {"label": "— No baseline",          "color": "#5c6577"},
    "Insufficient baseline":        {"label": "— Insufficient baseline","color": "#5c6577"},
    "Baseline N/A (window too long)": {"label": "— Baseline N/A",       "color": "#5c6577"},
}


# ============================================================
# NOTABLE EVENTS NARRATIVE
# ============================================================

def render_notable_events(bullets: List[str]) -> None:
    """
    Render the auto-generated narrative as a stack of styled cards.

    Each bullet appears in its own card with the cyan-dim left border —
    same .narrative-bullet class we already use elsewhere.

    Parameters
    ----------
    bullets : List[str]
        Output of risk_report.generate_notable_events_narrative().
    """
    if not bullets:
        st.markdown(
            "<div class='narrative-bullet'>No notable events recorded for this period.</div>",
            unsafe_allow_html=True,
        )
        return

    for bullet in bullets:
        st.markdown(
            f"<div class='narrative-bullet'>{bullet}</div>",
            unsafe_allow_html=True,
        )


# ============================================================
# TOP 10 RISK REPORT TABLE
# ============================================================

def render_risk_report(report_df: pd.DataFrame) -> None:
    """
    Render the Top 10 risk report as a styled HTML table.

    Built as a single HTML string with no leading whitespace per line —
    Streamlit's markdown parser treats indented lines as code blocks,
    which breaks complex HTML.
    """
    if report_df.empty:
        st.markdown(
            "<div class='narrative-bullet'>No risk data available for this period.</div>",
            unsafe_allow_html=True,
        )
        return

    # Build rows as a flat list of HTML strings — no indentation
    rows_html = []
    for _, row in report_df.iterrows():
        pill_class = RISK_PILL_CLASSES.get(row["risk_rating"], "risk-pill-moderate")
        risk_pill = f"<span class='risk-pill {pill_class}'>{row['risk_rating']}</span>"

        anomaly_info = ANOMALY_DISPLAY.get(row.get("anomaly_flag", "Normal"), ANOMALY_DISPLAY["Normal"])
        anomaly_html = (
            f"<span style='color:{anomaly_info['color']}; font-family: JetBrains Mono, monospace; "
            f"font-size: 0.75rem;'>{anomaly_info['label']}</span>"
        )

        country = row["country_name"]
        if len(country) > 28:
            country = country[:26] + "…"

        # IMPORTANT: keep this on a single logical line — no internal indentation
        row_html = (
            f"<tr>"
            f"<td class='rr-country'>{country}</td>"
            f"<td class='rr-num'>{int(row['event_count'])}</td>"
            f"<td class='rr-num'>M{row['max_magnitude']:.1f}</td>"
            f"<td class='rr-num'>{int(row['shallow_pct'])}%</td>"
            f"<td class='rr-num rr-score'>{row['risk_score']:.1f}</td>"
            f"<td>{risk_pill}</td>"
            f"<td>{anomaly_html}</td>"
            f"</tr>"
        )
        rows_html.append(row_html)

    # Assemble the complete table — no indentation anywhere
    table_html = (
        "<table class='risk-table'>"
        "<thead>"
        "<tr>"
        "<th>REGION</th>"
        "<th class='rr-num'>EVENTS</th>"
        "<th class='rr-num'>PEAK</th>"
        "<th class='rr-num'>SHALLOW</th>"
        "<th class='rr-num'>SCORE</th>"
        "<th>RATING</th>"
        "<th>VS BASELINE</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        + "".join(rows_html)
        + "</tbody>"
        "</table>"
    )

    st.markdown(table_html, unsafe_allow_html=True)


# ============================================================
# PER-REGION SUMMARIES
# ============================================================

def render_region_summaries(report_df: pd.DataFrame) -> None:
    """
    Render the plain-English per-region summary sentences as a stack of cards.
    Same .narrative-bullet styling as the notable events.
    """
    if report_df.empty or "summary" not in report_df.columns:
        return

    for _, row in report_df.iterrows():
        st.markdown(
            f"<div class='narrative-bullet'>{row['summary']}</div>",
            unsafe_allow_html=True,
        )