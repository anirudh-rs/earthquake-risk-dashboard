"""
Global Earthquake Risk & Seismic Activity Dashboard
====================================================
Streamlit entry point.

Architecture:
  - Sidebar: date range, magnitude filter, refresh button
  - Header: title, subtitle, live status indicator
  - Body: KPIs → narrative → map → charts → risk report
  - Footer: data source attribution + limitations link

Data pipeline:
  - On load: fetch current window from USGS, fetch historical baseline
  - Both are @st.cache_data'd so UI interactions don't re-fetch
  - "Refresh Data" button clears the cache and re-runs
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
from pathlib import Path
from components.kpi_cards import render_kpi_strip
from streamlit_folium import st_folium
from src.clustering import find_hotzones
from components.map_view import build_map

from src.data_pipeline import fetch_earthquakes
from src.data_cleaning import clean_earthquakes
from src.data_pipeline import fetch_earthquakes, QueryTooLargeError

from components.time_series import (
    daily_activity_chart,
    magnitude_trend_chart,
    magnitude_bucket_chart,
    depth_distribution_chart,
    depth_by_region_chart,
)
from src import analytics

from components.risk_brief import (
    render_notable_events,
    render_risk_report,
    render_region_summaries,
)
from src.risk_report import score_regions, generate_notable_events_narrative

# Safe defaults — used during reruns triggered by preset buttons
# before sidebar widgets have fully rendered
min_magnitude = 4.5

# Plotly chart config — shared across all charts in the dashboard.
# We KEEP the modebar (zoom box, pan, reset, download) because it's genuinely useful
# for time-series and distribution analysis. We hide just the "Edit in Plotly" link
# (which redirects to plot.ly's site) and the lasso/select tools (less useful for our use).
PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,  # hides the "plotly" branding
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d",
        "toggleSpikelines",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
    ],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "seismic_chart",
        "height": 500,
        "width": 1200,
        "scale": 2,  # 2x scale for crisp downloads
    },
}


# ============================================================
# PAGE CONFIG — must be the first Streamlit call
# ============================================================
st.set_page_config(
    page_title="Seismic Risk Monitor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS INJECTION
# ============================================================
def load_css():
    """Inject our Mission Control stylesheet into the page."""
    css_path = Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("Stylesheet not found — falling back to default theme.")


load_css()


# ============================================================
# CACHED DATA LOADERS
# ============================================================
# ttl=3600 means cache expires after 1 hour. USGS updates continuously,
# but caching for an hour balances freshness with API politeness.
# show_spinner=False because we render our own custom loader below.

@st.cache_data(ttl=3600, show_spinner=False)
def load_current_data(start_date: datetime, end_date: datetime, min_mag: float) -> pd.DataFrame:
    """Fetch + clean the user-selected current window."""
    raw = fetch_earthquakes(start_date, end_date, min_magnitude=min_mag)
    return clean_earthquakes(raw)


@st.cache_data(ttl=3600, show_spinner=False)
def load_baseline_data(end_date: datetime, min_mag: float, baseline_days: int = 90) -> pd.DataFrame:
    """
    Fetch + clean the historical baseline window.
    The baseline ends just before the current window starts to avoid overlap.
    """
    baseline_end = end_date - timedelta(days=1)
    baseline_start = baseline_end - timedelta(days=baseline_days)
    raw = fetch_earthquakes(baseline_start, baseline_end, min_magnitude=min_mag)
    return clean_earthquakes(raw)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ◈ CONTROLS")
    st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)

    today = datetime.utcnow().date()
    default_start = today - timedelta(days=30)

    # Split into two separate date inputs instead of one range picker.
    # Streamlit's range date picker has a known rendering bug where the calendar
    # flips upward and clips when near the top of the sidebar or when picking
    # early dates. Two separate pickers are more reliable and clearer to use.
    st.markdown(
        "<div style='font-family: JetBrains Mono, monospace; font-size: 0.7rem; "
        "color: var(--text-tertiary); text-transform: uppercase; "
        "letter-spacing: 0.1em; margin-bottom: 0.5rem;'>QUICK SELECT</div>",
        unsafe_allow_html=True,
    )

    # Preset buttons — three per row using columns
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    with preset_col1:
        if st.button("7D", use_container_width=True, help="Last 7 days"):
            st.session_state["start_date"] = today - timedelta(days=7)
            st.session_state["end_date"] = today
    with preset_col2:
        if st.button("30D", use_container_width=True, help="Last 30 days"):
            st.session_state["start_date"] = today - timedelta(days=30)
            st.session_state["end_date"] = today
    with preset_col3:
        if st.button("90D", use_container_width=True, help="Last 90 days"):
            st.session_state["start_date"] = today - timedelta(days=90)
            st.session_state["end_date"] = today

    preset_col4, preset_col5, preset_col6 = st.columns(3)
    with preset_col4:
        if st.button("6M", use_container_width=True, help="Last 6 months"):
            st.session_state["start_date"] = today - timedelta(days=180)
            st.session_state["end_date"] = today
    with preset_col5:
        if st.button("1Y", use_container_width=True, help="Last 1 year"):
            st.session_state["start_date"] = today - timedelta(days=365)
            st.session_state["end_date"] = today
    with preset_col6:
        if st.button("5Y", use_container_width=True, help="Last 5 years"):
            st.session_state["start_date"] = today - timedelta(days=1825)
            st.session_state["end_date"] = today

    st.markdown(
        "<div style='font-family: JetBrains Mono, monospace; font-size: 0.7rem; "
        "color: var(--text-tertiary); text-transform: uppercase; "
        "letter-spacing: 0.1em; margin: 0.75rem 0 0.25rem 0;'>CUSTOM RANGE (UTC)</div>",
        unsafe_allow_html=True,
    )

    # Manual date pickers — use session state so preset buttons can set them
    start_date = st.date_input(
        "From",
        value=st.session_state.get("start_date", default_start),
        min_value=date(1970, 1, 1),
        max_value=today,
        key="start_date",
    )

    end_date = st.date_input(
        "To",
        value=st.session_state.get("end_date", today),
        min_value=date(1970, 1, 1),
        max_value=today,
        key="end_date",
    )

    # Validate order
    if start_date > end_date:
        st.error("Start date must be before end date.")
        st.stop()

    # Dynamic guidance — changes based on the selected window + magnitude
    window_days_sidebar = (end_date - start_date).days + 1

    if window_days_sidebar > 365 and min_magnitude < 5.5:
        sidebar_note_color = "var(--risk-high)"
        sidebar_note_text = (
            f"⚠ {window_days_sidebar}-day window at M{min_magnitude} may exceed "
            f"USGS's 20,000-event limit. Consider raising to M5.5+."
        )
    elif window_days_sidebar > 180 and min_magnitude < 4.5:
        sidebar_note_color = "var(--risk-moderate)"
        sidebar_note_text = (
            f"⚠ {window_days_sidebar}-day window at M{min_magnitude} is large. "
            f"M4.5+ recommended for reliable results."
        )
    else:
        sidebar_note_color = "var(--text-tertiary)"
        sidebar_note_text = (
            "Coverage back to 1970. For windows >1 year, M5.5+ recommended."
        )

    st.markdown(
        f"<div class='map-note' style='margin: 0.5rem 0; font-size: 0.72rem; "
        f"border-left-color: {sidebar_note_color}; color: {sidebar_note_color};'>"
        f"{sidebar_note_text}"
        f"</div>",
        unsafe_allow_html=True,
    )

    min_magnitude = st.slider(
        "Minimum magnitude",
        min_value=2.5,
        max_value=7.0,
        value=4.5,
        step=0.1,
        help=(
            "Filter out events below this magnitude. "
            "For windows longer than 1 year, M5.5+ is recommended "
            "to stay within USGS's 20,000-event limit."
        ),
    )

    st.markdown("---")

    refresh = st.button("⟳ REFRESH DATA", use_container_width=True)
    if refresh:
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<div class='live-indicator'>"
        "<span class='live-dot'></span>"
        "Live data via USGS"
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# HEADER
# ============================================================
st.markdown("# SEISMIC RISK MONITOR")
st.markdown(
    "<p style='color: #8b96a8; font-family: Inter; font-size: 0.95rem; margin-top: -0.5rem;'>"
    "Live global earthquake risk intelligence — sourced fresh from USGS on every load."
    "</p>",
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================
# Validate the date range — st.date_input returns a tuple if a range is selected,
# or a single date if the user only picked one. Handle both.

# Convert to datetime for the API call (it wants datetime, not date)
start_dt = datetime.combine(start_date, datetime.min.time())
end_dt = datetime.combine(end_date, datetime.max.time())
window_days = (end_date - start_date).days + 1

# Load the data with our custom loading state
# Multi-stage loading with progress feedback
loading_placeholder = st.empty()

def show_loading(message: str, brief_pause: bool = True):
    """Render a loading message and optionally pause briefly to let the UI render."""
    loading_placeholder.markdown(
        f"<div class='narrative-bullet'>"
        f"<span class='live-dot' style='display:inline-block; margin-right:0.6rem; vertical-align:middle;'></span>"
        f"{message}"
        f"</div>",
        unsafe_allow_html=True,
    )
    if brief_pause:
        import time
        time.sleep(0.15)

# Forecast for large queries
estimated_seconds = max(3, window_days // 10)
size_warning = ""
if window_days > 180:
    size_warning = f" Larger windows take longer (~{estimated_seconds}s expected)."

try:
    show_loading(f"Connecting to USGS — fetching {window_days} days at M{min_magnitude}+...{size_warning}")
    current_df = load_current_data(start_dt, end_dt, min_magnitude)

    show_loading(f"Loaded {len(current_df):,} events. Fetching 90-day historical baseline...")
    baseline_df = load_baseline_data(end_dt, min_magnitude, baseline_days=90)

    load_error = None
    error_type = None
except QueryTooLargeError as e:
    current_df = pd.DataFrame()
    baseline_df = pd.DataFrame()
    load_error = str(e)
    error_type = "too_large"
except Exception as e:
    current_df = pd.DataFrame()
    baseline_df = pd.DataFrame()
    load_error = str(e)
    error_type = "generic"

loading_placeholder.empty()

# Show error if the load failed
if load_error:
    if error_type == "too_large":
        st.error(
            f"**Query too large for USGS API.** {load_error}\n\n"
            f"**Quick fixes:**\n"
            f"- Raise the magnitude filter (try M4.0+ or M4.5+)\n"
            f"- Shorten the date range (try 90 days or fewer for low magnitude queries)\n"
            f"- For long-term analysis, query M5.0+ or higher — covers the historically significant events"
        )
    else:
        st.error(f"Could not fetch data from USGS: {load_error}")
    st.stop()

# Empty-data handling
if current_df.empty:
    st.warning(
        f"No earthquakes found for {start_date} → {end_date} at M{min_magnitude}+. "
        "Try widening the date range or lowering the magnitude threshold."
    )
    st.stop()


# ============================================================
# KPI STRIP
# ============================================================
st.markdown(
    "<div class='section-divider'>// KEY METRICS</div>",
    unsafe_allow_html=True,
)
render_kpi_strip(current_df)

# ============================================================
# SYSTEM STATUS LINE
# ============================================================
last_event = current_df["datetime_utc"].max()
last_event_str = last_event.strftime("%Y-%m-%d %H:%M UTC")

st.markdown(
    f"<div class='section-divider'>// SYSTEM STATUS</div>"
    f"<div class='narrative-bullet'>"
    f"<strong>{len(current_df):,}</strong> events loaded for "
    f"<strong>{start_date} → {end_date}</strong> ({window_days} days) at M{min_magnitude}+. "
    f"Most recent event: <strong>{last_event_str}</strong>. "
    f"Historical baseline: {len(baseline_df):,} events from prior 90 days."
    f"</div>",
    unsafe_allow_html=True,
)

# ============================================================
# WORLD MAP
# ============================================================
st.markdown(
    "<div class='section-divider'>// GLOBAL SEISMIC MAP</div>",
    unsafe_allow_html=True,
)

# Compute hotzones for the current data
hotzones_df = find_hotzones(current_df, n_clusters=8)

# Build the map
plate_path = Path(__file__).parent / "assets" / "tectonic_plates.geojson"

# Persistent rendering note — sets honest expectations before the map appears.
# Folium renders client-side in an iframe; from Python we have no visibility
# into when that's actually finished, so we tell the user upfront.
st.markdown(
    "<div class='map-note'>"
    "<strong>Note:</strong> The map renders in your browser and may take a few seconds "
    "to fully paint after the loading indicator disappears, especially for large date "
    "ranges or low magnitude thresholds. This is normal — the Python pipeline finishes "
    "before the browser is done drawing all markers."
    "</div>",
    unsafe_allow_html=True,
)

with st.spinner(f"Rendering {len(current_df):,} events on the world map..."):
    earthquake_map = build_map(
        df=current_df,
        hotzones=hotzones_df,
        plate_boundaries_path=str(plate_path) if plate_path.exists() else None,
    )

    st_folium(
        earthquake_map,
        width=None,
        height=560,
        returned_objects=[],
        use_container_width=True,
    )

# ============================================================
# TIME-SERIES & DISTRIBUTIONS
# ============================================================
st.markdown(
    "<div class='section-divider'>// TEMPORAL ANALYSIS</div>",
    unsafe_allow_html=True,
)

# Compute analytics once, reuse across charts
daily_df = analytics.daily_activity(current_df)
buckets_df = analytics.magnitude_buckets(current_df)
depth_df = analytics.depth_distribution(current_df)
depth_region_df = analytics.depth_by_region(current_df, top_n=10)

# Row 1: Daily activity (full width)
st.plotly_chart(daily_activity_chart(daily_df), use_container_width=True, config=PLOTLY_CONFIG)

# Row 2: Magnitude trend (full width)
st.plotly_chart(magnitude_trend_chart(daily_df), use_container_width=True, config=PLOTLY_CONFIG)

# Row 3: Magnitude buckets + depth donut (side by side)
st.markdown(
    "<div class='map-note'>"
    "<strong>Depth context:</strong> Earthquake depth is measured from the surface "
    "to the rupture point. Shallow events (0–70 km) cause the most surface damage "
    "because energy hasn't dissipated before reaching the ground — a M5.5 shallow "
    "quake under a populated area is often more destructive than a M6.5 deep quake. "
    "This is why depth is treated as a primary risk factor alongside magnitude."
    "</div>",
    unsafe_allow_html=True,
)

col3, col4 = st.columns([3, 2], gap="small")
with col3:
    st.plotly_chart(magnitude_bucket_chart(buckets_df), use_container_width=True, config=PLOTLY_CONFIG)
with col4:
    st.plotly_chart(depth_distribution_chart(depth_df), use_container_width=True, config=PLOTLY_CONFIG)

# Row 4: Depth by region (full width)
st.plotly_chart(depth_by_region_chart(depth_region_df), use_container_width=True, config=PLOTLY_CONFIG)

# ============================================================
# RISK BRIEF — Notable Events + Top 10
# ============================================================

# Compute risk report and narrative once
report_df = score_regions(
    df=current_df,
    baseline_df=baseline_df,
    current_window_days=window_days,
    baseline_window_days=90,
    top_n=10,
)
narrative_bullets = generate_notable_events_narrative(
    df=current_df,
    baseline_df=baseline_df,
    current_window_days=window_days,
    baseline_window_days=90,
)

# Notable Events
st.markdown(
    "<div class='section-divider'>// NOTABLE EVENTS</div>",
    unsafe_allow_html=True,
)
render_notable_events(narrative_bullets)

# Top 10 Risk Report
st.markdown(
    "<div class='section-divider'>// TOP 10 REGIONAL RISK REPORT</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='map-note'>"
    "<strong>Risk score (0–100):</strong> blends event count, peak magnitude, USGS significance, "
    "and shallow event percentage into a single composite. Ratings are calibrated within the "
    "current dataset, so the leader always anchors near the upper end."
    "<br><br>"
    "<strong>Tiers:</strong> Low (0–25), Moderate (25–50), High (50–75), Severe (75–100). "
    "A region scoring 80+ typically combines high event volume, at least one major (M6+) event, "
    "and a high shallow-event percentage — the profile most associated with surface impact."
    "<br><br>"
    "<strong>Vs Baseline:</strong> compares the current window's per-day event rate against "
    "the prior 90 days. The comparison is suppressed (shown as <em>N/A</em>) when the current "
    "window exceeds 180 days, or when a region has fewer than 5 baseline events — in either "
    "case the historical sample is too small to give a reliable deviation."
    "</div>",
    unsafe_allow_html=True,
)

render_risk_report(report_df)

# Per-region summaries (collapsible)
with st.expander("Show plain-English summaries per region"):
    render_region_summaries(report_df)

# ============================================================
# CSV EXPORT
# ============================================================
st.markdown(
    "<div class='section-divider'>// EXPORT</div>",
    unsafe_allow_html=True,
)

# Prepare export DataFrame — clean column names, drop internal columns
export_df = report_df.copy()

# Rename columns to be human-readable in the CSV
export_df = export_df.rename(columns={
    "country_name":    "Region",
    "event_count":     "Event Count",
    "avg_magnitude":   "Avg Magnitude",
    "max_magnitude":   "Peak Magnitude",
    "avg_significance":"Avg Significance",
    "shallow_pct":     "Shallow %",
    "risk_score":      "Risk Score",
    "risk_rating":     "Risk Rating",
    "anomaly_flag":    "Vs Baseline",
    "deviation_pct":   "Deviation %",
    "summary":         "Summary",
})

# Add metadata rows at the top via a header dict
export_metadata = {
    "Generated":       [pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")],
    "Date Range":      [f"{start_date} to {end_date}"],
    "Min Magnitude":   [min_magnitude],
    "Total Events":    [len(current_df)],
    "Data Source":     ["USGS Earthquake Catalog API"],
}

col_export1, col_export2 = st.columns([2, 3])

with col_export1:
    # Risk report CSV
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ DOWNLOAD RISK REPORT (CSV)",
        data=csv_bytes,
        file_name=f"seismic_risk_report_{start_date}_{end_date}.csv",
        mime="text/csv",
        use_container_width=True,
        help="Downloads the Top 10 Regional Risk Report as a CSV file.",
    )

with col_export2:
    # Full event-level CSV
    event_export = current_df[[
        "datetime_utc", "magnitude", "place", "country_name",
        "continent", "depth_km", "depth_category", "risk_tier",
        "significance", "tsunami", "felt_reports", "alert", "url"
    ]].copy()
    event_export = event_export.rename(columns={
        "datetime_utc":  "Date/Time (UTC)",
        "magnitude":     "Magnitude",
        "place":         "Place",
        "country_name":  "Country",
        "continent":     "Continent",
        "depth_km":      "Depth (km)",
        "depth_category":"Depth Category",
        "risk_tier":     "Risk Tier",
        "significance":  "USGS Significance",
        "tsunami":       "Tsunami Flag",
        "felt_reports":  "Felt Reports",
        "alert":         "Alert Level",
        "url":           "USGS URL",
    })
    event_csv_bytes = event_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ DOWNLOAD ALL EVENTS (CSV)",
        data=event_csv_bytes,
        file_name=f"seismic_events_{start_date}_{end_date}.csv",
        mime="text/csv",
        use_container_width=True,
        help="Downloads every earthquake event in the current window as a CSV file.",
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class='dashboard-footer'>
        <div class='footer-grid'>
            <div>
                <div class='footer-section-title'>About this dashboard</div>
                <div class='footer-text'>
                    The Seismic Risk Monitor pulls live earthquake data from the USGS
                    Earthquake Catalog API on every load. No pre-downloaded datasets —
                    every analysis is built fresh from the source. Risk scores blend
                    event count, peak magnitude, USGS significance, and depth profile
                    into a transparent 0–100 composite.
                </div>
            </div>
            <div>
                <div class='footer-section-title'>Data sources</div>
                <div class='footer-text'>
                    <a class='footer-link'
                       href='https://earthquake.usgs.gov/fdsnws/event/1/'
                       target='_blank'>USGS Earthquake Catalog API</a><br>
                    Free · No API key · Updated continuously<br>
                    Coverage from 1900 to present.<br><br>
                    <a class='footer-link'
                       href='https://doi.org/10.1029/2001GC000252'
                       target='_blank'>PB2002 Tectonic Boundaries</a><br>
                    Bird (2003) · Geochemistry,<br>
                    Geophysics, Geosystems
                </div>
            </div>
            <div>
                <div class='footer-section-title'>Project</div>
                <div class='footer-text'>
                    <a class='footer-link'
                       href='https://github.com/anirudh-rs/earthquake-risk-dashboard'
                       target='_blank'>GitHub Repository</a><br><br>
                    <div class='footer-section-title' style='margin-top: 0.5rem;'>Stack</div>
                    Python · Streamlit · Folium<br>
                    Plotly · scikit-learn<br>
                    pandas · reverse_geocoder<br>
                    Tectonic data: PB2002 (Bird, 2003)
                </div>
            </div>
        </div>
        <div class='footer-bottom'>
            <span>◈ SEISMIC RISK MONITOR — LIVE DATA</span>
            <span>Data: USGS · Tectonics: PB2002 · Not for emergency use</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
