"""
KPI strip — four metric cards for the dashboard header.

Renders Mission Control-styled cards using the .kpi-card CSS class
defined in assets/styles.css. Each card has a label, a primary value,
and an optional sub-label for context.
"""

import streamlit as st
import pandas as pd


def render_kpi_strip(df: pd.DataFrame) -> None:
    """
    Render the four KPI cards across the top of the dashboard.

    Cards (left to right):
      1. Total Events — count + window context
      2. Average Magnitude — mean + max
      3. Strongest Event — peak magnitude + location
      4. Most Active Region — top country + event count

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned current-window earthquake DataFrame.
    """
    if df.empty:
        return

    # Compute KPI values
    total_events = len(df)
    avg_mag = df["magnitude"].mean()
    max_mag = df["magnitude"].max()

    strongest = df.loc[df["magnitude"].idxmax()]
    strongest_mag = strongest["magnitude"]
    strongest_country = strongest.get("country_name", "Unknown") or "Unknown"
    strongest_place = strongest.get("place", "") or ""

    top_country = df["country_name"].value_counts().head(1)
    top_country_name = top_country.index[0]
    top_country_count = top_country.iloc[0]
    top_country_pct = round(top_country_count / total_events * 100, 1)

    # Strong+ count (M5.0+) — useful sub-label for total events
    strong_plus = (df["risk_tier"].isin(["Strong", "Major", "Great"])).sum()

    # Render four equal columns
    col1, col2, col3, col4 = st.columns(4, gap="small")

    with col1:
        _render_card(
            label="TOTAL EVENTS",
            value=f"{total_events:,}",
            sublabel=f"{strong_plus} reached M5.0+",
        )

    with col2:
        _render_card(
            label="AVG MAGNITUDE",
            value=f"M{avg_mag:.2f}",
            sublabel=f"Peak: M{max_mag:.1f}",
        )

    with col3:
        place_attr = strongest_place.replace('"', "&quot;")
        country_display = _shorten_place(strongest_country, max_len=22)
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-label'>STRONGEST EVENT</div>"
            f"<div class='kpi-value'>M{strongest_mag:.1f}</div>"
            f"<div class='kpi-sublabel' title=\"{place_attr}\" style='cursor: help;'>"
            f"{country_display}"
            f"<span style='margin-left: 0.4rem; color: var(--text-tertiary);'>ⓘ</span>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col4:
        _render_card(
            label="MOST ACTIVE REGION",
            value=_shorten_place(top_country_name, max_len=18),
            sublabel=f"{top_country_count} events ({top_country_pct}%)",
            value_size="small",  # country names are longer than numbers
        )


def _render_card(label: str, value: str, sublabel: str = "", value_size: str = "default") -> None:
    """Render a single KPI card using our custom CSS classes."""
    # Allow a smaller value font for long text (like country names)
    value_class = "kpi-value-small" if value_size == "small" else "kpi-value"

    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-label'>{label}</div>
            <div class='{value_class}'>{value}</div>
            <div class='kpi-sublabel'>{sublabel}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _shorten_place(place: str, max_len: int = 24) -> str:
    """
    USGS place strings can be long ('142 km ENE of Vilyuchinsk, Russia').
    Truncate intelligently for KPI card display.
    """
    if not place or pd.isna(place):
        return "Unknown"
    place = str(place).strip()
    if len(place) <= max_len:
        return place
    # Try to keep the country/region (after the last comma)
    if "," in place:
        parts = place.rsplit(",", 1)
        country_part = parts[1].strip()
        # If just the country fits well, use that
        if len(country_part) <= max_len - 3:
            return f"…{country_part}"
    return place[:max_len - 1] + "…"