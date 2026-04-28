"""
Folium map component — the visual centerpiece of the dashboard.

Three layers:
  1. Earthquake markers (CircleMarkers) — sized by magnitude, colored by risk tier
  2. Tectonic plate boundaries (PolyLine from GeoJSON) — cyan reference layer
  3. K-Means hotzone centers — large annotated markers with cluster summaries

The map uses a dark CartoDB tile layer to match our Mission Control theme.
"""

import folium
import pandas as pd
from pathlib import Path
import json
from typing import Optional
from folium.plugins import MarkerCluster

# Risk tier → color (matches our CSS color tokens)
RISK_COLORS = {
    "Minor":    "#5c6577",  # muted gray
    "Light":    "#06d6f7",  # cyan
    "Moderate": "#fbbf24",  # amber
    "Strong":   "#fb923c",  # orange
    "Major":    "#ef4444",  # red
    "Great":    "#dc2626",  # deep red
}


def build_map(
    df: pd.DataFrame,
    hotzones: Optional[pd.DataFrame] = None,
    plate_boundaries_path: Optional[str] = None,
) -> folium.Map:
    """
    Build the full earthquake risk map with all three layers.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned earthquake DataFrame.
    hotzones : pd.DataFrame, optional
        Output of clustering.find_hotzones(). If None, hotzone layer is skipped.
    plate_boundaries_path : str, optional
        Path to the tectonic plate boundaries GeoJSON file.
        If None or missing, the layer is skipped silently.

    Returns
    -------
    folium.Map
        Configured Folium map ready for st_folium() rendering.
    """
    # Center the map at (0, 30) to give a good world view that includes the Pacific
    m = folium.Map(
        location=[15, 30],
        zoom_start=2,
        tiles="CartoDB dark_matter",  # matches our dark theme
        attr="© OpenStreetMap, © CartoDB",
        prefer_canvas=True,  # major perf boost when rendering many markers
        worldCopyJump=True,  # smooth scrolling across the antimeridian
    )

    # Layer 1: Earthquake markers
    _add_earthquake_markers(m, df)

    # Layer 2: Tectonic plate boundaries (if file is available)
    if plate_boundaries_path:
        _add_plate_boundaries(m, plate_boundaries_path)

    # Layer 3: K-Means hotzone centers (if provided)
    if hotzones is not None and not hotzones.empty:
        _add_hotzone_markers(m, hotzones)

    # Layer control — lets users toggle each layer on/off
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    return m


# ============================================================
# LAYER 1: EARTHQUAKE MARKERS
# ============================================================

def _add_earthquake_markers(m: folium.Map, df: pd.DataFrame, cluster_threshold: int = 500) -> None:
    """
    Add earthquake markers. Two modes:
      - Below threshold: every event gets a CircleMarker (best detail)
      - Above threshold: events are grouped via MarkerCluster (best performance)

    The threshold is configurable; 500 is the sweet spot where individual
    markers stay snappy and clustering kicks in before lag becomes noticeable.
    """
    if df.empty:
        return

    use_clustering = len(df) > cluster_threshold

    if use_clustering:
        layer = MarkerCluster(
            name=f"Earthquakes ({len(df):,} events)",
            options={
                "maxClusterRadius": 40,
                "spiderfyOnMaxZoom": True,
                "showCoverageOnHover": False,
                "disableClusteringAtZoom": 8,  # at high zoom, show individual markers
            }
        )
    else:
        layer = folium.FeatureGroup(name=f"Earthquakes ({len(df):,} events)", show=True)

    for _, row in df.iterrows():
        color = RISK_COLORS.get(row.get("risk_tier", "Minor"), "#5c6577")
        radius = _magnitude_to_radius(row["magnitude"])
        popup_html = _build_popup(row)

        marker = folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            weight=1,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"M{row['magnitude']:.1f} — {row.get('country_name', 'Unknown')}",
        )
        marker.add_to(layer)

    layer.add_to(m)


def _magnitude_to_radius(mag: float) -> float:
    """
    Convert magnitude to circle radius (pixels).
    Uses an exponential scale because earthquake energy scales exponentially with magnitude —
    each step up represents ~32x more energy. We compress that visually.
    """
    if pd.isna(mag):
        return 3
    # Magnitudes typically 2.5–8.0 → radii 3–24 pixels
    return max(3, min(24, 2 ** (mag - 2.5)))


def _build_popup(row: pd.Series) -> str:
    """Build the HTML popup for an earthquake marker."""
    risk_tier = row.get("risk_tier", "Unknown")
    risk_color = RISK_COLORS.get(risk_tier, "#5c6577")
    when = row["datetime_utc"].strftime("%Y-%m-%d %H:%M UTC") if pd.notna(row.get("datetime_utc")) else "Unknown"
    place = row.get("place", "Unknown location")
    depth_cat = row.get("depth_category", "Unknown")
    tsunami = "⚠️ Tsunami advisory issued" if row.get("tsunami") == 1 else ""
    felt = row.get("felt_reports")
    felt_str = f"{int(felt):,} felt reports" if pd.notna(felt) and felt > 0 else ""

    extras = " · ".join(filter(None, [tsunami, felt_str]))
    extras_html = f"<div style='margin-top:6px; color:#fbbf24; font-size:11px;'>{extras}</div>" if extras else ""

    # Inline styles since Folium popups don't pick up our CSS
    return f"""
    <div style='font-family: monospace; font-size: 12px; color: #1a1a1a; min-width: 220px;'>
        <div style='font-size: 16px; font-weight: 600; margin-bottom: 4px;'>
            M{row['magnitude']:.1f}
            <span style='display:inline-block; padding:2px 6px; margin-left:6px;
                         background:{risk_color}; color:white; font-size:10px;
                         text-transform:uppercase; letter-spacing:0.05em;'>
                {risk_tier}
            </span>
        </div>
        <div style='color:#4a4a4a; margin-bottom:6px; font-family: sans-serif;'>{place}</div>
        <div style='font-size: 11px; color: #6a6a6a;'>
            <div>{when}</div>
            <div>Depth: {row['depth_km']:.0f} km ({depth_cat})</div>
        </div>
        {extras_html}
    </div>
    """


# ============================================================
# LAYER 2: TECTONIC PLATE BOUNDARIES
# ============================================================

def _add_plate_boundaries(m: folium.Map, path: str) -> None:
    """Add tectonic plate boundaries as a thin cyan overlay."""
    plate_path = Path(path)
    if not plate_path.exists():
        return

    try:
        with open(plate_path, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)
    except Exception:
        return  # silently skip if the file is malformed

    folium.GeoJson(
        geojson_data,
        name="Tectonic Plates",
        style_function=lambda feature: {
            "color": "#06d6f7",
            "weight": 1.2,
            "opacity": 0.55,
        },
        show=True,
    ).add_to(m)


# ============================================================
# LAYER 3: K-MEANS HOTZONE CENTERS
# ============================================================

def _add_hotzone_markers(m: folium.Map, hotzones: pd.DataFrame) -> None:
    """
    Add a marker for each K-Means hotzone center.
    Larger circle outline + a label showing event count and dominant region.
    """
    layer = folium.FeatureGroup(name="K-Means Hotzones", show=True)

    for _, row in hotzones.iterrows():
        # Outer ring scales with event count (visually emphasizes the busy zones)
        radius = max(15, min(40, row["event_count"] * 1.5))

        popup_html = f"""
        <div style='font-family: monospace; font-size: 12px; color: #1a1a1a; min-width: 200px;'>
            <div style='font-size: 14px; font-weight: 600; margin-bottom: 4px;
                        text-transform: uppercase; letter-spacing: 0.05em;'>
                Hotzone
            </div>
            <div style='font-size: 13px; color: #2a2a2a; margin-bottom: 6px;'>
                {row['hotzone_label']}
            </div>
            <div style='font-size: 11px; color: #6a6a6a;'>
                <div>Events: {row['event_count']}</div>
                <div>Avg magnitude: M{row['avg_magnitude']:.2f}</div>
                <div>Peak magnitude: M{row['max_magnitude']:.1f}</div>
            </div>
        </div>
        """

        folium.CircleMarker(
            location=[row["center_lat"], row["center_lon"]],
            radius=radius,
            color="#06d6f7",
            fill=False,
            weight=2,
            opacity=0.85,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"Hotzone: {row['dominant_country']} ({row['event_count']} events)",
        ).add_to(layer)

    layer.add_to(m)