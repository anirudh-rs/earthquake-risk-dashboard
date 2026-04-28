"""
USGS Earthquake API pipeline.

Single responsibility: take a date range and minimum magnitude,
return a raw pandas DataFrame of earthquake events.

No cleaning, no risk tiering, no enrichment — just fetch and flatten.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

class QueryTooLargeError(Exception):
    """Raised when a USGS query exceeds the 20,000-event hard cap."""
    pass


USGS_BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_HARD_LIMIT = 20000  # USGS will silently truncate beyond this


def fetch_earthquakes(
    start_date: datetime,
    end_date: datetime,
    min_magnitude: float = 2.5,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Fetch earthquakes from USGS for the given window and return a DataFrame.

    Parameters
    ----------
    start_date : datetime
        Earliest event time (UTC).
    end_date : datetime
        Latest event time (UTC).
    min_magnitude : float
        Minimum magnitude to include. Default 2.5 — below this is mostly noise.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    pd.DataFrame
        One row per earthquake, raw fields preserved. Empty DataFrame if no events.

    Raises
    ------
    requests.RequestException
        If the API call fails (network, server error, timeout).
    ValueError
        If the API response isn't valid GeoJSON.
    """
    params = {
        "format": "geojson",
        "starttime": start_date.isoformat(),
        "endtime": end_date.isoformat(),
        "minmagnitude": min_magnitude,
        "orderby": "time",
    }

    response = requests.get(USGS_BASE_URL, params=params, timeout=timeout)

    # Special handling for 400 Bad Request — usually means the query
    # would return more than USGS's 20,000-event hard cap.
    if response.status_code == 400:
        raise QueryTooLargeError(
            f"USGS rejected the query — your selection would likely return more than "
            f"{USGS_HARD_LIMIT:,} events. Narrow the date range or raise the magnitude threshold."
        )

    response.raise_for_status()

    data = response.json()

    if "features" not in data:
        raise ValueError(f"Unexpected API response shape: keys={list(data.keys())}")

    features = data["features"]
    reported_count = data.get("metadata", {}).get("count", len(features))

    # Warn if we likely hit the hard limit
    if reported_count >= USGS_HARD_LIMIT:
        print(
            f"WARNING: USGS returned {reported_count} events — at or above the "
            f"{USGS_HARD_LIMIT} hard limit. Results may be truncated. "
            f"Consider narrowing the date range or raising min_magnitude."
        )

    if not features:
        # Return an empty DataFrame with the expected columns so downstream code doesn't break
        return _empty_dataframe()

    return _flatten_features(features)


def _flatten_features(features: list) -> pd.DataFrame:
    """Flatten the nested GeoJSON feature list into a flat DataFrame."""
    rows = []
    for f in features:
        props = f.get("properties", {})
        coords = f.get("geometry", {}).get("coordinates", [None, None, None])

        # Defensive: coordinates should be [lon, lat, depth] but pad if shorter
        lon = coords[0] if len(coords) > 0 else None
        lat = coords[1] if len(coords) > 1 else None
        depth = coords[2] if len(coords) > 2 else None

        rows.append({
            "id": f.get("id"),
            "time": props.get("time"),               # ms since epoch
            "magnitude": props.get("mag"),
            "place": props.get("place"),
            "longitude": lon,
            "latitude": lat,
            "depth_km": depth,
            "significance": props.get("sig"),
            "tsunami": props.get("tsunami"),
            "felt_reports": props.get("felt"),
            "cdi": props.get("cdi"),
            "mmi": props.get("mmi"),
            "alert": props.get("alert"),
            "event_type": props.get("type"),
            "status": props.get("status"),
            "url": props.get("url"),
        })

    return pd.DataFrame(rows)


def _empty_dataframe() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical column set."""
    return pd.DataFrame(columns=[
        "id", "time", "magnitude", "place", "longitude", "latitude", "depth_km",
        "significance", "tsunami", "felt_reports", "cdi", "mmi", "alert",
        "event_type", "status", "url",
    ])


# Convenience helper for the "last N days" pattern we'll use a lot
def fetch_recent(days: int = 30, min_magnitude: float = 2.5) -> pd.DataFrame:
    """Fetch the last N days of earthquakes ending now (UTC)."""
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    return fetch_earthquakes(start, end, min_magnitude=min_magnitude)