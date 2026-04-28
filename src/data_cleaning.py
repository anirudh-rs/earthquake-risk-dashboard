"""
Earthquake data cleaning and enrichment.

Takes the raw DataFrame from data_pipeline.fetch_earthquakes() and:
  - Converts timestamps to datetime
  - Filters to actual earthquakes (drops quarry blasts, explosions, etc.)
  - Adds risk tier, depth category, and country/continent enrichment

Returns a DataFrame ready for analytics and visualization.
"""

import pandas as pd
import numpy as np
import reverse_geocoder as rg
import pycountry
from typing import Optional


# Magnitude tiers — based on USGS / Richter scale conventions
MAGNITUDE_TIERS = [
    (0.0, 3.0, "Minor"),
    (3.0, 4.0, "Light"),
    (4.0, 5.0, "Moderate"),
    (5.0, 6.0, "Strong"),
    (6.0, 7.0, "Major"),
    (7.0, 11.0, "Great"),
]

# Depth categories — standard seismological boundaries
DEPTH_CATEGORIES = [
    (0, 70, "Shallow"),       # Most damaging at surface
    (70, 300, "Intermediate"),
    (300, 1000, "Deep"),       # Less surface damage typically
]

# Continent lookup — pycountry doesn't include continents, so we map ISO country codes
# This is a small static dict; keeps us offline and dependency-light
COUNTRY_TO_CONTINENT = {
    # We'll populate this lazily — for now use a simple region map via reverse_geocoder
    # reverse_geocoder gives us ISO country codes, and we'll group below
}


def clean_earthquakes(
    df: pd.DataFrame,
    only_reviewed: bool = False,
    only_earthquakes: bool = True,
) -> pd.DataFrame:
    """
    Clean and enrich the raw earthquake DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw output from fetch_earthquakes().
    only_reviewed : bool
        If True, drop events with status='automatic' (not yet human-verified).
        Default False — most users want to see the latest data even if not reviewed.
    only_earthquakes : bool
        If True, drop non-earthquake events (quarry blasts, explosions, etc.).
        Default True.

    Returns
    -------
    pd.DataFrame
        Cleaned and enriched DataFrame, sorted by time descending.
    """
    if df.empty:
        return df.copy()

    df = df.copy()

    # --- Timestamps ---
    # USGS gives Unix milliseconds. Convert to UTC datetime.
    df["datetime_utc"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["date"] = df["datetime_utc"].dt.date

    # Strip timezone for period conversion (we already know it's UTC).
    # Periods are calendar buckets and don't carry timezones — converting tz-aware
    # datetimes raises a warning. Going via tz-naive UTC is explicit and clean.
    naive_utc = df["datetime_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
    df["year_month"] = naive_utc.dt.to_period("M").astype(str)
    df["year_week"] = naive_utc.dt.to_period("W").astype(str)

    # --- Event type filtering ---
    if only_earthquakes and "event_type" in df.columns:
        df = df[df["event_type"] == "earthquake"].copy()

    if only_reviewed and "status" in df.columns:
        df = df[df["status"] == "reviewed"].copy()

    # --- Drop rows missing core fields (should be rare/zero) ---
    core_fields = ["magnitude", "latitude", "longitude", "depth_km"]
    before = len(df)
    df = df.dropna(subset=core_fields)
    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows missing core fields (magnitude/lat/lon/depth)")

    # --- Risk tier (magnitude-based) ---
    df["risk_tier"] = df["magnitude"].apply(_assign_risk_tier)

    # --- Depth category ---
    df["depth_category"] = df["depth_km"].apply(_assign_depth_category)

    # --- Country / continent enrichment via reverse geocoding ---
    if len(df) > 0:
        df = _enrich_with_country(df)

    # --- Sort newest first ---
    df = df.sort_values("datetime_utc", ascending=False).reset_index(drop=True)

    return df


def _assign_risk_tier(magnitude: float) -> str:
    """Map a magnitude to its named risk tier."""
    if pd.isna(magnitude):
        return "Unknown"
    for low, high, name in MAGNITUDE_TIERS:
        if low <= magnitude < high:
            return name
    return "Great"  # anything M7+


def _assign_depth_category(depth_km: float) -> str:
    """Map a depth (km) to shallow/intermediate/deep."""
    if pd.isna(depth_km):
        return "Unknown"
    for low, high, name in DEPTH_CATEGORIES:
        if low <= depth_km < high:
            return name
    return "Deep"  # anything >= 1000km (rare)


# Module-level coordinate cache. Keyed by (rounded_lat, rounded_lon).
# Rounding to 1 decimal means coords within ~11km share a result, which is
# fine — country boundaries don't change at that scale. Massive perf win
# for repeated queries that share many events (e.g. expanding a date range).
_GEOCODE_CACHE: dict = {}


def _enrich_with_country(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use offline reverse_geocoder to add country_code, country_name, and continent.

    Caches per-coordinate (rounded to 1 decimal) so repeat queries are nearly free.
    Only uncached coordinates hit reverse_geocoder.search().
    """
    # Round coordinates for caching — 0.1° ≈ 11km, well below country resolution
    rounded_coords = list(zip(
        df["latitude"].round(1).astype(float),
        df["longitude"].round(1).astype(float),
    ))

    # Find which coordinates we haven't seen before
    uncached = [c for c in set(rounded_coords) if c not in _GEOCODE_CACHE]

    if uncached:
        # Only geocode the uncached coordinates
        new_results = rg.search(uncached, mode=1)
        for coord, result in zip(uncached, new_results):
            _GEOCODE_CACHE[coord] = result.get("cc", "")

    # Look up every event's country code from the cache
    country_codes = [_GEOCODE_CACHE.get(c, "") for c in rounded_coords]

    df["country_code"] = country_codes
    df["country_name"] = df["country_code"].apply(_iso2_to_name)
    df["continent"] = df["country_code"].apply(_iso2_to_continent)
    df["region"] = df["country_name"].fillna("Unknown")

    return df


def _iso2_to_name(iso2: str) -> str:
    """ISO 3166 alpha-2 country code → full country name."""
    if not iso2 or pd.isna(iso2):
        return "Unknown"
    try:
        country = pycountry.countries.get(alpha_2=iso2.upper())
        return country.name if country else iso2
    except (KeyError, AttributeError):
        return iso2


# Static ISO2 → continent mapping. Compact and offline.
_CONTINENT_MAP = {
    # Africa
    "DZ": "Africa", "AO": "Africa", "BJ": "Africa", "BW": "Africa", "BF": "Africa",
    "BI": "Africa", "CM": "Africa", "CV": "Africa", "CF": "Africa", "TD": "Africa",
    "KM": "Africa", "CD": "Africa", "CG": "Africa", "CI": "Africa", "DJ": "Africa",
    "EG": "Africa", "GQ": "Africa", "ER": "Africa", "ET": "Africa", "GA": "Africa",
    "GM": "Africa", "GH": "Africa", "GN": "Africa", "GW": "Africa", "KE": "Africa",
    "LS": "Africa", "LR": "Africa", "LY": "Africa", "MG": "Africa", "MW": "Africa",
    "ML": "Africa", "MR": "Africa", "MU": "Africa", "MA": "Africa", "MZ": "Africa",
    "NA": "Africa", "NE": "Africa", "NG": "Africa", "RW": "Africa", "ST": "Africa",
    "SN": "Africa", "SC": "Africa", "SL": "Africa", "SO": "Africa", "ZA": "Africa",
    "SS": "Africa", "SD": "Africa", "SZ": "Africa", "TZ": "Africa", "TG": "Africa",
    "TN": "Africa", "UG": "Africa", "ZM": "Africa", "ZW": "Africa", "EH": "Africa",
    "SH": "Africa",  # Saint Helena, Ascension, Tristan da Cunha (South Atlantic)
    # Asia
    "AF": "Asia", "AM": "Asia", "AZ": "Asia", "BH": "Asia", "BD": "Asia",
    "BT": "Asia", "BN": "Asia", "KH": "Asia", "CN": "Asia", "CY": "Asia",
    "GE": "Asia", "IN": "Asia", "ID": "Asia", "IR": "Asia", "IQ": "Asia",
    "IL": "Asia", "JP": "Asia", "JO": "Asia", "KZ": "Asia", "KW": "Asia",
    "KG": "Asia", "LA": "Asia", "LB": "Asia", "MY": "Asia", "MV": "Asia",
    "MN": "Asia", "MM": "Asia", "NP": "Asia", "KP": "Asia", "OM": "Asia",
    "PK": "Asia", "PS": "Asia", "PH": "Asia", "QA": "Asia", "SA": "Asia",
    "SG": "Asia", "KR": "Asia", "LK": "Asia", "SY": "Asia", "TW": "Asia",
    "TJ": "Asia", "TH": "Asia", "TL": "Asia", "TR": "Asia", "TM": "Asia",
    "AE": "Asia", "UZ": "Asia", "VN": "Asia", "YE": "Asia",
    # Europe
    "AL": "Europe", "AD": "Europe", "AT": "Europe", "BY": "Europe", "BE": "Europe",
    "BA": "Europe", "BG": "Europe", "HR": "Europe", "CZ": "Europe", "DK": "Europe",
    "EE": "Europe", "FI": "Europe", "FR": "Europe", "DE": "Europe", "GR": "Europe",
    "HU": "Europe", "IS": "Europe", "IE": "Europe", "IT": "Europe", "XK": "Europe",
    "LV": "Europe", "LI": "Europe", "LT": "Europe", "LU": "Europe", "MT": "Europe",
    "MD": "Europe", "MC": "Europe", "ME": "Europe", "NL": "Europe", "MK": "Europe",
    "NO": "Europe", "PL": "Europe", "PT": "Europe", "RO": "Europe", "RU": "Europe",
    "SM": "Europe", "RS": "Europe", "SK": "Europe", "SI": "Europe", "ES": "Europe",
    "SE": "Europe", "CH": "Europe", "UA": "Europe", "GB": "Europe", "VA": "Europe",
    # North America
    "AG": "North America", "BS": "North America", "BB": "North America", "BZ": "North America",
    "CA": "North America", "CR": "North America", "CU": "North America", "DM": "North America",
    "DO": "North America", "SV": "North America", "GD": "North America", "GT": "North America",
    "HT": "North America", "HN": "North America", "JM": "North America", "MX": "North America",
    "NI": "North America", "PA": "North America", "KN": "North America", "LC": "North America",
    "VC": "North America", "TT": "North America", "US": "North America",
    "PR": "North America", "GL": "North America",
    # South America
    "AR": "South America", "BO": "South America", "BR": "South America", "CL": "South America",
    "CO": "South America", "EC": "South America", "GY": "South America", "PY": "South America",
    "PE": "South America", "SR": "South America", "UY": "South America", "VE": "South America",
    "GF": "South America",  # French Guiana
    # Oceania
    "AU": "Oceania", "FJ": "Oceania", "KI": "Oceania", "MH": "Oceania", "FM": "Oceania",
    "NR": "Oceania", "NZ": "Oceania", "PW": "Oceania", "PG": "Oceania", "WS": "Oceania",
    "SB": "Oceania", "TO": "Oceania", "TV": "Oceania", "VU": "Oceania", "NC": "Oceania",
    "PF": "Oceania", "GU": "Oceania",
    "MP": "Oceania",  # Northern Mariana Islands
    # Antarctica
    "AQ": "Antarctica",
    "TF": "Antarctica",  # French Southern Territories (sub-Antarctic)
    "GS": "Antarctica",  # South Georgia and South Sandwich Islands (sub-Antarctic)
}


def _iso2_to_continent(iso2: str) -> str:
    """ISO 3166 alpha-2 country code → continent name."""
    if not iso2 or pd.isna(iso2):
        return "Unknown"
    return _CONTINENT_MAP.get(iso2.upper(), "Unknown")