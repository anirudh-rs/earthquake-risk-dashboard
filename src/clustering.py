import os
import warnings

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Suppress the harmless Intel/LLVM OpenMP duplicate-library warning from threadpoolctl.
# This is a known Windows + MKL artifact; we've already pinned to single-threaded mode
# via OMP_NUM_THREADS, so no actual threading conflict can occur.
# This warning does not appear on Linux (Streamlit Cloud).
warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")

"""
Geographic hotspot clustering.

Uses K-Means on (latitude, longitude) to identify the top N earthquake hotzones
in any given dataset. Each cluster summary includes its center, event count,
average magnitude, max magnitude, and a representative country/region label.

Note: K-Means uses Euclidean distance, which treats the Earth as flat.
For global hotzone identification this is acceptable — clusters won't span
the antimeridian (180°) in our use case because earthquake activity is
naturally regionalized.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from typing import Optional


def find_hotzones(
    df: pd.DataFrame,
    n_clusters: int = 8,
    min_events: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Identify the top earthquake hotzones using K-Means clustering.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned earthquake DataFrame (output of clean_earthquakes()).
        Must contain 'latitude', 'longitude', 'magnitude', 'country_name'.
    n_clusters : int
        Number of hotzones to identify. Default 8 — a sensible balance between
        too few (vague) and too many (noisy) for a global view.
    min_events : int
        Minimum events required for a cluster to be reported. Filters out
        sparse clusters that K-Means sometimes creates. Default 3.
    random_state : int
        For reproducibility. K-Means initialization is stochastic.

    Returns
    -------
    pd.DataFrame
        One row per hotzone with columns:
          - cluster_id
          - center_lat, center_lon
          - event_count
          - avg_magnitude, max_magnitude
          - dominant_country, dominant_continent
          - hotzone_label (e.g., "Indonesia / Asia hotzone")
        Sorted by event_count descending.
    """
    if df.empty or len(df) < n_clusters:
        # Not enough data to cluster meaningfully
        return _empty_hotzones()

    # Adjust n_clusters down if dataset is small
    n_clusters = min(n_clusters, len(df) // 2)
    if n_clusters < 2:
        return _empty_hotzones()

    # Fit K-Means on coordinates
    coords = df[["latitude", "longitude"]].values
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,  # multiple random starts → most stable result
    )
    labels = kmeans.fit_predict(coords)

    df = df.copy()
    df["cluster_id"] = labels

    # Aggregate per cluster
    summaries = []
    for cluster_id in range(n_clusters):
        cluster_df = df[df["cluster_id"] == cluster_id]

        if len(cluster_df) < min_events:
            continue  # skip sparse clusters

        # Dominant country = most frequent country in the cluster
        dominant_country = cluster_df["country_name"].mode().iloc[0]
        dominant_continent = cluster_df["continent"].mode().iloc[0]

        center_lat, center_lon = kmeans.cluster_centers_[cluster_id]

        summaries.append({
            "cluster_id": int(cluster_id),
            "center_lat": float(center_lat),
            "center_lon": float(center_lon),
            "event_count": int(len(cluster_df)),
            "avg_magnitude": round(float(cluster_df["magnitude"].mean()), 2),
            "max_magnitude": round(float(cluster_df["magnitude"].max()), 2),
            "dominant_country": dominant_country,
            "dominant_continent": dominant_continent,
            "hotzone_label": f"{dominant_country} / {dominant_continent}",
        })

    if not summaries:
        return _empty_hotzones()

    result = pd.DataFrame(summaries)
    result = result.sort_values("event_count", ascending=False).reset_index(drop=True)
    return result


def assign_clusters_to_events(
    df: pd.DataFrame,
    n_clusters: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Same clustering logic, but returns the original DataFrame with a 'cluster_id'
    column added. Useful when we want to color the map markers by cluster.

    Returns the input df with one extra column. If clustering is not possible
    (too few events), returns the df with cluster_id = -1 for all rows.
    """
    if df.empty or len(df) < n_clusters:
        df = df.copy()
        df["cluster_id"] = -1
        return df

    n_clusters = min(n_clusters, len(df) // 2)
    if n_clusters < 2:
        df = df.copy()
        df["cluster_id"] = -1
        return df

    coords = df[["latitude", "longitude"]].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(coords)

    df = df.copy()
    df["cluster_id"] = labels
    return df


def _empty_hotzones() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "cluster_id", "center_lat", "center_lon", "event_count",
        "avg_magnitude", "max_magnitude", "dominant_country",
        "dominant_continent", "hotzone_label",
    ])