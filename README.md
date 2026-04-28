# 🌍 Global Earthquake Risk & Seismic Activity Dashboard

**Live dashboard:** https://earthquake-risk-dashboards.streamlit.app/

A real-time seismic risk intelligence platform powered by live data from the USGS Earthquake Catalog API. Every analysis is built fresh on each load — no pre-downloaded datasets.

---

## What it does

- **Live data pipeline** — calls the USGS API on every run, pulling real earthquake data for any user-specified date range and magnitude threshold
- **Interactive world map** — earthquake markers (sized by magnitude, colored by risk tier), tectonic plate boundaries (PB2002), and K-Means hotzone overlays
- **Time-series analysis** — daily event counts, magnitude trends, 7-day rolling averages
- **Magnitude & depth profiling** — Gutenberg-Richter distribution, shallow vs intermediate vs deep breakdown per region
- **Top 10 Risk Report** — composite risk score blending event count, peak magnitude, USGS significance, and shallow event percentage
- **Auto-generated narrative** — plain-English Notable Events summary, updated with every data load
- **Baseline anomaly detection** — compares current window against prior 90-day historical baseline, with statistical guardrails for low-sample periods
- **CSV export** — risk report and full event dataset downloadable on demand

---

## Tech stack

| Layer | Tools |
|---|---|
| Data collection | `requests`, USGS Earthquake Catalog API |
| Data processing | `pandas`, `numpy` |
| Geospatial enrichment | `reverse_geocoder`, `pycountry` |
| Clustering | `scikit-learn` K-Means |
| Mapping | `folium`, `streamlit-folium` |
| Charts | `plotly` |
| Dashboard | `streamlit` |
| Tectonic data | PB2002 (Bird, 2003) |

---

## Data sources

- **USGS Earthquake Catalog API** — https://earthquake.usgs.gov/fdsnws/event/1/
  Free, no API key required, updated continuously, coverage from 1900 to present.

- **PB2002 Tectonic Plate Boundaries** — Bird, P. (2003). An updated digital model of plate boundaries. *Geochemistry, Geophysics, Geosystems*, 4(3), 1027.
  https://doi.org/10.1029/2001GC000252

---

## Key findings (from live data, April 2026)

- Greece recorded **291% above its 90-day baseline** activity rate during the test period — flagged as anomalous by the baseline comparison engine
- The Nevada M4.78 earthquake near Silver Springs generated **764 public felt reports** despite its modest magnitude — shallow depth and proximity to population centers explain the discrepancy
- Three M7+ events occurred globally in the 30-day test window (Japan M7.4, Indonesia M7.4, Vanuatu M7.3) — all `mww` moment magnitude, the most authoritative scale. None generated widespread news coverage despite their severity, illustrating the dashboard's value: **news cycles cover approximately 1 in 5 major earthquakes; this dashboard surfaces all of them**
- K-Means clustering correctly and independently identified the Pacific Ring of Fire's major subduction zones as distinct hotzones, matching known seismological boundaries without any geographic prior

---

## Known limitations

- **USGS 20,000-event hard cap** — queries that would return more than 20,000 events are rejected by the API. The dashboard detects this and prompts the user to narrow the date range or raise the magnitude threshold. For global data at M2.5+, the cap is reached in approximately 4-6 months.

- **Baseline comparisons require stable sample sizes** — the baseline anomaly flag is suppressed when the current window exceeds 180 days or when a region has fewer than 5 baseline events. Low-frequency regions (e.g., Mongolia at M4.5+) cannot produce reliable deviation percentages.

- **K-Means uses Euclidean distance** — treats the Earth as flat, which is acceptable for identifying broad regional hotzones but means cluster boundaries are approximate near the antimeridian (180° longitude).

- **Offshore earthquake attribution** — events in the open ocean are attributed to the nearest land territory via `reverse_geocoder`. Sub-Antarctic events are classified as Antarctica per geographic convention.

- **Folium map renders client-side** — the Python pipeline completes before the browser finishes drawing all markers. A note is displayed to set expectations for large datasets.

- **Risk scores are relative, not absolute** — scores are normalised within the current dataset, so the top region always anchors near the upper end regardless of whether it was actually an unusually active period globally.

- **Not for emergency use** — this dashboard is an analytical and portfolio tool. For real-time emergency information, use the official USGS Earthquake Hazards Program at https://earthquake.usgs.gov

---

## Running locally

```bash
# Clone the repo
git clone https://github.com/anirudh-rs/earthquake-risk-dashboard.git
cd earthquake-risk-dashboard

# Create and activate environment
conda create -n earthquake-dash python=3.11 -y
conda activate earthquake-dash

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

---

## Project structure

```
earthquake-risk-dashboard/
├── app.py                      # Streamlit entry point
├── requirements.txt            # Dependencies
├── README.md
├── .gitignore
├── src/
│   ├── data_pipeline.py        # USGS API calls + JSON parsing
│   ├── data_cleaning.py        # Cleaning, enrichment, risk tiers
│   ├── clustering.py           # K-Means hotzone detection
│   ├── analytics.py            # Aggregations + baseline comparison
│   └── risk_report.py          # Risk scoring + narrative generation
├── components/
│   ├── kpi_cards.py            # KPI strip
│   ├── map_view.py             # Folium map builder
│   ├── time_series.py          # Plotly charts
│   └── risk_brief.py           # Risk table + narrative display
└── assets/
    ├── styles.css              # Mission Control design system
    └── tectonic_plates.geojson # PB2002 plate boundaries
```

---

## Methodology notes

**Risk score** blends four inputs into a 0-100 composite:
- Event count (weight 30) — raw activity volume
- Peak magnitude (weight 30) — single-event severity
- Average USGS significance (weight 20) — impact-weighted score
- Shallow event percentage (weight 20) — depth-weighted damage potential

Each input is normalised to 0-1 within the current dataset before weighting.

**Depth classification** follows standard seismological boundaries: shallow (0-70 km), intermediate (70-300 km), deep (300+ km). Shallow events are weighted most heavily in risk scoring because they cause the greatest surface impact at equivalent magnitudes.

**Baseline comparison** uses a 90-day window immediately preceding the current query period, normalised to events-per-day to allow fair comparison across windows of different lengths.

---

*Data: USGS Earthquake Catalog API · Tectonics: PB2002 (Bird, 2003) · Not for emergency use*
