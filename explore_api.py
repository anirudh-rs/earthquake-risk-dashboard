"""Smoke test for the risk report module."""

from src.data_pipeline import fetch_earthquakes, fetch_recent
from src.data_cleaning import clean_earthquakes
from src.risk_report import score_regions, generate_notable_events_narrative
from datetime import datetime, timedelta

print("Fetching current week...")
current = clean_earthquakes(fetch_recent(days=7, min_magnitude=4.5))
print(f"{len(current)} events\n")

print("Fetching 90-day baseline...")
end = datetime.utcnow() - timedelta(days=7)
start = end - timedelta(days=90)
baseline = clean_earthquakes(fetch_earthquakes(start, end, min_magnitude=4.5))
print(f"{len(baseline)} baseline events\n")

print("=== TOP 10 RISK REPORT ===")
report = score_regions(
    df=current,
    baseline_df=baseline,
    current_window_days=7,
    baseline_window_days=90,
)
print(report[[
    "country_name", "event_count", "max_magnitude",
    "shallow_pct", "risk_score", "risk_rating", "anomaly_flag"
]].to_string())

print("\n=== PER-REGION SUMMARIES ===")
for _, row in report.iterrows():
    print(f"• {row['summary']}")

print("\n=== NOTABLE EVENTS NARRATIVE ===")
bullets = generate_notable_events_narrative(
    df=current,
    baseline_df=baseline,
    current_window_days=7,
    baseline_window_days=90,
)
for b in bullets:
    print(f"• {b}")