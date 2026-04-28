"""
Plotly chart components for the dashboard.

All charts share a Mission Control color palette and dark theme.
Each function takes a DataFrame and returns a configured plotly.graph_objects.Figure.
The figures are then rendered in app.py via st.plotly_chart().
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# DESIGN TOKENS — match assets/styles.css
# ============================================================
COLORS = {
    "bg_base":        "#0a0e1a",
    "bg_surface":     "#141a26",
    "border":         "#2a3447",
    "text_primary":   "#e8ecf4",
    "text_secondary": "#8b96a8",
    "text_tertiary":  "#5c6577",
    "cyan":           "#06d6f7",
    "cyan_dim":       "#0891a8",
    "low":            "#4ade80",
    "moderate":       "#fbbf24",
    "high":           "#fb923c",
    "severe":         "#ef4444",
    "critical":       "#dc2626",
}

RISK_TIER_COLORS = {
    "Minor":    COLORS["text_tertiary"],
    "Light":    COLORS["cyan"],
    "Moderate": COLORS["moderate"],
    "Strong":   COLORS["high"],
    "Major":    COLORS["severe"],
    "Great":    COLORS["critical"],
}


def _apply_theme(fig: go.Figure, height: int = 320) -> go.Figure:
    """Apply our Mission Control theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor=COLORS["bg_surface"],
        plot_bgcolor=COLORS["bg_surface"],
        font=dict(
            family="Inter, sans-serif",
            color=COLORS["text_primary"],
            size=12,
        ),
        title_font=dict(
            family="JetBrains Mono, monospace",
            size=14,
            color=COLORS["text_primary"],
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        height=height,
        xaxis=dict(
            gridcolor=COLORS["border"],
            zerolinecolor=COLORS["border"],
            tickfont=dict(family="JetBrains Mono, monospace", size=10),
        ),
        yaxis=dict(
            gridcolor=COLORS["border"],
            zerolinecolor=COLORS["border"],
            tickfont=dict(family="JetBrains Mono, monospace", size=10),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=COLORS["text_secondary"]),
        ),
        hoverlabel=dict(
            bgcolor=COLORS["bg_base"],
            bordercolor=COLORS["cyan"],
            font=dict(family="JetBrains Mono, monospace", size=11, color=COLORS["text_primary"]),
        ),
    )
    return fig


# ============================================================
# DAILY ACTIVITY LINE CHART
# ============================================================

def daily_activity_chart(daily_df: pd.DataFrame) -> go.Figure:
    """
    Line chart: events per day with a 7-day rolling average overlay.
    Expects DataFrame with: date, event_count, avg_magnitude, rolling_avg.
    """
    fig = go.Figure()

    if daily_df.empty:
        fig.add_annotation(text="No data", showarrow=False)
        return _apply_theme(fig)

    # Daily count — primary line
    fig.add_trace(go.Scatter(
        x=daily_df["date"],
        y=daily_df["event_count"],
        mode="lines+markers",
        name="Daily count",
        line=dict(color=COLORS["cyan"], width=2),
        marker=dict(size=5, color=COLORS["cyan"]),
        hovertemplate="<b>%{x}</b><br>%{y} events<extra></extra>",
    ))

    # 7-day rolling avg (computed inline if not already present)
    rolling = daily_df["event_count"].rolling(window=7, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=daily_df["date"],
        y=rolling,
        mode="lines",
        name="7-day rolling avg",
        line=dict(color=COLORS["cyan_dim"], width=1.5, dash="dot"),
        hovertemplate="<b>%{x}</b><br>7d avg: %{y:.1f}<extra></extra>",
    ))

    fig.update_layout(
        title="DAILY EVENT COUNT",
        xaxis_title=None,
        yaxis_title="Events",
    )

    return _apply_theme(fig)


# ============================================================
# AVG MAGNITUDE TREND CHART
# ============================================================

def magnitude_trend_chart(daily_df: pd.DataFrame) -> go.Figure:
    """
    Line chart: daily average magnitude over time.
    Expects DataFrame with: date, avg_magnitude, max_magnitude.
    """
    fig = go.Figure()

    if daily_df.empty:
        fig.add_annotation(text="No data", showarrow=False)
        return _apply_theme(fig)

    # Avg magnitude line
    fig.add_trace(go.Scatter(
        x=daily_df["date"],
        y=daily_df["avg_magnitude"],
        mode="lines+markers",
        name="Daily avg",
        line=dict(color=COLORS["cyan"], width=2),
        marker=dict(size=5),
        hovertemplate="<b>%{x}</b><br>Avg: M%{y:.2f}<extra></extra>",
    ))

    # Max magnitude — secondary line
    fig.add_trace(go.Scatter(
        x=daily_df["date"],
        y=daily_df["max_magnitude"],
        mode="lines",
        name="Daily peak",
        line=dict(color=COLORS["high"], width=1.5, dash="dot"),
        hovertemplate="<b>%{x}</b><br>Peak: M%{y:.1f}<extra></extra>",
    ))

    fig.update_layout(
        title="MAGNITUDE TREND",
        xaxis_title=None,
        yaxis_title="Magnitude",
    )

    return _apply_theme(fig)


# ============================================================
# MAGNITUDE BUCKET HISTOGRAM
# ============================================================

def magnitude_bucket_chart(buckets_df: pd.DataFrame) -> go.Figure:
    """
    Bar chart: count of events per integer magnitude bucket.
    Demonstrates the Gutenberg-Richter exponential decay pattern.
    """
    fig = go.Figure()

    if buckets_df.empty:
        return _apply_theme(fig)

    # Color each bar by tier — M0-3 muted, M4-5 cyan, M5-6 amber, M6+ red
    bucket_colors = []
    for bucket in buckets_df["magnitude_bucket"]:
        if bucket in ["M0-1", "M1-2", "M2-3"]:
            bucket_colors.append(COLORS["text_tertiary"])
        elif bucket == "M3-4":
            bucket_colors.append(COLORS["cyan_dim"])
        elif bucket == "M4-5":
            bucket_colors.append(COLORS["cyan"])
        elif bucket == "M5-6":
            bucket_colors.append(COLORS["moderate"])
        elif bucket == "M6-7":
            bucket_colors.append(COLORS["high"])
        else:  # M7-8, M8+
            bucket_colors.append(COLORS["severe"])

    fig.add_trace(go.Bar(
        x=buckets_df["magnitude_bucket"],
        y=buckets_df["event_count"],
        marker=dict(color=bucket_colors, line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>%{y} events<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        title="MAGNITUDE DISTRIBUTION",
        xaxis_title=None,
        yaxis_title="Events",
        bargap=0.15,
    )

    return _apply_theme(fig)


# ============================================================
# DEPTH DISTRIBUTION DONUT
# ============================================================

def depth_distribution_chart(depth_df: pd.DataFrame) -> go.Figure:
    """
    Donut chart: shallow vs intermediate vs deep events.
    Donut > pie because it leaves room for a center label.
    """
    fig = go.Figure()

    if depth_df.empty or depth_df["event_count"].sum() == 0:
        return _apply_theme(fig)

    depth_color_map = {
        "Shallow":      COLORS["severe"],   # most damaging
        "Intermediate": COLORS["moderate"],
        "Deep":         COLORS["cyan"],
    }
    colors_ordered = [depth_color_map.get(c, COLORS["text_secondary"]) for c in depth_df["depth_category"]]

    total = depth_df["event_count"].sum()

    fig.add_trace(go.Pie(
        labels=depth_df["depth_category"],
        values=depth_df["event_count"],
        hole=0.6,
        marker=dict(colors=colors_ordered, line=dict(color=COLORS["bg_surface"], width=2)),
        textfont=dict(family="JetBrains Mono, monospace", color=COLORS["text_primary"], size=11),
        hovertemplate="<b>%{label}</b><br>%{value} events (%{percent})<extra></extra>",
        textinfo="label+percent",
    ))

    fig.update_layout(
        title="DEPTH DISTRIBUTION",
        annotations=[dict(
            text=f"<b>{total:,}</b><br><span style='font-size:11px; color:{COLORS['text_secondary']};'>events</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(family="JetBrains Mono, monospace", size=20, color=COLORS["text_primary"]),
        )],
        showlegend=False,
    )

    return _apply_theme(fig)


# ============================================================
# DEPTH BY REGION STACKED BAR
# ============================================================

def depth_by_region_chart(depth_region_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal stacked bar: shallow/intermediate/deep breakdown per top country.
    Reveals which regions are dominated by which depth profile.
    """
    fig = go.Figure()

    if depth_region_df.empty:
        return _apply_theme(fig)

    # Sort with most active at top of chart (Plotly puts first row at bottom by default)
    df = depth_region_df.sort_values("total", ascending=True)

    fig.add_trace(go.Bar(
        y=df["country_name"],
        x=df["Shallow"],
        name="Shallow",
        orientation="h",
        marker=dict(color=COLORS["severe"]),
        hovertemplate="<b>%{y}</b><br>Shallow: %{x}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        y=df["country_name"],
        x=df["Intermediate"],
        name="Intermediate",
        orientation="h",
        marker=dict(color=COLORS["moderate"]),
        hovertemplate="<b>%{y}</b><br>Intermediate: %{x}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        y=df["country_name"],
        x=df["Deep"],
        name="Deep",
        orientation="h",
        marker=dict(color=COLORS["cyan"]),
        hovertemplate="<b>%{y}</b><br>Deep: %{x}<extra></extra>",
    ))

    fig.update_layout(
        title="DEPTH BREAKDOWN BY REGION",
        barmode="stack",
        xaxis_title="Events",
        yaxis_title=None,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,            # below the x-axis
            xanchor="center",
            x=0.5,              # horizontally centered
        ),
    )

    fig.update_layout(margin=dict(l=40, r=20, t=50, b=70))  # extra bottom margin for the legend
    return _apply_theme(fig, height=400)