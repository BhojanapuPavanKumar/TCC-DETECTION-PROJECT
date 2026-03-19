# -*- coding: utf-8 -*-
"""
build_actual_dashboard.py
Builds actual behavior dashboard from pipeline JSON.
Shows what REALLY happened — ground truth.
"""

import os
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PIPELINE_JSON = "output/pipeline_day/pipeline_2025-01-05.json"
OUT_DIR       = "output/actual"
TOP_N         = 5   # show top N clusters by event count
os.makedirs(OUT_DIR, exist_ok=True)

BC = {
    "STABLE"      : "#4A90D9",
    "GROWING"     : "#27AE60",
    "SHRINKING"   : "#E8593C",
    "INTENSIFYING": "#9B59B6",
    "SPLITTING"   : "#F39C12",
    "MERGING"     : "#1ABC9C",
    "NEW"         : "#95A5A6",
    "NORMAL"      : "#4A90D9",
    "ORGANIZING"  : "#F39C12",
}


def build_map_trace(steps, name, color, show_legend=True):
    lats = [s["centroid_lat"] for s in steps]
    lons = [s["centroid_lon"] for s in steps]
    bhvs = [s["true_behavior"] for s in steps]
    times= [s["time"][11:16] for s in steps]
    return go.Scattergeo(
        lat=lats, lon=lons, mode="lines+markers",
        name=name,
        line=dict(color=color, width=1.5),
        marker=dict(
            size=7,
            color=[BC.get(b, "#888") for b in bhvs],
            line=dict(width=0.5, color="white"),
        ),
        text=[f"{times[i]}<br>{bhvs[i]}" for i in range(len(steps))],
        hoverinfo="text",
        showlegend=show_legend,
    )


def build_actual_dashboard(data):
    clusters = data["clusters"]
    summary  = data["summary"]
    date     = data["metadata"]["date"]

    # Sort clusters by event count
    sorted_clusters = sorted(
        clusters.values(),
        key=lambda c: c["n_events_true"],
        reverse=True
    )[:TOP_N]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f"Actual cloud tracks — {date} (top {TOP_N} by events)",
            "Behavior distribution — actual",
            "Cloud lifecycle timeline (cluster 6846)",
            "Day summary statistics",
        ],
        specs=[
            [{"type": "scattergeo", "colspan": 2}, None],
            [{"type": "xy"},                        {"type": "xy"}],
        ],
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    # ---- PANEL 1: MAP of all top clusters ----
    colors_map = ["#4A90D9","#27AE60","#E8593C","#9B59B6","#F39C12"]
    all_lats, all_lons = [], []

    for ci, cl in enumerate(sorted_clusters):
        steps = cl["steps"]
        color = colors_map[ci % len(colors_map)]
        cid   = cl["cluster_id"]

        fig.add_trace(go.Scattergeo(
            lat=[s["centroid_lat"] for s in steps],
            lon=[s["centroid_lon"] for s in steps],
            mode="lines+markers",
            name=f"Cluster {cid} (n={cl['n_events_true']} events)",
            line=dict(color=color, width=2),
            marker=dict(
                size=7,
                color=[BC.get(s["true_behavior"], "#888") for s in steps],
                line=dict(width=0.5, color="white"),
            ),
            text=[f"Cluster {cid}<br>{s['time'][11:16]}<br>{s['true_behavior']}"
                  for s in steps],
            hoverinfo="text",
        ), row=1, col=1)

        all_lats += [s["centroid_lat"] for s in steps]
        all_lons += [s["centroid_lon"] for s in steps]

    lat_c = float(np.mean(all_lats)) if all_lats else 0
    lon_c = float(np.mean(all_lons)) if all_lons else 70

    fig.update_geos(
        projection_type="mercator",
        center=dict(lat=lat_c, lon=lon_c),
        lataxis_range=[lat_c - 15, lat_c + 15],
        lonaxis_range=[lon_c - 20, lon_c + 20],
        showland=True,       landcolor="#F0EBE0",
        showocean=True,      oceancolor="#D6E8F5",
        showcoastlines=True, coastlinecolor="#AAAAAA",
        showcountries=True,  countrycolor="#CCCCCC",
        resolution=50,
    )

    # ---- PANEL 2: Behavior distribution bar ----
    all_behaviors = {}
    for cl in clusters.values():
        for beh, cnt in cl["true_behaviors"].items():
            all_behaviors[beh] = all_behaviors.get(beh, 0) + cnt

    beh_order = ["STABLE","GROWING","SHRINKING",
                 "INTENSIFYING","SPLITTING","MERGING","NEW"]
    beh_counts = [all_behaviors.get(b, 0) for b in beh_order]
    beh_colors = [BC.get(b, "#888") for b in beh_order]

    fig.add_trace(go.Bar(
        x=beh_order, y=beh_counts,
        marker_color=beh_colors,
        marker_line_color="white", marker_line_width=0.5,
        name="Actual behavior count",
        hovertemplate="%{x}: %{y}<extra></extra>",
    ), row=2, col=1)

    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Behavior", row=2, col=1)

    # ---- PANEL 3: Cluster 6846 lifecycle timeline ----
    cl6846 = clusters.get("6846")
    if cl6846:
        steps   = cl6846["steps"]
        times_x = list(range(len(steps)))
        tbs     = [s["mean_tb"] for s in steps]
        areas   = [s["area_km2"] for s in steps]
        bhvs    = [s["true_behavior"] for s in steps]

        fig.add_trace(go.Scatter(
            x=times_x, y=tbs, mode="lines+markers",
            name="TB (K) — cluster 6846",
            line=dict(color="#9B59B6", width=2),
            marker=dict(size=5,
                        color=[BC.get(b, "#888") for b in bhvs]),
            hovertemplate="Step %{x}: TB=%{y:.1f}K<extra></extra>",
            yaxis="y3",
        ), row=2, col=2)

        fig.add_trace(go.Scatter(
            x=times_x, y=areas, mode="lines",
            name="Area km2 — cluster 6846",
            line=dict(color="#4A90D9", width=1.5, dash="dot"),
            hovertemplate="Step %{x}: Area=%{y:.0f}km2<extra></extra>",
            yaxis="y4",
            opacity=0.7,
        ), row=2, col=2)

        # Mark events
        event_x = [i for i, b in enumerate(bhvs)
                   if b in ("INTENSIFYING","SPLITTING","MERGING")]
        event_y = [tbs[i] for i in event_x]
        event_b = [bhvs[i] for i in event_x]
        if event_x:
            fig.add_trace(go.Scatter(
                x=event_x, y=event_y, mode="markers",
                name="Events (6846)",
                marker=dict(
                    size=12, symbol="star",
                    color=[BC.get(b, "#888") for b in event_b],
                    line=dict(width=1, color="white"),
                ),
                text=event_b,
                hovertemplate="Event: %{text}<extra></extra>",
            ), row=2, col=2)

        tlabels_6846 = [steps[i]["time"][11:16]
                        for i in range(0, len(steps), max(1, len(steps)//8))]
        tvals_6846   = list(range(0, len(steps), max(1, len(steps)//8)))
        fig.update_xaxes(tickvals=tvals_6846, ticktext=tlabels_6846,
                         tickangle=45, row=2, col=2)
        fig.update_yaxes(title_text="mean_tb (K)", row=2, col=2)

    # ---- OVERALL LAYOUT ----
    total_ev = sum(c["n_events_true"] for c in clusters.values())
    fig.update_layout(
        title=dict(
            text=(f"ACTUAL Cloud Behavior — {date}<br>"
                  f"<sup>{len(clusters)} clusters predicted | "
                  f"{total_ev} real events | "
                  f"Overall accuracy: {summary['overall_accuracy']:.1%} | "
                  f"Event recall: {summary['event_recall']:.1%}</sup>"),
            font=dict(size=15),
        ),
        height=900,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.06,
                    xanchor="center", x=0.5, font=dict(size=9)),
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=11),
        margin=dict(t=90, b=100, l=60, r=60),
    )
    return fig


def main():
    print("Loading pipeline results...")
    with open(PIPELINE_JSON) as f:
        data = json.load(f)

    print(f"  Date     : {data['metadata']['date']}")
    print(f"  Clusters : {len(data['clusters'])}")
    print(f"  Summary  : {data['summary']}")

    print("\nBuilding actual dashboard...")
    fig = build_actual_dashboard(data)

    out = os.path.join(OUT_DIR, f"actual_{data['metadata']['date']}.html")
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()