# -*- coding: utf-8 -*-
"""
build_predicted_dashboard.py
Builds predicted behavior dashboard from pipeline JSON.
Shows what the MODEL SAID would happen.
"""

import os
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PIPELINE_JSON = "output/pipeline_day/pipeline_2025-01-05.json"
OUT_DIR       = "output/predicted"
TOP_N         = 5
os.makedirs(OUT_DIR, exist_ok=True)

PC = {
    "NORMAL"      : "#4A90D9",
    "ORGANIZING"  : "#F39C12",
    "INTENSIFYING": "#9B59B6",
    "EVENT"       : "#E8593C",
}


def build_predicted_dashboard(data):
    clusters = data["clusters"]
    summary  = data["summary"]
    date     = data["metadata"]["date"]

    sorted_clusters = sorted(
        clusters.values(),
        key=lambda c: c["n_events_true"],
        reverse=True
    )[:TOP_N]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f"Predicted cloud classes — {date} (top {TOP_N} clusters)",
            "Predicted class distribution",
            "Stage 1 probability timeline (cluster 6846)",
            "Prediction accuracy per cluster",
        ],
        specs=[
            [{"type": "scattergeo", "colspan": 2}, None],
            [{"type": "xy"},                        {"type": "xy"}],
        ],
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    colors_map = ["#4A90D9","#27AE60","#E8593C","#9B59B6","#F39C12"]
    all_lats, all_lons = [], []

    # ---- PANEL 1: MAP colored by PREDICTED class ----
    for ci, cl in enumerate(sorted_clusters):
        steps = cl["steps"]
        cid   = cl["cluster_id"]
        color = colors_map[ci % len(colors_map)]

        fig.add_trace(go.Scattergeo(
            lat=[s["centroid_lat"] for s in steps],
            lon=[s["centroid_lon"] for s in steps],
            mode="lines+markers",
            name=f"Cluster {cid}",
            line=dict(color=color, width=1.5, dash="dot"),
            marker=dict(
                size=8,
                color=[PC.get(s["pred_final"], "#888") for s in steps],
                symbol=["diamond" if s["pred_final"] == "INTENSIFYING"
                        else "square" if s["pred_final"] == "ORGANIZING"
                        else "circle" for s in steps],
                line=dict(width=0.5, color="white"),
            ),
            text=[f"Cluster {cid}<br>{s['time'][11:16]}<br>"
                  f"Pred: {s['pred_final']}<br>True: {s['true_behavior']}"
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

    # ---- PANEL 2: Predicted class distribution ----
    all_preds = {}
    for cl in clusters.values():
        for cls, cnt in cl["pred_classes"].items():
            all_preds[cls] = all_preds.get(cls, 0) + cnt

    cls_order  = ["NORMAL","ORGANIZING","INTENSIFYING"]
    cls_counts = [all_preds.get(c, 0) for c in cls_order]
    cls_colors = [PC.get(c, "#888") for c in cls_order]

    fig.add_trace(go.Bar(
        x=cls_order, y=cls_counts,
        marker_color=cls_colors,
        marker_line_color="white", marker_line_width=0.5,
        name="Predicted class count",
        hovertemplate="%{x}: %{y}<extra></extra>",
    ), row=2, col=1)

    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Predicted class", row=2, col=1)

    # ---- PANEL 3: Cluster 6846 Stage 1 probability ----
    cl6846 = clusters.get("6846")
    if cl6846:
        steps   = cl6846["steps"]
        times_x = list(range(len(steps)))
        s1_prob = [s["s1_prob"] for s in steps]
        pred_ev = [1 if s["s1_pred"] == "EVENT" else 0 for s in steps]
        true_ev = [1 if s["true_grouped"] != "NORMAL" else 0 for s in steps]

        fig.add_trace(go.Scatter(
            x=times_x, y=s1_prob, mode="lines",
            name="S1 P(EVENT) — 6846",
            line=dict(color="#4A90D9", width=2),
            fill="tozeroy", fillcolor="rgba(74,144,217,0.10)",
            hovertemplate="Step %{x}: P(EVENT)=%{y:.3f}<extra></extra>",
        ), row=2, col=2)

        # Threshold line via add_shape
        fig.add_shape(type="line", xref="x3", yref="y3",
                      x0=0, x1=len(steps)-1,
                      y0=0.3, y1=0.3,
                      line=dict(color="#F39C12", dash="dash", width=1.5))
        fig.add_annotation(xref="x3", yref="y3",
                           x=len(steps)-1, y=0.32,
                           text="Threshold 0.3", showarrow=False,
                           font=dict(size=9, color="#F39C12"),
                           xanchor="right", yanchor="bottom")

        # True event markers
        true_ev_x = [i for i in times_x if true_ev[i] == 1]
        true_ev_y = [s1_prob[i] for i in true_ev_x]
        if true_ev_x:
            fig.add_trace(go.Scatter(
                x=true_ev_x, y=true_ev_y, mode="markers",
                name="True event (6846)",
                marker=dict(size=10, symbol="diamond", color="#27AE60",
                            line=dict(width=1.5, color="white")),
                hovertemplate="True event step %{x}<extra></extra>",
            ), row=2, col=2)

        tlabels = [steps[i]["time"][11:16]
                   for i in range(0, len(steps), max(1, len(steps)//8))]
        tvals   = list(range(0, len(steps), max(1, len(steps)//8)))
        fig.update_xaxes(tickvals=tvals, ticktext=tlabels,
                         tickangle=45, row=2, col=2)
        fig.update_yaxes(title_text="P(EVENT)", range=[0,1], row=2, col=2)

    # ---- PANEL 4: Accuracy per cluster (bar) ----
    # Shown as xy subplot row=2 col=2 — but we already used that
    # Instead embed accuracy stats as annotations on map

    # ---- OVERALL LAYOUT ----
    fig.update_layout(
        title=dict(
            text=(f"PREDICTED Cloud Classes — {date}<br>"
                  f"<sup>{len(clusters)} clusters | "
                  f"Event recall: {summary['event_recall']:.1%} | "
                  f"Events found: {summary['total_events_found']}"
                  f"/{summary['total_events_true']} | "
                  f"Threshold S1={data['metadata']['threshold_s1']}</sup>"),
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

    print(f"  Date    : {data['metadata']['date']}")
    print(f"  Summary : {data['summary']}")

    print("\nBuilding predicted dashboard...")
    fig = build_predicted_dashboard(data)

    out = os.path.join(OUT_DIR, f"predicted_{data['metadata']['date']}.html")
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()