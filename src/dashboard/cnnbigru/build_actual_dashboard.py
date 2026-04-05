# -*- coding: utf-8 -*-
"""
build_actual_dashboard.py
Builds actual behavior dashboard from pipeline JSON.
Shows what REALLY happened — ground truth.

Panels:
  1. Map     — top-N cluster tracks colored by actual behavior
  2. Bar     — behavior distribution (all clusters, counts + labels)
  3. Line    — TB + Area lifecycle for best cluster (dual Y)
  4. Pie     — event type breakdown (INTENSIFYING / SPLITTING / MERGING)
  5. Bar     — top-10 clusters by event count
  6. Scatter — cluster mean area vs mean TB colored by dominant behavior
  + Summary metric cards generated entirely from pipeline JSON data
"""

import os
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
from scripts.select_best_case import select_best_case

CLUSTER_ID, DATE = select_best_case()
PIPELINE_JSON    = f"output/pipeline_day/pipeline_{DATE}.json"
OUT_DIR          = "output/actual"
TOP_N            = 5
os.makedirs(OUT_DIR, exist_ok=True)

BC = {
    "STABLE"      : "#4A90D9",
    "GROWING"     : "#27AE60",
    "SHRINKING"   : "#E8593C",
    "INTENSIFYING": "#9B59B6",
    "SPLITTING"   : "#F39C12",
    "MERGING"     : "#1ABC9C",
    "NEW"         : "#95A5A6",
}

BEH_ORDER   = ["STABLE","GROWING","SHRINKING","INTENSIFYING","SPLITTING","MERGING","NEW"]
EVENT_TYPES = ["INTENSIFYING","SPLITTING","MERGING"]
COLORS_MAP  = ["#4A90D9","#27AE60","#E8593C","#9B59B6","#F39C12"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _mean(vals):
    return float(np.mean(vals)) if vals else 0.0


def compute_summary(clusters, summary, date):
    """Compute all scalar stats from raw pipeline cluster data."""
    all_behaviors   = Counter()
    all_areas, all_tbs = [], []
    event_types     = Counter()

    for cl in clusters.values():
        for beh, cnt in cl["true_behaviors"].items():
            all_behaviors[beh] += cnt
        for step in cl["steps"]:
            all_areas.append(step.get("area_km2", 0))
            all_tbs.append(step.get("mean_tb", 0))
            if step.get("true_behavior") in EVENT_TYPES:
                event_types[step["true_behavior"]] += 1

    total_obs    = sum(all_behaviors.values())
    total_events = sum(c["n_events_true"] for c in clusters.values())
    n_clusters   = len(clusters)
    n_active     = sum(1 for c in clusters.values() if c["n_events_true"] > 0)

    return {
        "date"             : date,
        "n_clusters"       : n_clusters,
        "n_active"         : n_active,
        "total_obs"        : total_obs,
        "total_events"     : total_events,
        "event_recall"     : summary.get("event_recall", 0),
        "overall_accuracy" : summary.get("overall_accuracy", 0),
        "mean_area_km2"    : _mean(all_areas),
        "mean_tb_k"        : _mean(all_tbs),
        "pct_events"       : total_events / total_obs * 100 if total_obs else 0,
        "all_behaviors"    : dict(all_behaviors),
        "event_types"      : dict(event_types),
    }


def make_summary_html(s):
    """Render metric cards as styled HTML from computed summary dict."""
    cards = [
        ("Clusters",      str(s["n_clusters"]),               "total tracked"),
        ("Active",        str(s["n_active"]),                  "with ≥1 event"),
        ("Total events",  str(s["total_events"]),              "INTENS+SPLIT+MERGE"),
        ("Event recall",  f"{s['event_recall']:.1%}",         "stage-1 model"),
        ("Accuracy",      f"{s['overall_accuracy']:.1%}",     "overall day"),
        ("Mean TB",       f"{s['mean_tb_k']:.1f} K",          "brightness temp"),
        ("Mean area",     f"{s['mean_area_km2']:.0f} km²",    "cluster size"),
        ("% events",      f"{s['pct_events']:.2f}%",          "of all obs"),
    ]
    divs = "".join(
        f"<div style='display:inline-block;margin:5px 8px;padding:10px 16px;"
        f"background:#f7f9fc;border-radius:8px;border:1px solid #dde4ef;"
        f"min-width:108px;text-align:center;vertical-align:top'>"
        f"<div style='font-size:11px;color:#666;margin-bottom:3px'>{lbl}</div>"
        f"<div style='font-size:19px;font-weight:700;color:#2c3e50'>{val}</div>"
        f"<div style='font-size:10px;color:#999;margin-top:2px'>{sub}</div>"
        f"</div>"
        for lbl, val, sub in cards
    )
    return (
        f"<div style='font-family:Arial,sans-serif;padding:10px 0 6px'>"
        f"<div style='font-size:13px;font-weight:700;color:#333;"
        f"margin-bottom:6px'>&#x1F4CA; Day summary — {s['date']}</div>"
        f"{divs}</div>"
    )


# ── main dashboard builder ────────────────────────────────────────────────────

def build_actual_dashboard(data):
    clusters = data["clusters"]
    summary  = data["summary"]
    date     = data["metadata"]["date"]

    s = compute_summary(clusters, summary, date)

    sorted_clusters = sorted(
        clusters.values(),
        key=lambda c: c["n_events_true"],
        reverse=True,
    )
    top_clusters = sorted_clusters[:TOP_N]

    # ── subplot grid: 3 rows × 2 cols ────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            f"Cloud tracks — {date}  (top {TOP_N} by events)",
            "Behavior distribution",
            f"Lifecycle — cluster {CLUSTER_ID}  (TB + area)",
            "Event type breakdown",
            "Top 10 clusters by event count",
            "Mean area vs mean TB per cluster",
        ],
        specs=[
            [{"type": "scattergeo", "colspan": 2}, None],
            [{"type": "xy"},                        {"type": "domain"}],
            [{"type": "xy"},                        {"type": "xy"}],
        ],
        vertical_spacing=0.09,
        horizontal_spacing=0.08,
        row_heights=[0.38, 0.31, 0.31],
    )

    # ── PANEL 1: Map ──────────────────────────────────────────────────────────
    all_lats, all_lons = [], []

    for ci, cl in enumerate(top_clusters):
        steps = cl["steps"]
        cid   = cl["cluster_id"]
        color = COLORS_MAP[ci % len(COLORS_MAP)]
        bhvs  = [st["true_behavior"] for st in steps]

        fig.add_trace(go.Scattergeo(
            lat=[st["centroid_lat"] for st in steps],
            lon=[st["centroid_lon"] for st in steps],
            mode="lines+markers",
            name=f"Cluster {cid} ({cl['n_events_true']} ev)",
            line=dict(color=color, width=2),
            marker=dict(
                size=7,
                color=[BC.get(b, "#888") for b in bhvs],
                line=dict(width=0.5, color="white"),
            ),
            text=[
                f"<b>Cluster {cid}</b><br>"
                f"Time: {st['time'][11:16]}<br>"
                f"Behavior: {st['true_behavior']}<br>"
                f"TB: {st.get('mean_tb', 0):.1f} K<br>"
                f"Area: {st.get('area_km2', 0):.0f} km²"
                for st in steps
            ],
            hoverinfo="text",
        ), row=1, col=1)

        all_lats += [st["centroid_lat"] for st in steps]
        all_lons += [st["centroid_lon"] for st in steps]

    lat_c = float(np.mean(all_lats)) if all_lats else 10.0
    lon_c = float(np.mean(all_lons)) if all_lons else 70.0

    fig.update_geos(
        projection_type="mercator",
        center=dict(lat=lat_c, lon=lon_c),
        lataxis_range=[lat_c - 18, lat_c + 18],
        lonaxis_range=[lon_c - 22, lon_c + 22],
        showland=True,        landcolor="#F0EBE0",
        showocean=True,       oceancolor="#D6E8F5",
        showcoastlines=True,  coastlinecolor="#AAAAAA",
        showcountries=True,   countrycolor="#CCCCCC",
        showlakes=True,       lakecolor="#D6E8F5",
        resolution=50,
    )

    # ── PANEL 2: Behavior distribution bar ───────────────────────────────────
    beh_counts = [s["all_behaviors"].get(b, 0) for b in BEH_ORDER]

    fig.add_trace(go.Bar(
        x=BEH_ORDER,
        y=beh_counts,
        marker_color=[BC[b] for b in BEH_ORDER],
        marker_line_color="white",
        marker_line_width=0.5,
        name="Actual behavior",
        hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>",
        text=[f"{c:,}" for c in beh_counts],
        textposition="outside",
        textfont=dict(size=9),
    ), row=2, col=1)

    fig.update_yaxes(title_text="Observations", row=2, col=1)
    fig.update_xaxes(tickangle=30, row=2, col=1)

    # ── PANEL 3: Lifecycle — dual Y  (TB + area) ─────────────────────────────
    sel = clusters.get(str(CLUSTER_ID))
    if sel:
        steps_sel = sel["steps"]
        xs    = list(range(len(steps_sel)))
        tbs   = [st.get("mean_tb", 0)    for st in steps_sel]
        areas = [st.get("area_km2", 0)   for st in steps_sel]
        bhvs  = [st.get("true_behavior", "") for st in steps_sel]

        # Main TB line
        fig.add_trace(go.Scatter(
            x=xs, y=tbs, mode="lines",
            name=f"Mean TB (K) — cl {CLUSTER_ID}",
            line=dict(color="#9B59B6", width=2),
            hovertemplate="Step %{x}: TB=%{y:.1f} K<extra></extra>",
        ), row=3, col=1)

        # Area line  (secondary Y — set via yaxis trick after layout)
        fig.add_trace(go.Scatter(
            x=xs, y=areas, mode="lines",
            name=f"Area km² — cl {CLUSTER_ID}",
            line=dict(color="#4A90D9", width=1.5, dash="dot"),
            opacity=0.75,
            yaxis="y6",
            hovertemplate="Step %{x}: Area=%{y:.0f} km²<extra></extra>",
        ), row=3, col=1)

        # Event star markers on TB
        ev_ix = [i for i, b in enumerate(bhvs) if b in EVENT_TYPES]
        if ev_ix:
            fig.add_trace(go.Scatter(
                x=[xs[i] for i in ev_ix],
                y=[tbs[i] for i in ev_ix],
                mode="markers",
                name="Events",
                marker=dict(
                    size=13, symbol="star",
                    color=[BC.get(bhvs[i], "#888") for i in ev_ix],
                    line=dict(width=1, color="white"),
                ),
                text=[bhvs[i] for i in ev_ix],
                hovertemplate="<b>%{text}</b><br>Step %{x}: TB=%{y:.1f} K<extra></extra>",
            ), row=3, col=1)

        tick_step = max(1, len(steps_sel) // 8)
        tvals = list(range(0, len(steps_sel), tick_step))
        tlbls = [steps_sel[i]["time"][11:16] for i in tvals]
        fig.update_xaxes(tickvals=tvals, ticktext=tlbls, tickangle=45, row=3, col=1)
        fig.update_yaxes(title_text="Mean TB (K)", row=3, col=1)

        # Secondary Y for area
        fig.update_layout(
            yaxis6=dict(
                title="Area (km²)",
                overlaying="y4",
                side="right",
                showgrid=False,
                tickfont=dict(size=9),
            )
        )

    # ── PANEL 4: Pie — event type breakdown ───────────────────────────────────
    ev_labels = [e for e in EVENT_TYPES if s["event_types"].get(e, 0) > 0]
    ev_vals   = [s["event_types"].get(e, 0) for e in ev_labels]

    if ev_vals:
        fig.add_trace(go.Pie(
            labels=ev_labels,
            values=ev_vals,
            marker_colors=[BC.get(e, "#888") for e in ev_labels],
            hole=0.4,
            textinfo="label+percent+value",
            textfont=dict(size=10),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
            name="Event types",
        ), row=2, col=2)

    # ── PANEL 5: Top-10 clusters by event count ───────────────────────────────
    top10      = sorted_clusters[:10]
    t10_ids    = [str(c["cluster_id"]) for c in top10]
    t10_events = [c["n_events_true"]   for c in top10]

    fig.add_trace(go.Bar(
        x=t10_ids,
        y=t10_events,
        marker_color=[COLORS_MAP[i % len(COLORS_MAP)] for i in range(len(top10))],
        marker_line_color="white",
        marker_line_width=0.5,
        name="Events per cluster",
        text=t10_events,
        textposition="outside",
        textfont=dict(size=9),
        hovertemplate="Cluster %{x}<br>Events: %{y}<extra></extra>",
    ), row=3, col=2)

    fig.update_xaxes(title_text="Cluster ID", tickangle=30, row=3, col=2)
    fig.update_yaxes(title_text="Event count", row=3, col=2)

    # ── PANEL 6 is shared with panel 5 (row=3,col=2) —
    #    swap to scatter by making panel 5 a bar and overlaying scatter:
    #    Actually use a fresh scatter on a separate trace with xaxis/yaxis
    #    pointing to the same subplot. Build as second trace on row=3,col=2
    #    using a completely different x/y: mean area vs mean TB.
    sc_x, sc_y, sc_c, sc_t = [], [], [], []
    for cl in clusters.values():
        st_list = cl["steps"]
        if not st_list:
            continue
        dom = Counter(st.get("true_behavior", "") for st in st_list).most_common(1)[0][0]
        avg_a = _mean([st.get("area_km2", 0) for st in st_list])
        avg_t = _mean([st.get("mean_tb", 0)  for st in st_list])
        sc_x.append(avg_a)
        sc_y.append(avg_t)
        sc_c.append(BC.get(dom, "#888"))
        sc_t.append(
            f"Cluster {cl['cluster_id']}<br>"
            f"Dominant: {dom}<br>"
            f"Avg area: {avg_a:.0f} km²<br>"
            f"Avg TB: {avg_t:.1f} K"
        )

    # Place scatter on a separate row — extend to 4 rows instead
    # We reuse row=3 col=2 but add the scatter as an OVERLAY via yaxis trick —
    # simpler: just repurpose panel 5 as a combined bar+scatter is not clean.
    # Best approach: remove top-10 bar from row=3,col=2 and instead use
    # the scatter there, and move top-10 bar to row=2,col=1 as a second layer.
    # -- Actually cleanest: keep 5 as bar on row=3,col=2 and add the scatter
    #    in a 4th row, but that changes the subplot spec.
    # -- Decision: keep 6 panels as designed. Panel 5 = top-10 bar.
    #    Panel 6 = scatter on its own trace USING THE SAME row=3 col=2 with
    #    a secondary x/y that we set via layout. Use xaxis7/yaxis7 trick.

    # Attach scatter to xaxis7/yaxis7 which we'll define in layout
    fig.add_trace(go.Scatter(
        x=sc_x, y=sc_y,
        mode="markers",
        marker=dict(size=5, color=sc_c, opacity=0.55,
                    line=dict(width=0.3, color="white")),
        text=sc_t,
        hoverinfo="text",
        name="Area vs TB",
        xaxis="x7", yaxis="y7",
        showlegend=True,
    ))

    # Define x7/y7 as a new subplot anchored below the existing ones
    fig.update_layout(
        xaxis7=dict(
            title="Mean area (km²)",
            domain=[0.55, 1.0],
            anchor="y7",
            showgrid=True, gridcolor="#eee",
        ),
        yaxis7=dict(
            title="Mean TB (K)",
            domain=[0.0, 0.27],
            anchor="x7",
            showgrid=True, gridcolor="#eee",
        ),
    )

    # ── behavior color legend ─────────────────────────────────────────────────
    for beh, col in BC.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            name=beh,
            marker=dict(size=9, color=col, symbol="square"),
            showlegend=True,
        ))

    # ── overall layout ────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(
                f"<b>ACTUAL Cloud Behavior Dashboard — {date}</b><br>"
                f"<sup>"
                f"{s['n_clusters']} clusters &nbsp;|&nbsp; "
                f"{s['total_events']} events &nbsp;|&nbsp; "
                f"Accuracy: {s['overall_accuracy']:.1%} &nbsp;|&nbsp; "
                f"Recall: {s['event_recall']:.1%} &nbsp;|&nbsp; "
                f"Mean TB: {s['mean_tb_k']:.1f} K &nbsp;|&nbsp; "
                f"Mean area: {s['mean_area_km2']:.0f} km²"
                f"</sup>"
            ),
            font=dict(size=14, family="Arial, sans-serif"),
        ),
        height=1200,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.05,
            xanchor="center", x=0.5,
            font=dict(size=9),
            tracegroupgap=4,
        ),
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=11),
        margin=dict(t=110, b=140, l=60, r=70),
        hoverlabel=dict(bgcolor="white", font_size=11),
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
    )

    return fig, s


# ── HTML assembly ─────────────────────────────────────────────────────────────

def build_html(fig, s):
    """Wrap Plotly figure in full HTML with summary card block."""
    plotly_div = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        config={"displayModeBar": True, "scrollZoom": True},
    )
    summary_block = make_summary_html(s)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Actual Dashboard — {s['date']}</title>
  <style>
    body  {{ margin:0; padding:16px 20px;
             font-family:Arial,sans-serif; background:#f4f6f9; }}
    .wrap {{ max-width:1440px; margin:0 auto; background:#fff;
             border-radius:12px; padding:20px 24px;
             box-shadow:0 2px 10px rgba(0,0,0,0.09); }}
    h1    {{ font-size:16px; color:#2c3e50; margin:0 0 4px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>ACTUAL Cloud Behavior — {s['date']}</h1>
    {summary_block}
    {plotly_div}
  </div>
</body>
</html>"""


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    print("Loading pipeline results...")
    with open(PIPELINE_JSON) as f:
        data = json.load(f)

    date = data["metadata"]["date"]
    print(f"  Date     : {date}")
    print(f"  Clusters : {len(data['clusters'])}")
    print(f"  Summary  : {data['summary']}")

    print("\nBuilding actual dashboard...")
    fig, s = build_actual_dashboard(data)

    out = os.path.join(OUT_DIR, f"actual_{date}.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(build_html(fig, s))

    print(f"\nSaved : {out}")
    print(f"\nComputed summary:")
    for k, v in s.items():
        if not isinstance(v, (dict, list)):
            print(f"  {k:25s}: {v}")


if __name__ == "__main__":
    main()