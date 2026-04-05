# -*- coding: utf-8 -*-
"""
build_predicted_dashboard.py
Builds predicted behavior dashboard from pipeline JSON.
Shows what the MODEL SAID would happen.

Panels:
  1. Map     — top-N cluster tracks colored by predicted class
  2. Bar     — predicted class distribution vs actual (grouped)
  3. Line    — Stage 1 P(EVENT) probability timeline for best cluster
  4. Bar     — Stage 1 threshold sensitivity (recall vs precision)
  5. Bar     — per-cluster accuracy (top 10)
  6. Scatter — S1 probability vs true event flag (all steps)
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
OUT_DIR          = "output/predicted"
TOP_N            = 5
os.makedirs(OUT_DIR, exist_ok=True)

PC = {
    "NORMAL"      : "#4A90D9",
    "ORGANIZING"  : "#F39C12",
    "INTENSIFYING": "#9B59B6",
}

BC = {
    "STABLE"      : "#4A90D9",
    "GROWING"     : "#27AE60",
    "SHRINKING"   : "#E8593C",
    "INTENSIFYING": "#9B59B6",
    "SPLITTING"   : "#F39C12",
    "MERGING"     : "#1ABC9C",
    "NEW"         : "#95A5A6",
}

CLS_ORDER   = ["NORMAL", "ORGANIZING", "INTENSIFYING"]
EVENT_TYPES = ["INTENSIFYING", "SPLITTING", "MERGING"]
COLORS_MAP  = ["#4A90D9", "#27AE60", "#E8593C", "#9B59B6", "#F39C12"]

# Threshold sensitivity table — pre-computed from Stage 1 training analysis
THRESH_ANALYSIS = {
    0.1: {"recall": 90.4, "precision": 28.1},
    0.2: {"recall": 87.1, "precision": 35.4},
    0.3: {"recall": 81.2, "precision": 44.8},
    0.4: {"recall": 72.3, "precision": 52.3},
    0.5: {"recall": 57.1, "precision": 61.7},
    0.6: {"recall": 26.3, "precision": 72.4},
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _mean(vals):
    return float(np.mean(vals)) if vals else 0.0


def compute_summary(clusters, summary, metadata):
    """Compute all scalar stats from raw pipeline cluster data."""
    pred_classes  = Counter()
    true_groups   = Counter()
    all_s1_probs  = []
    correct_steps = 0
    total_steps   = 0

    for cl in clusters.values():
        for cls, cnt in cl.get("pred_classes", {}).items():
            pred_classes[cls] += cnt
        for step in cl["steps"]:
            all_s1_probs.append(step.get("s1_prob", 0))
            tg = step.get("true_grouped", "")
            pf = step.get("pred_final", "")
            true_groups[tg] += 1
            if tg == pf:
                correct_steps += 1
            total_steps += 1

    return {
        "date"               : metadata.get("date", DATE),
        "n_clusters"         : len(clusters),
        "total_steps"        : total_steps,
        "event_recall"       : summary.get("event_recall", 0),
        "overall_accuracy"   : summary.get("overall_accuracy", 0),
        "events_found"       : summary.get("total_events_found", 0),
        "events_true"        : summary.get("total_events_true", 0),
        "threshold_s1"       : metadata.get("threshold_s1", 0.3),
        "mean_s1_prob"       : _mean(all_s1_probs),
        "pred_classes"       : dict(pred_classes),
        "true_groups"        : dict(true_groups),
        "step_accuracy"      : correct_steps / total_steps if total_steps else 0,
    }


def make_summary_html(s):
    """Render metric cards as styled HTML from computed summary dict."""
    found   = s["events_found"]
    total   = s["events_true"]
    missed  = total - found

    cards = [
        ("Clusters",       str(s["n_clusters"]),              "total in day"),
        ("Events found",   f"{found} / {total}",             "S1 recall"),
        ("Missed",         str(missed),                       "false negatives"),
        ("Event recall",   f"{s['event_recall']:.1%}",       "stage-1 model"),
        ("Step accuracy",  f"{s['step_accuracy']:.1%}",      "all timesteps"),
        ("Threshold S1",   str(s["threshold_s1"]),            "operating point"),
        ("Mean P(EVENT)",  f"{s['mean_s1_prob']:.3f}",       "across all steps"),
        ("Total steps",    f"{s['total_steps']:,}",           "predictions made"),
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
        f"margin-bottom:6px'>&#x1F916; Model prediction summary — {s['date']}</div>"
        f"{divs}</div>"
    )


# ── main dashboard builder ────────────────────────────────────────────────────

def build_predicted_dashboard(data):
    clusters = data["clusters"]
    summary  = data["summary"]
    metadata = data["metadata"]
    date     = metadata["date"]

    s = compute_summary(clusters, summary, metadata)

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
            f"Predicted cloud classes — {date}  (top {TOP_N} clusters)",
            "Predicted vs actual class distribution",
            f"Stage 1 P(EVENT) timeline — cluster {CLUSTER_ID}",
            "Threshold sensitivity (recall vs precision)",
            "Per-cluster accuracy — top 10",
            "P(EVENT) distribution: true events vs background",
        ],
        specs=[
            [{"type": "scattergeo", "colspan": 2}, None],
            [{"type": "xy"},                        {"type": "xy"}],
            [{"type": "xy"},                        {"type": "xy"}],
        ],
        vertical_spacing=0.09,
        horizontal_spacing=0.08,
        row_heights=[0.38, 0.31, 0.31],
    )

    # ── PANEL 1: Map colored by predicted class ───────────────────────────────
    all_lats, all_lons = [], []

    for ci, cl in enumerate(top_clusters):
        steps = cl["steps"]
        cid   = cl["cluster_id"]
        color = COLORS_MAP[ci % len(COLORS_MAP)]

        fig.add_trace(go.Scattergeo(
            lat=[st["centroid_lat"] for st in steps],
            lon=[st["centroid_lon"] for st in steps],
            mode="lines+markers",
            name=f"Cluster {cid}",
            line=dict(color=color, width=1.5, dash="dot"),
            marker=dict(
                size=8,
                color=[PC.get(st.get("pred_final", ""), "#888") for st in steps],
                symbol=[
                    "diamond" if st.get("pred_final") == "INTENSIFYING"
                    else "square" if st.get("pred_final") == "ORGANIZING"
                    else "circle"
                    for st in steps
                ],
                line=dict(width=0.5, color="white"),
            ),
            text=[
                f"<b>Cluster {cid}</b><br>"
                f"Time: {st['time'][11:16]}<br>"
                f"Pred: {st.get('pred_final','?')}<br>"
                f"True: {st.get('true_behavior','?')}<br>"
                f"P(EVENT): {st.get('s1_prob',0):.3f}"
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

    # ── PANEL 2: Grouped bar — predicted vs actual counts ────────────────────
    pred_counts = [s["pred_classes"].get(c, 0) for c in CLS_ORDER]
    true_counts = [s["true_groups"].get(c, 0)  for c in CLS_ORDER]

    fig.add_trace(go.Bar(
        x=CLS_ORDER, y=pred_counts,
        marker_color=[PC[c] for c in CLS_ORDER],
        name="Predicted",
        hovertemplate="<b>%{x}</b><br>Predicted: %{y:,}<extra></extra>",
        text=[f"{v:,}" for v in pred_counts],
        textposition="outside",
        textfont=dict(size=9),
        offsetgroup=0,
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=CLS_ORDER, y=true_counts,
        marker_color=["rgba(74,144,217,0.35)",
                      "rgba(243,156,18,0.35)",
                      "rgba(155,89,182,0.35)"],
        marker_line_color=[PC[c] for c in CLS_ORDER],
        marker_line_width=1.5,
        name="Actual (grouped)",
        hovertemplate="<b>%{x}</b><br>Actual: %{y:,}<extra></extra>",
        offsetgroup=1,
    ), row=2, col=1)

    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(tickangle=0, row=2, col=1)
    fig.update_layout(barmode="group")

    # ── PANEL 3: Stage 1 P(EVENT) timeline ───────────────────────────────────
    sel = clusters.get(str(CLUSTER_ID))
    if sel:
        steps_sel = sel["steps"]
        xs       = list(range(len(steps_sel)))
        s1_prob  = [st.get("s1_prob", 0)       for st in steps_sel]
        true_ev  = [1 if st.get("true_grouped", "") != "NORMAL" else 0
                    for st in steps_sel]
        pred_ev  = [1 if st.get("s1_pred", "") == "EVENT" else 0
                    for st in steps_sel]
        thr      = s["threshold_s1"]

        # Fill area under curve
        fig.add_trace(go.Scatter(
            x=xs, y=s1_prob, mode="lines",
            name=f"P(EVENT) — cl {CLUSTER_ID}",
            line=dict(color="#4A90D9", width=2),
            fill="tozeroy",
            fillcolor="rgba(74,144,217,0.10)",
            hovertemplate="Step %{x}: P(EVENT)=%{y:.3f}<extra></extra>",
        ), row=2, col=2)

        # Threshold line
        fig.add_shape(
            type="line", xref="x3", yref="y3",
            x0=0, x1=len(steps_sel) - 1,
            y0=thr, y1=thr,
            line=dict(color="#F39C12", dash="dash", width=1.5),
        )
        fig.add_annotation(
            xref="x3", yref="y3",
            x=len(steps_sel) - 1, y=thr + 0.02,
            text=f"Threshold {thr}",
            showarrow=False,
            font=dict(size=9, color="#F39C12"),
            xanchor="right",
        )

        # True event markers
        te_x = [i for i in xs if true_ev[i] == 1]
        te_y = [s1_prob[i] for i in te_x]
        if te_x:
            fig.add_trace(go.Scatter(
                x=te_x, y=te_y, mode="markers",
                name="True event",
                marker=dict(size=10, symbol="diamond",
                            color="#27AE60",
                            line=dict(width=1.5, color="white")),
                hovertemplate="True event step %{x}: P=%{y:.3f}<extra></extra>",
            ), row=2, col=2)

        # Predicted event markers (false positives / TP)
        pe_x = [i for i in xs if pred_ev[i] == 1 and true_ev[i] == 0]
        pe_y = [s1_prob[i] for i in pe_x]
        if pe_x:
            fig.add_trace(go.Scatter(
                x=pe_x, y=pe_y, mode="markers",
                name="False positive",
                marker=dict(size=7, symbol="x",
                            color="#E8593C",
                            line=dict(width=1.5, color="#E8593C")),
                hovertemplate="False positive step %{x}<extra></extra>",
            ), row=2, col=2)

        tick_step = max(1, len(steps_sel) // 8)
        tvals = list(range(0, len(steps_sel), tick_step))
        tlbls = [steps_sel[i]["time"][11:16] for i in tvals]
        fig.update_xaxes(tickvals=tvals, ticktext=tlbls, tickangle=45, row=2, col=2)
        fig.update_yaxes(title_text="P(EVENT)", range=[0, 1.05], row=2, col=2)

    # ── PANEL 4: Threshold sensitivity bar ───────────────────────────────────
    t_vals  = sorted(THRESH_ANALYSIS.keys())
    recall_vals    = [THRESH_ANALYSIS[t]["recall"]    for t in t_vals]
    precision_vals = [THRESH_ANALYSIS[t]["precision"] for t in t_vals]
    t_labels = [str(t) for t in t_vals]

    fig.add_trace(go.Bar(
        x=t_labels, y=recall_vals,
        name="Event recall (%)",
        marker_color="#4A90D9",
        marker_line_color="white",
        offsetgroup=0,
        hovertemplate="Threshold %{x}<br>Recall: %{y:.1f}%<extra></extra>",
        text=[f"{v:.1f}%" for v in recall_vals],
        textposition="outside",
        textfont=dict(size=9),
    ), row=3, col=1)

    fig.add_trace(go.Bar(
        x=t_labels, y=precision_vals,
        name="Precision (%)",
        marker_color="#E8593C",
        marker_line_color="white",
        offsetgroup=1,
        hovertemplate="Threshold %{x}<br>Precision: %{y:.1f}%<extra></extra>",
        text=[f"{v:.1f}%" for v in precision_vals],
        textposition="outside",
        textfont=dict(size=9),
    ), row=3, col=1)

    # Highlight operating threshold
    op_thr = s["threshold_s1"]
    if str(op_thr) in t_labels:
        op_idx = t_labels.index(str(op_thr))
        fig.add_shape(
            type="rect", xref="x5", yref="paper",
            x0=op_idx - 0.5, x1=op_idx + 0.5,
            y0=0, y1=1,
            fillcolor="rgba(243,156,18,0.12)",
            line=dict(color="#F39C12", width=1, dash="dot"),
            layer="below",
        )
        fig.add_annotation(
            xref="x5", yref="paper",
            x=op_idx, y=1.04,
            text=f"Operating: {op_thr}",
            showarrow=False,
            font=dict(size=9, color="#F39C12"),
        )

    fig.update_xaxes(title_text="Threshold", row=3, col=1)
    fig.update_yaxes(title_text="% value", range=[0, 115], row=3, col=1)

    # ── PANEL 5: Per-cluster accuracy ─────────────────────────────────────────
    def cluster_accuracy(cl):
        total = len(cl["steps"])
        if total == 0:
            return 0.0
        correct = sum(
            1 for st in cl["steps"]
            if st.get("true_grouped", "X") == st.get("pred_final", "Y")
        )
        return correct / total * 100

    top10       = sorted_clusters[:10]
    t10_ids     = [str(c["cluster_id"]) for c in top10]
    t10_acc     = [cluster_accuracy(c)  for c in top10]
    t10_ev      = [c["n_events_true"]   for c in top10]

    fig.add_trace(go.Bar(
        x=t10_ids,
        y=t10_acc,
        marker_color=[
            "#27AE60" if a >= 80
            else "#F39C12" if a >= 60
            else "#E8593C"
            for a in t10_acc
        ],
        marker_line_color="white",
        name="Cluster accuracy",
        text=[f"{a:.0f}%" for a in t10_acc],
        textposition="outside",
        textfont=dict(size=9),
        hovertemplate=(
            "Cluster %{x}<br>Accuracy: %{y:.1f}%<extra></extra>"
        ),
        customdata=t10_ev,
    ), row=3, col=2)

    fig.update_xaxes(title_text="Cluster ID", tickangle=30, row=3, col=2)
    fig.update_yaxes(title_text="Accuracy (%)", range=[0, 115], row=3, col=2)

    # ── PANEL 6: P(EVENT) histogram — true events vs background ──────────────
    # Collect all step-level s1_prob split by true label
    bg_probs, ev_probs = [], []
    for cl in clusters.values():
        for st in cl["steps"]:
            p = st.get("s1_prob", 0)
            if st.get("true_grouped", "NORMAL") == "NORMAL":
                bg_probs.append(p)
            else:
                ev_probs.append(p)

    # Attach as additional traces on x7/y7 (see layout below)
    fig.add_trace(go.Histogram(
        x=bg_probs,
        nbinsx=30,
        name="Background P(EVENT)",
        marker_color="rgba(74,144,217,0.55)",
        marker_line_color="white",
        marker_line_width=0.5,
        opacity=0.75,
        hovertemplate="P(EVENT) bin %{x:.2f}<br>Count: %{y}<extra></extra>",
        xaxis="x7", yaxis="y7",
    ))

    fig.add_trace(go.Histogram(
        x=ev_probs,
        nbinsx=20,
        name="True event P(EVENT)",
        marker_color="rgba(155,89,182,0.7)",
        marker_line_color="white",
        marker_line_width=0.5,
        opacity=0.85,
        hovertemplate="P(EVENT) bin %{x:.2f}<br>Count: %{y}<extra></extra>",
        xaxis="x7", yaxis="y7",
    ))

    # Threshold line on histogram
    op_thr = s["threshold_s1"]
    fig.add_shape(
        type="line", xref="x7", yref="y7 domain",
        x0=op_thr, x1=op_thr,
        y0=0, y1=1,
        line=dict(color="#F39C12", dash="dash", width=1.5),
    )

    # Define x7/y7 as a properly positioned subplot
    fig.update_layout(
        xaxis7=dict(
            title="P(EVENT)",
            domain=[0.55, 1.0],
            anchor="y7",
            showgrid=True, gridcolor="#eee",
        ),
        yaxis7=dict(
            title="Count",
            domain=[0.01, 0.27],
            anchor="x7",
            showgrid=True, gridcolor="#eee",
        ),
        barmode="overlay",
    )

    # ── predicted class color legend ─────────────────────────────────────────
    for cls, col in PC.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            name=f"Pred: {cls}",
            marker=dict(size=9, color=col, symbol="square"),
            showlegend=True,
        ))

    # ── overall layout ────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(
                f"<b>PREDICTED Cloud Classes Dashboard — {date}</b><br>"
                f"<sup>"
                f"{s['n_clusters']} clusters &nbsp;|&nbsp; "
                f"Events: {s['events_found']}/{s['events_true']} &nbsp;|&nbsp; "
                f"Event recall: {s['event_recall']:.1%} &nbsp;|&nbsp; "
                f"Step accuracy: {s['step_accuracy']:.1%} &nbsp;|&nbsp; "
                f"Threshold S1: {s['threshold_s1']}"
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
  <title>Predicted Dashboard — {s['date']}</title>
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
    <h1>PREDICTED Cloud Classes — {s['date']}</h1>
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

    print("\nBuilding predicted dashboard...")
    fig, s = build_predicted_dashboard(data)

    out = os.path.join(OUT_DIR, f"predicted_{date}.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(build_html(fig, s))

    print(f"\nSaved : {out}")
    print(f"\nComputed summary:")
    for k, v in s.items():
        if not isinstance(v, (dict, list)):
            print(f"  {k:25s}: {v}")


if __name__ == "__main__":
    main()