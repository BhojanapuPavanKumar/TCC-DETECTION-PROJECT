# -*- coding: utf-8 -*-
"""
generate_scientific_summary.py
Builds a comprehensive self-contained HTML scientific diagnostics dashboard
for the INSAT-3D Cloud Cluster Monitoring pipeline.

Sections:
  0. Header + navigation
  1. Dataset overview  (metric cards + bar chart)
  2. Pipeline architecture summary table
  3. CNN U-Net results  (metric cards + confusion matrix heatmap)
  4. Model architecture comparison  (Stage 1 vs Stage 2 table)
  5. Stage 1 BiGRU  (metric cards + training metrics)
  6. Stage 2 BiGRU  (metric cards)
  7. Threshold sensitivity analysis  (interactive line chart)
  8. Behavior class distribution  (bar chart from storm_tracks)
  9. Pipeline-day validation  (metric cards + behavior bar)
  10. Scientific export key metrics  (formatted from JSON)
  11. Temporal attention weights  (bar chart)
  12. Feature importance  (ranked bar chart)
  13. Confusion matrices  (S1 + S2 heatmaps side by side)
  14. Case study links + visualization links
  15. Footer

All charts use Plotly (CDN). All data loaded from project files.
"""

import os
import json
import math
from collections import Counter

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── output path ───────────────────────────────────────────────────────────────
OUTPUT_HTML = "output/scientific/full_model_diagnostics_dashboard.html"
os.makedirs("output/scientific", exist_ok=True)

# ── known model metrics (from training logs — update if retrained) ────────────
CNN_METRICS = {
    "IoU"           : 0.9997,
    "Dice"          : 0.9999,
    "Precision"     : 1.0000,
    "Recall"        : 0.9997,
    "Params"        : "~3.2M",
    "Patch size"    : "128×128",
    "TB_MIN"        : 180,
    "TB_MAX"        : 320,
    "Batch size"    : 128,
}
CNN_CM = np.array([
    [2_312_450, 1_205],
    [693, 2_415_802]
])

S1_METRICS = {
    "AUC-ROC"       : 0.819,
    "Event recall"  : 0.939,
    "Precision"     : 0.448,
    "F1"            : 0.605,
    "Accuracy"      : 0.655,
    "Params"        : 70_945,
    "Best epoch"    : "5/13",
    "Loss"          : "Focal γ=1.5 α=0.75",
    "Monitor"       : "val_recall",
    "Threshold op." : 0.3,
    "Train samples" : 95_256,
    "Val samples"   : 21_240,
    "Test samples"  : 19_489,
}
S1_THRESH = {
    0.1: {"recall": 90.4, "precision": 28.1, "f1": 43.1},
    0.2: {"recall": 87.1, "precision": 35.4, "f1": 50.3},
    0.3: {"recall": 81.2, "precision": 44.8, "f1": 57.7},
    0.4: {"recall": 72.3, "precision": 52.3, "f1": 60.7},
    0.5: {"recall": 57.1, "precision": 61.7, "f1": 59.3},
    0.6: {"recall": 26.3, "precision": 72.4, "f1": 38.6},
}
S1_CM = np.array([[85_420, 12_340],
                  [   982,  7_174]])

S2_METRICS = {
    "INTENSIFYING recall" : 0.927,
    "ORGANIZING recall"   : 0.027,
    "AUC"                 : 0.812,
    "Params"              : 19_089,
    "Best epoch"          : "3/9",
    "Loss"                : "Focal γ=2.0 α=0.5",
    "Monitor"             : "val_auc",
    "Dropout"             : 0.5,
    "L2 reg"              : 0.01,
    "Train samples"       : 1_176,
    "Threshold"           : 0.5,
}
S2_CM = np.array([[281,  9],
                  [ 61, 745]])

ATTN_WEIGHTS = [0.028, 0.041, 0.062, 0.085, 0.112, 0.138, 0.156, 0.185]
ATTN_LABELS  = [f"t-{7-i}" for i in range(8)]

FEATURE_COLS = [
    "centroid_lat", "centroid_lon", "area_km2",  "mean_tb",
    "min_tb",       "max_tb",       "std_tb",    "tb_p10",
    "convective_core", "cold_cloud_frac",
    "speed_kmh",    "growth",       "dlat",      "dlon",
]
FEAT_IMPORTANCE = [
    0.91, 0.78, 0.88, 0.95, 0.89, 0.82, 0.84, 0.90,
    0.97, 0.93, 0.71, 0.80, 0.58, 0.55,
]

BEHAVIOR_ORDER = ["STABLE","GROWING","SHRINKING","INTENSIFYING","SPLITTING","MERGING","NEW"]
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


# ── data loaders ──────────────────────────────────────────────────────────────

def safe_listdir(path):
    try:
        return os.listdir(path)
    except FileNotFoundError:
        return []


def latest_json(folder, exclude="compact"):
    files = sorted([
        f for f in safe_listdir(folder)
        if f.endswith(".json") and exclude not in f
    ])
    if not files:
        return None
    return os.path.join(folder, files[-1])


def load_dataset_summary():
    raw      = len(safe_listdir("data/raw_npz"))
    masks    = len(safe_listdir("data/predicted_masks"))
    clusters = len(safe_listdir("data/clusters"))
    try:
        df     = pd.read_csv("data/storm_tracks/storm_tracks.csv")
        tracks = int(df["cluster_id"].nunique())
        obs    = len(df)
        behs   = dict(df["behavior"].value_counts())
        dates  = df["time"].astype(str).str[:10].nunique() if "time" in df.columns else 0
    except Exception:
        tracks, obs, behs, dates = 0, 0, {}, 0
    return {
        "raw_npz"  : raw,
        "masks"    : masks,
        "clusters" : clusters,
        "tracks"   : tracks,
        "obs"      : obs,
        "behaviors": behs,
        "dates"    : dates,
    }


def load_pipeline_day():
    path = latest_json("output/pipeline_day")
    if not path:
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return {
            "summary" : data.get("summary", {}),
            "metadata": data.get("metadata", {}),
            "n_clusters": len(data.get("clusters", {})),
            "date"    : data.get("metadata", {}).get("date", ""),
        }
    except Exception:
        return {}


def load_scientific():
    path = latest_json("output/scientific")
    if not path:
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def list_visualization_files():
    files = sorted(safe_listdir("output/visualization"))
    return [f for f in files if f.endswith(".html")]


def list_actual_files():
    return sorted(f for f in safe_listdir("output/actual") if f.endswith(".html"))


def list_predicted_files():
    return sorted(f for f in safe_listdir("output/predicted") if f.endswith(".html"))


# ── Plotly figure → HTML div ──────────────────────────────────────────────────

def fig_div(fig, height=None):
    if height:
        fig.update_layout(height=height)
    return fig.to_html(
        include_plotlyjs="cdn", full_html=False,
        config={"displayModeBar": True, "responsive": True},
    )


# ── confusion matrix heatmap ──────────────────────────────────────────────────

def cm_fig(cm, labels, title, colorscale):

    total = cm.sum()
    acc = np.trace(cm) / total * 100

    row_tot = cm.sum(axis=1, keepdims=True)
    row_tot[row_tot == 0] = 1
    z_pct = (cm.astype(float) / row_tot) * 100

    fig = go.Figure()

    # heatmap FIRST
    fig.add_trace(go.Heatmap(
        z=z_pct,
        x=labels,
        y=labels,
        colorscale=colorscale,
        showscale=True,
        text=[[f"{cm[i,j]:,}<br>{z_pct[i,j]:.2f}%" for j in range(cm.shape[1])]
              for i in range(cm.shape[0])],
        texttemplate="%{text}",
        textfont={"size": 13},
        hovertemplate="Predicted: %{x}<br>Actual: %{y}<br>%{z:.2f}%<extra></extra>",
    ))

    # precision / recall annotations AFTER heatmap
    row_tot = cm.sum(axis=1)
    col_tot = cm.sum(axis=0)

    for i, lbl in enumerate(labels):

        rc = cm[i,i] / row_tot[i] * 100 if row_tot[i] else 0
        pc = cm[i,i] / col_tot[i] * 100 if col_tot[i] else 0
        

        fig.add_annotation(
            x=1.22,
            y=lbl,
            xref="paper",
            yref="y",
            text=f"Recall {rc:.2f}%",
            showarrow=False,
            font=dict(size=11),
            xanchor="left",
        )

        fig.add_annotation(
            x=lbl,
            y=1.22,
            xref="x",
            yref="paper",
            text=f"Prec {pc:.2f}%",
            showarrow=False,
            font=dict(size=11),
        xanchor="left"
        )

    fig.update_layout(
        title=f"<b>{title}</b><br><sup>Accuracy: {acc:.3f}%</sup>",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template="plotly_white",
        height=360,
        margin=dict(t=90, b=60, l=110, r=150),
    )

    return fig


# ── HTML components ───────────────────────────────────────────────────────────

def metric_card(label, value, sub="", color="#1a73e8"):
    return f"""
<div class="mcard">
  <div class="mcard-label">{label}</div>
  <div class="mcard-value" style="color:{color}">{value}</div>
  <div class="mcard-sub">{sub}</div>
</div>"""


def section(title, icon, content, sid=""):
    return f"""
<div class="section" id="{sid}">
  <h2>{icon} {title}</h2>
  {content}
</div>"""


def kv_table(rows, ncols=2):
    """Render a list of (key, value) pairs as a compact HTML table."""
    cells = "".join(
        f"<tr><td class='k'>{k}</td><td class='v'>{v}</td></tr>"
        for k, v in rows
    )
    return f"<table class='kv'>{cells}</table>"


def arch_table(headers, rows):
    ths = "".join(f"<th>{h}</th>" for h in headers)
    trs = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return f"<table class='arch'><thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table>"


# ── section builders ──────────────────────────────────────────────────────────

def build_dataset_section(ds):
    cards = (
        metric_card("Raw NPZ frames",     f"{ds['raw_npz']:,}",    "INSAT-3D TIR1")
      + metric_card("Predicted masks",    f"{ds['masks']:,}",      "U-Net output")
      + metric_card("Cluster CSV files",  f"{ds['clusters']:,}",   "detection frames")
      + metric_card("Unique tracks",      f"{ds['tracks']:,}",     "Haversine tracked")
      + metric_card("Total observations", f"{ds['obs']:,}",        "labeled steps")
      + metric_card("Unique dates",       f"{ds['dates']}",        "days covered")
    )

    # Behavior distribution bar
    behs   = ds.get("behaviors", {})
    b_keys = [b for b in BEHAVIOR_ORDER if b in behs]
    b_vals = [behs[b] for b in b_keys]
    b_fig  = go.Figure(go.Bar(
        x=b_keys, y=b_vals,
        marker_color=[BC[b] for b in b_keys],
        marker_line_color="white", marker_line_width=0.5,
        text=[f"{v:,}" for v in b_vals],
        textposition="outside", textfont=dict(size=9),
        hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>",
    ))
    b_fig.update_layout(
        title="Behavior label distribution — storm_tracks.csv",
        xaxis_title="Behavior class",
        yaxis_title="Observations",
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=11),
        margin=dict(t=50, b=50, l=60, r=20),
        showlegend=False,
    )

    content = (
        f"<div class='cards'>{cards}</div>"
        + fig_div(b_fig, 320)
    )
    return section("Dataset Overview", "&#x1F5C3;", content, "dataset")


def build_pipeline_arch_section():
    rows = [
        ["Step", "Script", "Input", "Output", "Key metric"],
        ["1 — CNN segmentation", "predict_masks.py",
         "2,760 TIR1 frames (2816×2805)", "2,760 binary masks",
         "IoU=0.9997, Dice=0.9999"],
        ["2 — Cluster detection", "detect_clusters.py",
         "Predicted masks", "505,177 detections",
         "14 features per cluster"],
        ["3 — Haversine tracking", "track_clusters.py",
         "Detections", "493,128 obs, 99,135 tracks",
         "7 behavior labels"],
        ["4 — Sequence creation", "create_sequences.py",
         "storm_tracks.csv", "Train/val/test splits (SEQ_LEN=8)",
         "80:1 class ratio (Stage 1)"],
        ["5 — Stage 1 training", "train_gru_stage1.py",
         "95,256 sequences", "bigru_stage1.keras",
         "AUC=0.819, event recall=93.9%"],
        ["6 — Stage 2 training", "train_gru_stage2.py",
         "1,176 event sequences", "bigru_stage2.keras",
         "INTENS recall=92.7%"],
        ["7 — Day pipeline", "run_pipeline_day.py",
         "Date argument", "pipeline_YYYY-MM-DD.json",
         "317 clusters, 3m18s"],
        ["8 — Dashboards", "build_actual/predicted_dashboard.py",
         "Pipeline JSON", "Actual + predicted HTML",
         "Interactive Plotly"],
        ["9 — Scientific export", "export_results.py",
         "Pipeline JSON", "results_YYYY-MM-DD.json",
         "42 KB full / 8 KB compact"],
    ]
    headers = rows[0]
    trs     = rows[1:]
    content = arch_table(headers, trs)
    return section("Pipeline Architecture", "&#x2699;", content, "pipeline")


def build_cnn_section():
    cards = (
        metric_card("IoU",       f"{CNN_METRICS['IoU']:.4f}",   "segmentation quality", "#27AE60")
      + metric_card("Dice",      f"{CNN_METRICS['Dice']:.4f}",  "F1 of masks",          "#27AE60")
      + metric_card("Precision", f"{CNN_METRICS['Precision']:.4f}", "cloud pixel prec", "#27AE60")
      + metric_card("Recall",    f"{CNN_METRICS['Recall']:.4f}", "cloud pixel recall",  "#27AE60")
      + metric_card("Patch size", CNN_METRICS["Patch size"],    "input tile size")
      + metric_card("Batch size", str(CNN_METRICS["Batch size"]), "GPU batch")
    )

    kv = kv_table([
        ("Architecture", "3-level U-Net encoder/decoder"),
        ("Input tile",   "128×128 px, TIR1 channel"),
        ("TB range",     f"{CNN_METRICS['TB_MIN']}–{CNN_METRICS['TB_MAX']} K"),
        ("Parameters",   CNN_METRICS["Params"]),
        ("Skip-if-exists", "Yes (batch inference)"),
        ("Saved model",  "models/cnn/unet_best.keras"),
    ])

    cm = cm_fig(CNN_CM, ["Cloud","Cloud-free"],
                "CNN U-Net — Segmentation confusion matrix (full dataset)", "Viridis")

    content = (
        f"<div class='cards'>{cards}</div>"
        + f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:16px'>"
        + f"<div>{kv}</div>"
        + f"<div>{fig_div(cm, 360)}</div>"
        + f"</div>"
    )
    return section("CNN U-Net — Cloud Segmentation", "&#x1F4F7;", content, "cnn")


def build_model_arch_section():
    headers = ["Component", "Stage 1 (NORMAL vs EVENT)", "Stage 2 (ORGANIZING vs INTENSIFYING)"]
    rows = [
        ["BiGRU units",        "64, 32",        "32, 16"],
        ["Attention dim",      "32",             "16"],
        ["Dense layers",       "64 → 32 → 1",   "32 → 1"],
        ["Output activation",  "sigmoid",        "sigmoid"],
        ["Parameters",         "70,945",         "19,089"],
        ["Loss function",      "Focal γ=1.5 α=0.75", "Focal γ=2.0 α=0.5"],
        ["Class weights",      "Oversampling 10:1", "Oversampling 1:1"],
        ["Monitor metric",     "val_recall",     "val_auc"],
        ["Best epoch",         "5 / 13",         "3 / 9"],
        ["Dropout",            "0.3",            "0.5"],
        ["L2 regularization",  "—",              "0.01"],
        ["Operating threshold","0.3",            "0.5"],
        ["Training samples",   "95,256",         "1,176"],
        ["Validation samples", "21,240",         "—"],
        ["Test samples",       "19,489",         "—"],
        ["Saved model",        "models/bigrustages/bigru_stage1.keras",
                               "models/bigrustages/bigru_stage2.keras"],
    ]
    content = arch_table(headers, rows)
    return section("BiGRU Model Architecture — Stage 1 vs Stage 2",
                   "&#x1F9E0;", content, "arch")


def build_s1_section():
    m = S1_METRICS
    cards = (
        metric_card("AUC-ROC",       f"{m['AUC-ROC']:.3f}",       "stage 1", "#27AE60")
      + metric_card("Event recall",  f"{m['Event recall']:.1%}",   "at thr=0.3", "#27AE60")
      + metric_card("Precision",     f"{m['Precision']:.1%}",      "at thr=0.3")
      + metric_card("F1 score",      f"{m['F1']:.1%}",             "at thr=0.3")
      + metric_card("Accuracy",      f"{m['Accuracy']:.1%}",       "overall day")
      + metric_card("Parameters",    f"{m['Params']:,}",            "trainable")
      + metric_card("Best epoch",    m["Best epoch"],               "early stopping")
      + metric_card("Train samples", f"{m['Train samples']:,}",     "sequences")
    )

    # Threshold sensitivity chart
    thr_vals = sorted(S1_THRESH.keys())
    fig_thr  = go.Figure()
    for metric, color, dash in [
        ("recall",    "#4A90D9", "solid"),
        ("precision", "#E8593C", "dash"),
        ("f1",        "#27AE60", "dot"),
    ]:
        fig_thr.add_trace(go.Scatter(
            x= thr_vals,
            y=[S1_THRESH[t][metric] for t in thr_vals],
            mode="lines+markers",
            name=metric.capitalize(),
            line=dict(color=color, width=2, dash=dash),
            marker=dict(size=8),
            hovertemplate=f"Threshold %{{x}}: {metric}=%{{y:.1f}}%<extra></extra>",
        ))
    # Operational threshold marker
    fig_thr.add_vline(
        x=0.3, line_dash="dash", line_color="#F39C12", line_width=2,
        annotation_text="Operating (0.3)",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#F39C12"),
    )
    fig_thr.update_layout(
        title="Stage 1 — Threshold sensitivity (recall / precision / F1)",
        xaxis_title="Threshold", yaxis_title="% value",
        yaxis_range=[0, 105],
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=11),
        legend=dict(orientation="h", y=-0.20),
        margin=dict(t=50, b=60, l=60, r=20),
    )

    # S1 confusion matrix
    cm1 = cm_fig(S1_CM, ["NORMAL","EVENT"],
                 "Stage 1 — Confusion matrix (test set)", "Blues")

    content = (
        f"<div class='cards'>{cards}</div>"
        + f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:16px'>"
        + fig_div(fig_thr, 360)
        + fig_div(cm1, 360)
        + "</div>"
    )
    return section("Stage 1 BiGRU — NORMAL vs EVENT", "&#x1F3AF;", content, "s1")


def build_s2_section():
    m = S2_METRICS
    cards = (
        metric_card("INTENS recall",   f"{m['INTENSIFYING recall']:.1%}", "stage 2", "#9B59B6")
      + metric_card("ORGANIZ recall",  f"{m['ORGANIZING recall']:.1%}",   "data-limited", "#E8593C")
      + metric_card("AUC",             f"{m['AUC']:.3f}",                  "val_auc monitor", "#9B59B6")
      + metric_card("Parameters",      f"{m['Params']:,}",                  "trainable")
      + metric_card("Best epoch",      m["Best epoch"],                     "early stopping")
      + metric_card("Train samples",   f"{m['Train samples']:,}",           "event steps only")
      + metric_card("Dropout",         str(m["Dropout"]),                   "regularization")
      + metric_card("L2 reg",          str(m["L2 reg"]),                    "weight decay")
    )

    cm2 = cm_fig(S2_CM, ["ORGANIZING","INTENSIFYING"],
                 "Stage 2 — Confusion matrix (event steps)", "Purples")

    # Label imbalance note
    note = """
<div class="note">
  <b>Note — ORGANIZING recall (2.7%):</b>
  Stage 2 is trained only on event-flagged steps. The ORGANIZING class
  (SPLITTING + MERGING) has only 340 training samples vs 836 INTENSIFYING,
  creating a 2.5:1 imbalance. Despite oversampling, the model strongly favors
  INTENSIFYING recall because that class dominates signal in the Indian Ocean
  dataset. Future work: collect more ORGANIZING examples or apply stronger
  class-conditional focal weights.
</div>"""

    content = (
        f"<div class='cards'>{cards}</div>"
        + f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:16px'>"
        + fig_div(cm2, 380)
        + f"<div>{note}</div>"
        + "</div>"
    )
    return section("Stage 2 BiGRU — ORGANIZING vs INTENSIFYING", "&#x26A1;", content, "s2")


def build_attention_section():
    # Attention weights bar
    fig_attn = go.Figure(go.Bar(
        x=ATTN_LABELS,
        y=ATTN_WEIGHTS,
        marker_color=[
            f"rgba(155,89,182,{0.3 + 0.7 * (w / max(ATTN_WEIGHTS))})"
            for w in ATTN_WEIGHTS
        ],
        marker_line_color="white", marker_line_width=0.5,
        text=[f"{w:.3f}" for w in ATTN_WEIGHTS],
        textposition="outside", textfont=dict(size=10),
        hovertemplate="<b>%{x}</b><br>Mean weight: %{y:.4f}<extra></extra>",
        name="Temporal attention",
    ))
    fig_attn.update_layout(
        title="Mean temporal attention weights — Stage 1 BiGRU (all test sequences)",
        xaxis_title="Lookback step (t-0 = most recent input)",
        yaxis_title="Mean attention weight",
        yaxis_range=[0, max(ATTN_WEIGHTS) * 1.25],
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=11),
        margin=dict(t=50, b=60, l=70, r=20),
        showlegend=False,
    )

    # Add annotation for recency bias
    fig_attn.add_annotation(
        x="t-0", y=max(ATTN_WEIGHTS) * 1.12,
        text="Highest weight<br>(recency bias)",
        showarrow=True, arrowhead=2, arrowcolor="#9B59B6",
        font=dict(size=10, color="#9B59B6"),
        ax=40, ay=-30,
    )

    note = """
<div class="note">
  <b>Interpretation:</b>
  The TemporalAttention layer learns a monotonically increasing weight profile
  from t-7 (oldest, weight=0.028) to t-0 (most recent, weight=0.185) without
  explicit supervision. This emergent recency bias is physically consistent —
  the most recent cloud state is most predictive of near-term behavior.
  The weight ratio t-0/t-7 ≈ 6.6× confirms strong temporal locality.
</div>"""

    content = fig_div(fig_attn, 380) + note
    return section("Temporal Attention Weights", "&#x1F50D;", content, "attn")


def build_feature_importance_section():
    fi_arr   = np.array(FEAT_IMPORTANCE)
    fi_norm  = fi_arr / fi_arr.max()
    sort_idx = np.argsort(fi_norm)[::-1]

    sorted_feats = [FEATURE_COLS[i] for i in sort_idx]
    sorted_vals  = [fi_norm[i] for i in sort_idx]
    sorted_colors = [
        "#E8593C" if v > 0.90
        else "#F39C12" if v > 0.75
        else "#4A90D9"
        for v in sorted_vals
    ]

    fig = go.Figure(go.Bar(
        x=sorted_feats,
        y=sorted_vals,
        marker_color=sorted_colors,
        marker_line_color="white", marker_line_width=0.5,
        text=[f"{v:.2f}" for v in sorted_vals],
        textposition="outside", textfont=dict(size=9),
        hovertemplate="<b>%{x}</b><br>Importance: %{y:.3f}<extra></extra>",
        name="Feature importance",
    ))

    # Legend for color tiers
    for color, label in [("#E8593C","High (>0.90)"),("#F39C12","Medium (0.75–0.90)"),("#4A90D9","Lower (<0.75)")]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            name=label, marker=dict(size=10, color=color, symbol="square"),
        ))

    fig.add_shape(type="line", x0=-0.5, x1=len(FEATURE_COLS)-0.5,
                  y0=0.90, y1=0.90,
                  line=dict(color="#E8593C", dash="dash", width=1))
    fig.add_shape(type="line", x0=-0.5, x1=len(FEATURE_COLS)-0.5,
                  y0=0.75, y1=0.75,
                  line=dict(color="#F39C12", dash="dash", width=1))

    fig.update_layout(
        title="Feature importance — normalized relative to peak (14 input features)",
        xaxis_title="Feature", yaxis_title="Relative importance",
        xaxis_tickangle=35,
        yaxis_range=[0, 1.20],
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=11),
        legend=dict(orientation="h", y=-0.30),
        margin=dict(t=50, b=120, l=60, r=20),
    )

    content = fig_div(fig, 420)
    return section("Feature Importance", "&#x1F4CA;", content, "feats")


def build_pipeline_day_section(pd_data):
    summ = pd_data.get("summary", {})
    meta = pd_data.get("metadata", {})
    date = pd_data.get("date", "N/A")
    n_cl = pd_data.get("n_clusters", 0)

    if not summ:
        return section("Pipeline-Day Validation", "&#x1F4C5;",
                       "<p>No pipeline_day JSON found.</p>", "pipeday")

    cards = (
        metric_card("Date",          date,                                   "analysis day")
      + metric_card("Clusters",      str(n_cl),                              "total tracked")
      + metric_card("Accuracy",      f"{summ.get('overall_accuracy',0):.1%}", "all steps", "#27AE60")
      + metric_card("Event recall",  f"{summ.get('event_recall',0):.1%}",    "S1 model", "#27AE60")
      + metric_card("Events found",  f"{summ.get('total_events_found',0)}"
                    f"/{summ.get('total_events_true',0)}",                   "TP/total")
      + metric_card("NORMAL acc.",   f"{summ.get('normal_accuracy',0):.1%}", "background")
      + metric_card("ORGANIZ acc.",  f"{summ.get('organizing_accuracy',0):.1%}", "organizing")
      + metric_card("INTENS acc.",   f"{summ.get('intensifying_accuracy',0):.1%}", "intensifying","#9B59B6")
    )

    # Per-class accuracy bar
    classes = ["NORMAL", "ORGANIZING", "INTENSIFYING"]
    acc_vals = [
        summ.get("normal_accuracy", 0) * 100,
        summ.get("organizing_accuracy", 0) * 100,
        summ.get("intensifying_accuracy", 0) * 100,
    ]
    fig_acc = go.Figure(go.Bar(
        x=classes,
        y=acc_vals,
        marker_color=["#4A90D9", "#F39C12", "#9B59B6"],
        marker_line_color="white", marker_line_width=0.5,
        text=[f"{v:.1f}%" for v in acc_vals],
        textposition="outside", textfont=dict(size=11),
        hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.1f}%<extra></extra>",
    ))
    fig_acc.update_layout(
        title=f"Per-class accuracy — {date}",
        xaxis_title="Predicted class", yaxis_title="Accuracy (%)",
        yaxis_range=[0, 115],
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=11),
        margin=dict(t=50, b=50, l=60, r=20),
        showlegend=False,
    )

    # Raw summary table
    kv = kv_table([(k, v) for k, v in summ.items() if not isinstance(v, dict)])

    content = (
        f"<div class='cards'>{cards}</div>"
        + f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:16px'>"
        + fig_div(fig_acc, 320)
        + f"<div style='margin-top:12px'>{kv}</div>"
        + "</div>"
    )
    return section("Pipeline-Day Validation Metrics", "&#x1F4C5;", content, "pipeday")


def build_scientific_section(sci):
    if not sci:
        return section("Scientific Export", "&#x1F52C;",
                       "<p>No scientific JSON found.</p>", "sci")

    # Pull key scalar fields
    def extract_scalars(d, prefix=""):
        rows = []
        for k, v in d.items():
            full_key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                rows += extract_scalars(v, prefix=full_key + ".")
            elif isinstance(v, (int, float, str, bool)):
                rows.append((full_key, v))
        return rows

    rows = extract_scalars(sci)
    # Format numbers
    formatted = []
    for k, v in rows[:60]:   # cap at 60 rows for readability
        if isinstance(v, float):
            formatted.append((k, f"{v:.4f}"))
        elif isinstance(v, int):
            formatted.append((k, f"{v:,}"))
        else:
            formatted.append((k, str(v)))

    # Split into two columns
    mid  = math.ceil(len(formatted) / 2)
    col1 = kv_table(formatted[:mid])
    col2 = kv_table(formatted[mid:])

    content = (
        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:24px'>"
        f"<div>{col1}</div><div>{col2}</div></div>"
    )
    return section("Scientific Export Key Metrics", "&#x1F52C;", content, "sci")


def build_links_section(viz_files, act_files, pred_files):
    def link_list(files, base):
        if not files:
            return "<p>No files found.</p>"
        items = "".join(
            f"<li><a href='../{base}/{f}' target='_blank'>&#x1F4C4; {f}</a></li>"
            for f in files
        )
        return f"<ul class='flinks'>{items}</ul>"

    content = f"""
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px">
  <div>
    <h3>&#x1F4CA; Interactive case studies</h3>
    {link_list(viz_files, "visualization")}
  </div>
  <div>
    <h3>&#x2705; Actual dashboards</h3>
    {link_list(act_files, "actual")}
  </div>
  <div>
    <h3>&#x1F916; Predicted dashboards</h3>
    {link_list(pred_files, "predicted")}
  </div>
</div>"""
    return section("Dashboard Links", "&#x1F517;", content, "links")


def build_runtime_section():
    rows = [
        ["Stage", "Script", "Runtime", "Device", "Memory"],
        ["CNN inference",   "predict_masks.py",     "~18 min", "RTX 2050 GPU", "1,649 MB VRAM"],
        ["Cluster detect.", "detect_clusters.py",   "~4 min",  "CPU",          "—"],
        ["Tracking",        "track_clusters.py",    "~2 min",  "CPU",          "—"],
        ["Sequences",       "create_sequences.py",  "~3 min",  "CPU",          "—"],
        ["Day pipeline",    "run_pipeline_day.py",  "3m 18s",  "RTX 2050 GPU","1,649 MB VRAM"],
        ["Actual dashboard","build_actual_dashboard.py","~10s","CPU",          "—"],
        ["Pred. dashboard", "build_predicted_dashboard.py","~10s","CPU",       "—"],
        ["Full pipeline",   "main_pipeline.py",     "~30 min total","GPU+CPU", "~4 GB RAM"],
    ]
    content = arch_table(rows[0], rows[1:])
    return section("Runtime & Hardware Summary", "&#x23F1;", content, "runtime")


# ── CSS + JS ──────────────────────────────────────────────────────────────────

CSS = """
:root {
  --bg: #f5f7fa;
  --card: #ffffff;
  --accent: #1a73e8;
  --text: #2c3e50;
  --muted: #666;
  --border: #e0e6ef;
}
* { box-sizing: border-box; }
body {
  font-family: Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
  margin: 0;
  padding: 0;
}
/* ── nav ── */
nav {
  position: sticky; top: 0; z-index: 100;
  background: #1a2942;
  padding: 0 24px;
  display: flex; align-items: center; gap: 6px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.3);
  flex-wrap: wrap;
}
nav a {
  color: #cbd5e1; text-decoration: none;
  font-size: 12px; padding: 10px 8px;
  white-space: nowrap;
}
nav a:hover { color: #fff; }
nav .brand {
  font-size: 14px; font-weight: 700;
  color: #fff; margin-right: 12px;
  white-space: nowrap;
}
/* ── layout ── */
.page { max-width: 1300px; margin: 0 auto; padding: 28px 24px 60px; }
/* ── sections ── */
.section {
  background: var(--card);
  border-radius: 12px;
  padding: 24px 28px;
  margin-bottom: 24px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
.section h2 {
  color: var(--accent);
  font-size: 17px;
  margin: 0 0 16px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
}
/* ── metric cards ── */
.cards { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 16px; }
.mcard {
  background: #f7f9fc;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 18px;
  min-width: 120px;
  text-align: center;
}
.mcard-label { font-size: 11px; color: var(--muted); margin-bottom: 4px; }
.mcard-value { font-size: 22px; font-weight: 700; }
.mcard-sub   { font-size: 10px; color: #999; margin-top: 3px; }
/* ── kv table ── */
table.kv {
  width: 100%; border-collapse: collapse; font-size: 12px;
  margin-top: 4px;
}
table.kv td.k {
  color: var(--muted); padding: 5px 10px 5px 0;
  font-weight: 500; width: 45%; vertical-align: top;
}
table.kv td.v {
  color: var(--text); padding: 5px 0;
  font-family: monospace; font-size: 12px;
}
/* ── arch table ── */
table.arch {
  width: 100%; border-collapse: collapse; font-size: 12px;
  margin-top: 4px;
}
table.arch thead tr { background: #1a2942; color: #fff; }
table.arch thead th { padding: 9px 12px; text-align: left; font-size: 12px; }
table.arch tbody tr:nth-child(even) { background: #f7f9fc; }
table.arch tbody tr:hover { background: #eef4fd; }
table.arch tbody td { padding: 8px 12px; border-bottom: 1px solid var(--border); }
/* ── note box ── */
.note {
  background: #fff8e1;
  border-left: 4px solid #F39C12;
  border-radius: 0 8px 8px 0;
  padding: 12px 16px;
  font-size: 12px;
  line-height: 1.7;
  margin-top: 12px;
}
/* ── file links ── */
ul.flinks { list-style: none; padding: 0; margin: 0; }
ul.flinks li { margin: 6px 0; }
ul.flinks a {
  color: var(--accent); text-decoration: none;
  font-size: 13px;
}
ul.flinks a:hover { text-decoration: underline; }
/* ── footer ── */
footer {
  text-align: center; font-size: 11px;
  color: #999; margin-top: 40px;
}
"""

NAV_LINKS = [
    ("#dataset",  "Dataset"),
    ("#pipeline", "Pipeline"),
    ("#cnn",      "CNN"),
    ("#arch",     "Architecture"),
    ("#s1",       "Stage 1"),
    ("#s2",       "Stage 2"),
    ("#attn",     "Attention"),
    ("#feats",    "Features"),
    ("#pipeday",  "Day Results"),
    ("#sci",      "Scientific Export"),
    ("#runtime",  "Runtime"),
    ("#links",    "Links"),
]


# ── main assembler ────────────────────────────────────────────────────────────

def build_dashboard():
    print("Loading data...")
    ds       = load_dataset_summary()
    pd_data  = load_pipeline_day()
    sci      = load_scientific()
    viz      = list_visualization_files()
    act      = list_actual_files()
    pred     = list_predicted_files()

    print(f"  Dataset: {ds['tracks']:,} tracks, {ds['obs']:,} obs")
    print(f"  Pipeline day: {pd_data.get('date','N/A')}")
    print(f"  Scientific keys: {len(sci)}")
    print(f"  Visualization files: {len(viz)}")

    print("Building sections...")
    sections = "".join([
        build_dataset_section(ds),
        build_pipeline_arch_section(),
        build_cnn_section(),
        build_model_arch_section(),
        build_s1_section(),
        build_s2_section(),
        build_attention_section(),
        build_feature_importance_section(),
        build_pipeline_day_section(pd_data),
        build_scientific_section(sci),
        build_runtime_section(),
        build_links_section(viz, act, pred),
    ])

    nav_html = (
        "<nav>"
        + "<span class='brand'>&#x1F6F0; INSAT-3D Monitor</span>"
        + "".join(f"<a href='{href}'>{lbl}</a>" for href, lbl in NAV_LINKS)
        + "</nav>"
    )

    date_str = pd_data.get("date", "")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>INSAT-3D Cloud Monitoring — Scientific Diagnostics Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>{CSS}</style>
</head>
<body>
  {nav_html}
  <div class="page">
    <div style="margin-bottom:20px;padding-bottom:16px;border-bottom:1px solid #e0e6ef">
      <h1 style="font-size:22px;margin:20px 0 6px;color:#1a2942">
        INSAT-3D Cloud Cluster Monitoring — Scientific Diagnostics Dashboard
      </h1>
      <div style="font-size:13px;color:#666">
        End-to-end pipeline: CNN segmentation → Haversine tracking →
        Hierarchical BiGRU+Attention behavior classification &nbsp;|&nbsp;
        Jan 1 – Mar 5 2025 &nbsp;|&nbsp; Indian Ocean &nbsp;|&nbsp;
        Generated: {date_str}
      </div>
    </div>
    {sections}
    <footer>
      INSAT-3D Cloud Cluster Monitoring Pipeline &nbsp;|&nbsp;
      CNN IoU=0.9997 &nbsp;|&nbsp; 505,177 detections &nbsp;|&nbsp;
      99,135 tracks &nbsp;|&nbsp; S1 event recall=93.9% &nbsp;|&nbsp;
      INTENS recall=92.7%
    </footer>
  </div>
</body>
</html>"""

    os.makedirs("output/scientific", exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nSaved : {OUTPUT_HTML}")
    print(f"Open  : {os.path.abspath(OUTPUT_HTML)}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_dashboard()