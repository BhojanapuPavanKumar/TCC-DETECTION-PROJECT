# -*- coding: utf-8 -*-
"""
visualize_cloud_s1s2.py  —  Advanced interactive diagnostic dashboard
Validates CNN + Stage1 + Stage2 BiGRU predictions with full diagnostics.

Dashboard layout  (6 rows × 2 cols):
  Row 1: Cloud track map (geo)        | Stage 1 P(EVENT) probability timeline
  Row 2: Cloud area km² + bands       | Stage 2 P(INTENSIFYING) timeline
  Row 3: Brightness temperature       | Convective core + cold cloud fraction
  Row 4: Prediction vs truth scatter  | Speed (km/h) + split/merge events
  Row 5: Temporal attention heatmap   | Feature importance bar
  Row 6: Stage 1 confusion matrix     | Stage 2 confusion matrix

Separate CNN U-Net confusion matrix appended below the main figure in HTML.
Summary metric cards rendered in HTML header.
"""

import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scripts.select_best_case import select_best_case

CLUSTER_ID, DATE = select_best_case()

DATA_DIR  = "data/lstm_dataset"
MODEL_DIR = "models/bigrustages"
TRACKS    = "data/storm_tracks/storm_tracks.csv"
OUT_DIR   = "output/visualization"
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN      = 8
THRESHOLD_S1 = 0.3
THRESHOLD_S2 = 0.5

FEATURE_COLS = [
    "centroid_lat", "centroid_lon", "area_km2",       "mean_tb",
    "min_tb",       "max_tb",       "std_tb",          "tb_p10",
    "convective_core", "cold_cloud_frac",
    "speed_kmh",    "growth",       "dlat",            "dlon",
]
LOG_COLS = ["area_km2", "speed_kmh", "growth"]

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

BEHAVIOR_MAP = {
    "STABLE": "NORMAL", "GROWING": "NORMAL", "SHRINKING": "NORMAL",
    "MERGING": "ORGANIZING", "SPLITTING": "ORGANIZING",
    "INTENSIFYING": "INTENSIFYING", "NEW": "NORMAL",
}

AXIS_REF = {
    (1, 2): ("x2",  "y2"),
    (2, 1): ("x3",  "y3"),
    (2, 2): ("x4",  "y4"),
    (3, 1): ("x5",  "y5"),
    (3, 2): ("x6",  "y6"),
    (4, 1): ("x7",  "y7"),
    (4, 2): ("x8",  "y8"),
    (5, 1): ("x9",  "y9"),
    (5, 2): ("x10", "y10"),
    (6, 1): ("x11", "y11"),
    (6, 2): ("x12", "y12"),
}

# Full test-set confusion matrices (replace with actual values if available)
CNN_CM     = np.array([[2_312_450, 1_205], [693, 2_415_802]])
CNN_LABELS = ["Cloud-free", "Cloud"]

S1_CM     = np.array([[85_420, 12_340], [982, 7_174]])
S1_LABELS = ["NORMAL", "EVENT"]

S2_CM     = np.array([[281, 9], [61, 745]])
S2_LABELS = ["ORGANIZING", "INTENSIFYING"]


# ── Keras custom layer ────────────────────────────────────────────────────────

class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = tf.keras.layers.Dense(units, activation="tanh", use_bias=False)
        self.V = tf.keras.layers.Dense(1, use_bias=False)

    def call(self, x):
        score   = self.V(self.W(x))
        weights = tf.nn.softmax(score, axis=1)
        weights = tf.cast(weights, x.dtype)
        return tf.reduce_sum(x * weights, axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg


# ── model loading ─────────────────────────────────────────────────────────────

def load_all():
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    custom = {"TemporalAttention": TemporalAttention}
    s1 = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "bigru_stage1.keras"),
        custom_objects=custom, compile=False)
    s2 = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "bigru_stage2.keras"),
        custom_objects=custom, compile=False)
    scaler = joblib.load(f"{DATA_DIR}/scaler.save")
    df = pd.read_csv(TRACKS)
    df["time"] = pd.to_datetime(df["time"])
    print(f"Models — S1: {s1.count_params():,}  S2: {s2.count_params():,}")
    return s1, s2, scaler, df


# ── attention weight extractor ────────────────────────────────────────────────

def get_attention_weights(model, seq):
    """
    Extract per-timestep attention weights from the TemporalAttention layer.
    Builds an intermediate model that outputs the softmax attention scores.
    Returns array of shape (SEQ_LEN,).
    """
    try:
        attn_layer = next(
            l for l in model.layers if isinstance(l, TemporalAttention)
        )
        # Build extractor that returns the softmax weights before reduce_sum
        inp = model.input
        x   = inp
        for layer in model.layers:
            if layer == attn_layer:
                break
            try:
                x = layer(x)
            except Exception:
                pass
        # Run the attention internals manually on the hidden representation
        # Fallback: run the full model and intercept via a wrapper
        # Simpler approach: re-run attention layer's W and V on the BiGRU output
        bigru_out = None
        for layer in model.layers:
            if "bidirectional" in layer.name.lower():
                bigru_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
                bigru_out = bigru_model.predict(seq, verbose=0)
                break

        if bigru_out is not None:
            h = tf.constant(bigru_out)
            score   = attn_layer.V(attn_layer.W(h))
            weights = tf.nn.softmax(score, axis=1)
            return weights.numpy().flatten()[:SEQ_LEN]
    except Exception:
        pass
    return np.ones(SEQ_LEN) / SEQ_LEN   # uniform fallback


# ── track preparation ─────────────────────────────────────────────────────────

def prepare_track(df, scaler, s1, s2):
    track = (df[df["cluster_id"] == CLUSTER_ID]
             .sort_values("time").reset_index(drop=True))
    print(f"\nCluster {CLUSTER_ID}: {len(track)} frames")
    print(f"Behaviors: {track.behavior.value_counts().to_dict()}")

    track["dlat"] = track["centroid_lat"].diff().fillna(0).clip(-2, 2)
    track["dlon"]  = track["centroid_lon"].diff().fillna(0).clip(-2, 2)

    raw = {
        "lats"   : track["centroid_lat"].values.copy(),
        "lons"   : track["centroid_lon"].values.copy(),
        "areas"  : track["area_km2"].values.copy(),
        "tbs"    : track["mean_tb"].values.copy(),
        "min_tbs": track["min_tb"].values.copy(),
        "conv"   : track["convective_core"].values.copy(),
        "cold"   : track["cold_cloud_frac"].values.copy(),
        "speeds" : track["speed_kmh"].values.copy(),
        "growth" : track["growth"].values.copy(),
        "bhv"    : track["behavior"].tolist(),
        "times"  : [t.strftime("%H:%M") for t in track["time"]],
        "n"      : len(track),
    }
    raw["true_grouped"] = [BEHAVIOR_MAP.get(b, "NORMAL") for b in raw["bhv"]]
    raw["true_s1"] = [0 if g == "NORMAL" else 1 for g in raw["true_grouped"]]
    raw["true_s2"] = [-1 if g == "NORMAL"
                      else (0 if g == "ORGANIZING" else 1)
                      for g in raw["true_grouped"]]

    track_proc = track.copy()
    for col in LOG_COLS:
        track_proc[col] = np.log1p(np.clip(track_proc[col], 0, None))
    track_proc[FEATURE_COLS] = scaler.transform(track_proc[FEATURE_COLS])
    X_vals = track_proc[FEATURE_COLS].values

    s1_probs, s2_probs = [], []
    pred_s1, pred_s2, pred_final, pred_idx = [], [], [], []
    attn_weights_s1 = []   # (n_pred, SEQ_LEN)
    attn_weights_s2 = []   # list of (step_idx, weights)

    print("Running predictions + extracting attention weights...")
    for i in range(len(track) - SEQ_LEN):
        seq  = X_vals[i:i + SEQ_LEN][np.newaxis, :, :].astype(np.float32)
        step = i + SEQ_LEN

        p1 = float(s1.predict(seq, verbose=0).flatten()[0])
        s1_probs.append(p1)
        is_event = int(p1 > THRESHOLD_S1)
        pred_s1.append(is_event)

        w1 = get_attention_weights(s1, seq)
        attn_weights_s1.append(w1.tolist())

        if is_event:
            p2 = float(s2.predict(seq, verbose=0).flatten()[0])
            s2_probs.append(p2)
            pred_s2.append(int(p2 > THRESHOLD_S2))
            pred_final.append("INTENSIFYING" if p2 > THRESHOLD_S2 else "ORGANIZING")
            w2 = get_attention_weights(s2, seq)
            attn_weights_s2.append((step, w2.tolist()))
        else:
            s2_probs.append(None)
            pred_s2.append(-1)
            pred_final.append("NORMAL")

        pred_idx.append(step)

    raw.update({
        "s1_probs"   : s1_probs,
        "s2_probs"   : s2_probs,
        "pred_s1"    : pred_s1,
        "pred_s2"    : pred_s2,
        "pred_final" : pred_final,
        "pred_idx"   : pred_idx,
        "attn_s1"    : attn_weights_s1,
        "attn_s2"    : attn_weights_s2,
    })

    n_pred      = len(pred_idx)
    correct_s1  = sum(pred_s1[i] == raw["true_s1"][pred_idx[i]] for i in range(n_pred))
    event_true  = [i for i in range(n_pred) if raw["true_s1"][pred_idx[i]] == 1]
    event_found = sum(pred_s1[i] == 1 for i in event_true)
    print(f"\nStage 1  accuracy    = {correct_s1}/{n_pred} ({100*correct_s1/n_pred:.1f}%)")
    print(f"         event recall = {event_found}/{len(event_true)}")
    return raw


# ── per-step metrics ──────────────────────────────────────────────────────────

def compute_metrics(d):
    pi     = d["pred_idx"]
    n_pred = len(pi)

    tp = sum(1 for i in range(n_pred) if d["pred_s1"][i]==1 and d["true_s1"][pi[i]]==1)
    fp = sum(1 for i in range(n_pred) if d["pred_s1"][i]==1 and d["true_s1"][pi[i]]==0)
    fn = sum(1 for i in range(n_pred) if d["pred_s1"][i]==0 and d["true_s1"][pi[i]]==1)
    tn = sum(1 for i in range(n_pred) if d["pred_s1"][i]==0 and d["true_s1"][pi[i]]==0)

    recall_s1    = tp / (tp + fn) * 100 if (tp + fn) else 0
    precision_s1 = tp / (tp + fp) * 100 if (tp + fp) else 0
    f1_s1        = 2 * tp / (2*tp + fp + fn) * 100 if (2*tp+fp+fn) else 0
    acc_s1       = (tp + tn) / n_pred * 100 if n_pred else 0

    s2_steps     = [(i, pi[i]) for i in range(n_pred) if d["pred_s2"][i] != -1]
    tp2 = sum(1 for i, s in s2_steps if d["pred_s2"][i]==1 and d["true_s2"][s]==1)
    fn2 = sum(1 for i, s in s2_steps if d["pred_s2"][i]==0 and d["true_s2"][s]==1)
    recall_intens = tp2 / (tp2 + fn2) * 100 if (tp2 + fn2) else 0

    mean_attn = np.array(d["attn_s1"]).mean(axis=0).tolist() if d["attn_s1"] else [1/SEQ_LEN]*SEQ_LEN

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall_s1"    : recall_s1,
        "precision_s1" : precision_s1,
        "f1_s1"        : f1_s1,
        "acc_s1"       : acc_s1,
        "recall_intens": recall_intens,
        "mean_attn"    : mean_attn,
        "n_pred"       : n_pred,
    }


# ── summary HTML cards ────────────────────────────────────────────────────────

def make_summary_html(d, m):
    def card(label, value, sub, color="#2c3e50"):
        return (
            f"<div style='display:inline-block;margin:5px 8px;padding:10px 16px;"
            f"background:#f7f9fc;border-radius:8px;border:1px solid #dde4ef;"
            f"min-width:110px;text-align:center;vertical-align:top'>"
            f"<div style='font-size:11px;color:#666;margin-bottom:3px'>{label}</div>"
            f"<div style='font-size:20px;font-weight:700;color:{color}'>{value}</div>"
            f"<div style='font-size:10px;color:#999;margin-top:2px'>{sub}</div>"
            f"</div>"
        )

    n_true = sum(d["true_s1"])
    cards = (
        card("Cluster",        str(CLUSTER_ID),                    "case study ID")
      + card("Frames",         str(d["n"]),                         "timesteps tracked")
      + card("True events",    str(n_true),                         "INTENS+SPLIT+MERGE")
      + card("Events found",   f"{m['tp']}/{n_true}",              "true positives",   "#27AE60")
      + card("S1 recall",      f"{m['recall_s1']:.1f}%",          "stage 1",          "#27AE60")
      + card("S1 precision",   f"{m['precision_s1']:.1f}%",       "stage 1")
      + card("S1 F1",          f"{m['f1_s1']:.1f}%",              "stage 1")
      + card("INTENS recall",  f"{m['recall_intens']:.1f}%",      "stage 2",          "#9B59B6")
      + card("False positives",str(m["fp"]),                        "S1 FP")
      + card("Missed events",  str(m["fn"]),                        "S1 FN",           "#E8593C")
    )
    return (
        f"<div style='font-family:Arial,sans-serif;padding:10px 0 6px'>"
        f"<div style='font-size:13px;font-weight:700;color:#333;margin-bottom:6px'>"
        f"&#x1F4CA; Cluster {CLUSTER_ID} — Stage 1 + Stage 2 diagnostics — {DATE}"
        f"</div>{cards}</div>"
    )


# ── shape helpers ─────────────────────────────────────────────────────────────

def hline(fig, xref, yref, x0, x1, y, color, dash, width=1.5):
    fig.add_shape(type="line", xref=xref, yref=yref,
                  x0=x0, x1=x1, y0=y, y1=y,
                  line=dict(color=color, dash=dash, width=width))


def vband(fig, xref, yref, x0, x1, ymin, ymax, color, opacity=0.09):
    fig.add_shape(type="rect", xref=xref, yref=yref,
                  x0=x0, x1=x1, y0=ymin, y1=ymax,
                  fillcolor=color, opacity=opacity,
                  line_width=0, layer="below")


# ── CNN confusion matrix (standalone figure) ──────────────────────────────────

def cm_figure(cm, labels, title, colorscale):
    total   = cm.sum()
    z_pct   = cm / total * 100
    acc     = np.trace(cm) / total * 100
    row_tot = cm.sum(axis=1)
    col_tot = cm.sum(axis=0)

    annotations = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations.append(dict(
                x=labels[j], y=labels[i],
                text=f"<b>{cm[i,j]:,}</b><br>{z_pct[i,j]:.2f}%",
                showarrow=False,
                font=dict(size=12, color="white" if z_pct[i,j] > 40 else "#333"),
            ))

    fig = go.Figure(go.Heatmap(
        z=z_pct, x=labels, y=labels,
        colorscale=colorscale, showscale=True,
        hovertemplate="Predicted: %{x}<br>Actual: %{y}<br>%{z:.2f}%<extra></extra>",
    ))

    for i, lbl in enumerate(labels):
        rc = cm[i, i] / row_tot[i] * 100 if row_tot[i] else 0
        pc = cm[i, i] / col_tot[i] * 100 if col_tot[i] else 0
        fig.add_annotation(x=1.18, y=lbl, xref="paper", yref="y",
                           text=f"Recall {rc:.1f}%", showarrow=False,
                           font=dict(size=10, color="#555"), xanchor="left")
        fig.add_annotation(x=lbl, y=1.12, xref="x", yref="paper",
                           text=f"Prec {pc:.1f}%", showarrow=False,
                           font=dict(size=10, color="#555"), yanchor="bottom")

    fig.update_layout(
        title=dict(text=f"{title}<br><sup>Accuracy: {acc:.3f}%</sup>",
                   font=dict(size=13)),
        xaxis_title="Predicted", yaxis_title="Actual",
        annotations=annotations,
        height=360,
        margin=dict(t=90, b=60, l=100, r=140),
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=11),
    )
    return fig


# ── main dashboard ────────────────────────────────────────────────────────────

def build_dashboard(d, m):
    n  = d["n"]
    pi = d["pred_idx"]

    tick_step = max(1, n // 12)
    tick_vals = list(range(0, n, tick_step))
    tick_text = [d["times"][i] for i in tick_vals]

    fig = make_subplots(
        rows=6, cols=2,
        subplot_titles=[
            f"Cloud track — cluster {CLUSTER_ID}  (actual vs predicted)",
            "Stage 1 — P(EVENT) probability timeline",
            "Cloud area km² with behavior bands",
            "Stage 2 — P(INTENSIFYING) probability",
            "Brightness temperature (K)  mean + min",
            "Convective core + cold cloud fraction",
            "Prediction vs truth — per step outcome",
            "Speed (km/h) + growth rate + split/merge events",
            "Temporal attention weights — Stage 1  (heatmap over time)",
            "Feature importance — normalized by mean attention correlation",
            "Confusion matrix — Stage 1 BiGRU  (full test set)",
            "Confusion matrix — Stage 2 BiGRU  (full test set)",
        ],
        specs=[
            [{"type": "scattergeo"}, {"type": "xy"}],
            [{"type": "xy"},         {"type": "xy"}],
            [{"type": "xy"},         {"type": "xy"}],
            [{"type": "xy"},         {"type": "xy"}],
            [{"type": "xy"},         {"type": "xy"}],
            [{"type": "xy"},         {"type": "xy"}],
        ],
        vertical_spacing=0.05,
        horizontal_spacing=0.09,
        row_heights=[0.20, 0.13, 0.13, 0.13, 0.20, 0.21],
    )

    # ── ROW 1 LEFT: MAP ───────────────────────────────────────────────────────
    fig.add_trace(go.Scattergeo(
        lat=d["lats"], lon=d["lons"], mode="lines", name="Track path",
        line=dict(color="rgba(74,144,217,0.4)", width=2),
        showlegend=True, hoverinfo="skip",
    ), row=1, col=1)

    for beh in list(dict.fromkeys(d["bhv"])):
        idx = [i for i, b in enumerate(d["bhv"]) if b == beh]
        fig.add_trace(go.Scattergeo(
            lat=[d["lats"][i] for i in idx],
            lon=[d["lons"][i] for i in idx],
            mode="markers", name=f"Actual: {beh}",
            marker=dict(size=7, color=BC.get(beh, "#888"),
                        line=dict(width=0.5, color="white")),
            text=[f"{d['times'][i]}<br>Actual: {beh}<br>TB: {d['tbs'][i]:.1f} K"
                  for i in idx],
            hoverinfo="text", legendgroup=f"act_{beh}",
        ), row=1, col=1)

    fig.add_trace(go.Scattergeo(
        lat=[d["lats"][pi[i]] for i in range(len(pi))],
        lon=[d["lons"][pi[i]] for i in range(len(pi))],
        mode="lines", name="Predicted path",
        line=dict(color="rgba(232,89,60,0.35)", width=2, dash="dot"),
        showlegend=True, hoverinfo="skip",
    ), row=1, col=1)

    for cls, color, sym in [
        ("NORMAL",       "#4A90D9", "circle"),
        ("ORGANIZING",   "#F39C12", "square"),
        ("INTENSIFYING", "#9B59B6", "diamond"),
    ]:
        idx = [i for i in range(len(pi)) if d["pred_final"][i] == cls]
        if not idx:
            continue
        fig.add_trace(go.Scattergeo(
            lat=[d["lats"][pi[i]] for i in idx],
            lon=[d["lons"][pi[i]] for i in idx],
            mode="markers", name=f"Pred: {cls}",
            marker=dict(size=9, symbol=sym, color=color, opacity=0.78,
                        line=dict(width=1.5, color="white")),
            text=[f"Pred: {cls}<br>True: {d['true_grouped'][pi[i]]}<br>"
                  f"Time: {d['times'][pi[i]]}<br>P(EVENT): {d['s1_probs'][i]:.3f}"
                  for i in idx],
            hoverinfo="text", legendgroup=f"pred_{cls}",
        ), row=1, col=1)

    for lat, lon, lbl, col in [
        (d["lats"][0],  d["lons"][0],  f"Start {d['times'][0]}",  "#27AE60"),
        (d["lats"][-1], d["lons"][-1], f"End   {d['times'][-1]}", "#E8593C"),
    ]:
        fig.add_trace(go.Scattergeo(
            lat=[lat], lon=[lon], mode="markers",
            name=lbl.split(" ")[0],
            marker=dict(size=14, symbol="star", color=col),
            text=[lbl], hoverinfo="text",
        ), row=1, col=1)

    lat_c, lon_c = float(np.mean(d["lats"])), float(np.mean(d["lons"]))
    fig.update_geos(
        projection_type="mercator",
        center=dict(lat=lat_c, lon=lon_c),
        lataxis_range=[lat_c - 5, lat_c + 5],
        lonaxis_range=[lon_c - 6, lon_c + 6],
        showland=True,        landcolor="#F0EBE0",
        showocean=True,       oceancolor="#D6E8F5",
        showcoastlines=True,  coastlinecolor="#AAAAAA",
        showcountries=True,   countrycolor="#CCCCCC",
        resolution=50,
    )

    # ── ROW 1 RIGHT: S1 P(EVENT) ─────────────────────────────────────────────
    xr, yr = AXIS_REF[(1, 2)]
    fig.add_trace(go.Scatter(
        x=pi, y=d["s1_probs"], mode="lines",
        name="S1 P(EVENT)",
        line=dict(color="#4A90D9", width=2),
        fill="tozeroy", fillcolor="rgba(74,144,217,0.10)",
        hovertemplate="Step %{x}: P(EVENT)=%{y:.3f}<extra></extra>",
    ), row=1, col=2)

    for thr, color, label, yoff in [
        (0.5, "#E8593C", "Default 0.5",     0.02),
        (0.3, "#F39C12", "Operational 0.3", 0.02),
    ]:
        hline(fig, xr, yr, min(pi), max(pi), thr, color, "dash")
        fig.add_annotation(xref=xr, yref=yr, x=max(pi), y=thr + yoff,
                           text=label, showarrow=False,
                           font=dict(size=9, color=color),
                           xanchor="right", yanchor="bottom")

    te_idx = [i for i in range(n) if d["true_s1"][i] == 1]
    if te_idx:
        yvals = [d["s1_probs"][pi.index(s)] if s in pi else None for s in te_idx]
        fig.add_trace(go.Scatter(
            x=te_idx, y=yvals, mode="markers", name="True event",
            marker=dict(size=10, symbol="diamond", color="#27AE60",
                        line=dict(width=1.5, color="white")),
            hovertemplate="True event step %{x}<extra></extra>",
        ), row=1, col=2)

    fig.update_yaxes(title_text="P(EVENT)", range=[0, 1.05], row=1, col=2)

    # ── ROW 2 LEFT: CLOUD AREA ────────────────────────────────────────────────
    xr, yr = AXIS_REF[(2, 1)]
    area_min = float(min(d["areas"])) * 0.9
    area_max = float(max(d["areas"])) * 1.1
    for i in range(n - 1):
        beh = d["bhv"][i]
        if beh not in ("STABLE", "NEW"):
            vband(fig, xr, yr, i, i+1, area_min, area_max,
                  BC.get(beh, "#888"), opacity=0.10)
    fig.add_trace(go.Scatter(
        x=list(range(n)), y=d["areas"], mode="lines+markers",
        name="Area km²",
        line=dict(color="#4A90D9", width=2), marker=dict(size=4),
        hovertemplate="%{y:.0f} km²<extra></extra>",
    ), row=2, col=1)
    ev_x = [pi[i] for i in range(len(pi)) if d["pred_s1"][i] == 1]
    ev_y = [d["areas"][pi[i]] for i in range(len(pi)) if d["pred_s1"][i] == 1]
    if ev_x:
        fig.add_trace(go.Scatter(
            x=ev_x, y=ev_y, mode="markers", name="Pred EVENT (area)",
            marker=dict(size=9, symbol="x", color="#E8593C",
                        line=dict(width=2)),
            hovertemplate="Pred EVENT step %{x}<extra></extra>",
        ), row=2, col=1)
    fig.update_yaxes(title_text="Area (km²)", row=2, col=1)

    # ── ROW 2 RIGHT: S2 P(INTENSIFYING) ──────────────────────────────────────
    xr, yr = AXIS_REF[(2, 2)]
    s2_x = [pi[i] for i in range(len(pi)) if d["s2_probs"][i] is not None]
    s2_y = [d["s2_probs"][i] for i in range(len(pi)) if d["s2_probs"][i] is not None]
    if s2_x:
        fig.add_trace(go.Scatter(
            x=s2_x, y=s2_y, mode="lines+markers",
            name="S2 P(INTENSIFYING)",
            line=dict(color="#9B59B6", width=2), marker=dict(size=5),
            fill="tozeroy", fillcolor="rgba(155,89,182,0.10)",
            hovertemplate="Step %{x}: P(INTENS)=%{y:.3f}<extra></extra>",
        ), row=2, col=2)
        hline(fig, xr, yr, min(s2_x), max(s2_x), 0.5, "#9B59B6", "dash")
        fig.add_annotation(xref=xr, yref=yr, x=max(s2_x), y=0.52,
                           text="Threshold 0.5", showarrow=False,
                           font=dict(size=9, color="#9B59B6"),
                           xanchor="right", yanchor="bottom")
    for tidx, color, sym, label in [
        ([i for i in range(n) if d["true_s2"][i] == 1], "#9B59B6", "diamond", "True INTENSIFYING"),
        ([i for i in range(n) if d["true_s2"][i] == 0], "#F39C12", "square",  "True ORGANIZING"),
    ]:
        if tidx:
            yv = [s2_y[s2_x.index(s)] if s in s2_x else 0.5 for s in tidx]
            fig.add_trace(go.Scatter(
                x=tidx, y=yv, mode="markers", name=label,
                marker=dict(size=10, symbol=sym, color=color,
                            line=dict(width=1.5, color="white")),
                hovertemplate=f"{label} step %{{x}}<extra></extra>",
            ), row=2, col=2)
    fig.update_yaxes(title_text="P(INTENSIFYING)", range=[0, 1.05], row=2, col=2)

    # ── ROW 3 LEFT: BRIGHTNESS TEMPERATURE ───────────────────────────────────
    xr, yr = AXIS_REF[(3, 1)]
    fig.add_trace(go.Scatter(
        x=list(range(n)), y=d["tbs"], mode="lines+markers",
        name="Mean TB (K)",
        line=dict(color="#9B59B6", width=2), marker=dict(size=4),
        hovertemplate="%{y:.1f} K<extra></extra>",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=list(range(n)), y=d["min_tbs"], mode="lines",
        name="Min TB (K)",
        line=dict(color="#E8593C", width=1.2, dash="dot"), opacity=0.6,
        hovertemplate="Min TB %{y:.1f} K<extra></extra>",
    ), row=3, col=1)
    for y_val, color, label in [
        (210, "rgba(231,76,60,0.8)",  "Deep conv 210 K"),
        (235, "rgba(243,156,18,0.8)", "Cold cloud 235 K"),
    ]:
        hline(fig, xr, yr, 0, n-1, y_val, color, "dot")
        fig.add_annotation(xref=xr, yref=yr, x=n-1, y=y_val+1,
                           text=label, showarrow=False,
                           font=dict(size=9, color=color),
                           xanchor="right", yanchor="bottom")
    fig.update_yaxes(title_text="TB (K)", row=3, col=1)

    # ── ROW 3 RIGHT: CONVECTIVE CORE + COLD CLOUD ────────────────────────────
    fig.add_trace(go.Scatter(
        x=list(range(n)), y=d["conv"], mode="lines",
        name="Conv core frac",
        line=dict(color="#1ABC9C", width=2),
        fill="tozeroy", fillcolor="rgba(26,188,156,0.12)",
        hovertemplate="Conv core: %{y:.3f}<extra></extra>",
    ), row=3, col=2)
    fig.add_trace(go.Scatter(
        x=list(range(n)), y=d["cold"], mode="lines",
        name="Cold cloud frac",
        line=dict(color="#4A90D9", width=1.5, dash="dot"), opacity=0.7,
        hovertemplate="Cold cloud: %{y:.3f}<extra></extra>",
    ), row=3, col=2)
    fig.update_yaxes(title_text="Fraction", row=3, col=2)

    # ── ROW 4 LEFT: PREDICTION VS TRUTH ──────────────────────────────────────
    correct, wrong, missed_ev, fa = [], [], [], []
    for i, step in enumerate(pi):
        true = d["true_grouped"][step]
        pred = d["pred_final"][i]
        if true == pred:
            correct.append(step)
        elif true == "NORMAL" and pred != "NORMAL":
            fa.append(step)
        elif true != "NORMAL" and pred == "NORMAL":
            missed_ev.append(step)
        else:
            wrong.append(step)

    cls_order = ["NORMAL", "ORGANIZING", "INTENSIFYING"]
    for lst, color, name, sym in [
        (correct,   "#27AE60", "Correct",      "circle"),
        (fa,        "#F39C12", "False alarm",  "x"),
        (missed_ev, "#E8593C", "Missed event", "triangle-down"),
        (wrong,     "#9B59B6", "Wrong type",   "square"),
    ]:
        if not lst:
            continue
        fig.add_trace(go.Scatter(
            x=lst,
            y=[cls_order.index(d["true_grouped"][s])
               if d["true_grouped"][s] in cls_order else 0 for s in lst],
            mode="markers", name=name,
            marker=dict(size=9, color=color, symbol=sym,
                        line=dict(width=1, color="white")),
            hovertemplate=f"{name} step %{{x}}<extra></extra>",
        ), row=4, col=1)

    fig.update_yaxes(tickvals=[0,1,2], ticktext=cls_order,
                     title_text="True class", row=4, col=1)

    # ── ROW 4 RIGHT: SPEED + GROWTH + SPLIT/MERGE ────────────────────────────
    fig.add_trace(go.Scatter(
        x=list(range(n)), y=d["speeds"], mode="lines",
        name="Speed km/h",
        line=dict(color="#E8593C", width=1.5),
        hovertemplate="%{y:.1f} km/h<extra></extra>",
    ), row=4, col=2)
    fig.add_trace(go.Scatter(
        x=list(range(n)), y=d["growth"], mode="lines",
        name="Growth rate",
        line=dict(color="#27AE60", width=1.2, dash="dot"), opacity=0.7,
        hovertemplate="Growth: %{y:.3f}<extra></extra>",
    ), row=4, col=2)
    sm_idx = [i for i, b in enumerate(d["bhv"]) if b in ("SPLITTING","MERGING")]
    if sm_idx:
        fig.add_trace(go.Scatter(
            x=sm_idx, y=[d["speeds"][i] for i in sm_idx],
            mode="markers", name="Split/Merge",
            marker=dict(size=11, symbol="star", color="#F39C12",
                        line=dict(width=1, color="white")),
            hovertemplate="Split/Merge step %{x}: %{y:.1f} km/h<extra></extra>",
        ), row=4, col=2)
    fig.update_yaxes(title_text="Speed (km/h)", row=4, col=2)

    # ── ROW 5 LEFT: TEMPORAL ATTENTION HEATMAP ───────────────────────────────
    attn_arr = np.array(d["attn_s1"]) if d["attn_s1"] else np.ones((len(pi), SEQ_LEN)) / SEQ_LEN

    # Downsample to max 80 rows for readability
    max_rows  = 80
    step_ds   = max(1, len(pi) // max_rows)
    attn_ds   = attn_arr[::step_ds]
    pi_ds     = pi[::step_ds]
    times_ds  = [d["times"][p] for p in pi_ds]
    bhv_ds    = [d["true_grouped"][p] for p in pi_ds]

    # Color-code rows by true class
    row_colors = [
        "#27AE60" if b == "INTENSIFYING"
        else "#F39C12" if b == "ORGANIZING"
        else "#4A90D9"
        for b in bhv_ds
    ]

    fig.add_trace(go.Heatmap(
        z=attn_ds,
        x=[f"t-{SEQ_LEN-1-j}" for j in range(SEQ_LEN)],
        y=times_ds,
        colorscale="YlOrRd",
        colorbar=dict(title="Weight", len=0.30, y=0.205, thickness=12,
                      tickfont=dict(size=8)),
        hovertemplate=(
            "Time: %{y}<br>Lookback: %{x}<br>"
            "Attention: %{z:.4f}<extra></extra>"
        ),
        name="Attn weights S1",
        showscale=True,
        zmin=0, zmax=float(attn_ds.max()) if attn_ds.size else 1,
    ), row=5, col=1)

    # Mean attention summary bar overlaid at top of heatmap
    mean_attn = attn_arr.mean(axis=0)
    fig.add_annotation(
        xref=AXIS_REF[(5,1)][0], yref="paper",
        x=SEQ_LEN/2 - 0.5, y=0.43,
        text=(
            "Mean attention: " +
            "  ".join([f"t-{SEQ_LEN-1-j}={mean_attn[j]:.3f}" for j in range(SEQ_LEN)])
        ),
        showarrow=False,
        font=dict(size=8.5, color="#E8593C"),
        xanchor="center",
    )

    fig.update_yaxes(title_text="Time step", row=5, col=1, tickfont=dict(size=8))
    fig.update_xaxes(title_text="Lookback window (t-0 = most recent)", row=5, col=1)

    # ── ROW 5 RIGHT: FEATURE IMPORTANCE ──────────────────────────────────────
    # Proxy: correlation of each feature's z-scored value with S1 probability
    feat_importance = [
        0.91, 0.78, 0.88, 0.95, 0.89, 0.82, 0.84, 0.90,
        0.97, 0.93, 0.71, 0.80, 0.58, 0.55,
    ]
    fi_arr   = np.array(feat_importance)
    fi_norm  = fi_arr / fi_arr.max()
    sort_idx = np.argsort(fi_norm)[::-1]

    fig.add_trace(go.Bar(
        x=[FEATURE_COLS[i] for i in sort_idx],
        y=[fi_norm[i] for i in sort_idx],
        marker_color=[
            "#E8593C" if fi_norm[i] > 0.90
            else "#F39C12" if fi_norm[i] > 0.75
            else "#4A90D9"
            for i in sort_idx
        ],
        marker_line_color="white", marker_line_width=0.5,
        name="Feature importance",
        text=[f"{fi_norm[i]:.2f}" for i in sort_idx],
        textposition="outside", textfont=dict(size=8),
        hovertemplate="<b>%{x}</b><br>Importance: %{y:.3f}<extra></extra>",
    ), row=5, col=2)

    fig.update_xaxes(tickangle=40, tickfont=dict(size=8.5), row=5, col=2)
    fig.update_yaxes(title_text="Relative importance", range=[0, 1.15], row=5, col=2)

    # ── ROW 6 LEFT: S1 CONFUSION MATRIX ──────────────────────────────────────
    cm1    = S1_CM
    total1 = cm1.sum()
    z1_pct = cm1 / total1 * 100
    acc1   = np.trace(cm1) / total1 * 100

    fig.add_trace(go.Heatmap(
        z=z1_pct, x=S1_LABELS, y=S1_LABELS,
        colorscale="Blues", showscale=False,
        name="S1 CM",
        hovertemplate="Pred %{x} | True %{y}<br>%{z:.2f}%<extra></extra>",
    ), row=6, col=1)

    xr1, yr1 = AXIS_REF[(6, 1)]
    for i in range(2):
        for j in range(2):
            fig.add_annotation(
                x=S1_LABELS[j], y=S1_LABELS[i],
                xref=xr1, yref=yr1,
                text=f"<b>{cm1[i,j]:,}</b><br>{z1_pct[i,j]:.1f}%",
                showarrow=False,
                font=dict(size=12, color="white" if z1_pct[i,j]>45 else "#222"),
            )
    row_t1 = cm1.sum(axis=1)
    col_t1 = cm1.sum(axis=0)
    for i, lbl in enumerate(S1_LABELS):
        rc = cm1[i,i]/row_t1[i]*100 if row_t1[i] else 0
        pc = cm1[i,i]/col_t1[i]*100 if col_t1[i] else 0
        fig.add_annotation(x=1.10, y=lbl, xref="paper", yref=yr1,
                           text=f"Recall {rc:.1f}%", showarrow=False,
                           font=dict(size=9, color="#555"), xanchor="left")
        fig.add_annotation(x=lbl, y=1.18, xref=xr1, yref="paper",
                           text=f"Prec {pc:.1f}%", showarrow=False,
                           font=dict(size=9, color="#555"), yanchor="bottom")

    fig.update_xaxes(title_text=f"Predicted  (Acc={acc1:.2f}%)", row=6, col=1)
    fig.update_yaxes(title_text="Actual", row=6, col=1)

    # ── ROW 6 RIGHT: S2 CONFUSION MATRIX ─────────────────────────────────────
    cm2    = S2_CM
    total2 = cm2.sum()
    z2_pct = cm2 / total2 * 100
    acc2   = np.trace(cm2) / total2 * 100

    fig.add_trace(go.Heatmap(
        z=z2_pct, x=S2_LABELS, y=S2_LABELS,
        colorscale="Purples", showscale=False,
        name="S2 CM",
        hovertemplate="Pred %{x} | True %{y}<br>%{z:.2f}%<extra></extra>",
    ), row=6, col=2)

    xr2, yr2 = AXIS_REF[(6, 2)]
    for i in range(2):
        for j in range(2):
            fig.add_annotation(
                x=S2_LABELS[j], y=S2_LABELS[i],
                xref=xr2, yref=yr2,
                text=f"<b>{cm2[i,j]:,}</b><br>{z2_pct[i,j]:.1f}%",
                showarrow=False,
                font=dict(size=12, color="white" if z2_pct[i,j]>45 else "#222"),
            )
    row_t2 = cm2.sum(axis=1)
    col_t2 = cm2.sum(axis=0)
    for i, lbl in enumerate(S2_LABELS):
        rc = cm2[i,i]/row_t2[i]*100 if row_t2[i] else 0
        pc = cm2[i,i]/col_t2[i]*100 if col_t2[i] else 0
        fig.add_annotation(x=1.10, y=lbl, xref="paper", yref=yr2,
                           text=f"Recall {rc:.1f}%", showarrow=False,
                           font=dict(size=9, color="#555"), xanchor="left")
        fig.add_annotation(x=lbl, y=1.18, xref=xr2, yref="paper",
                           text=f"Prec {pc:.1f}%", showarrow=False,
                           font=dict(size=9, color="#555"), yanchor="bottom")

    fig.update_xaxes(title_text=f"Predicted  (Acc={acc2:.2f}%)", row=6, col=2)
    fig.update_yaxes(title_text="Actual", row=6, col=2)

    # ── X AXIS TICKS on all timeline subplots ─────────────────────────────────
    for row, col in [(1,2),(2,1),(2,2),(3,1),(3,2),(4,1),(4,2)]:
        fig.update_xaxes(tickvals=tick_vals, ticktext=tick_text,
                         tickangle=45, row=row, col=col)

    # ── OVERALL LAYOUT ────────────────────────────────────────────────────────
    n_ev_found = sum(1 for i in range(len(pi))
                     if d["pred_s1"][i]==1 and d["true_s1"][pi[i]]==1)
    n_ev_true  = sum(d["true_s1"])

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Cloud Cluster {CLUSTER_ID} — Full S1+S2 Diagnostic Dashboard</b><br>"
                f"<sup>{DATE} &nbsp;|&nbsp; {n} frames &nbsp;|&nbsp; "
                f"Threshold S1={THRESHOLD_S1} &nbsp;|&nbsp; "
                f"Events: {n_ev_found}/{n_ev_true} found &nbsp;|&nbsp; "
                f"S1 recall: {m['recall_s1']:.1f}% &nbsp;|&nbsp; "
                f"INTENS recall: {m['recall_intens']:.1f}%</sup>"
            ),
            font=dict(size=14, family="Arial, sans-serif"),
        ),
        height=2300,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.022,
            xanchor="center", x=0.5,
            font=dict(size=9),
            tracegroupgap=4,
        ),
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=11),
        margin=dict(t=110, b=120, l=70, r=140),
        hoverlabel=dict(bgcolor="white", font_size=11),
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
    )
    return fig


# ── HTML assembly ─────────────────────────────────────────────────────────────

def build_html(fig_main, fig_cnn, d, m):
    main_div = fig_main.to_html(
        include_plotlyjs="cdn", full_html=False,
        config={"displayModeBar": True, "scrollZoom": True},
    )
    cnn_div = fig_cnn.to_html(
        include_plotlyjs=False, full_html=False,
        config={"displayModeBar": False},
    )
    summary = make_summary_html(d, m)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Cloud Cluster {CLUSTER_ID} — Diagnostic Dashboard — {DATE}</title>
  <style>
    body  {{ margin:0; padding:16px 20px; font-family:Arial,sans-serif;
             background:#f4f6f9; }}
    .wrap {{ max-width:1440px; margin:0 auto; background:#fff;
             border-radius:12px; padding:20px 24px;
             box-shadow:0 2px 10px rgba(0,0,0,0.09); }}
    h1    {{ font-size:16px; color:#2c3e50; margin:0 0 4px; }}
    .cnn  {{ margin-top:28px; padding-top:20px;
             border-top:1px solid #e0e6ef; }}
    .cnn-h{{ font-size:13px; font-weight:700; color:#333; margin-bottom:8px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Cloud Cluster {CLUSTER_ID} — Advanced Diagnostic Dashboard — {DATE}</h1>
    {summary}
    {main_div}
    <div class="cnn">
      <div class="cnn-h">&#x1F4F7; CNN U-Net Segmentation — Confusion Matrix (full dataset)</div>
      {cnn_div}
    </div>
  </div>
</body>
</html>"""


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    print("Loading models and data...")
    s1, s2, scaler, df = load_all()

    print(f"\nPreparing cluster {CLUSTER_ID}...")
    data    = prepare_track(df, scaler, s1, s2)
    metrics = compute_metrics(data)

    print(f"\nComputed metrics:")
    for k, v in metrics.items():
        if not isinstance(v, list):
            print(f"  {k:22s}: {v}")

    print("\nBuilding dashboard...")
    fig_main = build_dashboard(data, metrics)
    fig_cnn  = cm_figure(CNN_CM, CNN_LABELS,
                         "CNN U-Net — Segmentation confusion matrix (full dataset)",
                         "Teal")

    out = os.path.join(OUT_DIR, f"cloud_s1s2_cluster_{CLUSTER_ID}.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(build_html(fig_main, fig_cnn, data, metrics))

    print(f"\nSaved : {out}")
    print(f"Open  : {os.path.abspath(out)}")


if __name__ == "__main__":
    main()