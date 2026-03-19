# -*- coding: utf-8 -*-
"""
visualize_cloud_s1s2.py
Validates Stage1 + Stage2 predictions against actual cloud behavior
Cluster 6846 — Indian Ocean, Jan 5-6 2025
"""

import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_DIR   = "data/lstm_dataset"
MODEL_DIR  = "models/bigrustages"
TRACKS     = "data/storm_tracks/storm_tracks.csv"
OUT_DIR    = "output/visualization"
os.makedirs(OUT_DIR, exist_ok=True)

CLUSTER_ID   = 6846
SEQ_LEN      = 8
THRESHOLD_S1 = 0.3

FEATURE_COLS = [
    "centroid_lat","centroid_lon","area_km2","mean_tb",
    "min_tb","max_tb","std_tb","tb_p10",
    "convective_core","cold_cloud_frac",
    "speed_kmh","growth","dlat","dlon",
]
LOG_COLS = ["area_km2","speed_kmh","growth"]

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
    "STABLE"      : "NORMAL",
    "GROWING"     : "NORMAL",
    "SHRINKING"   : "NORMAL",
    "MERGING"     : "ORGANIZING",
    "SPLITTING"   : "ORGANIZING",
    "INTENSIFYING": "INTENSIFYING",
    "NEW"         : "NORMAL",
}

# axis refs for each subplot position (row, col) -> (xref, yref)
# row1col1 is geo — skip shapes there
# row1col2 -> x2,y2  row2col1 -> x3,y3  row2col2 -> x4,y4
# row3col1 -> x5,y5  row3col2 -> x6,y6  row4col1 -> x7,y7  row4col2 -> x8,y8
AXIS_REF = {
    (1,2): ("x2","y2"),
    (2,1): ("x3","y3"),
    (2,2): ("x4","y4"),
    (3,1): ("x5","y5"),
    (3,2): ("x6","y6"),
    (4,1): ("x7","y7"),
    (4,2): ("x8","y8"),
}


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


def load_all():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
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
    print(f"Models loaded. Stage1={s1.count_params():,}  Stage2={s2.count_params():,}")
    return s1, s2, scaler, df


def prepare_track(df, scaler, s1, s2):
    track = (df[df["cluster_id"] == CLUSTER_ID]
             .sort_values("time").reset_index(drop=True))
    print(f"\nCluster {CLUSTER_ID}: {len(track)} frames")
    print(f"Behaviors: {track.behavior.value_counts().to_dict()}")

    track["dlat"] = track["centroid_lat"].diff().fillna(0).clip(-2, 2)
    track["dlon"]  = track["centroid_lon"].diff().fillna(0).clip(-2, 2)

    raw = {
        "lats"  : track["centroid_lat"].values.copy(),
        "lons"  : track["centroid_lon"].values.copy(),
        "areas" : track["area_km2"].values.copy(),
        "tbs"   : track["mean_tb"].values.copy(),
        "conv"  : track["convective_core"].values.copy(),
        "speeds": track["speed_kmh"].values.copy(),
        "bhv"   : track["behavior"].tolist(),
        "times" : [t.strftime("%H:%M") for t in track["time"]],
        "n"     : len(track),
    }
    raw["true_grouped"] = [BEHAVIOR_MAP.get(b, "NORMAL") for b in raw["bhv"]]
    raw["true_s1"] = [0 if g == "NORMAL" else 1 for g in raw["true_grouped"]]
    raw["true_s2"] = [-1 if g == "NORMAL" else
                      (0 if g == "ORGANIZING" else 1)
                      for g in raw["true_grouped"]]

    track_proc = track.copy()
    for col in LOG_COLS:
        track_proc[col] = np.log1p(np.clip(track_proc[col], 0, None))
    track_proc[FEATURE_COLS] = scaler.transform(track_proc[FEATURE_COLS])

    s1_probs, s2_probs = [], []
    pred_s1, pred_s2, pred_final, pred_idx = [], [], [], []
    X_vals = track_proc[FEATURE_COLS].values

    print("Running predictions...")
    for i in range(len(track) - SEQ_LEN):
        seq  = X_vals[i:i+SEQ_LEN][np.newaxis, :, :].astype(np.float32)
        step = i + SEQ_LEN
        p1   = float(s1.predict(seq, verbose=0).flatten()[0])
        s1_probs.append(p1)
        is_event = int(p1 > THRESHOLD_S1)
        pred_s1.append(is_event)
        if is_event:
            p2 = float(s2.predict(seq, verbose=0).flatten()[0])
            s2_probs.append(p2)
            pred_s2.append(int(p2 > 0.5))
            pred_final.append("INTENSIFYING" if p2 > 0.5 else "ORGANIZING")
        else:
            s2_probs.append(None)
            pred_s2.append(-1)
            pred_final.append("NORMAL")
        pred_idx.append(step)

    raw.update({
        "s1_probs": s1_probs, "s2_probs": s2_probs,
        "pred_s1": pred_s1,   "pred_s2": pred_s2,
        "pred_final": pred_final, "pred_idx": pred_idx,
    })

    n_pred = len(pred_idx)
    correct_s1  = sum(pred_s1[i] == raw["true_s1"][pred_idx[i]] for i in range(n_pred))
    event_true  = [i for i in range(n_pred) if raw["true_s1"][pred_idx[i]] == 1]
    event_found = sum(pred_s1[i] == 1 for i in event_true)
    print(f"\nStage 1:  accuracy={correct_s1}/{n_pred} ({100*correct_s1/n_pred:.1f}%)")
    print(f"          event recall={event_found}/{len(event_true)} found")
    return raw


def hline(fig, xref, yref, x0, x1, y, color, dash, width=1.5):
    """Safe horizontal line using add_shape with explicit axis refs."""
    fig.add_shape(type="line",
                  xref=xref, yref=yref,
                  x0=x0, x1=x1, y0=y, y1=y,
                  line=dict(color=color, dash=dash, width=width))


def vband(fig, xref, yref, x0, x1, ymin, ymax, color, opacity=0.08):
    """Safe vertical band using add_shape with explicit axis refs."""
    fig.add_shape(type="rect",
                  xref=xref, yref=yref,
                  x0=x0, x1=x1, y0=ymin, y1=ymax,
                  fillcolor=color, opacity=opacity,
                  line_width=0, layer="below")


def build_dashboard(d):
    n  = d["n"]
    pi = d["pred_idx"]

    tick_step = max(1, n // 12)
    tick_vals = list(range(0, n, tick_step))
    tick_text = [d["times"][i] for i in tick_vals]

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            "Cloud track — actual vs predicted",
            "Stage 1 probability (EVENT detection)",
            "Cloud area km2 with behavior overlay",
            "Stage 2 probability (ORGANIZING vs INTENSIFYING)",
            "Brightness temperature K",
            "Convective core fraction",
            "Prediction vs truth — per step",
            "Speed and split/merge events",
        ],
        specs=[
            [{"type": "scattergeo"}, {"type": "xy"}],
            [{"type": "xy"},         {"type": "xy"}],
            [{"type": "xy"},         {"type": "xy"}],
            [{"type": "xy"},         {"type": "xy"}],
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    # ================================================================
    # PANEL 1: MAP — actual track + predicted classes
    # ================================================================
    fig.add_trace(go.Scattergeo(
        lat=d["lats"], lon=d["lons"], mode="lines",
        name="Actual track",
        line=dict(color="rgba(74,144,217,0.5)", width=2),
        showlegend=True, hoverinfo="skip",
    ), row=1, col=1)

    for beh in list(dict.fromkeys(d["bhv"])):
        idx = [i for i, b in enumerate(d["bhv"]) if b == beh]
        fig.add_trace(go.Scattergeo(
            lat=[d["lats"][i] for i in idx],
            lon=[d["lons"][i] for i in idx],
            mode="markers", name=f"Actual: {beh}",
            marker=dict(size=8, color=BC.get(beh, "#888"),
                        line=dict(width=0.5, color="white")),
            text=[f"{d['times'][i]}<br>Actual: {beh}" for i in idx],
            hoverinfo="text", legendgroup=f"actual_{beh}",
        ), row=1, col=1)

    # Predicted path
    fig.add_trace(go.Scattergeo(
        lat=[d["lats"][pi[i]] for i in range(len(pi))],
        lon=[d["lons"][pi[i]] for i in range(len(pi))],
        mode="lines", name="Predicted path",
        line=dict(color="rgba(232,89,60,0.4)", width=2, dash="dot"),
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
            marker=dict(size=9, symbol=sym, color=color, opacity=0.75,
                        line=dict(width=1.5, color="white")),
            text=[f"Pred: {cls}<br>True: {d['true_grouped'][pi[i]]}<br>"
                  f"Time: {d['times'][pi[i]]}" for i in idx],
            hoverinfo="text", legendgroup=f"pred_{cls}",
        ), row=1, col=1)

    fig.add_trace(go.Scattergeo(
        lat=[d["lats"][0]], lon=[d["lons"][0]], mode="markers", name="Start",
        marker=dict(size=14, symbol="star", color="#27AE60"),
        text=[f"Start: {d['times'][0]}"], hoverinfo="text",
    ), row=1, col=1)
    fig.add_trace(go.Scattergeo(
        lat=[d["lats"][-1]], lon=[d["lons"][-1]], mode="markers", name="End",
        marker=dict(size=14, symbol="star", color="#E8593C"),
        text=[f"End: {d['times'][-1]}"], hoverinfo="text",
    ), row=1, col=1)

    lat_c = float(np.mean(d["lats"]))
    lon_c = float(np.mean(d["lons"]))
    fig.update_geos(
        projection_type="mercator",
        center=dict(lat=lat_c, lon=lon_c),
        lataxis_range=[lat_c - 4, lat_c + 4],
        lonaxis_range=[lon_c - 5, lon_c + 5],
        showland=True,       landcolor="#F0EBE0",
        showocean=True,      oceancolor="#D6E8F5",
        showcoastlines=True, coastlinecolor="#AAAAAA",
        showcountries=True,  countrycolor="#CCCCCC",
        resolution=50,
    )

    # ================================================================
    # PANEL 2: STAGE 1 PROBABILITY  (x2, y2)
    # ================================================================
    xr, yr = AXIS_REF[(1,2)]
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
        fig.add_annotation(xref=xr, yref=yr, x=max(pi), y=thr+yoff,
                           text=label, showarrow=False,
                           font=dict(size=9, color=color),
                           xanchor="right", yanchor="bottom")

    true_event_idx = [i for i in range(n) if d["true_s1"][i] == 1]
    if true_event_idx:
        yvals = [d["s1_probs"][pi.index(s)] if s in pi else None
                 for s in true_event_idx]
        fig.add_trace(go.Scatter(
            x=true_event_idx, y=yvals, mode="markers",
            name="True event",
            marker=dict(size=10, symbol="diamond", color="#27AE60",
                        line=dict(width=1.5, color="white")),
            hovertemplate="True event step %{x}<extra></extra>",
        ), row=1, col=2)

    fig.update_yaxes(title_text="P(EVENT)", range=[0, 1], row=1, col=2)

    # ================================================================
    # PANEL 3: CLOUD AREA  (x3, y3)
    # ================================================================
    xr, yr = AXIS_REF[(2,1)]
    area_min = float(min(d["areas"])) * 0.9
    area_max = float(max(d["areas"])) * 1.1

    for i in range(n - 1):
        beh = d["bhv"][i]
        if beh not in ("STABLE", "NEW"):
            vband(fig, xr, yr, i, i+1, area_min, area_max,
                  BC.get(beh, "#888"), opacity=0.08)

    fig.add_trace(go.Scatter(
        x=list(range(n)), y=d["areas"], mode="lines+markers",
        name="Area km2", line=dict(color="#4A90D9", width=2),
        marker=dict(size=4),
        hovertemplate="%{y:.0f} km2<extra></extra>",
    ), row=2, col=1)

    ev_x = [pi[i] for i in range(len(pi)) if d["pred_s1"][i] == 1]
    ev_y = [d["areas"][pi[i]] for i in range(len(pi)) if d["pred_s1"][i] == 1]
    if ev_x:
        fig.add_trace(go.Scatter(
            x=ev_x, y=ev_y, mode="markers",
            name="Pred event (area)",
            marker=dict(size=9, symbol="x", color="#E8593C",
                        line=dict(width=2)),
            hovertemplate="Pred EVENT step %{x}<extra></extra>",
        ), row=2, col=1)

    fig.update_yaxes(title_text="Area (km2)", row=2, col=1)

    # ================================================================
    # PANEL 4: STAGE 2 PROBABILITY  (x4, y4)
    # ================================================================
    xr, yr = AXIS_REF[(2,2)]
    s2_x = [pi[i] for i in range(len(pi)) if d["s2_probs"][i] is not None]
    s2_y = [d["s2_probs"][i] for i in range(len(pi))
             if d["s2_probs"][i] is not None]

    if s2_x:
        fig.add_trace(go.Scatter(
            x=s2_x, y=s2_y, mode="lines+markers",
            name="S2 P(INTENSIFYING)",
            line=dict(color="#9B59B6", width=2), marker=dict(size=6),
            fill="tozeroy", fillcolor="rgba(155,89,182,0.10)",
            hovertemplate="Step %{x}: P(INTENSIFYING)=%{y:.3f}<extra></extra>",
        ), row=2, col=2)
        hline(fig, xr, yr, min(s2_x), max(s2_x), 0.5, "#9B59B6", "dash")
        fig.add_annotation(xref=xr, yref=yr, x=max(s2_x), y=0.52,
                           text="Threshold 0.5", showarrow=False,
                           font=dict(size=9, color="#9B59B6"),
                           xanchor="right", yanchor="bottom")

    for tidx, color, sym, label in [
        ([i for i in range(n) if d["true_s2"][i] == 1],
         "#9B59B6", "diamond", "True INTENSIFYING"),
        ([i for i in range(n) if d["true_s2"][i] == 0],
         "#F39C12", "square",  "True ORGANIZING"),
    ]:
        if tidx:
            yvals = [s2_y[s2_x.index(s)] if s in s2_x else 0.5 for s in tidx]
            fig.add_trace(go.Scatter(
                x=tidx, y=yvals, mode="markers", name=label,
                marker=dict(size=10, symbol=sym, color=color,
                            line=dict(width=1.5, color="white")),
                hovertemplate=f"{label} step %{{x}}<extra></extra>",
            ), row=2, col=2)

    fig.update_yaxes(title_text="P(INTENSIFYING)", range=[0, 1], row=2, col=2)

    # ================================================================
    # PANEL 5: BRIGHTNESS TEMPERATURE  (x5, y5)
    # ================================================================
    xr, yr = AXIS_REF[(3,1)]
    fig.add_trace(go.Scatter(
        x=list(range(n)), y=d["tbs"], mode="lines+markers",
        name="mean_tb K", line=dict(color="#9B59B6", width=2),
        marker=dict(size=4),
        hovertemplate="%{y:.1f} K<extra></extra>",
    ), row=3, col=1)

    for y_val, color, label, yoff in [
        (210, "rgba(231,76,60,0.8)",  "Deep conv 210K", 1),
        (235, "rgba(243,156,18,0.8)", "Cold cloud 235K", 1),
    ]:
        hline(fig, xr, yr, 0, n-1, y_val, color, "dot")
        fig.add_annotation(xref=xr, yref=yr, x=n-1, y=y_val+yoff,
                           text=label, showarrow=False,
                           font=dict(size=9, color=color),
                           xanchor="right", yanchor="bottom")

    fig.update_yaxes(title_text="TB (K)", row=3, col=1)

    # ================================================================
    # PANEL 6: CONVECTIVE CORE  (x6, y6)
    # ================================================================
    fig.add_trace(go.Scatter(
        x=list(range(n)), y=d["conv"], mode="lines",
        name="Conv core",
        line=dict(color="#1ABC9C", width=2),
        fill="tozeroy", fillcolor="rgba(26,188,156,0.12)",
        hovertemplate="Conv core: %{y:.3f}<extra></extra>",
    ), row=3, col=2)
    fig.update_yaxes(title_text="Conv fraction", row=3, col=2)

    # ================================================================
    # PANEL 7: PREDICTION vs TRUTH  (x7, y7)
    # ================================================================
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

    class_order = ["NORMAL", "ORGANIZING", "INTENSIFYING"]

    def scatter_verdict(steps, color, name, symbol):
        if not steps:
            return
        fig.add_trace(go.Scatter(
            x=steps,
            y=[class_order.index(d["true_grouped"][s])
               if d["true_grouped"][s] in class_order else 0
               for s in steps],
            mode="markers", name=name,
            marker=dict(size=9, color=color, symbol=symbol,
                        line=dict(width=1, color="white")),
            hovertemplate=f"{name} step %{{x}}<extra></extra>",
        ), row=4, col=1)

    scatter_verdict(correct,   "#27AE60", "Correct",      "circle")
    scatter_verdict(fa,        "#F39C12", "False alarm",  "x")
    scatter_verdict(missed_ev, "#E8593C", "Missed event", "triangle-down")
    scatter_verdict(wrong,     "#9B59B6", "Wrong type",   "square")

    fig.update_yaxes(tickvals=[0,1,2], ticktext=class_order,
                     title_text="True class", row=4, col=1)

    # ================================================================
    # PANEL 8: SPEED & SPLIT/MERGE  (x8, y8)
    # ================================================================
    fig.add_trace(go.Scatter(
        x=list(range(n)), y=d["speeds"], mode="lines",
        name="Speed km/h",
        line=dict(color="#E8593C", width=1.5),
        hovertemplate="%{y:.1f} km/h<extra></extra>",
    ), row=4, col=2)

    split_idx = [i for i, b in enumerate(d["bhv"])
                 if b in ("SPLITTING", "MERGING")]
    if split_idx:
        fig.add_trace(go.Scatter(
            x=split_idx,
            y=[d["speeds"][i] for i in split_idx],
            mode="markers", name="Split/Merge",
            marker=dict(size=11, symbol="star", color="#F39C12",
                        line=dict(width=1, color="white")),
            hovertemplate="Split/Merge step %{x}: %{y:.1f} km/h<extra></extra>",
        ), row=4, col=2)

    fig.update_yaxes(title_text="Speed (km/h)", row=4, col=2)

    # ================================================================
    # X AXIS LABELS on all xy subplots
    # ================================================================
    for row, col in [(1,2),(2,1),(2,2),(3,1),(3,2),(4,1),(4,2)]:
        fig.update_xaxes(tickvals=tick_vals, ticktext=tick_text,
                         tickangle=45, row=row, col=col)

    # ================================================================
    # OVERALL LAYOUT
    # ================================================================
    n_events_found = sum(1 for i in range(len(pi))
                         if d["pred_s1"][i] == 1
                         and d["true_s1"][pi[i]] == 1)
    n_true_events  = sum(d["true_s1"])

    fig.update_layout(
        title=dict(
            text=(f"Cloud Cluster {CLUSTER_ID} — Stage 1 + Stage 2 Validation<br>"
                  f"<sup>Jan 5-6 2025 | {n} frames | "
                  f"Threshold S1={THRESHOLD_S1} | "
                  f"Events found: {n_events_found}/{n_true_events} | "
                  f"Indian Ocean (5.7N, 69.2E)</sup>"),
            font=dict(size=14),
        ),
        height=1400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.04,
                    xanchor="center", x=0.5, font=dict(size=9)),
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=11),
        margin=dict(t=90, b=90, l=60, r=60),
    )
    return fig


def main():
    print("Loading models and data...")
    s1, s2, scaler, df = load_all()

    print(f"\nPreparing cluster {CLUSTER_ID}...")
    data = prepare_track(df, scaler, s1, s2)

    print("\nBuilding dashboard...")
    fig = build_dashboard(data)

    out = os.path.join(OUT_DIR, f"cloud_s1s2_cluster_{CLUSTER_ID}.html")
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    print(f"\nSaved: {out}")
    print(f"Open : {os.path.abspath(out)}")


if __name__ == "__main__":
    main()