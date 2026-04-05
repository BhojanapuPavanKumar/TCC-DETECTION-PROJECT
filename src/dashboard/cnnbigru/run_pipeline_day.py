# -*- coding: utf-8 -*-
"""
run_pipeline_day.py
Takes 1 day of storm_tracks data, runs Stage1+Stage2 predictions
on all clusters with enough frames, saves results to JSON.
"""


import os,sys
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.select_best_case import select_best_case
_, TARGET_DATE = select_best_case()

DATA_DIR   = "data/lstm_dataset"
MODEL_DIR  = "models/bigrustages"
TRACKS     = "data/storm_tracks/storm_tracks.csv"
OUT_DIR    = "output/pipeline_day"
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN      = 8
THRESHOLD_S1 = 0.3

FEATURE_COLS = [
    "centroid_lat","centroid_lon","area_km2","mean_tb",
    "min_tb","max_tb","std_tb","tb_p10",
    "convective_core","cold_cloud_frac",
    "speed_kmh","growth","dlat","dlon",
]
LOG_COLS = ["area_km2","speed_kmh","growth"]

BEHAVIOR_MAP = {
    "STABLE":"NORMAL","GROWING":"NORMAL","SHRINKING":"NORMAL",
    "MERGING":"ORGANIZING","SPLITTING":"ORGANIZING",
    "INTENSIFYING":"INTENSIFYING","NEW":"NORMAL",
}


class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = tf.keras.layers.Dense(units, activation="tanh", use_bias=False)
        self.V = tf.keras.layers.Dense(1, use_bias=False)
    def call(self, x):
        s = self.V(self.W(x))
        w = tf.nn.softmax(s, axis=1)
        w = tf.cast(w, x.dtype)
        return tf.reduce_sum(x * w, axis=1)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg


def load_models():
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
    return s1, s2, scaler


def predict_cluster(track, scaler, s1, s2):
    """Run full pipeline on one cluster track. Returns per-step predictions."""
    track = track.sort_values("time").reset_index(drop=True)

    # velocity
    track["dlat"] = track["centroid_lat"].diff().fillna(0).clip(-2, 2)
    track["dlon"]  = track["centroid_lon"].diff().fillna(0).clip(-2, 2)

    # log + scale
    tp = track.copy()
    for col in LOG_COLS:
        tp[col] = np.log1p(np.clip(tp[col], 0, None))
    tp[FEATURE_COLS] = scaler.transform(tp[FEATURE_COLS])
    X = tp[FEATURE_COLS].values

    steps = []
    for i in range(len(track) - SEQ_LEN):
        seq      = X[i:i+SEQ_LEN][np.newaxis, :, :].astype(np.float32)
        step_idx = i + SEQ_LEN
        row      = track.iloc[step_idx]

        p1       = float(s1.predict(seq, verbose=0).flatten()[0])
        is_event = p1 > THRESHOLD_S1

        if is_event:
            p2    = float(s2.predict(seq, verbose=0).flatten()[0])
            pred  = "INTENSIFYING" if p2 > 0.5 else "ORGANIZING"
            p2_val = p2
        else:
            pred   = "NORMAL"
            p2_val = None

        true_beh     = row["behavior"]
        true_grouped = BEHAVIOR_MAP.get(true_beh, "NORMAL")

        steps.append({
            "step"            : step_idx,
            "time"            : str(row["time"]),
            "centroid_lat"    : round(float(row["centroid_lat"]), 4),
            "centroid_lon"    : round(float(row["centroid_lon"]), 4),
            "area_km2"        : round(float(row["area_km2"]), 1),
            "mean_tb"         : round(float(row["mean_tb"]), 2),
            "convective_core" : round(float(row["convective_core"]), 4),
            "speed_kmh"       : round(float(row["speed_kmh"]), 2),
            "true_behavior"   : true_beh,
            "true_grouped"    : true_grouped,
            "s1_prob"         : round(p1, 4),
            "s1_pred"         : "EVENT" if is_event else "NORMAL",
            "s2_prob"         : round(p2_val, 4) if p2_val is not None else None,
            "pred_final"      : pred,
            "correct"         : pred == true_grouped,
        })

    return steps


def main():
    print(f"Running pipeline for {TARGET_DATE}")
    print(f"Threshold S1 = {THRESHOLD_S1}")

    s1, s2, scaler = load_models()

    df = pd.read_csv(TRACKS)
    df["time"] = pd.to_datetime(df["time"])

    day = df[df["time"].dt.date == pd.to_datetime(TARGET_DATE).date()].copy()
    print(f"\nDay data: {len(day)} rows, {day['cluster_id'].nunique()} clusters")

    # Only clusters with enough frames for at least 1 prediction
    lengths = day.groupby("cluster_id").size()
    valid   = lengths[lengths >= SEQ_LEN + 1].index
    print(f"Clusters with {SEQ_LEN+1}+ frames: {len(valid)}")

    results = {
        "metadata": {
            "date"           : TARGET_DATE,
            "threshold_s1"   : THRESHOLD_S1,
            "seq_len"        : SEQ_LEN,
            "n_clusters_total": int(day["cluster_id"].nunique()),
            "n_clusters_predicted": int(len(valid)),
            "generated_at"   : datetime.now().isoformat(),
            "model_stage1"   : "models/bigrustages/bigru_stage1.keras",
            "model_stage2"   : "models/bigrustages/bigru_stage2.keras",
            "features"       : FEATURE_COLS,
        },
        "clusters": {}
    }

    # Aggregate stats
    total_steps   = 0
    total_correct = 0
    total_events_true  = 0
    total_events_found = 0

    for cid in tqdm(valid, desc="Predicting clusters"):
        track = day[day["cluster_id"] == cid].copy()
        steps = predict_cluster(track, scaler, s1, s2)

        if not steps:
            continue

        # Cluster summary
        n_correct     = sum(s["correct"] for s in steps)
        n_events_true = sum(s["true_grouped"] != "NORMAL" for s in steps)
        n_events_found= sum(s["s1_pred"] == "EVENT" and
                            s["true_grouped"] != "NORMAL"
                            for s in steps)
        behaviors_true = {}
        for s in steps:
            b = s["true_behavior"]
            behaviors_true[b] = behaviors_true.get(b, 0) + 1

        pred_classes = {}
        for s in steps:
            p = s["pred_final"]
            pred_classes[p] = pred_classes.get(p, 0) + 1

        results["clusters"][str(cid)] = {
            "cluster_id"      : int(cid),
            "n_frames"        : len(track),
            "n_predictions"   : len(steps),
            "lat_mean"        : round(float(track["centroid_lat"].mean()), 3),
            "lon_mean"        : round(float(track["centroid_lon"].mean()), 3),
            "time_start"      : str(track["time"].min()),
            "time_end"        : str(track["time"].max()),
            "accuracy"        : round(n_correct / len(steps), 4),
            "event_recall"    : round(n_events_found / n_events_true, 4)
                                if n_events_true > 0 else None,
            "n_events_true"   : n_events_true,
            "n_events_found"  : n_events_found,
            "true_behaviors"  : behaviors_true,
            "pred_classes"    : pred_classes,
            "steps"           : steps,
        }

        total_steps    += len(steps)
        total_correct  += n_correct
        total_events_true  += n_events_true
        total_events_found += n_events_found

    # Day-level summary
    # Day-level summary (with per-class accuracy)

    true_counts = {
        "NORMAL": 0,
        "ORGANIZING": 0,
        "INTENSIFYING": 0
    }

    correct_counts = {
        "NORMAL": 0,
        "ORGANIZING": 0,
        "INTENSIFYING": 0
    }

    # iterate through all clusters + steps
    for cluster in results["clusters"].values():
        for step in cluster["steps"]:
            true_cls = step["true_grouped"]
            pred_cls = step["pred_final"]

            true_counts[true_cls] += 1

            if true_cls == pred_cls:
                correct_counts[true_cls] += 1


    results["summary"] = {
        "total_predictions": total_steps,

        "overall_accuracy":
            round(total_correct / total_steps, 4)
            if total_steps > 0 else 0,

        "total_events_true": total_events_true,

        "total_events_found": total_events_found,

        "event_recall":
            round(total_events_found / total_events_true, 4)
            if total_events_true > 0 else 0,

        "normal_accuracy":
            round(correct_counts["NORMAL"] / true_counts["NORMAL"], 4)
            if true_counts["NORMAL"] > 0 else 0,

        "organizing_accuracy":
            round(correct_counts["ORGANIZING"] / true_counts["ORGANIZING"], 4)
            if true_counts["ORGANIZING"] > 0 else 0,

        "intensifying_accuracy":
            round(correct_counts["INTENSIFYING"] / true_counts["INTENSIFYING"], 4)
            if true_counts["INTENSIFYING"] > 0 else 0,
    }

    print(f"\nDay summary:")
    print(f"  Total predictions : {total_steps}")
    print(f"  Overall accuracy  : {results['summary']['overall_accuracy']:.4f}")
    print(f"  Event recall      : {results['summary']['event_recall']:.4f}")
    print(f"  Events found      : {total_events_found}/{total_events_true}")
    print(f"Normal accuracy   : {results['summary']['normal_accuracy']:.4f}")
    print(f"Organizing acc    : {results['summary']['organizing_accuracy']:.4f}")
    print(f"Intensifying acc  : {results['summary']['intensifying_accuracy']:.4f}")

    out_json = os.path.join(OUT_DIR, f"pipeline_{TARGET_DATE}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_json}")
    print(f"Size : {os.path.getsize(out_json)/1e6:.1f} MB")


if __name__ == "__main__":
    main()