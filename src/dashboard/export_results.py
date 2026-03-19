# -*- coding: utf-8 -*-
"""
export_results.py
Exports clean scientific JSON from pipeline results.
Designed for publication / sharing with other researchers.
"""

import os
import json
from datetime import datetime

PIPELINE_JSON = "output/pipeline_day/pipeline_2025-01-05.json"
OUT_DIR       = "output/scientific"
os.makedirs(OUT_DIR, exist_ok=True)


def export_scientific_json(data):
    """
    Clean JSON structured for scientific publication.
    Strips raw step-level data, keeps summary + top clusters.
    """
    clusters = data["clusters"]
    date     = data["metadata"]["date"]

    # Top clusters by event count
    top = sorted(clusters.values(),
                 key=lambda c: c["n_events_true"], reverse=True)[:10]

    # Per-class accuracy across all clusters
    class_correct = {"NORMAL":0,"ORGANIZING":0,"INTENSIFYING":0}
    class_total   = {"NORMAL":0,"ORGANIZING":0,"INTENSIFYING":0}

    for cl in clusters.values():
        for step in cl["steps"]:
            tg = step["true_grouped"]
            if tg in class_total:
                class_total[tg]   += 1
                if step["correct"]:
                    class_correct[tg] += 1

    class_accuracy = {
        cls: round(class_correct[cls]/class_total[cls], 4)
             if class_total[cls] > 0 else None
        for cls in class_correct
    }

    # Behavior event summary
    event_types = {}
    for cl in clusters.values():
        for beh, cnt in cl["true_behaviors"].items():
            event_types[beh] = event_types.get(beh, 0) + cnt

    out = {
        "paper_metadata": {
            "title"    : "INSAT-3D Cold Cloud Cluster Monitoring using BiGRU+Attention",
            "date"     : date,
            "system"   : "INSAT-3D TIR1 Channel (10.8 μm)",
            "region"   : "Indian Ocean (40-120E, -60-60N)",
            "interval" : "30 minutes",
            "generated": datetime.now().isoformat(),
        },
        "model_config": {
            "cnn_segmentation": {
                "architecture": "U-Net (shallow, 3 encoder levels)",
                "parameters"  : 51025,
                "iou"         : 0.9997,
                "dice"        : 0.9999,
                "precision"   : 1.0000,
                "recall"      : 0.9997,
            },
            "stage1_classifier": {
                "architecture"    : "BiGRU(64,32) + TemporalAttention(32)",
                "parameters"      : 70945,
                "task"            : "NORMAL vs EVENT detection",
                "loss"            : "Focal loss (gamma=1.5, alpha=0.75)",
                "threshold"       : data["metadata"]["threshold_s1"],
                "sequence_length" : data["metadata"]["seq_len"],
                "auc_roc"         : 0.819,
                "event_recall_at_threshold": data["summary"]["event_recall"],
            },
            "stage2_classifier": {
                "architecture"    : "BiGRU(32,16) + TemporalAttention(16)",
                "parameters"      : 19089,
                "task"            : "ORGANIZING vs INTENSIFYING",
                "loss"            : "Focal loss (gamma=2.0, alpha=0.5)",
                "intensifying_recall": 0.927,
                "organizing_recall"  : 0.027,
                "auc_roc"            : 0.668,
            },
        },
        "day_results": {
            "date"                : date,
            "n_clusters_total"    : data["metadata"]["n_clusters_total"],
            "n_clusters_predicted": data["metadata"]["n_clusters_predicted"],
            "total_predictions"   : data["summary"]["total_predictions"],
            "overall_accuracy"    : data["summary"]["overall_accuracy"],
            "event_recall"        : data["summary"]["event_recall"],
            "events_found"        : data["summary"]["total_events_found"],
            "events_true"         : data["summary"]["total_events_true"],
            "class_accuracy"      : class_accuracy,
            "class_sample_counts" : class_total,
            "behavior_distribution": event_types,
        },
        "top_clusters": [
            {
                "cluster_id"    : cl["cluster_id"],
                "lat_mean"      : cl["lat_mean"],
                "lon_mean"      : cl["lon_mean"],
                "time_start"    : cl["time_start"],
                "time_end"      : cl["time_end"],
                "n_frames"      : cl["n_frames"],
                "n_predictions" : cl["n_predictions"],
                "accuracy"      : cl["accuracy"],
                "event_recall"  : cl["event_recall"],
                "n_events_true" : cl["n_events_true"],
                "n_events_found": cl["n_events_found"],
                "true_behaviors": cl["true_behaviors"],
                "pred_classes"  : cl["pred_classes"],
            }
            for cl in top
        ],
        "case_study_cluster_6846": None,
    }

    # Add detailed steps for cluster 6846
    if "6846" in clusters:
        cl = clusters["6846"]
        out["case_study_cluster_6846"] = {
            "cluster_id"    : 6846,
            "description"   : "Best validation cluster — all 6 behaviors present",
            "lat_mean"      : cl["lat_mean"],
            "lon_mean"      : cl["lon_mean"],
            "time_start"    : cl["time_start"],
            "time_end"      : cl["time_end"],
            "n_frames"      : cl["n_frames"],
            "accuracy"      : cl["accuracy"],
            "event_recall"  : cl["event_recall"],
            "true_behaviors": cl["true_behaviors"],
            "pred_classes"  : cl["pred_classes"],
            "per_step"      : [
                {
                    "time"        : s["time"],
                    "lat"         : s["centroid_lat"],
                    "lon"         : s["centroid_lon"],
                    "area_km2"    : s["area_km2"],
                    "mean_tb"     : s["mean_tb"],
                    "conv_core"   : s["convective_core"],
                    "true_behavior"  : s["true_behavior"],
                    "true_grouped"   : s["true_grouped"],
                    "s1_prob"        : s["s1_prob"],
                    "pred_final"     : s["pred_final"],
                    "correct"        : s["correct"],
                }
                for s in cl["steps"]
            ],
        }

    return out


def main():
    print("Loading pipeline results...")
    with open(PIPELINE_JSON) as f:
        data = json.load(f)

    print("Exporting scientific JSON...")
    result = export_scientific_json(data)

    # Full scientific JSON
    out_full = os.path.join(OUT_DIR, f"results_{data['metadata']['date']}.json")
    with open(out_full, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved: {out_full}  ({os.path.getsize(out_full)/1e3:.0f} KB)")

    # Compact version — no per-step data, just summaries
    compact = {k: v for k, v in result.items()
               if k != "case_study_cluster_6846"}
    compact["case_study_cluster_6846_summary"] = {
        k: v for k, v in result["case_study_cluster_6846"].items()
        if k != "per_step"
    } if result["case_study_cluster_6846"] else None

    out_compact = os.path.join(OUT_DIR, f"results_{data['metadata']['date']}_compact.json")
    with open(out_compact, "w") as f:
        json.dump(compact, f, indent=2, default=str)
    print(f"Saved: {out_compact}  ({os.path.getsize(out_compact)/1e3:.0f} KB)")

    print("\nKey results:")
    print(f"  Date              : {result['day_results']['date']}")
    print(f"  Clusters predicted: {result['day_results']['n_clusters_predicted']}")
    print(f"  Overall accuracy  : {result['day_results']['overall_accuracy']:.4f}")
    print(f"  Event recall      : {result['day_results']['event_recall']:.4f}")
    print(f"  Class accuracy    : {result['day_results']['class_accuracy']}")


if __name__ == "__main__":
    main()