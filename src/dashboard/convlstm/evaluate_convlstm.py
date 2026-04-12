"""
evaluate_convlstm.py
====================
Full standalone evaluation script for the ConvLSTM BINARY model
(Non-TCC vs TCC, 2-class).

Mirrors evaluate_convlstm_mc.py exactly in style, layout and naming
so outputs can be directly compared via comparison_matrix.py.

Produces
--------
  1. confusion_matrix.png          — row-normalised + raw counts
  2. roc_pr_curves.png             — ROC + PR with AUC annotations
  3. per_class_metrics.png         — precision / recall / F1 bar chart
  4. confidence_distribution.png   — posterior prob histogram + violin
  5. gradcam_samples.png           — Grad-CAM saliency (3 rows)
  6. learning_curves.png           — accuracy / loss / AUC-ROC / AUC-PR
  7. generalization_gap.png        — train-val gap analysis
  8. training_stability.png        — epoch-to-epoch variance
  9. auc_progression.png           — AUC-ROC & AUC-PR over epochs
 10. combined_summary.png          — 6-panel publication figure
 11. dashboard.html                — interactive HTML dashboard
 12. test_metrics.json             — all numeric metrics
 13. scientific_report.json        — full statistics report
 14. stats_table.tex               — LaTeX-ready statistics table
"""

import os, sys, json, math, shutil, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"]     = "cuda_malloc_async"

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
from datetime import datetime
from pathlib import Path

from sklearn.model_selection    import train_test_split
from sklearn.metrics            import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing      import label_binarize

tf.get_logger().setLevel("ERROR")

# ── resolve project root ──────────────────────────────────────────────────────
_SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR
for _p in [_SCRIPT_DIR] + list(_SCRIPT_DIR.parents):
    if (_p / "logs").exists() or (_p / "src").exists():
        PROJECT_ROOT = _p; break
os.chdir(PROJECT_ROOT)
print(f"[INFO] Working directory: {PROJECT_ROOT}")

# =============================================================================
# CONFIG
# =============================================================================

DATA_X_PATH      = "data/convlstm_dataset/X_memmap.dat"   # or .npy path
DATA_NPZ_PATH    = "data/convlstm_dataset/convlstm_sequences.npz"
MODEL_PATH       = "models/convlstm/best_convlstm.keras"
TRAINING_CSV     = "logs/convlstm/training_log_convlstm.csv"

# Pre-computed arrays (faster re-runs)
Y_TRUE_PATH      = "output/plots/convlstm/y_true.npy"
Y_PRED_PATH      = "output/plots/convlstm/y_pred_proba.npy"

SEED             = 42
BATCH_SIZE       = 4        # keep low for 2 GB VRAM
TARGET_H         = 64
TARGET_W         = 64
NUM_CLASSES      = 2
CLASS_NAMES      = ["Non-TCC", "TCC"]
CLASS_COLORS     = ["#4e79a7", "#e15759"]
MODEL_NAME       = "ConvLSTM (Binary)"
TIMESTAMP        = datetime.now().strftime("%Y-%m-%d %H:%M")

OUTPUT_DIR  = Path("output/plots/convlstm")
PLOTS_DIR   = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"
for d in [OUTPUT_DIR, PLOTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)

# =============================================================================
# CUSTOM OBJECTS  (needed to load saved model)
# =============================================================================

class TrainingOnlyAugmentation(tf.keras.layers.Layer):
    def call(self, x, training=False):
        if training:
            B = tf.shape(x)[0]; T = tf.shape(x)[1]
            H = tf.shape(x)[2]; W = tf.shape(x)[3]; C = tf.shape(x)[4]
            flat  = tf.reshape(x, [B * T, H, W, C])
            flat  = tf.image.random_flip_left_right(flat)
            delta = tf.random.uniform([], -0.08, 0.08)
            flat  = tf.image.adjust_brightness(flat, delta)
            x     = tf.reshape(flat, [B, T, H, W, C])
        return x

CUSTOM_OBJECTS = {"TrainingOnlyAugmentation": TrainingOnlyAugmentation}

# =============================================================================
# UTILITY
# =============================================================================

def _mean(a): return float(np.mean(a))
def _std(a):  return float(np.std(a, ddof=0))
def _sem(a):  return float(np.std(a, ddof=1) / math.sqrt(len(a)))
def _ci95(a): return 1.96 * _sem(a)
def _cohens_d(a, b):
    p = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (_mean(a) - _mean(b)) / p if p > 0 else 0.0

def _style_ax(ax):
    ax.set_facecolor("#111722")
    ax.tick_params(colors="#c8d0df", labelsize=9)
    ax.xaxis.label.set_color("#7a8499")
    ax.yaxis.label.set_color("#7a8499")
    ax.title.set_color("#c8d0df")
    for sp in ax.spines.values(): sp.set_edgecolor("#2e3347")
    ax.yaxis.grid(True, color="#1e2535", lw=0.7, ls="--")
    ax.set_axisbelow(True)

# =============================================================================
# STEP 1 — GPU SETUP
# =============================================================================

def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"✓ GPU detected ({len(gpus)} device(s))")
    else:
        print("⚠  Running on CPU")

# =============================================================================
# STEP 2 — LOAD DATA & PREDICTIONS
# =============================================================================

def load_predictions():
    """
    Returns (y_true, y_pred_proba).
    y_pred_proba shape: (N, 2)  — probability of each class.
    Tries: pre-computed arrays → keras model → fails gracefully.
    """
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("float32")

    # Fast path: pre-computed arrays
    if os.path.exists(Y_TRUE_PATH) and os.path.exists(Y_PRED_PATH):
        y_true = np.load(Y_TRUE_PATH)
        y_pred = np.load(Y_PRED_PATH)
        # normalise to (N, 2) if needed
        if y_pred.ndim == 1 or (y_pred.ndim == 2 and y_pred.shape[1] == 1):
            p = y_pred.ravel()
            y_pred = np.stack([1 - p, p], axis=1)
        print(f"  Loaded pre-computed arrays  y_true={y_true.shape}  y_pred={y_pred.shape}")
        return y_true, y_pred

    # Load model + data
    if not os.path.exists(MODEL_PATH):
        print(f"  ⚠  Model not found: {MODEL_PATH}")
        return None, None

    print("  Loading Keras model …")
    model = tf.keras.models.load_model(
        MODEL_PATH, custom_objects=CUSTOM_OBJECTS, compile=False)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    print("  Model loaded ✓")

    # Load dataset
    npz = DATA_NPZ_PATH
    if not os.path.exists(npz):
        print(f"  ⚠  Dataset not found: {npz}")
        return None, None

    print("  Loading dataset …")
    data   = np.load(npz, mmap_mode="r")
    X      = data["X"]
    y_true = np.array(data["y"])
    print(f"  Dataset: X{X.shape}  y{y_true.shape}")

    # Batched inference (avoids OOM on 2 GB VRAM)
    T = X.shape[1]
    probs_list = []
    n = len(X)
    print(f"  Running inference: {n} samples …")
    for i in range(0, n, BATCH_SIZE):
        seq = X[i:i+BATCH_SIZE].astype(np.float32)
        b   = seq.shape[0]
        resized = np.stack([
            tf.image.resize(seq[:, t], [TARGET_H, TARGET_W]).numpy()
            for t in range(T)
        ], axis=1)
        p = model(tf.cast(resized, tf.float32), training=False).numpy()
        # normalise output to (b, 2)
        if p.ndim == 1 or (p.ndim == 2 and p.shape[1] == 1):
            p = p.ravel()
            p = np.stack([1 - p, p], axis=1)
        probs_list.append(p)
        if (i // BATCH_SIZE + 1) % max(1, (n // BATCH_SIZE) // 10) == 0:
            print(f"    {min(i+BATCH_SIZE, n)}/{n}", flush=True)

    y_pred = np.concatenate(probs_list, axis=0)
    np.save(Y_TRUE_PATH, y_true)
    np.save(Y_PRED_PATH, y_pred)
    print(f"  Saved arrays → {Y_TRUE_PATH}")
    return y_true, y_pred

# =============================================================================
# PLOT 1 — CONFUSION MATRIX
# =============================================================================

def plot_confusion_matrix(y_true, y_pred_proba):
    print("\n[1/9] Plotting confusion matrix …")
    preds   = np.argmax(y_pred_proba, axis=1)
    cm      = confusion_matrix(y_true, preds, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    n_test  = len(y_true)

    fig = plt.figure(figsize=(15, 7.5))
    fig.patch.set_facecolor("#0d1117")
    fig.text(0.5, 0.985,
             f"Multi-Class Confusion Matrix — {MODEL_NAME}  |  Binary TCC Detection",
             ha="center", va="top", color="white", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.945,
             f"Binary classification evaluation  ·  n = {n_test:,} test samples  "
             f"·  Classes: Non-TCC (0)  /  TCC (1)",
             ha="center", va="top", color="#7a8499", fontsize=9)

    gs = gridspec.GridSpec(1, 2, figure=fig, left=0.07, right=0.97,
                           top=0.86, bottom=0.13, wspace=0.30)
    subtitles = [
        "Row-Normalised Classification Rate  (diagonal = recall per class)",
        "Absolute Classification Frequency   (diagonal = true positives)",
    ]
    datas = [cm_norm, cm]; fmts = [".2f", "d"]
    cmaps = [
        LinearSegmentedColormap.from_list("n", ["#0d1117","#0f2a47","#1a5c9e","#38a3d1","#e8f4fd"]),
        LinearSegmentedColormap.from_list("r", ["#0d1117","#1a1a2e","#4a1942","#8b2fc9","#e8d5ff"]),
    ]

    import seaborn as sns
    for ax_idx, data, fmt, subtitle, cmap in zip(
            [gs[0], gs[1]], datas, fmts, subtitles, cmaps):
        ax = fig.add_subplot(ax_idx)
        ax.set_facecolor("#0d1117")
        if fmt == ".2f":
            annot = np.array([[f"{v:.2f}" for v in row] for row in data])
        else:
            annot = np.array([[f"{v:,}" for v in row] for row in data])

        sns.heatmap(data, annot=annot, fmt="", cmap=cmap,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    linewidths=0.6, linecolor="#1e2535",
                    cbar_kws={"shrink": 0.78, "pad": 0.02}, ax=ax,
                    annot_kws={"size": 11, "color": "white", "fontweight": "bold"},
                    vmin=0, vmax=(1.0 if fmt == ".2f" else cm.max()))
        ax.set_title(subtitle, color="#c8d0df", fontsize=9, pad=10, loc="left", style="italic")
        ax.set_xlabel("Predicted Class Label",     color="#7a8499", fontsize=9, labelpad=8)
        ax.set_ylabel("Ground-Truth Class Label",  color="#7a8499", fontsize=9, labelpad=8)
        ax.tick_params(colors="#c8d0df", labelsize=9.5)
        plt.setp(ax.get_xticklabels(), rotation=0,  fontsize=9.5)
        plt.setp(ax.get_yticklabels(), rotation=0,  fontsize=9.5)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(colors="#7a8499", labelsize=8)
        cbar.outline.set_edgecolor("#2e3347")
        cbar.set_label("Recall (sensitivity)" if fmt == ".2f" else "Sample count",
                       color="#7a8499", fontsize=8)
        for j in range(NUM_CLASSES):
            ax.add_patch(plt.Rectangle((j, j), 1, 1, fill=False,
                         edgecolor="white", lw=1.5, clip_on=False))

    path = str(PLOTS_DIR / "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {path}")

    print("\n── Classification Report ──────────────────────────")
    print(classification_report(y_true, preds, target_names=CLASS_NAMES, digits=4))

# =============================================================================
# PLOT 2 — ROC + PR CURVES
# =============================================================================

def plot_roc_pr(y_true, y_pred_proba):
    print("[2/9] Plotting ROC / PR curves …")
    n_test  = len(y_true)
    scores  = y_pred_proba[:, 1]   # probability of TCC class

    fpr, tpr, _     = roc_curve(y_true, scores)
    roc_auc         = auc(fpr, tpr)
    prec, rec, _    = precision_recall_curve(y_true, scores)
    pr_auc          = average_precision_score(y_true, scores)

    fig = plt.figure(figsize=(15, 7.5))
    fig.patch.set_facecolor("#0d1117")
    fig.text(0.5, 0.985,
             f"Discriminative Performance Curves — {MODEL_NAME}  |  Binary TCC Detection",
             ha="center", va="top", color="white", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.945,
             f"Binary classification evaluation  ·  n = {n_test:,} test samples  "
             f"·  Positive class = TCC",
             ha="center", va="top", color="#7a8499", fontsize=9)

    gs   = gridspec.GridSpec(1, 2, figure=fig, left=0.07, right=0.97,
                             top=0.86, bottom=0.10, wspace=0.28)
    ax_r = fig.add_subplot(gs[0])
    ax_p = fig.add_subplot(gs[1])
    _style_ax(ax_r); _style_ax(ax_p)
    ax_r.xaxis.grid(True, color="#1e2535", lw=0.7, ls="--")
    ax_p.xaxis.grid(True, color="#1e2535", lw=0.7, ls="--")

    # ROC
    ax_r.plot(fpr, tpr, color=CLASS_COLORS[1], lw=2.2,
              label=f"TCC (positive)  AUROC = {roc_auc:.4f}")
    ax_r.fill_between(fpr, fpr, tpr, alpha=0.12, color=CLASS_COLORS[1])
    ax_r.plot([0, 1], [0, 1], color="#3a3f52", lw=1.0, ls="--", label="Random classifier")
    ax_r.set(title="Receiver Operating Characteristic\n(Binary: TCC vs Non-TCC)",
             xlabel="False Positive Rate  (1 − Specificity)",
             ylabel="True Positive Rate  (Sensitivity / Recall)",
             xlim=[-0.01, 1.01], ylim=[-0.01, 1.03])
    ax_r.title.set_color("#c8d0df"); ax_r.title.set_fontsize(10)

    # annotate AUC on curve
    ax_r.annotate(f"AUROC = {roc_auc:.4f}",
                  xy=(0.4, 0.6), color="white", fontsize=10, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#181e2c",
                            edgecolor="#2e3347", alpha=0.9))

    # PR
    baseline = float(y_true.mean())
    ax_p.plot(rec, prec, color=CLASS_COLORS[1], lw=2.2,
              label=f"TCC  AP = {pr_auc:.4f}")
    ax_p.fill_between(rec, baseline, prec, alpha=0.12, color=CLASS_COLORS[1])
    ax_p.axhline(baseline, color="#3a3f52", lw=1.0, ls=":",
                 label=f"Random baseline (prevalence = {baseline:.2f})")

    # iso-F1 contours
    for f1_val in [0.5, 0.6, 0.7, 0.8, 0.9]:
        x_f1 = np.linspace(0.01, 1.0, 200)
        y_f1 = f1_val * x_f1 / (2 * x_f1 - f1_val + 1e-9)
        mask = (y_f1 >= 0) & (y_f1 <= 1)
        ax_p.plot(x_f1[mask], y_f1[mask], color="#2e3347", lw=0.8, ls="--")
        ax_p.text(x_f1[mask][-1] + 0.01, y_f1[mask][-1],
                  f"F₁={f1_val}", color="#3e4a60", fontsize=7, va="center")

    ax_p.annotate(f"AP = {pr_auc:.4f}",
                  xy=(0.35, 0.55), color="white", fontsize=10, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#181e2c",
                            edgecolor="#2e3347", alpha=0.9))

    ax_p.set(title="Precision-Recall Curve\n(OvR · Average Precision = interpolated AUPRC)",
             xlabel="Recall  (True Positive Rate)",
             ylabel="Precision  (Positive Predictive Value)",
             xlim=[-0.01, 1.01], ylim=[-0.01, 1.03])
    ax_p.title.set_color("#c8d0df"); ax_p.title.set_fontsize(10)

    for ax in [ax_r, ax_p]:
        ax.legend(fontsize=8.5, facecolor="#181e2c", edgecolor="#2e3347",
                  labelcolor="#c8d0df", loc="lower right", framealpha=0.9)

    path = str(PLOTS_DIR / "roc_pr_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {path}")
    return roc_auc, pr_auc

# =============================================================================
# PLOT 3 — PER-CLASS METRICS BAR CHART
# =============================================================================

def plot_per_class_metrics(y_true, y_pred_proba):
    print("[3/9] Plotting per-class metrics …")
    preds   = np.argmax(y_pred_proba, axis=1)
    report  = classification_report(y_true, preds, target_names=CLASS_NAMES,
                                    output_dict=True)
    n_test  = len(y_true)

    metrics    = ["precision", "recall", "f1-score"]
    met_labels = ["Precision  (PPV)", "Recall  (Sensitivity / TPR)", "F₁-Score  (Harmonic Mean)"]
    bar_colors = ["#4e9bd4", "#e8923a", "#5cb87a"]
    x      = np.arange(NUM_CLASSES)
    bar_w  = 0.24

    vals_all = {m: [report[cn][m] for cn in CLASS_NAMES] for m in metrics}
    ymin = max(0.0, min(min(v) for v in vals_all.values()) - 0.08)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#111722")

    for i, (metric, label, color) in enumerate(zip(metrics, met_labels, bar_colors)):
        vals = vals_all[metric]
        bars = ax.bar(x + i * bar_w, vals, bar_w, label=label,
                      color=color, alpha=0.88, edgecolor="#0d1117", lw=0.6, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=9, color="white", fontweight="bold")

    # macro reference lines — staggered
    y_offsets = [0.006, -0.012, 0.006]
    for (metric, color, ls), y_off in zip(
            zip(metrics, bar_colors, ["-", "--", "-."]), y_offsets):
        macro = report["macro avg"][metric]
        ax.axhline(macro, color=color, lw=0.9, ls=ls, alpha=0.45, zorder=2)
        ax.text(x[-1] + bar_w * 2.8, macro + y_off,
                f"macro {metric[:3]}={macro:.3f}", color=color, fontsize=7.5,
                ha="left", alpha=0.85, clip_on=False)

    ax.set_xticks(x + bar_w)
    ax.set_xticklabels(CLASS_NAMES, color="#c8d0df", fontsize=11)
    ax.set_xlim(-0.3, NUM_CLASSES + 0.5)
    ax.set_ylim(ymin, 1.07)
    ax.set_ylabel("Score", color="#7a8499", fontsize=10)
    ax.tick_params(colors="#c8d0df", labelsize=9)
    ax.yaxis.grid(True, color="#1e2535", lw=0.7, ls="--", zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_edgecolor("#2e3347")

    for ci, cn in enumerate(CLASS_NAMES):
        support = int(report[cn]["support"])
        ax.text(ci + bar_w, ymin + 0.005, f"n={support:,}",
                ha="center", va="bottom", fontsize=8, color="#7a8499")

    ax.set_title(
        f"Per-Class Discriminative Performance  ·  Precision · Recall · F₁-Score\n"
        f"{MODEL_NAME}  |  Binary TCC Detection  |  n = {n_test:,} test samples",
        color="white", fontsize=10, pad=12)
    ax.legend(fontsize=9, facecolor="#181e2c", edgecolor="#2e3347",
              labelcolor="#c8d0df", loc="lower right", framealpha=0.9)

    path = str(PLOTS_DIR / "per_class_metrics.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {path}")
    return report

# =============================================================================
# PLOT 4 — CONFIDENCE DISTRIBUTION
# =============================================================================

def plot_confidence_distribution(y_pred_proba, y_true, preds):
    print("[4/9] Plotting confidence distributions …")
    confidence = np.max(y_pred_proba, axis=1)
    correct    = preds == y_true
    n_test     = len(y_true)
    acc        = correct.mean()

    fig = plt.figure(figsize=(15, 7.5))
    fig.patch.set_facecolor("#0d1117")
    fig.text(0.5, 0.985,
             f"Posterior Predictive Probability Distribution — {MODEL_NAME}  |  Binary TCC Detection",
             ha="center", va="top", color="white", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.945,
             f"Softmax posterior p̂ = max_c P(y=c | x)  ·  n = {n_test:,} test samples  "
             f"·  Accuracy = {acc:.4f}  ({correct.sum():,} correct / {(~correct).sum():,} misclassified)",
             ha="center", va="top", color="#7a8499", fontsize=9)

    gs   = gridspec.GridSpec(1, 2, figure=fig, left=0.07, right=0.97,
                             top=0.86, bottom=0.10, wspace=0.28)
    ax_h = fig.add_subplot(gs[0])
    ax_v = fig.add_subplot(gs[1])
    _style_ax(ax_h); _style_ax(ax_v)

    # LEFT — histogram
    bins = np.linspace(0.45, 1.0, 30)
    ax_h.hist(confidence[correct],  bins=bins, alpha=0.78, color="#4a9e6b",
              label=f"Correct classification  (n={correct.sum():,})",
              edgecolor="#0d1117", lw=0.4, zorder=3)
    ax_h.hist(confidence[~correct], bins=bins, alpha=0.78, color="#c94f4f",
              label=f"Misclassification  (n={(~correct).sum():,})",
              edgecolor="#0d1117", lw=0.4, zorder=3)

    med_correct   = np.median(confidence[correct])
    med_incorrect = np.median(confidence[~correct])
    ax_h.axvline(med_correct,   color="#6fcf97", lw=1.5, ls="--", zorder=4)
    ax_h.axvline(med_incorrect, color="#eb5757", lw=1.5, ls="--", zorder=4)
    ymax = ax_h.get_ylim()[1] if ax_h.get_ylim()[1] > 0 else 1
    ax_h.set_ylim(0, ymax)
    ymax = ax_h.get_ylim()[1]
    ax_h.text(med_correct   + 0.005, ymax * 0.88,
              f"Median correct = {med_correct:.3f}",
              color="#6fcf97", fontsize=7.5, va="top")
    ax_h.text(med_incorrect + 0.005, ymax * 0.70,
              f"Median incorrect = {med_incorrect:.3f}",
              color="#eb5757", fontsize=7.5, va="top")

    ax_h.set_xlabel("Posterior Predictive Probability  max_c P(y=c | x)", fontsize=9)
    ax_h.set_ylabel("Sample Frequency", fontsize=9)
    ax_h.set_title("Predictive Confidence Distribution\nby Classification Outcome",
                   fontsize=10, pad=8)
    ax_h.legend(fontsize=8.5, facecolor="#181e2c", edgecolor="#2e3347",
                labelcolor="#c8d0df", framealpha=0.9)

    # RIGHT — per-class violin
    data_by_class = [confidence[y_true == c] for c in range(NUM_CLASSES)]
    parts = ax_v.violinplot(data_by_class, positions=range(NUM_CLASSES),
                            showmedians=True, showextrema=True, widths=0.55)
    for pc, color in zip(parts["bodies"], CLASS_COLORS):
        pc.set_facecolor(color); pc.set_alpha(0.72); pc.set_edgecolor("#0d1117")
    parts["cmedians"].set_color("white"); parts["cmedians"].set_lw(2.2)
    parts["cmins"].set_color("#5a6275");  parts["cmaxes"].set_color("#5a6275")
    parts["cbars"].set_color("#5a6275");  parts["cbars"].set_lw(0.8)

    for c in range(NUM_CLASSES):
        med = np.median(data_by_class[c])
        ax_v.text(c, med + 0.012, f"{med:.3f}",
                  ha="center", va="bottom", fontsize=8.5,
                  color="white", fontweight="bold")

    ax_v.axhline(1 / NUM_CLASSES, color="#3a3f52", lw=1.0, ls=":",
                 label=f"Random-chance baseline (p = 1/{NUM_CLASSES} = 0.50)")
    ax_v.set_xticks(range(NUM_CLASSES))
    ax_v.set_xticklabels(CLASS_NAMES, color="#c8d0df", fontsize=10)
    ax_v.set_ylabel("Posterior Predictive Probability  P(ŷ | x)", fontsize=9)
    ax_v.set_title("Class-Conditional Predictive Uncertainty\n"
                   "(Softmax posterior per ground-truth class)",
                   fontsize=10, pad=8)
    ax_v.legend(fontsize=8, facecolor="#181e2c", edgecolor="#2e3347",
                labelcolor="#c8d0df", loc="lower right", framealpha=0.9)

    path = str(PLOTS_DIR / "confidence_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {path}")

# =============================================================================
# GRAD-CAM
# =============================================================================

def compute_gradcam(model, seq_batch, class_idx):
    FALLBACK_LAYERS = ["bn3", "res2", "sattn_scale", "bn2", "res1", "bn1",
                       "clstm3", "clstm2", "clstm1"]

    def _try(lname):
        try:
            gm = tf.keras.Model(
                inputs  = model.input,
                outputs = [model.get_layer(lname).output, model.output])
        except (ValueError, KeyError):
            return None
        seq_t = tf.cast(seq_batch, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(seq_t)
            conv_out, logits = gm(seq_t, training=False)
            # binary: use logit of positive class
            score = logits[:, min(class_idx, logits.shape[-1]-1)]
        grads = tape.gradient(score, conv_out)
        if grads is None: return None
        if grads.ndim == 5:
            grads = grads[:, -1]; conv_out = conv_out[:, -1]
        pooled = tf.reduce_mean(grads, axis=[1, 2])
        cam    = tf.reduce_sum(conv_out * pooled[:, tf.newaxis, tf.newaxis, :], axis=-1)
        cam    = tf.nn.relu(cam)[0].numpy()
        if cam.max() < 1e-6: return None
        cam = cam / cam.max()
        cam = tf.image.resize(cam[..., np.newaxis],
                              [seq_batch.shape[2], seq_batch.shape[3]]).numpy()[..., 0]
        return cam, lname

    for lname in FALLBACK_LAYERS:
        r = _try(lname)
        if r is not None: return r
    H, W = seq_batch.shape[2], seq_batch.shape[3]
    return np.ones((H, W)) * 0.3, "fallback"


def plot_gradcam(model, y_true, y_pred_proba, X_sample, y_sample):
    print("[5/9] Computing Grad-CAM …")
    np.random.seed(SEED)

    preds = np.argmax(y_pred_proba, axis=1)
    # pick 5 correct + 5 misclassified
    correct_idx   = np.where((preds == y_true) & (y_true == 1))[0]
    incorrect_idx = np.where((preds != y_true))[0]
    nontcc_idx    = np.where((preds == y_true) & (y_true == 0))[0]

    chosen = []
    for pool, label in [
        (correct_idx,   1),
        (incorrect_idx, None),
        (nontcc_idx,    0),
    ]:
        if len(pool) == 0: continue
        sel = np.random.choice(pool, size=min(4, len(pool)), replace=False)
        for s in sel:
            chosen.append((s, int(y_true[s]), int(preds[s])))
    chosen = chosen[:10]
    n_cols = len(chosen)
    if n_cols == 0:
        print("  ⚠  No samples for Grad-CAM"); return

    T      = X_sample.shape[1]
    fig    = plt.figure(figsize=(n_cols * 2.9, 11))
    fig.patch.set_facecolor("#0d1117")
    fig.text(0.5, 0.99,
             "Gradient-weighted Class Activation Mapping (Grad-CAM)  —  ConvLSTM Binary Feature Attribution",
             ha="center", va="top", color="white", fontsize=12, fontweight="bold")
    fig.text(0.5, 0.965,
             "Row 1: IR brightness temperature (final temporal frame, t = T)  ·  "
             "Row 2: Grad-CAM saliency overlay  ·  Row 3: Isolated saliency heatmap",
             ha="center", va="top", color="#7a8499", fontsize=8)
    fig.text(0.5, 0.945,
             "Warm regions = spatial features most influential to predicted class  ·  "
             "Layer = deepest layer with non-zero gradient  ·  "
             "Green title = correct  /  Red title = misclassified",
             ha="center", va="top", color="#5a6275", fontsize=7.5)

    gs = gridspec.GridSpec(3, n_cols, figure=fig,
                           left=0.045, right=0.90,
                           top=0.92, bottom=0.03,
                           hspace=0.06, wspace=0.04)

    cmap_cam = LinearSegmentedColormap.from_list(
        "cam", ["#000000","#0d1117","#1a3a5c","#e15759","#f0a500","#fff176"])

    for col, (idx, true_cls, pred_cls) in enumerate(chosen):
        seq     = X_sample[idx].astype(np.float32)
        resized = np.stack([
            tf.image.resize(seq[t], [TARGET_H, TARGET_W]).numpy()
            for t in range(T)
        ], axis=0)
        seq_b      = resized[np.newaxis]
        last_frame = resized[-1, :, :, 0]

        try:
            cam, layer_used = compute_gradcam(model, seq_b, true_cls)
        except Exception as e:
            cam, layer_used = np.zeros((TARGET_H, TARGET_W)), "error"

        title_color = "#4a9e6b" if true_cls == pred_cls else "#c94f4f"
        title_text  = f"{CLASS_NAMES[true_cls]}"
        if true_cls != pred_cls:
            title_text += f"\n→{CLASS_NAMES[pred_cls]}"

        ax0 = fig.add_subplot(gs[0, col])
        ax0.imshow(last_frame, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax0.set_title(title_text, color=title_color, fontsize=7.5, pad=3, fontweight="bold")
        ax0.axis("off")
        if col == 0:
            ax0.set_ylabel("IR (t=T)", color="#7a8499", fontsize=7.5)
            ax0.yaxis.set_label_coords(-0.12, 0.5)

        ax1 = fig.add_subplot(gs[1, col])
        ax1.imshow(last_frame, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax1.imshow(cam, cmap=cmap_cam, alpha=0.58, vmin=0, vmax=1, interpolation="bilinear")
        ax1.axis("off")
        if col == 0:
            ax1.set_ylabel("Overlay", color="#7a8499", fontsize=7.5)
            ax1.yaxis.set_label_coords(-0.12, 0.5)

        ax2 = fig.add_subplot(gs[2, col])
        ax2.imshow(cam, cmap=cmap_cam, vmin=0, vmax=1, interpolation="bilinear")
        ax2.axis("off")
        ax2.set_xlabel(f"[{layer_used}]", color="#3e4a60", fontsize=6.5)
        ax2.xaxis.set_label_coords(0.5, -0.04)
        if col == 0:
            ax2.set_ylabel("Grad-CAM", color="#7a8499", fontsize=7.5)
            ax2.yaxis.set_label_coords(-0.12, 0.5)

    cbar_ax = fig.add_axes([0.915, 0.25, 0.012, 0.40])
    sm = plt.cm.ScalarMappable(cmap=cmap_cam, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Gradient-weighted\nactivation magnitude",
                   color="#7a8499", fontsize=7.5, labelpad=6)
    cbar.ax.tick_params(colors="#7a8499", labelsize=7)
    cbar.outline.set_edgecolor("#2e3347")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["Low", "", "Mid", "", "High"])

    path = str(PLOTS_DIR / "gradcam_samples.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {path}")

# =============================================================================
# TRAINING HISTORY PLOTS  (kept from original, styled to match MC)
# =============================================================================

PALETTE = {
    "train": "#3266AD", "val": "#E24B4A",
    "auc_roc": "#534AB7", "auc_pr": "#D85A30",
    "gap_pos": "#3266AD", "gap_neg": "#E24B4A",
    "var": "#534AB7", "best": "#1D9E75", "neutral": "#888780",
}

def load_training_csv():
    if not os.path.exists(TRAINING_CSV):
        print(f"  ⚠  Training CSV not found: {TRAINING_CSV}")
        return None
    df = pd.read_csv(TRAINING_CSV)
    print(f"  Loaded training history — {len(df)} epochs, {len(df.columns)} columns")
    return df


def plot_learning_curves(df):
    print("[6/9] Plotting learning curves …")
    epochs  = df["epoch"].values if "epoch" in df.columns else np.arange(len(df))
    best_ep = int(df["val_accuracy"].idxmax()) if "val_accuracy" in df.columns else len(df)-1

    fig = plt.figure(figsize=(16, 9.5))
    fig.patch.set_facecolor("#0d1117")
    fig.text(0.5, 0.985,
             f"Training History — {MODEL_NAME}  |  Binary TCC Detection",
             ha="center", va="top", color="white", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.950,
             f"Epoch-by-epoch training and validation metrics  ·  "
             f"Best epoch = {best_ep}  ·  Total epochs = {len(df)}",
             ha="center", va="top", color="#7a8499", fontsize=9)

    gs = gridspec.GridSpec(2, 2, figure=fig, left=0.07, right=0.97,
                           top=0.90, bottom=0.08, hspace=0.38, wspace=0.28)

    pairs = [
        ("accuracy", "val_accuracy", "Accuracy",   (0.85, 1.002)),
        ("loss",     "val_loss",     "Loss",        None),
        ("auc_roc",  "val_auc_roc",  "AUROC",       (0.95, 1.002)),
        ("auc_pr",   "val_auc_pr",   "AUPRC",       (0.90, 1.002)),
    ]
    for i, (tc, vc, title, ylim) in enumerate(pairs):
        if tc not in df.columns: continue
        ax = fig.add_subplot(gs[i//2, i%2])
        ax.set_facecolor("#111722")
        ax.plot(epochs, df[tc], color=PALETTE["train"], lw=1.8,
                label="Training", zorder=3)
        if vc in df.columns:
            ax.plot(epochs, df[vc], color=PALETTE["val"], lw=1.8,
                    label="Validation", zorder=3)
            ax.fill_between(epochs, df[tc], df[vc], alpha=0.07,
                            color=PALETTE["val"])
        ax.axvline(best_ep, color=PALETTE["best"], lw=1.2, ls="--",
                   label=f"Best epoch ({best_ep})", zorder=2)
        if ylim: ax.set_ylim(*ylim)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel(title, fontsize=9)
        ax.set_title(title, color="#c8d0df", fontsize=10, fontweight="bold", pad=6)
        ax.tick_params(colors="#c8d0df", labelsize=8)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.legend(fontsize=7.5, facecolor="#181e2c", edgecolor="#2e3347",
                  labelcolor="#c8d0df", framealpha=0.9)
        ax.yaxis.grid(True, color="#1e2535", lw=0.6, ls="--")
        ax.set_axisbelow(True)
        for sp in ax.spines.values(): sp.set_edgecolor("#2e3347")

    path = str(PLOTS_DIR / "learning_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {path}")
    return best_ep


def plot_generalization_gap(df):
    print("[7/9] Plotting generalization gap …")
    epochs = df["epoch"].values if "epoch" in df.columns else np.arange(len(df))
    gap    = (df["accuracy"] - df["val_accuracy"]).values
    roll   = pd.Series(gap).rolling(3, min_periods=1).mean().values
    colors = [PALETTE["gap_pos"] if g >= 0 else PALETTE["gap_neg"] for g in gap]

    fig = plt.figure(figsize=(15, 6.5))
    fig.patch.set_facecolor("#0d1117")
    fig.text(0.5, 0.985,
             f"Generalization Gap Analysis — {MODEL_NAME}  |  Binary TCC Detection",
             ha="center", va="top", color="white", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.945,
             f"Train accuracy − Validation accuracy  ·  "
             f"Mean gap = {gap.mean()*100:.3f}%  ·  "
             f"Max gap = {gap.max()*100:.3f}% at epoch {int(gap.argmax())}",
             ha="center", va="top", color="#7a8499", fontsize=9)

    gs  = gridspec.GridSpec(1, 2, figure=fig, left=0.07, right=0.97,
                            top=0.86, bottom=0.12, wspace=0.28)
    ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])
    for ax in [ax1, ax2]: _style_ax(ax)

    ax1.bar(epochs, gap * 100, color=colors, alpha=0.75, width=0.7, zorder=3)
    ax1.axhline(0, color="#c8d0df", lw=0.8, ls="--")
    ax1.axhline(gap.mean()*100, color=PALETTE["neutral"], lw=1.2, ls=":",
                label=f"Mean gap = {gap.mean()*100:.3f}%")
    ax1.set_xlabel("Epoch", fontsize=9)
    ax1.set_ylabel("Accuracy gap  (Train − Validation)  %", fontsize=9)
    ax1.set_title("Per-epoch Generalization Gap", color="#c8d0df", fontsize=10, pad=8)
    ax1.legend(fontsize=8, facecolor="#181e2c", edgecolor="#2e3347", labelcolor="#c8d0df")

    ax2.plot(epochs, gap*100, color=PALETTE["gap_pos"], lw=1.2, alpha=0.4, label="Gap")
    ax2.plot(epochs, roll*100, color=PALETTE["auc_roc"], lw=2.0,
             label="3-epoch rolling mean")
    ax2.axhline(0, color="#c8d0df", lw=0.8, ls="--")
    ax2.set_xlabel("Epoch", fontsize=9)
    ax2.set_ylabel("Gap  (%)", fontsize=9)
    ax2.set_title("Generalization Gap — Rolling Mean (w=3)", color="#c8d0df", fontsize=10, pad=8)
    ax2.legend(fontsize=8, facecolor="#181e2c", edgecolor="#2e3347", labelcolor="#c8d0df")

    path = str(PLOTS_DIR / "generalization_gap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {path}")


def plot_auc_progression(df, best_ep):
    print("[8/9] Plotting AUC progression …")
    epochs = df["epoch"].values if "epoch" in df.columns else np.arange(len(df))

    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#111722")

    if "val_auc_roc" in df.columns:
        ax.plot(epochs, df["val_auc_roc"], color=PALETTE["auc_roc"],
                lw=2.0, label="Validation AUROC")
        ax.fill_between(epochs, df["val_auc_roc"].min() * 0.999,
                        df["val_auc_roc"], alpha=0.10, color=PALETTE["auc_roc"])
    if "val_auc_pr" in df.columns:
        ax.plot(epochs, df["val_auc_pr"], color=PALETTE["auc_pr"],
                lw=2.0, ls="--", label="Validation AUPRC")
    if "auc_roc" in df.columns:
        ax.plot(epochs, df["auc_roc"], color=PALETTE["auc_roc"],
                lw=1.2, alpha=0.35, ls=":", label="Training AUROC")
    ax.axvline(best_ep, color=PALETTE["best"], lw=1.2, ls="--",
               label=f"Best epoch ({best_ep})")
    ax.set_xlabel("Epoch", color="#7a8499", fontsize=10)
    ax.set_ylabel("Area Under Curve", color="#7a8499", fontsize=10)
    ax.set_title(f"AUROC & AUPRC Progression Over Training\n{MODEL_NAME}  |  Binary TCC Detection",
                 color="white", fontsize=11, pad=10)
    ax.tick_params(colors="#c8d0df", labelsize=9)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax.yaxis.grid(True, color="#1e2535", lw=0.6, ls="--")
    ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_edgecolor("#2e3347")
    ax.legend(fontsize=9, facecolor="#181e2c", edgecolor="#2e3347",
              labelcolor="#c8d0df", framealpha=0.9)

    path = str(PLOTS_DIR / "auc_progression.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {path}")


def plot_combined_summary(df, best_ep, stats):
    print("[9/9] Plotting combined summary …")
    epochs = df["epoch"].values if "epoch" in df.columns else np.arange(len(df))
    gap    = (df["accuracy"] - df["val_accuracy"]).values

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0d1117")
    fig.text(0.5, 0.985,
             f"Scientific Evaluation Summary — {MODEL_NAME}  |  Binary TCC Detection  "
             f"|  Best epoch: {best_ep}  |  {TIMESTAMP}",
             ha="center", va="top", color="white", fontsize=12, fontweight="bold")

    gs = gridspec.GridSpec(3, 3, figure=fig, left=0.06, right=0.97,
                           top=0.95, bottom=0.07, hspace=0.50, wspace=0.32)

    def _mini_ax(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor("#111722")
        ax.tick_params(colors="#c8d0df", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#2e3347")
        ax.yaxis.grid(True, color="#1e2535", lw=0.5, ls="--")
        ax.set_axisbelow(True)
        return ax

    # A — Accuracy
    ax1 = _mini_ax(gs[0, 0])
    ax1.plot(epochs, df["accuracy"],     color=PALETTE["train"], lw=1.6, label="Train")
    ax1.plot(epochs, df["val_accuracy"], color=PALETTE["val"],   lw=1.6, label="Val")
    ax1.axvline(best_ep, color=PALETTE["best"], lw=1.0, ls="--")
    ax1.set_ylim(0.88, 1.002)
    ax1.set_title("(A) Accuracy", color="#c8d0df", fontsize=9, fontweight="bold")
    ax1.legend(fontsize=7, facecolor="#181e2c", edgecolor="#2e3347", labelcolor="#c8d0df")
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    # B — Loss
    ax2 = _mini_ax(gs[0, 1])
    ax2.plot(epochs, df["loss"],     color=PALETTE["train"], lw=1.6)
    ax2.plot(epochs, df["val_loss"], color=PALETTE["val"],   lw=1.6)
    ax2.axvline(best_ep, color=PALETTE["best"], lw=1.0, ls="--")
    ax2.set_title("(B) Loss", color="#c8d0df", fontsize=9, fontweight="bold")

    # C — AUC
    ax3 = _mini_ax(gs[0, 2])
    if "val_auc_roc" in df.columns:
        ax3.plot(epochs, df["val_auc_roc"], color=PALETTE["auc_roc"], lw=1.6, label="AUROC")
    if "val_auc_pr" in df.columns:
        ax3.plot(epochs, df["val_auc_pr"],  color=PALETTE["auc_pr"],  lw=1.6, ls="--", label="AUPRC")
    ax3.axvline(best_ep, color=PALETTE["best"], lw=1.0, ls="--")
    ax3.set_title("(C) Validation AUC", color="#c8d0df", fontsize=9, fontweight="bold")
    ax3.legend(fontsize=7, facecolor="#181e2c", edgecolor="#2e3347", labelcolor="#c8d0df")
    ax3.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))

    # D — Generalization gap
    ax4 = _mini_ax(gs[1, 0:2])
    colors = [PALETTE["gap_pos"] if g >= 0 else PALETTE["gap_neg"] for g in gap]
    ax4.bar(epochs, gap*100, color=colors, alpha=0.72, width=0.7, zorder=3)
    ax4.axhline(0, color="#c8d0df", lw=0.8, ls="--")
    roll = pd.Series(gap).rolling(3, min_periods=1).mean().values
    ax4.plot(epochs, roll*100, color=PALETTE["neutral"], lw=1.5,
             label="3-ep rolling mean")
    ax4.set_title("(D) Generalization gap  (Train − Val accuracy %)",
                  color="#c8d0df", fontsize=9, fontweight="bold")
    ax4.set_ylabel("Gap (%)", color="#7a8499", fontsize=8)
    ax4.legend(fontsize=7, facecolor="#181e2c", edgecolor="#2e3347", labelcolor="#c8d0df")

    # E — Epoch delta
    ax5 = _mini_ax(gs[1, 2])
    delta = np.abs(np.diff(df["val_accuracy"].values))
    ax5.bar(epochs[1:], delta*100, color=PALETTE["var"], alpha=0.7, width=0.7, zorder=3)
    ax5.set_title("(E) |Δ Val accuracy| per epoch",
                  color="#c8d0df", fontsize=9, fontweight="bold")
    ax5.set_ylabel("Change (%)", color="#7a8499", fontsize=8)

    # F — Stats table
    ax6 = fig.add_subplot(gs[2, :])
    ax6.set_facecolor("#0d1117"); ax6.axis("off")
    rows = []
    for metric in ["accuracy", "auc_roc", "auc_pr", "loss"]:
        e = stats.get(metric, {})
        if not e: continue
        tr = e.get("train", {}); va = e.get("val", {}); gn = e.get("generalization", {})
        rows.append([
            metric.replace("_", " ").upper(),
            f"{tr.get('mean',0):.4f} ± {tr.get('std',0):.4f}",
            f"{tr.get('final',0):.4f}",
            f"{va.get('mean',0):.4f} ± {va.get('std',0):.4f}",
            f"{va.get('best',0):.4f}",
            f"{gn.get('gap_mean',0):.4f}",
            f"{gn.get('cohens_d',0):.3f}",
        ])
    cols = ["Metric", "Train μ ± σ", "Train final",
            "Val μ ± σ", "Val best", "Gap μ", "Cohen d"]
    tbl = ax6.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.7)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#2e3347")
        if r == 0:
            cell.set_facecolor("#1a3a5c"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#151c27")
        else:
            cell.set_facecolor("#0d1117")
        if r > 0: cell.set_text_props(color="#c8d0df")
    ax6.set_title("(F) Descriptive statistics summary",
                  color="#c8d0df", fontsize=9, fontweight="bold", pad=10)

    path = str(PLOTS_DIR / "combined_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {path}")

# =============================================================================
# STATISTICS
# =============================================================================

def compute_statistics(df):
    metrics_map = {
        "accuracy":  ("accuracy",  "val_accuracy"),
        "auc_roc":   ("auc_roc",   "val_auc_roc"),
        "auc_pr":    ("auc_pr",    "val_auc_pr"),
        "loss":      ("loss",      "val_loss"),
    }
    stats = {}
    for name, (tc, vc) in metrics_map.items():
        if tc not in df.columns: continue
        tr = df[tc].values
        va = df[vc].values if vc in df.columns else None
        entry = {"train": {
            "mean": _mean(tr), "std": _std(tr), "sem": _sem(tr),
            "ci95": _ci95(tr), "min": float(tr.min()),
            "max":  float(tr.max()), "final": float(tr[-1]),
        }}
        if va is not None:
            entry["val"] = {
                "mean": _mean(va), "std": _std(va), "sem": _sem(va),
                "ci95": _ci95(va), "min": float(va.min()),
                "max":  float(va.max()), "final": float(va[-1]),
                "best": float(va.max()) if name != "loss" else float(va.min()),
            }
            entry["generalization"] = {
                "gap_mean":   _mean(tr - va),
                "gap_std":    _std(tr - va),
                "gap_max":    float((tr - va).max()),
                "gap_max_ep": int((tr - va).argmax()),
                "cohens_d":   _cohens_d(tr, va),
            }
        stats[name] = entry

    best_ep = int(df["val_accuracy"].idxmax()) if "val_accuracy" in df.columns else len(df)-1
    tr_loss = df["loss"].values
    va_loss = df["val_loss"].values if "val_loss" in df.columns else tr_loss
    thresh  = tr_loss[0] * 0.01
    conv_ep = next((i for i, v in enumerate(tr_loss) if v <= thresh), len(df)-1)
    va_acc  = df["val_accuracy"].values if "val_accuracy" in df.columns else None
    eoe_var = float(np.var(np.diff(va_acc), ddof=0)) if va_acc is not None else None

    stats["diagnostics"] = {
        "best_epoch":               best_ep,
        "best_val_accuracy":        float(df["val_accuracy"].max()) if "val_accuracy" in df.columns else None,
        "convergence_epoch_1pct":   conv_ep,
        "train_loss_reduction_pct": float((tr_loss[0]-tr_loss[-1])/tr_loss[0]*100),
        "val_loss_reduction_pct":   float((va_loss[0]-va_loss[-1])/va_loss[0]*100),
        "epoch_to_epoch_variance":  eoe_var,
        "overfitting_index":        stats.get("accuracy",{}).get("generalization",{}).get("gap_mean"),
        "total_epochs":             len(df),
    }
    return stats

# =============================================================================
# SAVE METRICS JSON
# =============================================================================

def save_metrics(y_true, preds, y_pred_proba, roc_auc, pr_auc, report, stats):
    metrics = {
        "model":        MODEL_NAME,
        "timestamp":    TIMESTAMP,
        "n_test":       int(len(y_true)),
        "accuracy":     float(np.mean(preds == y_true)),
        "roc_auc":      float(roc_auc),
        "pr_auc":       float(pr_auc),
        "macro_f1":     float(report["macro avg"]["f1-score"]),
        "per_class": {
            CLASS_NAMES[c]: {
                "precision": report[CLASS_NAMES[c]]["precision"],
                "recall":    report[CLASS_NAMES[c]]["recall"],
                "f1":        report[CLASS_NAMES[c]]["f1-score"],
                "support":   int(report[CLASS_NAMES[c]]["support"]),
            }
            for c in range(NUM_CLASSES)
        },
        "classification_report": report,
        "training_statistics":   stats,
    }
    path = str(REPORTS_DIR / "test_metrics.json")
    with open(path, "w") as f: json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved → {path}")

    # LaTeX table
    lines = [
        r"\begin{table}[h!]", r"\centering",
        r"\caption{" + MODEL_NAME.replace("_", " ") + r" — Descriptive Statistics}",
        r"\label{tab:convlstm_binary_stats}",
        r"\begin{tabular}{lcccccc}", r"\hline",
        r"Metric & Train $\mu$ & Train $\sigma$ & Val $\mu$ & Val best & Gap $\mu$ & Cohen $d$ \\",
        r"\hline",
    ]
    for metric in ["accuracy", "auc_roc", "auc_pr", "loss"]:
        e = stats.get(metric)
        if not e or "train" not in e: continue
        tr = e["train"]; va = e.get("val", {}); gn = e.get("generalization", {})
        lines.append(
            f"{metric.replace('_',' ').title()} & "
            f"{tr['mean']:.4f} & {tr['std']:.4f} & "
            f"{va.get('mean',0):.4f} & {va.get('best',0):.4f} & "
            f"{gn.get('gap_mean',0):.4f} & {gn.get('cohens_d',0):.3f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    tex_path = str(REPORTS_DIR / "stats_table.tex")
    with open(tex_path, "w") as f: f.write("\n".join(lines))
    print(f"  LaTeX table → {tex_path}")

# =============================================================================
# PRINT CONSOLE SUMMARY
# =============================================================================

def print_summary(y_true, preds, roc_auc, pr_auc, report, stats):
    acc  = float(np.mean(preds == y_true))
    diag = stats.get("diagnostics", {})
    sep  = "=" * 62
    print(f"\n{sep}")
    print(f"  TEST SET RESULTS — {MODEL_NAME}")
    print(f"{sep}")
    print(f"  Accuracy        : {acc:.4f}  ({acc*100:.2f} %)")
    print(f"  AUROC           : {roc_auc:.4f}")
    print(f"  AUPRC           : {pr_auc:.4f}")
    print(f"  Macro F1        : {report['macro avg']['f1-score']:.4f}")
    print(f"  Best epoch      : {diag.get('best_epoch')}")
    print(f"  Best val acc    : {diag.get('best_val_accuracy', 0)*100:.2f}%")
    print(f"  Overfit index   : {diag.get('overfitting_index', 0):.5f}")
    print(f"-{'-'*61}")
    print(f"  {'Class':10s}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Support':>8}")
    for c, name in enumerate(CLASS_NAMES):
        p = report[name]["precision"]; r = report[name]["recall"]
        f = report[name]["f1-score"];  s = int(report[name]["support"])
        print(f"  {name:10s}  {p:6.3f}  {r:6.3f}  {f:6.3f}  {s:8,}")
    print(f"{sep}\n")

# =============================================================================
# MAIN
# =============================================================================

def main():
    setup_gpu()
    print(f"\n{'='*62}")
    print(f"  {MODEL_NAME} — Full Scientific Evaluation")
    print(f"  {TIMESTAMP}")
    print(f"{'='*62}\n")

    # ── 1. Load predictions ───────────────────────────────────────
    print("[DATA] Loading predictions …")
    y_true, y_pred_proba = load_predictions()

    if y_true is None:
        print("  ⚠  Could not load predictions. Check DATA_NPZ_PATH or MODEL_PATH.")
        return

    preds = np.argmax(y_pred_proba, axis=1)
    print(f"  Test samples: {len(y_true):,}  |  "
          f"Positive (TCC): {int(y_true.sum()):,}  |  "
          f"Negative: {int((y_true==0).sum()):,}")

    # ── 2. Static evaluation plots ────────────────────────────────
    plot_confusion_matrix(y_true, y_pred_proba)
    roc_auc, pr_auc = plot_roc_pr(y_true, y_pred_proba)
    report          = plot_per_class_metrics(y_true, y_pred_proba)
    plot_confidence_distribution(y_pred_proba, y_true, preds)

    # ── 3. Grad-CAM — need model + a sample of data ───────────────
    if os.path.exists(MODEL_PATH):
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("float32")
        model = tf.keras.models.load_model(
            MODEL_PATH, custom_objects=CUSTOM_OBJECTS, compile=False)

        # load a small random sample for Grad-CAM
        try:
            npz    = np.load(DATA_NPZ_PATH, mmap_mode="r")
            X_all  = npz["X"]; y_all = np.array(npz["y"])
            rng    = np.random.default_rng(SEED)
            idx    = rng.choice(len(X_all), size=min(200, len(X_all)), replace=False)
            X_samp = np.array(X_all[idx]); y_samp = y_all[idx]
            # get predictions for this sample
            probs_samp = []
            for i in range(0, len(X_samp), BATCH_SIZE):
                seq = X_samp[i:i+BATCH_SIZE].astype(np.float32)
                T   = seq.shape[1]
                rsz = np.stack([
                    tf.image.resize(seq[:, t], [TARGET_H, TARGET_W]).numpy()
                    for t in range(T)
                ], axis=1)
                p = model(tf.cast(rsz, tf.float32), training=False).numpy()
                if p.ndim == 1 or (p.ndim == 2 and p.shape[1] == 1):
                    p = np.stack([1-p.ravel(), p.ravel()], axis=1)
                probs_samp.append(p)
            probs_samp = np.concatenate(probs_samp, axis=0)
            plot_gradcam(model, y_samp, probs_samp, X_samp, y_samp)
        except Exception as e:
            print(f"  ⚠  Grad-CAM skipped: {e}")
    else:
        print("  ⚠  Grad-CAM skipped — model not found")

    # ── 4. Training history plots ─────────────────────────────────
    df = load_training_csv()
    if df is not None:
        stats   = compute_statistics(df)
        best_ep = plot_learning_curves(df)
        plot_generalization_gap(df)
        plot_auc_progression(df, best_ep)
        plot_combined_summary(df, best_ep, stats)
    else:
        stats   = {}
        best_ep = 0

    # ── 5. Save metrics ───────────────────────────────────────────
    print_summary(y_true, preds, roc_auc, pr_auc, report, stats)
    save_metrics(y_true, preds, y_pred_proba, roc_auc, pr_auc, report, stats)

    print(f"\n✓ All outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()