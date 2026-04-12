"""
evaluate_convlstm_mc.py
=======================
Full standalone evaluation script for the ConvLSTM 5-class TCC model.

Produces:
  1. Console — test loss, accuracy, per-class classification report
  2. confusion_matrix_mc.png       — normalised + raw counts heatmap
  3. roc_pr_curves_mc.png          — per-class ROC + PR curves
  4. confidence_distribution_mc.png— prediction confidence per class
  5. per_class_metrics_mc.png      — bar chart precision / recall / F1
  6. gradcam_samples_mc.png        — Grad-CAM on 10 random test samples
  7. test_metrics_mc.json          — all numeric metrics saved to disk
"""

import os, json, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"]     = "cuda_malloc_async"

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from sklearn.model_selection    import train_test_split
from sklearn.metrics            import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing      import label_binarize

# ── suppress noisy TF logs ────────────────────────────────────────────────────
tf.get_logger().setLevel("ERROR")

# =============================================================================
# CONFIG  — adjust paths to match your project
# =============================================================================

DATA_X_PATH = "data/convlstm_mc_dataset/X.npy"
DATA_Y_PATH = "data/convlstm_mc_dataset/y.npy"
MODEL_PATH  = "models/convlstm_mc/best_convlstm_mc.keras"
PLOTS_DIR   = "output/plots/convlstm_mc"
METRICS_JSON= "logs/convlstm_mc/test_metrics_mc.json"

SEED        = 42            # must match train script
BATCH_SIZE  = 4             # keep low for 2 GB VRAM
TARGET_H    = 64
TARGET_W    = 64
NUM_CLASSES = 5
CLASS_NAMES = ["NON-TCC", "ORGANIZING", "INTENSIFYING", "MATURE", "DISSIPATING"]

# colour palette — one distinct colour per class
CLASS_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1"]

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(METRICS_JSON), exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)


# =============================================================================
# CUSTOM OBJECTS  (needed to load the saved model)
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


class MacroF1Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes=NUM_CLASSES, name="macro_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight("tp", shape=(num_classes,), initializer="zeros")
        self.fp = self.add_weight("fp", shape=(num_classes,), initializer="zeros")
        self.fn = self.add_weight("fn", shape=(num_classes,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_idx = tf.argmax(y_true, axis=-1)
        y_pred_idx = tf.argmax(y_pred, axis=-1)
        cm = tf.math.confusion_matrix(y_true_idx, y_pred_idx,
                                       num_classes=self.num_classes,
                                       dtype=tf.float32)
        tp = tf.linalg.diag_part(cm)
        fp = tf.reduce_sum(cm, axis=0) - tp
        fn = tf.reduce_sum(cm, axis=1) - tp
        self.tp.assign_add(tp); self.fp.assign_add(fp); self.fn.assign_add(fn)

    def result(self):
        p  = self.tp / (self.tp + self.fp + 1e-7)
        r  = self.tp / (self.tp + self.fn + 1e-7)
        f1 = 2 * p * r / (p + r + 1e-7)
        return tf.reduce_mean(f1)

    def reset_state(self):
        self.tp.assign(tf.zeros_like(self.tp))
        self.fp.assign(tf.zeros_like(self.fp))
        self.fn.assign(tf.zeros_like(self.fn))


CUSTOM_OBJECTS = {
    "TrainingOnlyAugmentation": TrainingOnlyAugmentation,
    "MacroF1Score":              MacroF1Score,
}


# =============================================================================
# STEP 1 — GPU setup
# =============================================================================

def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"✓ GPU detected ({len(gpus)} device(s))")
    else:
        print("⚠  Running on CPU — inference will be slow")


# =============================================================================
# STEP 2 — Rebuild the same test split used during training
# =============================================================================

def load_test_split():
    print("\n[1/6] Loading data & rebuilding test split …")
    X = np.load(DATA_X_PATH, mmap_mode="r")
    y = np.load(DATA_Y_PATH, mmap_mode="r")
    print(f"  Dataset: X{X.shape}  y{y.shape}")

    idx = np.arange(len(X))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=0.30, stratify=y, random_state=SEED)
    idx_val, idx_test = train_test_split(
        idx_test, test_size=0.50, stratify=y[idx_test], random_state=SEED)

    print(f"  Test samples: {len(idx_test):,}")
    for c, name in enumerate(CLASS_NAMES):
        n = int(np.sum(y[idx_test] == c))
        print(f"    {name:14s} → {n:,}")
    return X, y, idx_test


# =============================================================================
# STEP 3 — tf.data pipeline (identical to training script)
# =============================================================================

def make_test_dataset(X, y, idx_test):
    T, H, W, C = X.shape[1], X.shape[2], X.shape[3], X.shape[4]

    def gen():
        for idx in idx_test:
            seq   = X[idx].astype(np.float32)
            label = int(y[idx])
            resized = np.stack([
                tf.image.resize(seq[t], [TARGET_H, TARGET_W]).numpy()
                for t in range(seq.shape[0])
            ], axis=0)
            yield resized, label

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(T, TARGET_H, TARGET_W, C), dtype=tf.float32),
            tf.TensorSpec(shape=(),                          dtype=tf.int32),
        ),
    )
    ds = ds.batch(BATCH_SIZE)
    ds = ds.map(
        lambda x, lbl: (x, tf.one_hot(lbl, NUM_CLASSES)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


# =============================================================================
# STEP 4 — Load model & collect predictions
# =============================================================================

def load_model_and_predict(test_ds):
    print("\n[2/6] Loading model …")

    # force float32 globally — model was saved under mixed_float16,
    # but we run inference in pure float32 to avoid ConvLSTM index errors
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("float32")

    model = tf.keras.models.load_model(
        MODEL_PATH, custom_objects=CUSTOM_OBJECTS, compile=False)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("  Model loaded ✓  (inference policy: float32)")

    print("\n[3/6] Running inference on test set …")
    probs_list, labels_list, inputs_list = [], [], []
    for x_batch, y_batch in test_ds:
        # cast explicitly to float32 regardless of dataset dtype
        x_f32 = tf.cast(x_batch, tf.float32)
        p = model(x_f32, training=False).numpy()
        probs_list.append(p)
        labels_list.append(np.argmax(y_batch.numpy(), axis=1))
        if len(inputs_list) < 10:         # keep a few batches for Grad-CAM
            inputs_list.append(x_f32.numpy())

    probs  = np.concatenate(probs_list,  axis=0)
    y_true = np.concatenate(labels_list, axis=0)
    preds  = np.argmax(probs, axis=1)
    return model, probs, preds, y_true, inputs_list


# =============================================================================
# PLOT 1 — Confusion matrix  (normalised heat + raw counts)
# =============================================================================

def plot_confusion_matrix(y_true, preds):
    print("\n[4/6] Plotting confusion matrix …")
    cm      = confusion_matrix(y_true, preds, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    n_test  = len(y_true)

    fig = plt.figure(figsize=(17, 8))
    fig.patch.set_facecolor("#0d1117")

    # title block — reserved top 14% of figure, plots start at 0.86
    fig.text(0.5, 0.985, "Multi-Class Confusion Matrix — ConvLSTM-Residual-SE  |  TCC Lifecycle Classification",
             ha="center", va="top", color="white", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.945,
             f"5-class one-vs-rest evaluation  ·  n = {n_test:,} held-out test samples  "
             f"·  Balanced class distribution (6,000 samples / class)",
             ha="center", va="top", color="#7a8499", fontsize=9)

    gs = gridspec.GridSpec(1, 2, figure=fig, left=0.06, right=0.97,
                           top=0.86, bottom=0.13, wspace=0.28)

    subtitles = [
        "Row-Normalised Classification Rate  (diagonal = recall per class)",
        "Absolute Classification Frequency   (diagonal = true positives)",
    ]
    fmts   = [".2f", "d"]
    datas  = [cm_norm, cm]
    cmaps  = [
        LinearSegmentedColormap.from_list("n", ["#0d1117","#0f2a47","#1a5c9e","#38a3d1","#e8f4fd"]),
        LinearSegmentedColormap.from_list("r", ["#0d1117","#1a1a2e","#4a1942","#8b2fc9","#e8d5ff"]),
    ]

    for i, (ax_idx, data, fmt, subtitle, cmap) in enumerate(
            zip([gs[0], gs[1]], datas, fmts, subtitles, cmaps)):
        ax = fig.add_subplot(ax_idx)
        ax.set_facecolor("#0d1117")

        # choose annotation text colour per cell brightness
        if fmt == ".2f":
            annot_arr = np.array([[f"{v:.2f}" for v in row] for row in data])
        else:
            annot_arr = np.array([[f"{v:,}" for v in row] for row in data])

        sns.heatmap(
            data, annot=annot_arr, fmt="", cmap=cmap,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.6, linecolor="#1e2535",
            cbar_kws={"shrink": 0.78, "pad": 0.02},
            ax=ax,
            annot_kws={"size": 9.5, "color": "white", "fontweight": "bold"},
            vmin=0, vmax=(1.0 if fmt == ".2f" else cm.max()),
        )

        ax.set_title(subtitle, color="#c8d0df", fontsize=9, pad=10, loc="left", style="italic")
        ax.set_xlabel("Predicted Class Label", color="#7a8499", fontsize=9, labelpad=8)
        ax.set_ylabel("Ground-Truth Class Label", color="#7a8499", fontsize=9, labelpad=8)
        ax.tick_params(colors="#c8d0df", labelsize=8.5)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8.5)
        plt.setp(ax.get_yticklabels(), rotation=0,  fontsize=8.5)

        # colour-bar styling
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(colors="#7a8499", labelsize=8)
        cbar.outline.set_edgecolor("#2e3347")
        if fmt == ".2f":
            cbar.set_label("Recall (sensitivity)", color="#7a8499", fontsize=8)
        else:
            cbar.set_label("Sample count", color="#7a8499", fontsize=8)

        # highlight diagonal with a white border
        for j in range(NUM_CLASSES):
            ax.add_patch(plt.Rectangle((j, j), 1, 1,
                         fill=False, edgecolor="white", lw=1.5, clip_on=False))

    path = os.path.join(PLOTS_DIR, "confusion_matrix_mc.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {path}")


# =============================================================================
# PLOT 2 — ROC + PR curves  (per class + macro average)
# =============================================================================

def plot_roc_pr(y_true, probs):
    print("    Plotting ROC / PR curves …")
    y_bin  = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    n_test = len(y_true)

    fig = plt.figure(figsize=(17, 8))
    fig.patch.set_facecolor("#0d1117")
    fig.text(0.5, 0.985,
             "Discriminative Performance Curves — ConvLSTM-Residual-SE  |  TCC Lifecycle Classification",
             ha="center", va="top", color="white", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.945,
             f"One-vs-Rest (OvR) evaluation strategy  ·  n = {n_test:,} held-out test samples  "
             f"·  Balanced classes (6,000 samples / class)",
             ha="center", va="top", color="#7a8499", fontsize=9)

    gs   = gridspec.GridSpec(1, 2, figure=fig, left=0.06, right=0.97,
                             top=0.86, bottom=0.10, wspace=0.28)
    ax_r = fig.add_subplot(gs[0])
    ax_p = fig.add_subplot(gs[1])

    def _style(ax):
        ax.set_facecolor("#111722")
        ax.tick_params(colors="#c8d0df", labelsize=9)
        ax.xaxis.label.set_color("#7a8499")
        ax.yaxis.label.set_color("#7a8499")
        ax.title.set_color("#c8d0df")
        for sp in ax.spines.values():
            sp.set_edgecolor("#2e3347")
        ax.yaxis.grid(True, color="#1e2535", lw=0.7, ls="--")
        ax.xaxis.grid(True, color="#1e2535", lw=0.7, ls="--")
        ax.set_axisbelow(True)

    _style(ax_r); _style(ax_p)

    roc_aucs, pr_aucs = {}, {}
    dash_styles = ["-", "--", "-.", ":", (0,(3,1,1,1)), "-"]

    for c in range(NUM_CLASSES):
        col = CLASS_COLORS[c]
        ls  = dash_styles[c % len(dash_styles)]

        fpr, tpr, _ = roc_curve(y_bin[:, c], probs[:, c])
        roc_aucs[c] = auc(fpr, tpr)
        ax_r.plot(fpr, tpr, color=col, lw=1.8, ls=ls,
                  label=f"{CLASS_NAMES[c]}  AUROC = {roc_aucs[c]:.3f}")

        prec, rec, _ = precision_recall_curve(y_bin[:, c], probs[:, c])
        pr_aucs[c]   = average_precision_score(y_bin[:, c], probs[:, c])
        ax_p.plot(rec, prec, color=col, lw=1.8, ls=ls,
                  label=f"{CLASS_NAMES[c]}  AP = {pr_aucs[c]:.3f}")

    # macro-average ROC
    all_fpr  = np.unique(np.concatenate([
        roc_curve(y_bin[:, c], probs[:, c])[0] for c in range(NUM_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, c], probs[:, c])
        mean_tpr   += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= NUM_CLASSES
    macro_roc = auc(all_fpr, mean_tpr)
    ax_r.plot(all_fpr, mean_tpr, color="white", lw=2.4, ls="--",
              label=f"Macro-average  AUROC = {macro_roc:.3f}")
    ax_r.plot([0, 1], [0, 1], color="#3a3f52", lw=1.0, ls="--", label="Random classifier")

    # macro-average PR (iso-F1 contours)
    for f1_val in [0.5, 0.6, 0.7, 0.8, 0.9]:
        x_f1 = np.linspace(0.01, 1.0, 200)
        y_f1 = f1_val * x_f1 / (2 * x_f1 - f1_val + 1e-9)
        mask = (y_f1 >= 0) & (y_f1 <= 1)
        ax_p.plot(x_f1[mask], y_f1[mask], color="#2e3347", lw=0.8, ls="--")
        ax_p.text(x_f1[mask][-1] + 0.01, y_f1[mask][-1],
                  f"F₁={f1_val}", color="#3e4a60", fontsize=7, va="center")

    # chance baseline for PR (balanced = 1/K)
    ax_p.axhline(1 / NUM_CLASSES, color="#3a3f52", lw=1.0, ls=":",
                 label=f"Random baseline (P = 1/{NUM_CLASSES})")

    ax_r.set(
        title="Receiver Operating Characteristic  (One-vs-Rest OvR, per class + macro average)",
        xlabel="False Positive Rate  (1 − Specificity)",
        ylabel="True Positive Rate  (Sensitivity / Recall)",
        xlim=[-0.01, 1.01], ylim=[-0.01, 1.03],
    )
    ax_p.set(
        title="Precision-Recall Curves  (OvR · Average Precision = interpolated AUPRC)",
        xlabel="Recall  (True Positive Rate)",
        ylabel="Precision  (Positive Predictive Value)",
        xlim=[-0.01, 1.01], ylim=[-0.01, 1.03],
    )
    ax_r.title.set_color("#c8d0df"); ax_r.title.set_fontsize(9)
    ax_p.title.set_color("#c8d0df"); ax_p.title.set_fontsize(9)

    for ax in [ax_r, ax_p]:
        leg = ax.legend(fontsize=8, facecolor="#181e2c", edgecolor="#2e3347",
                        labelcolor="#c8d0df", loc="lower right",
                        framealpha=0.9, borderpad=0.8)

    path = os.path.join(PLOTS_DIR, "roc_pr_curves_mc.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {path}")
    return roc_aucs, pr_aucs


# =============================================================================
# PLOT 3 — Per-class metrics bar chart
# =============================================================================

def plot_per_class_metrics(y_true, preds):
    print("    Plotting per-class metrics …")
    report = classification_report(y_true, preds,
                                   target_names=CLASS_NAMES,
                                   output_dict=True)
    n_test = len(y_true)

    metrics    = ["precision", "recall", "f1-score"]
    met_labels = ["Precision  (PPV)", "Recall  (Sensitivity / TPR)", "F₁-Score  (Harmonic Mean)"]
    bar_colors = ["#4e9bd4", "#e8923a", "#5cb87a"]
    x      = np.arange(NUM_CLASSES)
    bar_w  = 0.24

    fig, ax = plt.subplots(figsize=(14, 6.5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#111722")

    vals_all = {m: [report[cn][m] for cn in CLASS_NAMES] for m in metrics}
    ymin = max(0.0, min(min(v) for v in vals_all.values()) - 0.08)

    for i, (metric, label, color) in enumerate(zip(metrics, met_labels, bar_colors)):
        vals = vals_all[metric]
        bars = ax.bar(x + i * bar_w, vals, bar_w,
                      label=label, color=color,
                      alpha=0.88, edgecolor="#0d1117", lw=0.6,
                      zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8, color="white", fontweight="bold")

    # macro-avg reference lines — stagger labels vertically so they don't stack
    y_offsets = [0.006, -0.012, 0.006]
    for (metric, color, ls), y_off in zip(
            zip(metrics, bar_colors, ["-", "--", "-."]), y_offsets):
        macro = report["macro avg"][metric]
        ax.axhline(macro, color=color, lw=0.9, ls=ls, alpha=0.45, zorder=2)
        ax.text(x[-1] + bar_w * 2.8, macro + y_off,
                f"macro {metric[:3]}={macro:.3f}", color=color, fontsize=7,
                ha="left", alpha=0.85, clip_on=False)

    ax.set_xticks(x + bar_w)
    ax.set_xticklabels(CLASS_NAMES, color="#c8d0df", fontsize=10)
    ax.set_xlim(-0.3, NUM_CLASSES + 0.5)   # extra right space for macro labels
    ax.set_ylim(ymin, 1.07)
    ax.set_ylabel("Score", color="#7a8499", fontsize=10)
    ax.tick_params(colors="#c8d0df", labelsize=9)
    ax.yaxis.grid(True, color="#1e2535", lw=0.7, ls="--", zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2e3347")

    # support annotation below each class
    for ci, cn in enumerate(CLASS_NAMES):
        support = int(report[cn]["support"])
        ax.text(ci + bar_w, ymin + 0.005,
                f"n={support:,}", ha="center", va="bottom",
                fontsize=7.5, color="#7a8499")

    ax.set_title(
        "Per-Class Discriminative Performance  ·  Precision · Recall · F₁-Score\n"
        f"ConvLSTM-Residual-SE  |  TCC Lifecycle  |  n = {n_test:,} test samples",
        color="white", fontsize=11, pad=12,
    )

    leg = ax.legend(fontsize=9, facecolor="#181e2c", edgecolor="#2e3347",
                    labelcolor="#c8d0df", loc="lower right",
                    framealpha=0.9, borderpad=0.8)

    path = os.path.join(PLOTS_DIR, "per_class_metrics_mc.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {path}")
    return report


# =============================================================================
# PLOT 4 — Prediction confidence distribution
# =============================================================================

def plot_confidence_distribution(probs, y_true, preds):
    print("    Plotting confidence distributions …")
    confidence = np.max(probs, axis=1)
    correct    = preds == y_true
    n_test     = len(y_true)
    acc        = correct.mean()

    fig = plt.figure(figsize=(17, 8))
    fig.patch.set_facecolor("#0d1117")
    fig.text(0.5, 0.985,
             "Posterior Predictive Probability Distribution — ConvLSTM-Residual-SE  |  TCC Lifecycle",
             ha="center", va="top", color="white", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.945,
             f"Softmax posterior p̂ = max_c P(y=c | x)  ·  n = {n_test:,} test samples  "
             f"·  Overall accuracy = {acc:.4f}  ({correct.sum():,} correct / {(~correct).sum():,} misclassified)",
             ha="center", va="top", color="#7a8499", fontsize=9)

    gs   = gridspec.GridSpec(1, 2, figure=fig, left=0.06, right=0.97,
                             top=0.86, bottom=0.10, wspace=0.28)
    ax_h = fig.add_subplot(gs[0])
    ax_v = fig.add_subplot(gs[1])

    def _style(ax):
        ax.set_facecolor("#111722")
        ax.tick_params(colors="#c8d0df", labelsize=9)
        ax.xaxis.label.set_color("#7a8499")
        ax.yaxis.label.set_color("#7a8499")
        ax.title.set_color("#c8d0df")
        for sp in ax.spines.values(): sp.set_edgecolor("#2e3347")
        ax.yaxis.grid(True, color="#1e2535", lw=0.7, ls="--")
        ax.set_axisbelow(True)
    _style(ax_h); _style(ax_v)

    # — LEFT: stacked histogram correct vs misclassified —
    bins = np.linspace(0.18, 1.0, 35)
    ax_h.hist(confidence[correct],  bins=bins, alpha=0.78,
              color="#4a9e6b", label=f"Correct classification  (n={correct.sum():,})",
              edgecolor="#0d1117", lw=0.4, zorder=3)
    ax_h.hist(confidence[~correct], bins=bins, alpha=0.78,
              color="#c94f4f", label=f"Misclassification  (n={(~correct).sum():,})",
              edgecolor="#0d1117", lw=0.4, zorder=3)

    # median lines — draw after hist so ylim is set, place labels at fixed fractions
    med_correct   = np.median(confidence[correct])
    med_incorrect = np.median(confidence[~correct])
    ax_h.axvline(med_correct,   color="#6fcf97", lw=1.5, ls="--", zorder=4)
    ax_h.axvline(med_incorrect, color="#eb5757", lw=1.5, ls="--", zorder=4)
    # stagger labels: correct at 88% height, incorrect at 70% height
    ymax = ax_h.get_ylim()[1]
    ax_h.text(med_correct   + 0.01, ymax * 0.88,
              f"Median correct = {med_correct:.3f}",
              color="#6fcf97", fontsize=7.5, va="top")
    ax_h.text(med_incorrect + 0.01, ymax * 0.70,
              f"Median incorrect = {med_incorrect:.3f}",
              color="#eb5757", fontsize=7.5, va="top")

    ax_h.set_xlabel("Posterior Predictive Probability  max_c P(y=c | x)", fontsize=9)
    ax_h.set_ylabel("Sample Frequency", fontsize=9)
    ax_h.set_title(
        "Predictive Confidence Distribution\nby Classification Outcome",
        fontsize=10, pad=8,
    )
    ax_h.legend(fontsize=8.5, facecolor="#181e2c", edgecolor="#2e3347",
                labelcolor="#c8d0df", framealpha=0.9)

    # — RIGHT: per-class violin —
    data_by_class = [confidence[y_true == c] for c in range(NUM_CLASSES)]
    parts = ax_v.violinplot(data_by_class, positions=range(NUM_CLASSES),
                            showmedians=True, showextrema=True, widths=0.7)
    for pc, color in zip(parts["bodies"], CLASS_COLORS):
        pc.set_facecolor(color); pc.set_alpha(0.72); pc.set_edgecolor("#0d1117")
    parts["cmedians"].set_color("white"); parts["cmedians"].set_lw(2.2)
    parts["cmins"].set_color("#5a6275");  parts["cmaxes"].set_color("#5a6275")
    parts["cbars"].set_color("#5a6275");  parts["cbars"].set_lw(0.8)

    # annotate median value
    for c in range(NUM_CLASSES):
        med = np.median(data_by_class[c])
        ax_v.text(c, med + 0.016, f"{med:.3f}",
                  ha="center", va="bottom", fontsize=7.5,
                  color="white", fontweight="bold")

    # chance line
    ax_v.axhline(1 / NUM_CLASSES, color="#3a3f52", lw=1.0, ls=":",
                 label=f"Random-chance baseline (p = 1/{NUM_CLASSES} = 0.20)")

    ax_v.set_xticks(range(NUM_CLASSES))
    ax_v.set_xticklabels(CLASS_NAMES, color="#c8d0df", fontsize=9,
                         rotation=18, ha="right")
    ax_v.set_ylabel("Posterior Predictive Probability  P(ŷ | x)", fontsize=9)
    ax_v.set_title(
        "Class-Conditional Predictive Uncertainty\n(Softmax posterior per ground-truth class)",
        fontsize=10, pad=8,
    )
    ax_v.legend(fontsize=8, facecolor="#181e2c", edgecolor="#2e3347",
                labelcolor="#c8d0df", loc="lower right", framealpha=0.9)

    path = os.path.join(PLOTS_DIR, "confidence_distribution_mc.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {path}")


# =============================================================================
# PLOT 5 — Grad-CAM on ConvLSTM last spatial layer
# =============================================================================

def compute_gradcam(model, seq_batch, class_idx, layer_name="bn3"):
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for ConvLSTM.
    Tries layer_name first; falls back through a candidate list if it
    returns a zero map (common when dark-channel samples have near-zero
    activations in late layers).
    seq_batch shape: (1, T, H, W, C)
    """
    FALLBACK_LAYERS = ["bn3", "res2", "sattn_scale", "bn2", "res1", "bn1"]

    def _try_layer(lname):
        try:
            grad_model = tf.keras.Model(
                inputs  = model.input,
                outputs = [model.get_layer(lname).output, model.output],
            )
        except ValueError:
            return None

        seq_t = tf.cast(seq_batch, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(seq_t)
            conv_out, logits = grad_model(seq_t, training=False)
            score = logits[:, class_idx]

        grads = tape.gradient(score, conv_out)
        if grads is None:
            return None

        # handle (1,T,H,W,F) and (1,H,W,F)
        if grads.ndim == 5:
            grads    = grads[:, -1]
            conv_out = conv_out[:, -1]

        pooled = tf.reduce_mean(grads, axis=[1, 2])                        # (1, F)
        cam    = tf.reduce_sum(
            conv_out * pooled[:, tf.newaxis, tf.newaxis, :], axis=-1)      # (1, H, W)
        cam    = tf.nn.relu(cam)[0].numpy()

        if cam.max() < 1e-6:       # zero map — try next layer
            return None

        cam = cam / cam.max()
        cam = tf.image.resize(cam[..., np.newaxis],
                               [seq_batch.shape[2], seq_batch.shape[3]]
                              ).numpy()[..., 0]
        return cam, lname

    for lname in FALLBACK_LAYERS:
        result = _try_layer(lname)
        if result is not None:
            return result   # (cam_array, layer_used)

    # absolute fallback: uniform heat map to avoid blank panel
    H, W = seq_batch.shape[2], seq_batch.shape[3]
    return np.ones((H, W)) * 0.3, "fallback"


def plot_gradcam(model, inputs_list, X, y, idx_test):
    print("    Computing Grad-CAM …")
    np.random.seed(SEED)

    # 2 samples per class = 10 total
    samples = []
    for cls in range(NUM_CLASSES):
        cls_idx = idx_test[y[idx_test] == cls]
        chosen  = np.random.choice(cls_idx, size=min(2, len(cls_idx)), replace=False)
        for ci in chosen:
            samples.append((ci, cls))
    samples = samples[:10]

    T      = X.shape[1]
    n_cols = len(samples)

    fig    = plt.figure(figsize=(n_cols * 2.9, 11))
    fig.patch.set_facecolor("#0d1117")

    # title zone — top 10%
    fig.text(0.5, 0.99,
             "Gradient-weighted Class Activation Mapping (Grad-CAM)  —  ConvLSTM Spatial Feature Attribution",
             ha="center", va="top", color="white", fontsize=12, fontweight="bold")
    fig.text(0.5, 0.965,
             "Row 1: IR brightness temperature (final temporal frame, t = T)  ·  "
             "Row 2: Grad-CAM saliency overlay  ·  Row 3: Isolated saliency heatmap",
             ha="center", va="top", color="#7a8499", fontsize=8)
    fig.text(0.5, 0.945,
             "Warm regions = spatial features most influential to predicted class  ·  "
             "Layer = deepest ConvLSTM layer with non-zero gradient",
             ha="center", va="top", color="#5a6275", fontsize=7.5)

    # plot zone — leave right 5% for colorbar
    gs = gridspec.GridSpec(3, n_cols, figure=fig,
                           left=0.045, right=0.90,
                           top=0.92, bottom=0.03,
                           hspace=0.06, wspace=0.04)

    cmap_cam = LinearSegmentedColormap.from_list(
        "cam", ["#000000", "#0d1117", "#1a3a5c", "#e15759", "#f0a500", "#fff176"])
    cmap_img = "gray"

    for col, (sample_idx, true_cls) in enumerate(samples):
        seq     = X[sample_idx].astype(np.float32)
        resized = np.stack([
            tf.image.resize(seq[t], [TARGET_H, TARGET_W]).numpy()
            for t in range(T)
        ], axis=0)
        seq_batch  = resized[np.newaxis]        # (1, T, H, W, C)
        last_frame = resized[-1, :, :, 0]

        try:
            cam, layer_used = compute_gradcam(model, seq_batch, true_cls)
        except Exception as e:
            print(f"    ⚠ sample {sample_idx}: {e}")
            cam, layer_used = np.zeros((TARGET_H, TARGET_W)), "error"

        # ── ROW 0: raw IR image ──────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, col])
        ax0.imshow(last_frame, cmap=cmap_img, vmin=0, vmax=1,
                   interpolation="nearest")
        ax0.set_title(CLASS_NAMES[true_cls], color=CLASS_COLORS[true_cls],
                      fontsize=8, pad=3, fontweight="bold")
        ax0.axis("off")
        if col == 0:
            ax0.set_ylabel("IR (t=T)", color="#7a8499", fontsize=7.5)
            ax0.yaxis.set_label_coords(-0.12, 0.5)

        # ── ROW 1: overlay ──────────────────────────────────────────
        ax1 = fig.add_subplot(gs[1, col])
        ax1.imshow(last_frame, cmap=cmap_img, vmin=0, vmax=1,
                   interpolation="nearest")
        ax1.imshow(cam, cmap=cmap_cam, alpha=0.58,
                   vmin=0, vmax=1, interpolation="bilinear")
        ax1.axis("off")
        if col == 0:
            ax1.set_ylabel("Overlay", color="#7a8499", fontsize=7.5)
            ax1.yaxis.set_label_coords(-0.12, 0.5)

        # ── ROW 2: heatmap only ─────────────────────────────────────
        ax2 = fig.add_subplot(gs[2, col])
        ax2.imshow(cam, cmap=cmap_cam, vmin=0, vmax=1,
                   interpolation="bilinear")
        ax2.axis("off")
        # layer label on bottom
        ax2.set_xlabel(f"[{layer_used}]", color="#3e4a60", fontsize=6.5)
        ax2.xaxis.set_label_coords(0.5, -0.04)
        if col == 0:
            ax2.set_ylabel("Grad-CAM", color="#7a8499", fontsize=7.5)
            ax2.yaxis.set_label_coords(-0.12, 0.5)

    # shared colour-bar — sits in reserved right 5% well clear of last column
    cbar_ax = fig.add_axes([0.915, 0.25, 0.012, 0.40])
    sm = plt.cm.ScalarMappable(cmap=cmap_cam,
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Gradient-weighted\nactivation magnitude", color="#7a8499",
                   fontsize=7.5, labelpad=6)
    cbar.ax.tick_params(colors="#7a8499", labelsize=7)
    cbar.outline.set_edgecolor("#2e3347")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["Low", "", "Mid", "", "High"])

    path = os.path.join(PLOTS_DIR, "gradcam_samples_mc.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {path}")


# =============================================================================
# SUMMARY CONSOLE PRINT
# =============================================================================

def print_summary(y_true, preds, probs, roc_aucs, pr_aucs, report):
    acc  = float(np.mean(preds == y_true))
    f1   = float(report["macro avg"]["f1-score"])
    print("\n" + "="*60)
    print("  TEST SET RESULTS")
    print("="*60)
    print(f"  Accuracy      : {acc:.4f}  ({acc*100:.2f} %)")
    print(f"  Macro F1      : {f1:.4f}")
    print(f"  Macro ROC-AUC : {np.mean(list(roc_aucs.values())):.4f}")
    print(f"  Macro PR-AUC  : {np.mean(list(pr_aucs.values())):.4f}")
    print("-"*60)
    print(f"  {'Class':14s}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  "
          f"{'ROC-AUC':>8}  {'PR-AUC':>7}")
    for c, name in enumerate(CLASS_NAMES):
        p  = report[name]["precision"]
        r  = report[name]["recall"]
        f  = report[name]["f1-score"]
        ra = roc_aucs[c]; pa = pr_aucs[c]
        print(f"  {name:14s}  {p:6.3f}  {r:6.3f}  {f:6.3f}  "
              f"{ra:8.4f}  {pa:7.4f}")
    print("="*60)


# =============================================================================
# SAVE METRICS JSON
# =============================================================================

def save_metrics(y_true, preds, probs, roc_aucs, pr_aucs, report):
    metrics = {
        "accuracy"   : float(np.mean(preds == y_true)),
        "macro_f1"   : float(report["macro avg"]["f1-score"]),
        "macro_roc_auc": float(np.mean(list(roc_aucs.values()))),
        "macro_pr_auc" : float(np.mean(list(pr_aucs.values()))),
        "per_class": {
            CLASS_NAMES[c]: {
                "precision": report[CLASS_NAMES[c]]["precision"],
                "recall"   : report[CLASS_NAMES[c]]["recall"],
                "f1"       : report[CLASS_NAMES[c]]["f1-score"],
                "roc_auc"  : roc_aucs[c],
                "pr_auc"   : pr_aucs[c],
            }
            for c in range(NUM_CLASSES)
        },
        "classification_report": report,
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved → {METRICS_JSON}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    setup_gpu()

    # data
    X, y, idx_test = load_test_split()
    test_ds        = make_test_dataset(X, y, idx_test)

    # inference
    model, probs, preds, y_true, inputs_list = load_model_and_predict(test_ds)

    # plots
    print("\n[5/6] Generating plots …")
    plot_confusion_matrix(y_true, preds)
    roc_aucs, pr_aucs = plot_roc_pr(y_true, probs)
    report            = plot_per_class_metrics(y_true, preds)
    plot_confidence_distribution(probs, y_true, preds)

    print("\n[6/6] Grad-CAM visualisation …")
    plot_gradcam(model, inputs_list, X, y, idx_test)

    # console + disk
    print_summary(y_true, preds, probs, roc_aucs, pr_aucs, report)
    save_metrics(y_true, preds, probs, roc_aucs, pr_aucs, report)

    print(f"\n✓ All outputs saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()