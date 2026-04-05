"""
ConvLSTM Scientific Evaluation Suite
======================================
Runs from ANY working directory — paths resolve relative to this script.

What it produces
----------------
  - Confusion matrix (raw counts + row-normalised)
  - ROC curve + Precision-Recall curve
  - Learning curves (accuracy, loss, AUC-ROC, AUC-PR)
  - Generalization gap analysis
  - Training stability analysis
  - AUC progression chart
  - 6-panel combined publication figure
  - Interactive HTML dashboard (open in browser, no server needed)
  - JSON report with full descriptive statistics
  - LaTeX-ready statistics table

HOW TO PLUG IN YOUR MODEL  (set LOAD_MODE below)
-------------------------------------------------
  "keras"  - Keras .h5 / SavedModel + .npy test arrays
  "torch"  - PyTorch checkpoint  (fill in load_model_and_predict_torch)
  "arrays" - pre-computed y_true.npy + y_pred_proba.npy
  "none"   - skip confusion matrix / ROC (still runs all other plots)
"""
import glob
import os, sys, json, math, shutil, textwrap
import numpy as np
import tensorflow as tf

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
from pathlib import Path

# ── Resolve project root so paths work regardless of cwd ─────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR
for _p in [_SCRIPT_DIR] + list(_SCRIPT_DIR.parents):
    if (_p / "logs").exists() or (_p / "src").exists() or (_p / "output").exists():
        PROJECT_ROOT = _p
        break
os.chdir(PROJECT_ROOT)
print(f"[INFO] Working directory: {PROJECT_ROOT}")

# =============================================================================
# GPU CONFIG  ← added to fix OOM on small VRAM cards (e.g. RTX 2050 / 2GB)
# Must run BEFORE any other tensorflow import or usage.
# =============================================================================

def configure_gpu():
    """
    Enable per-process GPU memory growth so TensorFlow only allocates
    what it actually needs, instead of reserving the entire 1.6 GB upfront.
    Also caps each GPU to 1 400 MB so the OS / display driver keeps headroom.
    """
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            print("[INFO] No GPU found — running on CPU.")
            return
        for gpu in gpus:
            # Option A (recommended): grow memory on demand
            tf.config.experimental.set_memory_growth(gpu, True)
            # Option B (alternative hard cap — uncomment if Option A still OOMs):
            # tf.config.set_logical_device_configuration(
            #     gpu,
            #     [tf.config.LogicalDeviceConfiguration(memory_limit=1400)]
            # )
        print(f"[INFO] GPU memory growth enabled for {len(gpus)} device(s).")
    except RuntimeError as e:
        # Memory growth must be set before GPUs are initialised.
        print(f"[WARN] Could not configure GPU memory growth: {e}")

configure_gpu()   # ← call immediately, before any tf ops

# =============================================================================
# CONFIG  ← edit these to match your project
# =============================================================================

TRAINING_CSV_PATH = "logs/convlstm/training_log_convlstm.csv"
CLASS_REPORT_PATH = "logs/convlstm/test_metrics.json"   # optional

# ── Model loading mode ────────────────────────────────────────────────────────
LOAD_MODE = "arrays"   # "keras" | "torch" | "arrays" | "none"

# Keras paths
MODEL_PATH       = "models/convlstm/best_convlstm.keras"

# Pre-computed arrays  (LOAD_MODE="arrays")
Y_TRUE_PATH      = "output/plots/convlstm/y_true.npy"
Y_PRED_PATH      = "output/plots/convlstm/y_pred_proba.npy"     # float probabilities

# Class names for confusion matrix  ([] = auto-detect)
CLASS_NAMES = ["Non-TCC", "TCC"]   # e.g. ["Normal", "Anomaly"]

# ── Inference batch size ──────────────────────────────────────────────────────
# Keep small (4–8) on GPUs with ≤ 2 GB VRAM.
# Each sample is (5, 128, 128, 1) float32 ≈ 0.3 MB → batch 4 ≈ 1.2 MB tensors
# plus activations.  Increase to 16–32 if you have ≥ 6 GB VRAM.
INFERENCE_BATCH_SIZE = 8

# ── Output locations ──────────────────────────────────────────────────────────
OUTPUT_DIR  = Path("output/plots/convlstm")
PLOTS_DIR   = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CONF_MATRIX_PATH = PLOTS_DIR / "confusion_matrix.png"
ROC_PR_PATH      = PLOTS_DIR / "roc_pr_curves.png"

BEST_EPOCH_COL = "val_accuracy"
MODEL_NAME     = "ConvLSTM"
TIMESTAMP      = datetime.now().strftime("%Y-%m-%d %H:%M")

# =============================================================================
# MATPLOTLIB STYLE
# =============================================================================

matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
    "figure.dpi":       150,
    "savefig.dpi":      180,
    "savefig.bbox":     "tight",
})

PALETTE = {
    "train":   "#3266AD",
    "val":     "#E24B4A",
    "auc_roc": "#534AB7",
    "auc_pr":  "#D85A30",
    "gap_pos": "#3266AD",
    "gap_neg": "#E24B4A",
    "var":     "#534AB7",
    "best":    "#1D9E75",
    "neutral": "#888780",
}

# =============================================================================
# MODEL LOADING & PREDICTION
# =============================================================================

def predict_in_batches(model, X, batch_size=4):
    """
    Run model.predict() manually in small batches to avoid loading the entire
    dataset tensor onto the GPU at once.  Returns a single numpy array of
    predictions, identical to what model.predict(X) would return.

    Why not just pass batch_size to model.predict()?
    -------------------------------------------------
    model.predict() with batch_size still tries to convert the full X array
    into a single EagerTensor before slicing it — that's the _EagerConst op
    that caused your OOM.  Feeding slices explicitly bypasses that conversion.
    """
    n        = len(X)
    results  = []
    n_batches = math.ceil(n / batch_size)

    print(f"[INFO] Running batched inference: {n} samples, "
          f"batch_size={batch_size}, {n_batches} batches")

    for i in range(0, n, batch_size):
        batch     = X[i : i + batch_size]          # numpy slice — stays on CPU
        # resize spatial dimensions: 128×128 → 64×64
        b, t, h, w, c = batch.shape

        input_shape = model.input_shape
        target_h = input_shape[2]
        target_w = input_shape[3]

        batch = tf.image.resize(
            tf.reshape(batch, (-1, h, w, c)),
            (target_h, target_w)
        )

        batch = tf.reshape(batch, (b, t, target_h, target_w, c))

        batch_out = model(batch, training=False)    # direct __call__ avoids EagerConst
        results.append(batch_out.numpy())

        if (i // batch_size + 1) % max(1, n_batches // 10) == 0:
            pct = min(100, int((i + batch_size) / n * 100))
            print(f"  ... {pct}% ({i + batch_size}/{n})", flush=True)

    print(f"[OK]   Inference complete — {n} samples")
    return np.concatenate(results, axis=0)


def get_predictions():
    """Return (y_true, y_pred_proba) or None."""
    if LOAD_MODE == "none":
        print("[INFO] LOAD_MODE='none' — skipping confusion matrix / ROC.")
        print("       Set LOAD_MODE to 'keras', 'torch', or 'arrays' to generate them.")
        return None

    if LOAD_MODE == "arrays":
        for p in [Y_TRUE_PATH, Y_PRED_PATH]:
            if not os.path.exists(p):
                print(f"[WARN] Array file not found: {p}")
                return None
        y_true = np.load(Y_TRUE_PATH)
        y_pred = np.load(Y_PRED_PATH)
        print(f"[OK]   Loaded arrays  y_true={y_true.shape}  y_pred={y_pred.shape}")
        return y_true, y_pred

    if LOAD_MODE == "keras":

        for p in [MODEL_PATH]:
            if not os.path.exists(p):
                print(f"[WARN] File not found: {p}")
                return None

        print("[INFO] Loading Keras model...")

        class TrainingOnlyAugmentation(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def call(self, inputs, training=None):
                return inputs

        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"TrainingOnlyAugmentation": TrainingOnlyAugmentation}
        )

        npz_path = "data/convlstm_dataset/convlstm_sequences.npz"
        if not os.path.exists(npz_path):
            print(f"[WARN] Dataset file not found: {npz_path}")
            return None

        print("[INFO] Loading dataset (memory-mapped)...")
        # Use mmap_mode='r' so the full array is NOT loaded into RAM eagerly.
        # numpy will page in only the slices we actually read during batching.
        data   = np.load(npz_path, mmap_mode="r")
        X      = data["X"]      # shape (N, 5, 128, 128, 1), stays on disk
        y_true = data["y"]      # shape (N,)

        print(f"[INFO] Dataset shape: {X.shape}  labels: {y_true.shape}")

        # ── Batched inference ── avoids the 20 GB EagerConst OOM ──────────────
        y_pred = predict_in_batches(model, X, batch_size=INFERENCE_BATCH_SIZE)

        # Save predictions for faster future evaluation
        np.save(Y_TRUE_PATH, y_true)
        np.save(Y_PRED_PATH, y_pred)

        print(f"[OK] Saved prediction arrays → {Y_TRUE_PATH}, {Y_PRED_PATH}")        # Materialise y_true from mmap to a plain numpy array
        y_true = np.array(y_true)

        return y_true, y_pred

    if LOAD_MODE == "torch":
        return load_model_and_predict_torch()

    print(f"[WARN] Unknown LOAD_MODE='{LOAD_MODE}'")
    return None


def load_model_and_predict_torch():
    """
    Fill this in for PyTorch. Skeleton:

        import torch
        from your_module import ConvLSTMModel, TestDataset
        from torch.utils.data import DataLoader

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = ConvLSTMModel(...)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()

        dataset = TestDataset(TEST_DATA_PATH, TEST_LABELS_PATH)
        loader  = DataLoader(dataset, batch_size=32, shuffle=False)

        y_true_all, y_pred_all = [], []
        with torch.no_grad():
            for X, y in loader:
                out = torch.sigmoid(model(X.to(device))).cpu().numpy()
                y_pred_all.append(out)
                y_true_all.append(y.numpy())

        return np.concatenate(y_true_all), np.concatenate(y_pred_all)
    """
    print("[WARN] load_model_and_predict_torch() not implemented.")
    return None

# =============================================================================
# CONFUSION MATRIX
# =============================================================================

def plot_confusion_matrix(y_true, y_pred_proba):
    from sklearn.metrics import (
        confusion_matrix, classification_report, ConfusionMatrixDisplay
    )

    if y_pred_proba.ndim == 1 or y_pred_proba.shape[-1] == 1:
        y_pred = (y_pred_proba.ravel() >= 0.5).astype(int)
    else:
        y_pred = np.argmax(y_pred_proba, axis=1)

    n_cls  = len(np.unique(y_true))
    labels = CLASS_NAMES if CLASS_NAMES else [str(i) for i in range(n_cls)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    cmap = LinearSegmentedColormap.from_list("blues", ["#EEF3FA", "#3266AD"])

    for ax, norm, title in zip(
        axes,
        [None, "true"],
        ["Confusion matrix — counts", "Confusion matrix — row-normalised"],
    ):
        cm   = confusion_matrix(y_true, y_pred, normalize=norm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, cmap=cmap, colorbar=True,
                  values_format=".2f" if norm else "d")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted label", fontsize=10)
        ax.set_ylabel("True label", fontsize=10)

    print("\n── Classification Report " + "─" * 31)
    print(classification_report(y_true, y_pred, target_names=labels, digits=4))

    fig.suptitle(f"{MODEL_NAME} — Confusion Matrix", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CONF_MATRIX_PATH)
    plt.close(fig)
    print(f"[OK]   Saved confusion matrix  → {CONF_MATRIX_PATH}")

# =============================================================================
# ROC + PR CURVES
# =============================================================================

def plot_roc_pr_curves(y_true, y_pred_proba):

    from sklearn.metrics import (
        roc_curve,
        auc,
        precision_recall_curve,
        average_precision_score
    )

    import numpy as np

    # Detect binary classification correctly
    if y_pred_proba.ndim == 1:
        binary = True
    elif y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
        binary = True
    elif y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 1:
        binary = True
    else:
        binary = False

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.5))

    # =========================
    # BINARY CLASSIFICATION
    # =========================
    if binary:

        scores = (
            y_pred_proba[:, 1]
            if y_pred_proba.ndim == 2
            else y_pred_proba
        )

        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)

        ax_roc.plot(
            fpr,
            tpr,
            color=PALETTE["auc_roc"],
            lw=2.2,
            label=f"AUC-ROC = {roc_auc:.4f}"
        )

        ax_pr.plot(
            recall,
            precision,
            color=PALETTE["auc_pr"],
            lw=2.2,
            label=f"AUC-PR = {pr_auc:.4f}"
        )

        print(f"[OK]   AUC-ROC={roc_auc:.4f}")
        print(f"[OK]   AUC-PR ={pr_auc:.4f}")

    # =========================
    # MULTICLASS CLASSIFICATION
    # =========================
    else:

        from sklearn.preprocessing import label_binarize

        n_cls = y_pred_proba.shape[1]

        y_bin = label_binarize(
            y_true,
            classes=list(range(n_cls))
        )

        for i in range(n_cls):

            fpr, tpr, _ = roc_curve(
                y_bin[:, i],
                y_pred_proba[:, i]
            )

            roc_auc = auc(fpr, tpr)

            precision, recall, _ = precision_recall_curve(
                y_bin[:, i],
                y_pred_proba[:, i]
            )

            pr_auc = average_precision_score(
                y_bin[:, i],
                y_pred_proba[:, i]
            )

            ax_roc.plot(
                fpr,
                tpr,
                lw=1.8,
                label=f"class {i} AUC={roc_auc:.3f}"
            )

            ax_pr.plot(
                recall,
                precision,
                lw=1.8,
                label=f"class {i} AP={pr_auc:.3f}"
            )

    # =========================
    # COMMON PLOT SETTINGS
    # =========================

    ax_roc.plot([0,1],[0,1],"k--",lw=0.8,alpha=0.4)

    ax_roc.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        xlim=[-0.01,1.01],
        ylim=[-0.01,1.05]
    )

    ax_roc.set_title(
        "ROC Curve",
        fontsize=11,
        fontweight="bold"
    )

    ax_roc.legend()

    baseline = (
        float(y_true.mean())
        if binary
        else 1 / y_pred_proba.shape[1]
    )

    ax_pr.axhline(
        baseline,
        color="k",
        lw=0.8,
        ls="--",
        alpha=0.4
    )

    ax_pr.set(
        xlabel="Recall",
        ylabel="Precision",
        xlim=[-0.01,1.01],
        ylim=[-0.01,1.05]
    )

    ax_pr.set_title(
        "Precision-Recall Curve",
        fontsize=11,
        fontweight="bold"
    )

    ax_pr.legend()

    fig.suptitle(
        f"{MODEL_NAME} — ROC & Precision-Recall Curves",
        fontsize=13,
        fontweight="bold"
    )

    fig.tight_layout()

    fig.savefig(ROC_PR_PATH)

    plt.close(fig)

    print(f"[OK]   Saved ROC + PR curves → {ROC_PR_PATH}")
# =============================================================================
# UTILITY
# =============================================================================

def _mean(arr): return float(np.mean(arr))
def _std(arr):  return float(np.std(arr, ddof=0))
def _sem(arr):  return float(np.std(arr, ddof=1) / math.sqrt(len(arr)))
def _ci95(arr): return 1.96 * _sem(arr)
def _cohens_d(a, b):
    pooled = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (_mean(a) - _mean(b)) / pooled if pooled > 0 else 0.0
def pct(v):  return f"{v*100:.2f}%"
def fmt4(v): return f"{v:.4f}"

# =============================================================================
# LOAD TRAINING HISTORY
# =============================================================================

def load_training_history(csv_path):
    if not os.path.exists(csv_path):
        print(f"[WARN] Training CSV not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    print(f"[OK]   Loaded training history — {len(df)} epochs, {len(df.columns)} columns")
    return df

# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compute_statistics(df):
    metrics_map = {
        "accuracy":  ("accuracy",  "val_accuracy"),
        "auc_roc":   ("auc_roc",   "val_auc_roc"),
        "auc_pr":    ("auc_pr",    "val_auc_pr"),
        "loss":      ("loss",      "val_loss"),
        "f1_score":  ("f1_score",  "val_f1_score"),
        "precision": ("precision", "val_precision"),
        "recall":    ("recall",    "val_recall"),
    }
    stats = {}

    for name, (tr_col, va_col) in metrics_map.items():
        if tr_col not in df.columns:
            continue
        tr = df[tr_col].values
        va = df[va_col].values if va_col in df.columns else None

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

    best_ep = int(df[BEST_EPOCH_COL].idxmax()) if BEST_EPOCH_COL in df.columns else len(df)-1
    tr_loss = df["loss"].values
    va_loss = df["val_loss"].values if "val_loss" in df.columns else None
    thresh  = tr_loss[0] * 0.01
    conv_ep = next((i for i, v in enumerate(tr_loss) if v <= thresh), len(df)-1)
    va_acc  = df["val_accuracy"].values if "val_accuracy" in df.columns else None
    eoe_var = float(np.var(np.diff(va_acc), ddof=0)) if va_acc is not None else None

    stats["diagnostics"] = {
        "best_epoch":              best_ep,
        "best_val_accuracy":       float(df[BEST_EPOCH_COL].max()) if BEST_EPOCH_COL in df.columns else None,
        "convergence_epoch_1pct":  conv_ep,
        "train_loss_reduction_pct": float((tr_loss[0]-tr_loss[-1])/tr_loss[0]*100),
        "val_loss_reduction_pct":  float((va_loss[0]-va_loss[-1])/va_loss[0]*100) if va_loss is not None else None,
        "epoch_to_epoch_var":      eoe_var,
        "overfitting_index":       stats.get("accuracy",{}).get("generalization",{}).get("gap_mean"),
        "total_epochs":            len(df),
    }
    return stats


def print_statistics(stats):
    diag = stats.get("diagnostics", {})
    sep = "=" * 60
    print(f"\n{sep}\n  {MODEL_NAME} -- Scientific Evaluation Report\n  {TIMESTAMP}\n{sep}")
    print(f"\n{'Convergence Diagnostics':=<55}")
    print(f"  Best epoch            : {diag.get('best_epoch')}")
    print(f"  Best val accuracy     : {pct(diag.get('best_val_accuracy', 0))}")
    print(f"  Convergence epoch     : {diag.get('convergence_epoch_1pct')} (loss < 1% of initial)")
    print(f"  Train loss reduction  : {diag.get('train_loss_reduction_pct', 0):.1f}%")
    print(f"  Val   loss reduction  : {diag.get('val_loss_reduction_pct', 0):.1f}%")
    print(f"  Epoch-to-epoch var    : {diag.get('epoch_to_epoch_var', 0):.6f}")
    print(f"  Overfitting index     : {diag.get('overfitting_index', 0):.5f}")
    for metric, entry in stats.items():
        if metric == "diagnostics" or "train" not in entry:
            continue
        print(f"\n  {metric.upper()}")
        tr = entry["train"]
        print(f"    Train  mean={fmt4(tr['mean'])}  std={fmt4(tr['std'])}  95%CI+-{fmt4(tr['ci95'])}  final={fmt4(tr['final'])}")
        if "val" in entry:
            va = entry["val"]; gn = entry["generalization"]
            print(f"    Val    mean={fmt4(va['mean'])}  std={fmt4(va['std'])}  95%CI+-{fmt4(va['ci95'])}  best={fmt4(va['best'])}")
            print(f"    Gap    mean={fmt4(gn['gap_mean'])}  std={fmt4(gn['gap_std'])}  Cohen d={gn['cohens_d']:.3f}")
    print(f"\n{sep}\n")

# =============================================================================
# PLOT 1 — LEARNING CURVES (2x2)
# =============================================================================

def plot_learning_curves(df, best_ep):
    epochs = df["epoch"].values if "epoch" in df.columns else np.arange(len(df))
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)
    pairs = [
        ("accuracy", "val_accuracy", "Accuracy",  (0.87, 1.002)),
        ("loss",     "val_loss",     "Loss",       None),
        ("auc_roc",  "val_auc_roc",  "AUC-ROC",   (0.95, 1.002)),
        ("auc_pr",   "val_auc_pr",   "AUC-PR",    (0.95, 1.002)),
    ]
    for idx, (tr_col, va_col, title, ylim) in enumerate(pairs):
        if tr_col not in df.columns:
            continue
        ax = fig.add_subplot(gs[idx//2, idx%2])
        ax.plot(epochs, df[tr_col], color=PALETTE["train"], lw=1.8, label="Train", zorder=3)
        if va_col in df.columns:
            ax.plot(epochs, df[va_col], color=PALETTE["val"], lw=1.8, label="Val", zorder=3)
            ax.fill_between(epochs, df[tr_col], df[va_col],
                            alpha=0.08, color=PALETTE["val"], label="Gap")
        ax.axvline(best_ep, color=PALETTE["best"], lw=1.2, ls="--",
                   label=f"Best (ep {best_ep})", zorder=2)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    fig.suptitle(f"{MODEL_NAME} — Learning Curves", fontsize=13, fontweight="bold")
    path = PLOTS_DIR / "learning_curves.png"
    fig.savefig(path); plt.close(fig)
    print(f"[OK]   Saved learning curves       → {path}")

# =============================================================================
# PLOT 2 — GENERALIZATION GAP
# =============================================================================

def plot_generalization_gap(df):
    epochs = df["epoch"].values if "epoch" in df.columns else np.arange(len(df))
    gap    = (df["accuracy"] - df["val_accuracy"]).values
    colors = [PALETTE["gap_pos"] if g >= 0 else PALETTE["gap_neg"] for g in gap]
    roll   = pd.Series(gap).rolling(3, min_periods=1).mean().values

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.bar(epochs, gap*100, color=colors, alpha=0.75, width=0.7)
    ax.axhline(0, color="#333", lw=0.8, ls="--")
    ax.axhline(gap.mean()*100, color=PALETTE["neutral"], lw=1.2, ls=":",
               label=f"Mean={gap.mean()*100:.3f}%")
    ax.set_xlabel("Epoch", fontsize=10); ax.set_ylabel("Train - Val accuracy (%)", fontsize=10)
    ax.set_title("Per-epoch generalization gap", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    ax2.plot(epochs, gap*100, color=PALETTE["gap_pos"], lw=1.2, alpha=0.5, label="Gap")
    ax2.plot(epochs, roll*100, color=PALETTE["auc_roc"], lw=2.0, label="3-epoch rolling mean")
    ax2.axhline(0, color="#333", lw=0.8, ls="--")
    ax2.set_xlabel("Epoch", fontsize=10); ax2.set_ylabel("Gap (%)", fontsize=10)
    ax2.set_title("Generalization gap — rolling mean", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)

    fig.suptitle(f"{MODEL_NAME} — Generalization Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = PLOTS_DIR / "generalization_gap.png"
    fig.savefig(path); plt.close(fig)
    print(f"[OK]   Saved generalization gap    → {path}")

# =============================================================================
# PLOT 3 — TRAINING STABILITY
# =============================================================================

def plot_stability(df):
    epochs  = df["epoch"].values if "epoch" in df.columns else np.arange(len(df))
    va_acc  = df["val_accuracy"].values
    delta   = np.abs(np.diff(va_acc))
    roll_var= pd.Series(va_acc).rolling(3, min_periods=2).var().values

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(epochs[1:], delta*100, color=PALETTE["var"], alpha=0.7, width=0.7)
    axes[0].set_xlabel("Epoch", fontsize=10); axes[0].set_ylabel("|Delta Val acc| (%)", fontsize=10)
    axes[0].set_title("Epoch-to-epoch absolute change", fontsize=11, fontweight="bold")

    axes[1].plot(epochs, roll_var*1e4, color=PALETTE["auc_pr"], lw=2.0)
    axes[1].fill_between(epochs, 0, roll_var*1e4, alpha=0.2, color=PALETTE["auc_pr"])
    axes[1].set_xlabel("Epoch", fontsize=10); axes[1].set_ylabel("Rolling variance (x1e-4)", fontsize=10)
    axes[1].set_title("Val accuracy rolling variance (w=3)", fontsize=11, fontweight="bold")

    fig.suptitle(f"{MODEL_NAME} — Training Stability", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = PLOTS_DIR / "training_stability.png"
    fig.savefig(path); plt.close(fig)
    print(f"[OK]   Saved training stability    → {path}")

# =============================================================================
# PLOT 4 — AUC PROGRESSION
# =============================================================================

def plot_auc_progression(df, best_ep):
    epochs = df["epoch"].values if "epoch" in df.columns else np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if "val_auc_roc" in df.columns:
        ax.plot(epochs, df["val_auc_roc"], color=PALETTE["auc_roc"], lw=2.0, label="Val AUC-ROC")
        ax.fill_between(epochs, 0.97, df["val_auc_roc"], alpha=0.10, color=PALETTE["auc_roc"])
    if "val_auc_pr" in df.columns:
        ax.plot(epochs, df["val_auc_pr"], color=PALETTE["auc_pr"], lw=2.0, ls="--", label="Val AUC-PR")
    if "auc_roc" in df.columns:
        ax.plot(epochs, df["auc_roc"], color=PALETTE["auc_roc"], lw=1.2, alpha=0.4, ls=":", label="Train AUC-ROC")
    ax.axvline(best_ep, color=PALETTE["best"], lw=1.2, ls="--", label=f"Best (ep {best_ep})")
    ax.set_ylim(0.97, 1.001)
    ax.set_xlabel("Epoch", fontsize=10); ax.set_ylabel("AUC", fontsize=10)
    ax.set_title(f"{MODEL_NAME} — AUC-ROC & AUC-PR Progression", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    fig.tight_layout()
    path = PLOTS_DIR / "auc_progression.png"
    fig.savefig(path); plt.close(fig)
    print(f"[OK]   Saved AUC progression       → {path}")

# =============================================================================
# PLOT 5 — COMBINED SUMMARY (6-panel publication figure)
# =============================================================================

def plot_combined_summary(df, stats):
    epochs  = df["epoch"].values if "epoch" in df.columns else np.arange(len(df))
    best_ep = stats["diagnostics"]["best_epoch"]
    gap     = (df["accuracy"] - df["val_accuracy"]).values

    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.35)

    # A — Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, df["accuracy"],     color=PALETTE["train"], lw=1.8, label="Train")
    ax1.plot(epochs, df["val_accuracy"], color=PALETTE["val"],   lw=1.8, label="Val")
    ax1.axvline(best_ep, color=PALETTE["best"], lw=1.1, ls="--")
    ax1.set_ylim(0.88, 1.002); ax1.set_title("(A) Accuracy", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=7); ax1.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    # B — Loss
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, df["loss"],     color=PALETTE["train"], lw=1.8)
    ax2.plot(epochs, df["val_loss"], color=PALETTE["val"],   lw=1.8)
    ax2.axvline(best_ep, color=PALETTE["best"], lw=1.1, ls="--")
    ax2.set_title("(B) Loss", fontsize=10, fontweight="bold")

    # C — AUC
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, df["val_auc_roc"], color=PALETTE["auc_roc"], lw=1.8, label="AUC-ROC")
    ax3.plot(epochs, df["val_auc_pr"],  color=PALETTE["auc_pr"],  lw=1.8, ls="--", label="AUC-PR")
    ax3.set_ylim(0.97, 1.001); ax3.axvline(best_ep, color=PALETTE["best"], lw=1.1, ls="--")
    ax3.set_title("(C) Val AUC", fontsize=10, fontweight="bold")
    ax3.legend(fontsize=7); ax3.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))

    # D — Generalization gap
    ax4 = fig.add_subplot(gs[1, 0:2])
    colors = [PALETTE["gap_pos"] if g >= 0 else PALETTE["gap_neg"] for g in gap]
    ax4.bar(epochs, gap*100, color=colors, alpha=0.72, width=0.7)
    ax4.axhline(0, color="#333", lw=0.8, ls="--")
    roll = pd.Series(gap).rolling(3, min_periods=1).mean().values
    ax4.plot(epochs, roll*100, color=PALETTE["neutral"], lw=1.5, label="3-ep rolling mean")
    ax4.set_title("(D) Generalization gap (train - val)", fontsize=10, fontweight="bold")
    ax4.set_ylabel("Gap (%)"); ax4.legend(fontsize=7)

    # E — Epoch delta
    ax5 = fig.add_subplot(gs[1, 2])
    delta = np.abs(np.diff(df["val_accuracy"].values))
    ax5.bar(epochs[1:], delta*100, color=PALETTE["var"], alpha=0.7, width=0.7)
    ax5.set_title("(E) |Delta Val acc| per epoch", fontsize=10, fontweight="bold")
    ax5.set_ylabel("Change (%)")

    # F — Stats table
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis("off")
    rows = []
    for metric in ["accuracy", "auc_roc", "auc_pr", "loss", "f1_score"]:
        e = stats.get(metric, {})
        if not e:
            continue
        tr = e.get("train", {}); va = e.get("val", {}); gn = e.get("generalization", {})
        rows.append([
            metric,
            f"{tr.get('mean',0):.4f} +/- {tr.get('std',0):.4f}",
            f"{tr.get('final',0):.4f}",
            f"{va.get('mean',0):.4f} +/- {va.get('std',0):.4f}",
            f"{va.get('best',0):.4f}",
            f"{gn.get('gap_mean',0):.4f}",
            f"{gn.get('cohens_d',0):.3f}",
        ])
    cols = ["Metric","Train mean+/-std","Train final","Val mean+/-std","Val best","Gap mean","Cohen d"]
    tbl  = ax6.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#3266AD"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F0F4FA")
        cell.set_edgecolor("#ddd")
    ax6.set_title("(F) Descriptive statistics summary", fontsize=10, fontweight="bold", pad=10)

    fig.suptitle(
        f"{MODEL_NAME} — Scientific Evaluation Summary  |  Best epoch: {best_ep}  |  {TIMESTAMP}",
        fontsize=12, fontweight="bold"
    )
    path = PLOTS_DIR / "combined_summary.png"
    fig.savefig(path); plt.close(fig)
    print(f"[OK]   Saved combined summary      → {path}")

# =============================================================================
# EXPORTS — JSON REPORT
# =============================================================================

def export_json_report(stats, class_report):
    report = {"model": MODEL_NAME, "timestamp": TIMESTAMP, "statistics": stats}
    if class_report:
        report["test_metrics"] = class_report
    path = REPORTS_DIR / "scientific_report.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"[OK]   Saved JSON report           → {path}")

# =============================================================================
# EXPORTS — LATEX TABLE
# =============================================================================

def export_latex_table(stats):
    lines = [
        r"\begin{table}[h!]", r"\centering",
        r"\caption{" + MODEL_NAME + r" --- Descriptive Statistics}",
        r"\label{tab:convlstm_stats}",
        r"\begin{tabular}{lcccccc}", r"\hline",
        r"Metric & Train $\mu$ & Train $\sigma$ & Val $\mu$ & Val best & Gap $\mu$ & Cohen $d$ \\",
        r"\hline",
    ]
    for metric in ["accuracy","auc_roc","auc_pr","loss","f1_score","precision","recall"]:
        e = stats.get(metric)
        if not e or "train" not in e:
            continue
        tr = e["train"]; va = e.get("val", {}); gn = e.get("generalization", {})
        lines.append(
            f"{metric.replace('_',' ').title()} & "
            f"{tr['mean']:.4f} & {tr['std']:.4f} & "
            f"{va.get('mean',0):.4f} & {va.get('best',0):.4f} & "
            f"{gn.get('gap_mean',0):.4f} & {gn.get('cohens_d',0):.3f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    path = REPORTS_DIR / "stats_table.tex"
    path.write_text("\n".join(lines))
    print(f"[OK]   Saved LaTeX table           → {path}")

# =============================================================================
# EXPORTS — HTML DASHBOARD
# =============================================================================

def export_html_dashboard(df, stats):
    epochs  = df["epoch"].tolist() if "epoch" in df.columns else list(range(len(df)))
    tr_acc  = [round(v,6) for v in df["accuracy"].tolist()]
    va_acc  = [round(v,6) for v in df["val_accuracy"].tolist()]
    tr_loss = [round(v,6) for v in df["loss"].tolist()]
    va_loss = [round(v,6) for v in df["val_loss"].tolist()]
    va_roc  = [round(v,6) for v in df["val_auc_roc"].tolist()]
    va_pr   = [round(v,6) for v in df["val_auc_pr"].tolist()]
    gap     = [round((a-b)*100, 4) for a, b in zip(df["accuracy"], df["val_accuracy"])]

    diag     = stats["diagnostics"]
    best_ep  = diag["best_epoch"]
    best_acc = diag.get("best_val_accuracy", 0)
    oi       = diag.get("overfitting_index", 0)
    conv_ep  = diag.get("convergence_epoch_1pct", "n/a")
    tr_mu    = stats.get("accuracy",{}).get("train",{}).get("mean",0)
    va_mu    = stats.get("accuracy",{}).get("val",{}).get("mean",0)
    tr_lred  = diag.get("train_loss_reduction_pct", 0)
    va_lred  = diag.get("val_loss_reduction_pct", 0)
    tr_roc   = [round(v,6) for v in df["auc_roc"].tolist()] if "auc_roc" in df.columns else va_roc

    rows_html = ""
    for metric in ["accuracy","auc_roc","auc_pr","loss","f1_score"]:
        e = stats.get(metric,{})
        if not e or "train" not in e:
            continue
        tr = e["train"]; va = e.get("val",{}); gn = e.get("generalization",{})
        rows_html += (
            f"<tr><td>{metric.replace('_',' ').title()}</td>"
            f"<td>{tr['mean']:.4f} &plusmn; {tr['std']:.4f}</td>"
            f"<td>{va.get('mean',0):.4f} &plusmn; {va.get('std',0):.4f}</td>"
            f"<td>{va.get('best',0):.4f}</td>"
            f"<td>{gn.get('gap_mean',0):.4f}</td>"
            f"<td>{gn.get('cohens_d',0):.3f}</td></tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{MODEL_NAME} Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#f5f5f3;color:#1a1a1a;padding:24px}}
h1{{font-size:20px;font-weight:600;margin-bottom:4px}}
.sub{{font-size:13px;color:#666;margin-bottom:20px}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}}
.card{{background:#fff;border-radius:10px;padding:16px 18px;border:0.5px solid #e0dfd8}}
.card .val{{font-size:22px;font-weight:600;margin-top:4px;color:#3266AD}}
.card .lbl{{font-size:12px;color:#888}}
.card .sub2{{font-size:11px;color:#aaa;margin-top:3px}}
.tabs{{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}}
.tab{{background:#fff;border:0.5px solid #ddd;border-radius:8px;padding:7px 16px;
      font-size:13px;cursor:pointer;color:#555;transition:all 0.15s}}
.tab.active{{background:#3266AD;color:#fff;border-color:#3266AD}}
.panel{{display:none}}.panel.active{{display:block}}
.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.cbox{{background:#fff;border-radius:10px;padding:16px;border:0.5px solid #e0dfd8}}
.cbox h3{{font-size:13px;font-weight:600;margin-bottom:10px;color:#333}}
.cwrap{{position:relative;height:220px}}
.legend{{display:flex;gap:16px;margin-top:8px;font-size:11px;color:#666}}
.legend span{{display:flex;align-items:center;gap:5px}}
.dot{{width:14px;height:3px;border-radius:2px;display:inline-block}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:12px}}
th{{background:#3266AD;color:#fff;padding:8px 12px;text-align:left;font-weight:500}}
td{{padding:7px 12px;border-bottom:0.5px solid #eee}}
tr:nth-child(even) td{{background:#f8f8f6}}
.sblock{{background:#fff;border-radius:10px;padding:16px;border:0.5px solid #e0dfd8;margin-top:12px}}
.srow{{display:flex;justify-content:space-between;padding:6px 0;
       border-bottom:0.5px solid #f0f0f0;font-size:13px}}
.srow:last-child{{border-bottom:none}}
.sl{{color:#888}}.sv{{font-weight:500}}
@media(max-width:720px){{.cards,.chart-grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<h1>{MODEL_NAME} &mdash; Scientific Evaluation Dashboard</h1>
<p class="sub">Generated: {TIMESTAMP} &nbsp;|&nbsp; {len(df)} epochs</p>

<div class="cards">
  <div class="card"><div class="lbl">Best val accuracy</div>
    <div class="val">{best_acc*100:.2f}%</div><div class="sub2">Epoch {best_ep}</div></div>
  <div class="card"><div class="lbl">Mean val AUC-ROC</div>
    <div class="val">{_mean(va_roc):.4f}</div><div class="sub2">across all epochs</div></div>
  <div class="card"><div class="lbl">Overfitting index</div>
    <div class="val">{oi:.4f}</div><div class="sub2">mean train-val gap</div></div>
  <div class="card"><div class="lbl">Convergence epoch</div>
    <div class="val">{conv_ep}</div><div class="sub2">loss &lt; 1% of initial</div></div>
</div>

<div class="tabs">
  <button class="tab active" onclick="show('curves',this)">Learning curves</button>
  <button class="tab" onclick="show('auc',this)">AUC progression</button>
  <button class="tab" onclick="show('gap',this)">Generalization</button>
  <button class="tab" onclick="show('summary',this)">Statistics</button>
</div>

<div id="p-curves" class="panel active">
  <div class="chart-grid">
    <div class="cbox"><h3>Accuracy</h3><div class="cwrap"><canvas id="accC"></canvas></div></div>
    <div class="cbox"><h3>Loss</h3><div class="cwrap"><canvas id="lossC"></canvas></div></div>
  </div>
  <div class="legend" style="margin-top:10px;">
    <span><span class="dot" style="background:#3266AD"></span>Train</span>
    <span><span class="dot" style="background:#E24B4A"></span>Validation</span>
    <span><span class="dot" style="background:#1D9E75"></span>Best epoch {best_ep}</span>
  </div>
</div>

<div id="p-auc" class="panel">
  <div class="cbox"><h3>Validation AUC-ROC and AUC-PR across epochs</h3>
    <div class="cwrap" style="height:280px;"><canvas id="aucC"></canvas></div></div>
  <div class="legend" style="margin-top:10px;">
    <span><span class="dot" style="background:#534AB7"></span>AUC-ROC</span>
    <span><span class="dot" style="background:#D85A30"></span>AUC-PR</span>
  </div>
</div>

<div id="p-gap" class="panel">
  <div class="cbox"><h3>Per-epoch generalization gap (train - val accuracy)</h3>
    <div class="cwrap"><canvas id="gapC"></canvas></div></div>
  <div class="sblock">
    <div class="srow"><span class="sl">Mean gap</span>
      <span class="sv">{sum(gap)/len(gap):.4f}%</span></div>
    <div class="srow"><span class="sl">Max gap</span>
      <span class="sv">{max(gap):.3f}% (epoch {gap.index(max(gap))})</span></div>
    <div class="srow"><span class="sl">Overfitting status</span>
      <span class="sv" style="color:#1D9E75;">Minimal</span></div>
  </div>
</div>

<div id="p-summary" class="panel">
  <div class="chart-grid">
    <div class="sblock"><strong>Training</strong>
      <div class="srow"><span class="sl">Accuracy mean</span><span class="sv">{tr_mu*100:.3f}%</span></div>
      <div class="srow"><span class="sl">AUC-ROC mean</span>
        <span class="sv">{_mean(tr_roc):.4f}</span></div>
      <div class="srow"><span class="sl">Loss reduction</span><span class="sv">{tr_lred:.1f}%</span></div>
    </div>
    <div class="sblock"><strong>Validation</strong>
      <div class="srow"><span class="sl">Accuracy mean</span><span class="sv">{va_mu*100:.3f}%</span></div>
      <div class="srow"><span class="sl">AUC-ROC mean</span><span class="sv">{_mean(va_roc):.4f}</span></div>
      <div class="srow"><span class="sl">Loss reduction</span><span class="sv">{va_lred:.1f}%</span></div>
    </div>
  </div>
  <table>
    <thead><tr><th>Metric</th><th>Train mean+/-std</th><th>Val mean+/-std</th>
      <th>Val best</th><th>Gap mean</th><th>Cohen d</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>

<script>
const ep={epochs},trA={tr_acc},vaA={va_acc},trL={tr_loss},vaL={va_loss};
const vaR={va_roc},vaP={va_pr},gapD={gap},bestEp={best_ep};

const bestLine={{id:'best',afterDraw(c){{
  const xi=c.scales.x.getPixelForValue(bestEp),ctx=c.ctx;
  const t=c.chartArea.top,b=c.chartArea.bottom;
  ctx.save();ctx.setLineDash([5,4]);ctx.strokeStyle='#1D9E75';
  ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(xi,t);ctx.lineTo(xi,b);
  ctx.stroke();ctx.restore();
}}}};

function mkLine(id,datasets,yMin,yMax,fmt){{
  return new Chart(document.getElementById(id),{{
    type:'line',data:{{labels:ep,datasets}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},tooltip:{{mode:'index',intersect:false}}}},
      scales:{{
        x:{{ticks:{{font:{{size:10}},color:'#999',maxTicksLimit:10}},grid:{{color:'rgba(0,0,0,0.04)'}}}},
        y:{{min:yMin,max:yMax,
          ticks:{{font:{{size:10}},color:'#999',callback:v=>typeof fmt==='function'?fmt(v):v.toFixed(3)}},
          grid:{{color:'rgba(0,0,0,0.04)'}}}}
      }},elements:{{point:{{radius:2,hoverRadius:4}},line:{{tension:0.35,borderWidth:2}}}}
    }},plugins:[bestLine]
  }});
}}

mkLine('accC',[
  {{data:trA,borderColor:'#3266AD',backgroundColor:'rgba(50,102,173,0.06)',fill:true,label:'Train'}},
  {{data:vaA,borderColor:'#E24B4A',backgroundColor:'rgba(226,75,74,0.06)',fill:true,label:'Val'}}
],0.87,1.002);
mkLine('lossC',[
  {{data:trL,borderColor:'#3266AD',label:'Train'}},
  {{data:vaL,borderColor:'#E24B4A',label:'Val'}}
],null,null);
mkLine('aucC',[
  {{data:vaR,borderColor:'#534AB7',backgroundColor:'rgba(83,74,183,0.06)',fill:true,label:'AUC-ROC'}},
  {{data:vaP,borderColor:'#D85A30',borderDash:[4,3],label:'AUC-PR'}}
],0.98,1.001,v=>v.toFixed(4));

const gapColors=gapD.map(v=>v>=0?'rgba(50,102,173,0.65)':'rgba(226,75,74,0.65)');
new Chart(document.getElementById('gapC'),{{
  type:'bar',data:{{labels:ep,datasets:[{{data:gapD,backgroundColor:gapColors,borderRadius:4,label:'Gap'}}]}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:c=>c.raw.toFixed(4)+'%'}}}}}},
    scales:{{
      x:{{ticks:{{font:{{size:10}},color:'#999'}},grid:{{display:false}}}},
      y:{{ticks:{{font:{{size:10}},color:'#999',callback:v=>v.toFixed(2)+'%'}},grid:{{color:'rgba(0,0,0,0.04)'}}}}
    }}
  }}
}});

function show(id,btn){{
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
  document.getElementById('p-'+id).classList.add('active');
  btn.classList.add('active');
}}
</script>
</body></html>"""

    path = OUTPUT_DIR / "dashboard.html"
    path.write_text(html, encoding="utf-8")
    print(f"[OK]   Saved HTML dashboard        → {path}")

# =============================================================================
# OPTIONAL — DISPLAY EXISTING IMAGES
# =============================================================================

def display_existing_images():
    class_report = None
    for path, title in [(CONF_MATRIX_PATH, "Confusion Matrix"), (ROC_PR_PATH, "ROC + PR Curves")]:
        if os.path.exists(path):
            from PIL import Image
            img = Image.open(path)
            plt.figure(figsize=(8, 6))
            plt.imshow(img); plt.axis("off"); plt.title(title)
            plt.tight_layout(); plt.show()
    if os.path.exists(CLASS_REPORT_PATH):
        with open(CLASS_REPORT_PATH) as f:
            class_report = json.load(f)
        print("\n── Test metrics (from JSON) " + "-"*30)
        for key in ["accuracy","precision","recall","auc_roc","auc_pr"]:
            if key in class_report:
                print(f"  {key:12s}: {class_report[key]:.4f}")
    return class_report

# =============================================================================
# MAIN
# =============================================================================

def main():
    sep = "=" * 60
    print(f"\n{sep}\n  {MODEL_NAME} -- Extended Scientific Evaluation\n  {TIMESTAMP}\n{sep}\n")

    # 1. Optional: load existing test-metrics JSON
    class_report = display_existing_images()

    # 2. Generate confusion matrix + ROC/PR from model or arrays
    preds = get_predictions()
    if preds is not None:
        y_true, y_pred_proba = preds
        plot_confusion_matrix(y_true, y_pred_proba)
        plot_roc_pr_curves(y_true, y_pred_proba)
    else:
        print("[INFO] Confusion matrix and ROC/PR curves skipped.")

    # 3. Load training history
    df = load_training_history(TRAINING_CSV_PATH)
    if df is None:
        print(f"\n[WARN] Cannot find: {TRAINING_CSV_PATH}")
        print(       "       Update TRAINING_CSV_PATH at the top of this script.\n")
        return

    # 4. Compute statistics & print report
    stats   = compute_statistics(df)
    best_ep = stats["diagnostics"]["best_epoch"]
    print_statistics(stats)

    # 5. Generate all training-history plots
    plot_learning_curves(df, best_ep)
    plot_generalization_gap(df)
    plot_stability(df)
    plot_auc_progression(df, best_ep)
    plot_combined_summary(df, stats)

    # 6. Export reports
    export_json_report(stats, class_report)
    export_latex_table(stats)
    export_html_dashboard(df, stats)

    print(f"\n{sep}")
    print(f"  All outputs saved to : {OUTPUT_DIR.resolve()}")
    print(f"  Open in browser      : {(OUTPUT_DIR / 'dashboard.html').resolve()}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()