# -*- coding: utf-8 -*-
"""
visualize_cnn.py
Visualizes U-Net CNN predictions vs actual cloud masks.
Shows: raw TIR1 image, predicted mask, overlay side by side.
Picks a random sample of NPZ files and runs the CNN on them.
"""

import os
import glob
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from datetime import datetime

MODEL_PATH = "models/cnn/unet_best.keras"
INPUT_DIR  = "data/raw_npz"
MASK_DIR   = "data/predicted_masks"
OUT_DIR    = "output/cnn_visualization"
os.makedirs(OUT_DIR, exist_ok=True)

PATCH_SIZE = 128
BATCH_SIZE = 128
TB_MIN     = 180
TB_MAX     = 320
N_SAMPLES  = 6    # number of full images to visualize


# ================================
# GPU
# ================================

def enable_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU enabled")


# ================================
# PATCH EXTRACTION + RECONSTRUCTION
# ================================

def extract_patches(image):
    img = tf.convert_to_tensor(image, dtype=tf.float32)
    img = tf.expand_dims(tf.expand_dims(img, 0), -1)
    patches = tf.image.extract_patches(
        images=img,
        sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
        strides=[1, PATCH_SIZE, PATCH_SIZE, 1],
        rates=[1, 1, 1, 1],
        padding="SAME"
    )
    return tf.reshape(patches, [-1, PATCH_SIZE, PATCH_SIZE, 1]).numpy()


def reconstruct(patches, h, w):
    mask = np.zeros((h, w), dtype=np.float32)
    idx  = 0
    for y in range(0, h, PATCH_SIZE):
        for x in range(0, w, PATCH_SIZE):
            patch = patches[idx]
            mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = patch[:h-y, :w-x]
            idx += 1
    return mask


# ================================
# LOAD AND PREDICT ONE FILE
# ================================

def predict_file(npz_path, model):
    data = np.load(npz_path)
    tb   = data["TIR1_TEMP"].astype(np.float32)
    h, w = tb.shape

    # Normalize
    tb_norm = np.clip(tb, TB_MIN, TB_MAX)
    tb_norm = (tb_norm - TB_MIN) / (TB_MAX - TB_MIN)

    # Predict
    patches  = extract_patches(tb_norm)
    preds    = model.predict(patches, batch_size=BATCH_SIZE, verbose=0)
    prob_map = reconstruct(preds.squeeze(axis=-1), h, w)
    bin_mask = (prob_map > 0.5).astype(np.uint8)

    return tb, tb_norm, prob_map, bin_mask


# ================================
# LOAD EXISTING MASK (if available)
# ================================

def load_existing_mask(npz_path):
    name     = os.path.basename(npz_path).replace(".npz", "")
    mask_path = os.path.join(MASK_DIR, name + "_mask.npy")
    if os.path.exists(mask_path):
        return np.load(mask_path)
    return None


# ================================
# PLOT ONE SAMPLE
# ================================

def plot_sample(tb_raw, tb_norm, prob_map, bin_mask,
                existing_mask, filename, ax_row):
    """
    Row layout:
    [0] Raw TIR1 TB    [1] Normalized    [2] Probability map
    [3] Predicted mask [4] Existing mask [5] Overlay
    """
    # Subsample for display (images are 2816x2805 — too large to render)
    step = 8
    tb_d   = tb_raw[::step, ::step]
    norm_d = tb_norm[::step, ::step]
    prob_d = prob_map[::step, ::step]
    pred_d = bin_mask[::step, ::step]
    exist_d = existing_mask[::step, ::step] if existing_mask is not None else None

    # Panel 0: Raw TB
    im0 = ax_row[0].imshow(tb_d, cmap="RdYlBu_r", vmin=TB_MIN, vmax=TB_MAX)
    ax_row[0].set_title("Raw TIR1 (K)", fontsize=8)
    ax_row[0].axis("off")
    plt.colorbar(im0, ax=ax_row[0], fraction=0.046, pad=0.04)

    # Panel 1: Normalized
    ax_row[1].imshow(norm_d, cmap="gray_r", vmin=0, vmax=1)
    ax_row[1].set_title("Normalized input", fontsize=8)
    ax_row[1].axis("off")

    # Panel 2: Probability map
    im2 = ax_row[2].imshow(prob_d, cmap="hot", vmin=0, vmax=1)
    ax_row[2].set_title("CNN probability", fontsize=8)
    ax_row[2].axis("off")
    plt.colorbar(im2, ax=ax_row[2], fraction=0.046, pad=0.04)

    # Panel 3: Predicted binary mask
    ax_row[3].imshow(pred_d, cmap="Blues", vmin=0, vmax=1)
    ax_row[3].set_title("Predicted mask", fontsize=8)
    ax_row[3].axis("off")

    # Panel 4: Existing saved mask (actual) or empty
    if exist_d is not None:
        ax_row[4].imshow(exist_d, cmap="Blues", vmin=0, vmax=1)
        ax_row[4].set_title("Saved mask (actual)", fontsize=8)

        # Compare predicted vs existing
        match = (pred_d == exist_d).mean() * 100
        ax_row[4].set_xlabel(f"Match: {match:.1f}%", fontsize=7)
    else:
        ax_row[4].imshow(np.zeros_like(pred_d), cmap="gray")
        ax_row[4].set_title("No saved mask", fontsize=8)
    ax_row[4].axis("off")

    # Panel 5: Overlay — raw TB + predicted mask contour
    ax_row[5].imshow(tb_d, cmap="RdYlBu_r", vmin=TB_MIN, vmax=TB_MAX, alpha=0.8)
    ax_row[5].contour(pred_d, levels=[0.5], colors=["white"], linewidths=0.5)
    if exist_d is not None:
        ax_row[5].contour(exist_d, levels=[0.5], colors=["yellow"],
                          linewidths=0.5, linestyles="dashed")
    ax_row[5].set_title("TB + predicted (white) vs saved (yellow)", fontsize=7)
    ax_row[5].axis("off")

    # Row label
    ts = os.path.basename(filename).replace(".npz", "")
    ax_row[0].set_ylabel(ts, fontsize=7, rotation=0,
                          labelpad=60, va="center")


# ================================
# MAIN
# ================================

def main():
    enable_gpu()

    print("Loading CNN model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Parameters : {model.count_params():,}")

    # Find all NPZ files
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.npz")))
    print(f"\nTotal NPZ files: {len(files)}")

    # Pick random sample
    random.seed(42)
    sample_files = random.sample(files, min(N_SAMPLES, len(files)))
    sample_files = sorted(sample_files)
    print(f"Visualizing : {N_SAMPLES} random samples")

    # ---- Figure setup ----
    n_cols = 6
    n_rows = len(sample_files)
    fig    = plt.figure(figsize=(n_cols * 3.5, n_rows * 3.2))
    fig.suptitle(
        f"U-Net CNN — Actual vs Predicted Cloud Masks\n"
        f"Model: {MODEL_PATH}  |  "
        f"White contour = predicted  |  Yellow dashed = saved mask",
        fontsize=10, fontweight="bold", y=1.01
    )

    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.4, wspace=0.3)

    for row, npz_path in enumerate(sample_files):
        print(f"\n[{row+1}/{n_rows}] Processing: {os.path.basename(npz_path)}")

        try:
            tb_raw, tb_norm, prob_map, bin_mask = predict_file(npz_path, model)
            existing_mask = load_existing_mask(npz_path)

            ax_row = [fig.add_subplot(gs[row, col]) for col in range(n_cols)]
            plot_sample(tb_raw, tb_norm, prob_map, bin_mask,
                        existing_mask, npz_path, ax_row)

            # Stats
            cloud_pct = bin_mask.mean() * 100
            print(f"  Image size    : {tb_raw.shape}")
            print(f"  Cloud cover   : {cloud_pct:.1f}%")
            if existing_mask is not None:
                match = (bin_mask == existing_mask).mean() * 100
                print(f"  Match (saved) : {match:.2f}%")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # ---- Save ----
    ts       = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = os.path.join(OUT_DIR, f"cnn_actual_vs_predicted_{ts}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"\nSaved: {out_path}")

    # ---- Also save individual panels for 1 sample ----
    print("\nSaving detailed single-sample comparison...")
    sample = sample_files[0]
    tb_raw, tb_norm, prob_map, bin_mask = predict_file(sample, model)
    existing_mask = load_existing_mask(sample)
    ts_name = os.path.basename(sample).replace(".npz", "")

    fig2, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig2.suptitle(
        f"Detailed CNN Analysis — {ts_name}\n"
        f"U-Net model: {MODEL_PATH}",
        fontsize=12, fontweight="bold"
    )

    step  = 4
    tb_d  = tb_raw[::step, ::step]
    prob_d = prob_map[::step, ::step]
    pred_d = bin_mask[::step, ::step]
    exist_d = existing_mask[::step, ::step] if existing_mask is not None else None

    # Row 1
    im = axes[0,0].imshow(tb_d, cmap="RdYlBu_r", vmin=TB_MIN, vmax=TB_MAX)
    axes[0,0].set_title("Raw TIR1 Brightness Temperature (K)", fontsize=10)
    plt.colorbar(im, ax=axes[0,0])
    axes[0,0].axis("off")

    axes[0,1].imshow(prob_d, cmap="hot", vmin=0, vmax=1)
    axes[0,1].set_title("CNN Cloud Probability Map", fontsize=10)
    axes[0,1].axis("off")

    axes[0,2].imshow(pred_d, cmap="Blues", vmin=0, vmax=1)
    cloud_pct = bin_mask.mean() * 100
    axes[0,2].set_title(f"Predicted Cloud Mask (cloud={cloud_pct:.1f}%)", fontsize=10)
    axes[0,2].axis("off")

    # Row 2
    if exist_d is not None:
        axes[1,0].imshow(exist_d, cmap="Blues", vmin=0, vmax=1)
        exist_pct = existing_mask.mean() * 100
        axes[1,0].set_title(f"Saved Mask (cloud={exist_pct:.1f}%)", fontsize=10)
        axes[1,0].axis("off")

        # Difference map
        diff = pred_d.astype(int) - exist_d.astype(int)
        axes[1,1].imshow(diff, cmap="RdBu", vmin=-1, vmax=1)
        fp = (diff == 1).mean() * 100
        fn = (diff == -1).mean() * 100
        axes[1,1].set_title(
            f"Difference: FP={fp:.1f}% FN={fn:.1f}%\n"
            f"Blue=False Positive  Red=False Negative",
            fontsize=9
        )
        axes[1,1].axis("off")
    else:
        axes[1,0].axis("off")
        axes[1,1].axis("off")

    # Overlay
    axes[1,2].imshow(tb_d, cmap="RdYlBu_r", vmin=TB_MIN, vmax=TB_MAX, alpha=0.75)
    axes[1,2].contour(pred_d,  levels=[0.5], colors=["white"],  linewidths=1.0)
    if exist_d is not None:
        axes[1,2].contour(exist_d, levels=[0.5], colors=["yellow"],
                          linewidths=1.0, linestyles="dashed")
        axes[1,2].set_title("Overlay: white=predicted  yellow=saved", fontsize=9)
    else:
        axes[1,2].set_title("Overlay: white=predicted contour", fontsize=9)
    axes[1,2].axis("off")

    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, f"cnn_detailed_{ts_name}.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out2}")
    print(f"\nAll outputs in: {OUT_DIR}/")


if __name__ == "__main__":
    main()