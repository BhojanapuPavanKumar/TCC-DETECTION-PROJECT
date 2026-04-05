import os
import numpy as np
from scipy.ndimage import gaussian_filter

# =========================================================
# CONFIG
# =========================================================

INPUT_DIR = "data/raw_npz"
IMG_DIR = "data/cnn_dataset/images"
MASK_DIR = "data/cnn_dataset/masks"

PATCH_SIZE = 128
STRIDE = 128

MIN_TEMP = 150
MAX_TEMP = 330
SMOOTH_SIGMA = 1.0
EDGE_MARGIN = 20
CLOUD_THRESHOLD = 235
MIN_CLOUD_PIXELS = 20

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)


# =========================================================
# LOAD NPZ
# =========================================================

def load_npz(file_path):
    try:
        data = np.load(file_path, mmap_mode="r")
        return data["TIR1_TEMP"]
    except Exception as e:
        print("Error loading:", file_path, e)
        return None


# =========================================================
# PREPROCESS IMAGE
# =========================================================

def preprocess_image(tb):

    tb = tb.astype(np.float32)

    # 1️⃣ Remove unrealistic values
    invalid_mask = (tb < MIN_TEMP) | (tb > MAX_TEMP)
    tb[invalid_mask] = np.nan

    # 2️⃣ Fill NaNs
    if np.isnan(tb).any():
        mean_val = np.nanmean(tb)
        tb = np.nan_to_num(tb, nan=mean_val)

    # 3️⃣ Smooth (IMPORTANT for satellite noise)
    tb = gaussian_filter(tb, sigma=SMOOTH_SIGMA)

    # 4️⃣ Remove edge artifacts
    tb[:EDGE_MARGIN, :] = np.nan
    tb[-EDGE_MARGIN:, :] = np.nan
    tb[:, :EDGE_MARGIN] = np.nan
    tb[:, -EDGE_MARGIN:] = np.nan

    # Replace NaNs again after edge removal
    tb = np.nan_to_num(tb, nan=np.nanmean(tb))

    return tb


# =========================================================
# NORMALIZE
# =========================================================

def normalize_image(tb):
    tb = np.clip(tb, 180, 320)
    tb = (tb - 180) / (320 - 180)
    return tb.astype(np.float32)


# =========================================================
# CREATE MASK
# =========================================================

def create_mask(tb_raw):
    return (tb_raw < CLOUD_THRESHOLD).astype(np.float32)


# =========================================================
# PATCH EXTRACTION
# =========================================================

def extract_patches(image, mask):

    h, w = image.shape
    patches = []

    for i in range(0, h - PATCH_SIZE + 1, STRIDE):
        for j in range(0, w - PATCH_SIZE + 1, STRIDE):

            img_patch = image[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            mask_patch = mask[i:i+PATCH_SIZE, j:j+PATCH_SIZE]

            # Skip low cloud patches
            if np.sum(mask_patch) < MIN_CLOUD_PIXELS:
                continue

            patches.append((img_patch, mask_patch))

    return patches


# =========================================================
# MAIN DATASET CREATION
# =========================================================

def create_dataset():

    files = sorted(os.listdir(INPUT_DIR))

    created = 0
    skipped = 0
    corrupted = 0

    for f in files:

        if not f.endswith(".npz"):
            continue

        path = os.path.join(INPUT_DIR, f)

        # ----------------------------
        # LOAD
        # ----------------------------
        tb_raw = load_npz(path)

        if tb_raw is None:
            corrupted += 1
            continue

        # ----------------------------
        # PREPROCESS
        # ----------------------------
        tb_processed = preprocess_image(tb_raw)

        # ----------------------------
        # NORMALIZE
        # ----------------------------
        image = normalize_image(tb_processed)

        # ----------------------------
        # MASK
        # ----------------------------
        mask = create_mask(tb_raw)

        # ----------------------------
        # PATCHES
        # ----------------------------
        patches = extract_patches(image, mask)

        base_name = os.path.splitext(f)[0]

        for idx, (img_patch, mask_patch) in enumerate(patches):

            img_name = f"{base_name}_patch_{idx:04d}.npy"

            img_path = os.path.join(IMG_DIR, img_name)
            mask_path = os.path.join(MASK_DIR, img_name)

            if os.path.exists(img_path):
                skipped += 1
                continue

            img_patch = np.expand_dims(img_patch, axis=-1)
            mask_patch = np.expand_dims(mask_patch, axis=-1)

            np.save(img_path, img_patch)
            np.save(mask_path, mask_patch)

            created += 1

    print("\n✅ Patch dataset generation complete")
    print("Created patches:", created)
    print("Skipped existing:", skipped)
    print("Corrupted files:", corrupted)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    create_dataset()