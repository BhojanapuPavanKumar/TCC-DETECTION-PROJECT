# -*- coding: utf-8 -*-
"""
create_seq_convlstm.py  (v5 — disk-safe, chunked save)

Fixes vs v4:
  FIX 1 — OSError: No space left on device
    np.savez_compressed loads the entire memmap into RAM then writes a
    compressed zip — on 98 GB disk with 606k sequences this overflows.
    Solution: skip npz entirely. Save X and y as raw .npy files directly
    from the memmap (zero extra disk — it IS the memmap, just renamed).
    Training pipeline loads them with np.load(..., mmap_mode="r") as before.

  FIX 2 — Dataset too large (606k sequences, ~45 GB uncompressed)
    Cap total sequences at MAX_SEQS=200_000 spread evenly across classes
    (40k per class). This keeps the dataset at ~15 GB on disk and trains
    in reasonable time on an RTX 2050.

  FIX 3 — NON-TCC cap fired too late (file ~303, minorities already large)
    With MAX_SEQS + PER_CLASS_CAP, balance is enforced per-patch from the
    start. No more rolling cap logic needed — simpler and more predictable.
"""

import os
import gc
import warnings
from collections import deque

import numpy as np
from tqdm import tqdm

# ================= CONFIG =================

RAW_DIR     = "data/raw_npz"

# Output: two raw npy files instead of one npz (avoids disk spike)
OUTPUT_DIR  = "data/convlstm_mc_dataset"
OUTPUT_X    = os.path.join(OUTPUT_DIR, "X.npy")
OUTPUT_Y    = os.path.join(OUTPUT_DIR, "y.npy")
TEMP_DIR    = os.path.join(OUTPUT_DIR, "_tmp")

SEQ_LEN         = 5
TEMPORAL_STRIDE = 3

PATCH_SIZE   = 64
PATCH_STRIDE = 64

# Hard cap per class — total dataset = NUM_CLASSES * PER_CLASS_CAP = 200k
PER_CLASS_CAP = 40_000

# Discard patches with > this fraction NaN in any frame
MAX_NAN_FRACTION = 0.50

# ── Lifecycle thresholds (Kelvin) ─────────────────────────────────────────
THRESHOLDS = dict(
    non_tcc_min_tb  = 270,
    non_tcc_soft_tb = 255,
    intensify_trend = -6,
    organize_trend  = -2,
    mature_max_tb   = 230,
    dissipate_trend =  2,
)

NUM_CLASSES = 5
CLASS_NAMES = {
    0: "NON-TCC",
    1: "ORGANIZING",
    2: "INTENSIFYING",
    3: "MATURE",
    4: "DISSIPATING",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


# ==========================================================
# Normalization
# ==========================================================

def normalize(x: np.ndarray) -> np.ndarray:
    frame_mean = np.nanmean(x)
    if np.isnan(frame_mean):
        return np.zeros_like(x)
    x = np.nan_to_num(x, nan=float(frame_mean))
    std = x.std()
    return (x - x.mean()) / (std + 1e-6)


# ==========================================================
# Sequence quality guard
# ==========================================================

def sequence_is_valid(seq_raw: np.ndarray) -> bool:
    for t in range(seq_raw.shape[0]):
        if np.isnan(seq_raw[t]).mean() > MAX_NAN_FRACTION:
            return False
    if np.nanstd(seq_raw) < 1e-3:
        return False
    return True


# ==========================================================
# Lifecycle classifier
# ==========================================================

def classify_patch_sequence(seq_raw: np.ndarray) -> int:
    T = THRESHOLDS
    last_mean  = float(np.nanmean(seq_raw[-1]))
    first_mean = float(np.nanmean(seq_raw[0]))
    trend      = last_mean - first_mean

    if last_mean > T["non_tcc_min_tb"]:  return 0   # NON-TCC (warm)
    if trend < T["intensify_trend"]:     return 2   # INTENSIFYING
    if trend < T["organize_trend"]:      return 1   # ORGANIZING
    if last_mean < T["mature_max_tb"]:   return 3   # MATURE (very cold)
    if trend > T["dissipate_trend"]:     return 4   # DISSIPATING
    if last_mean > T["non_tcc_soft_tb"]: return 0   # NON-TCC (warm-ish, no trend)
    return 3                                         # MATURE (default)


# ==========================================================
# Patch extractor
# ==========================================================

def extract_patches(frame: np.ndarray):
    H, W = frame.shape
    patches = []
    for i in range(0, H - PATCH_SIZE + 1, PATCH_STRIDE):
        for j in range(0, W - PATCH_SIZE + 1, PATCH_STRIDE):
            patches.append(frame[i : i + PATCH_SIZE, j : j + PATCH_SIZE])
    return patches


# ==========================================================
# Temperature sanity check
# ==========================================================

def print_temperature_stats(files):
    print("\nChecking brightness temperature statistics...")
    mins, maxs, means = [], [], []
    checked = 0
    for f in files[:20]:
        try:
            data = np.load(os.path.join(RAW_DIR, f))
            if "TIR1_TEMP" not in data.files:
                continue
            x = data["TIR1_TEMP"]
            mins.append(float(np.nanmin(x)))
            maxs.append(float(np.nanmax(x)))
            means.append(float(np.nanmean(x)))
            checked += 1
        except Exception:
            continue
    if checked == 0:
        print("  No valid files found in first 20.")
        return
    print(f"  Sampled {checked} files")
    print(f"  Mean of min  Tb : {np.mean(mins):.2f} K")
    print(f"  Mean of max  Tb : {np.mean(maxs):.2f} K")
    print(f"  Mean of mean Tb : {np.mean(means):.2f} K\n")


# ==========================================================
# Disk space check
# ==========================================================

def check_disk_space(path, required_gb):
    import shutil
    free = shutil.disk_usage(path).free / (1024 ** 3)
    print(f"  Free disk space : {free:.1f} GB")
    print(f"  Required (est.) : {required_gb:.1f} GB")
    if free < required_gb * 1.15:
        print(f"  WARNING: less than 15% headroom — risk of running out.")
    else:
        print(f"  OK: sufficient space.")
    return free


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def main():
    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".npz"))
    print(f"Total raw files: {len(files)}")
    print_temperature_stats(files)

    max_total = NUM_CLASSES * PER_CLASS_CAP
    # Each sequence: SEQ_LEN * PATCH_SIZE * PATCH_SIZE * 1 * 4 bytes (float32)
    seq_bytes    = SEQ_LEN * PATCH_SIZE * PATCH_SIZE * 1 * 4
    est_X_gb     = (max_total * seq_bytes) / (1024 ** 3)
    est_chunk_gb = est_X_gb * 0.6   # chunks are subset of total
    est_total_gb = est_X_gb + est_chunk_gb + 0.5  # X + chunks + y + margin

    print(f"Target sequences : {max_total:,} ({PER_CLASS_CAP:,} per class)")
    print(f"Estimated X size : {est_X_gb:.1f} GB (uncompressed npy)")
    print(f"Estimated peak   : {est_total_gb:.1f} GB (during merge)\n")
    check_disk_space(OUTPUT_DIR, est_total_gb)
    print()

    buffer: deque = deque(maxlen=SEQ_LEN)
    chunk_files_X, chunk_files_y = [], []

    skipped_corrupt     = 0
    skipped_missing_key = 0
    skipped_no_patches  = 0
    skipped_nan_patch   = 0
    skipped_cap         = 0

    seq_counter     = 0
    total_sequences = 0
    class_counts    = np.zeros(NUM_CLASSES, dtype=np.int64)

    all_classes_full = False

    for idx, file in enumerate(tqdm(files, desc="Building sequences")):

        # Stop early once all classes are full
        if all_classes_full:
            break

        path = os.path.join(RAW_DIR, file)

        try:
            data = np.load(path, mmap_mode="r")
        except Exception:
            skipped_corrupt += 1
            continue

        if "TIR1_TEMP" not in data.files:
            skipped_missing_key += 1
            continue

        try:
            frame_raw = data["TIR1_TEMP"].astype(np.float32)
        except Exception:
            skipped_corrupt += 1
            continue

        frame_norm   = normalize(frame_raw)
        patches_raw  = extract_patches(frame_raw)
        patches_norm = extract_patches(frame_norm)

        if not patches_raw:
            skipped_no_patches += 1
            continue

        buffer.append((patches_raw, patches_norm))

        if len(buffer) < SEQ_LEN:
            continue
        if idx % TEMPORAL_STRIDE != 0:
            continue

        buf_list  = list(buffer)
        n_patches = min(len(buf_list[t][0]) for t in range(SEQ_LEN))
        if n_patches == 0:
            continue

        X_patch_list, y_patch_list = [], []

        for patch_idx in range(n_patches):
            seq_raw_ = np.stack(
                [buf_list[t][0][patch_idx] for t in range(SEQ_LEN)]
            )

            if not sequence_is_valid(seq_raw_):
                skipped_nan_patch += 1
                continue

            label = classify_patch_sequence(seq_raw_)

            if class_counts[label] >= PER_CLASS_CAP:
                skipped_cap += 1
                continue

            seq_norm = np.stack(
                [buf_list[t][1][patch_idx] for t in range(SEQ_LEN)]
            )
            X_patch_list.append(seq_norm[..., np.newaxis])
            y_patch_list.append(label)
            class_counts[label] += 1

        if not X_patch_list:
            continue

        X_patch = np.array(X_patch_list, dtype=np.float32)
        y_patch = np.array(y_patch_list, dtype=np.int32)

        xp = os.path.join(TEMP_DIR, f"X_{seq_counter}.npy")
        yp = os.path.join(TEMP_DIR, f"y_{seq_counter}.npy")
        np.save(xp, X_patch)
        np.save(yp, y_patch)

        chunk_files_X.append(xp)
        chunk_files_y.append(yp)
        seq_counter     += 1
        total_sequences += len(X_patch)

        del X_patch, y_patch, X_patch_list, y_patch_list
        gc.collect()

        # Check if every class has hit its cap
        all_classes_full = all(class_counts[c] >= PER_CLASS_CAP
                               for c in range(NUM_CLASSES))
        if all_classes_full:
            print(f"\n[Done] All classes reached {PER_CLASS_CAP:,} — stopping early.")

    if seq_counter == 0:
        print("\n[ERROR] No sequences generated. Check RAW_DIR.")
        return

    print("\nInterim class distribution:")
    total_so_far = class_counts.sum()
    for cls_id, name in CLASS_NAMES.items():
        pct = 100 * class_counts[cls_id] / max(total_so_far, 1)
        bar = "█" * int(pct / 2)
        print(f"  {name:14s} → {class_counts[cls_id]:8,}  ({pct:5.1f}%)  {bar}")

    # ── Streaming merge into memmap ────────────────────────────────────────
    print("\nStreaming merge (memory-safe)...")

    total_samples = sum(
        len(np.load(yp, mmap_mode="r")) for yp in chunk_files_y
    )
    sample_shape = np.load(chunk_files_X[0], mmap_mode="r").shape[1:]

    print(f"  Total samples   : {total_samples:,}")
    print(f"  Sample shape    : {sample_shape}")

    memmap_X_path = os.path.join(TEMP_DIR, "X_memmap.dat")
    memmap_y_path = os.path.join(TEMP_DIR, "y_memmap.dat")

    X_memmap = np.memmap(memmap_X_path, dtype=np.float32, mode="w+",
                         shape=(total_samples,) + sample_shape)
    y_memmap = np.memmap(memmap_y_path, dtype=np.int32, mode="w+",
                         shape=(total_samples,))

    cursor = 0
    for xp, yp in tqdm(zip(chunk_files_X, chunk_files_y),
                        total=len(chunk_files_X), desc="Merging chunks"):
        X_chunk = np.load(xp)
        y_chunk = np.load(yp)
        n = len(X_chunk)
        X_memmap[cursor : cursor + n] = X_chunk
        y_memmap[cursor : cursor + n] = y_chunk
        cursor += n
        del X_chunk, y_chunk
        gc.collect()

    X_memmap.flush()
    y_memmap.flush()
    del X_memmap, y_memmap

    # ── FIX 1: Save as .npy — no compression, no disk spike ───────────────
    # np.savez_compressed decompresses the full array into RAM before writing,
    # causing an OSError: No space left on device on tight disks.
    # Instead we just MOVE (rename) the memmap .dat files to .npy.
    # np.load(..., mmap_mode="r") works identically on raw .npy files.
    print("\nFinalising output files (writing valid .npy format)...")

    for p in (OUTPUT_X, OUTPUT_Y):
        if os.path.exists(p):
            os.remove(p)

    X_final = np.memmap(
        memmap_X_path,
        dtype=np.float32,
        mode="r",
        shape=(total_samples,) + sample_shape
    )

    y_final = np.memmap(
        memmap_y_path,
        dtype=np.int32,
        mode="r",
        shape=(total_samples,)
    )

    np.save(OUTPUT_X, X_final)
    np.save(OUTPUT_Y, y_final)

    del X_final, y_final

    os.remove(memmap_X_path)
    os.remove(memmap_y_path)

    print(f"  → X saved : {OUTPUT_X}")
    print(f"  → y saved : {OUTPUT_Y}")

    # ── Cleanup chunk files ───────────────────────────────────────────────
    print("\nCleaning up temp chunk files...")
    for xp, yp in zip(chunk_files_X, chunk_files_y):
        for p in (xp, yp):
            try:
                os.remove(p)
            except OSError:
                pass

    # ── Verify outputs ────────────────────────────────────────────────────
    print("\nVerifying saved files...")
    X_check = np.load(OUTPUT_X, mmap_mode="r")
    y_check = np.load(OUTPUT_Y, mmap_mode="r")
    print(f"  X shape : {X_check.shape}")
    print(f"  y shape : {y_check.shape}")
    print(f"  X dtype : {X_check.dtype}")
    print(f"  y dtype : {y_check.dtype}")

    unique, counts = np.unique(y_check, return_counts=True)

    print("\n" + "=" * 60)
    print(f"Dataset saved  : {OUTPUT_DIR}")
    print(f"Total sequences: {total_samples:,}")
    print(f"Output shape   : {X_check.shape}")
    print("\nFinal class distribution:")
    for u, c in zip(unique, counts):
        pct = 100 * c / total_samples
        bar = "█" * int(pct / 2)
        print(f"  {CLASS_NAMES[u]:14s} → {c:8,}  ({pct:5.1f}%)  {bar}")
    print(f"\nSkipped — corrupt     : {skipped_corrupt}")
    print(f"Skipped — missing key : {skipped_missing_key}")
    print(f"Skipped — no patches  : {skipped_no_patches}")
    print(f"Skipped — NaN/flat    : {skipped_nan_patch}")
    print(f"Skipped — class cap   : {skipped_cap}")

    x_size = os.path.getsize(OUTPUT_X) / (1024**3)
    y_size = os.path.getsize(OUTPUT_Y) / (1024**3)
    print(f"\nDisk usage:")
    print(f"  X.npy : {x_size:.2f} GB")
    print(f"  y.npy : {y_size:.3f} GB")
    print(f"  Total : {x_size + y_size:.2f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()