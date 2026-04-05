import os
import numpy as np
from tqdm import tqdm
import gc

# ================= CONFIG =================

RAW_DIR = "data/raw_npz"
OUTPUT_FILE = "data/convlstm_dataset/convlstm_sequences.npz"
TEMP_DIR = "data/convlstm_dataset/_tmp"

SEQ_LEN = 5
TEMPORAL_STRIDE = 5

PATCH_SIZE = 128
PATCH_STRIDE = 256   # disk-safe

TB_THRESHOLD = 245  # Kelvin scale (correct for your dataset)

MAX_SEQS = None

# ==========================================


def normalize(x):

    x = np.nan_to_num(x, nan=np.nanmean(x))

    return (x - np.mean(x)) / (np.std(x) + 1e-6)


def classify_patch(raw_patch):

    valid_pixels = raw_patch[np.isfinite(raw_patch)]

    if valid_pixels.size == 0:
        return 0

    return 1 if np.nanmean(valid_pixels) <= TB_THRESHOLD else 0


def extract_patches(frame):

    H, W = frame.shape

    patches = []

    for i in range(0, H - PATCH_SIZE + 1, PATCH_STRIDE):
        for j in range(0, W - PATCH_SIZE + 1, PATCH_STRIDE):
            patches.append(frame[i:i + PATCH_SIZE, j:j + PATCH_SIZE])

    return patches


def print_temperature_stats(files):

    print("\nChecking brightness temperature statistics...\n")

    mins, maxs, means = [], [], []

    checked = 0

    for f in files[:20]:

        try:

            data = np.load(os.path.join(RAW_DIR, f))

            if "TIR1_TEMP" not in data.files:
                continue

            x = data["TIR1_TEMP"]

            mins.append(np.nanmin(x))
            maxs.append(np.nanmax(x))
            means.append(np.nanmean(x))

            checked += 1

        except:
            continue

    print("Sample statistics from", checked, "files")

    print("Min:", np.mean(mins))
    print("Max:", np.mean(maxs))
    print("Mean:", np.mean(means))


def main():

    os.makedirs("data/convlstm_dataset", exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    files = sorted(
        f for f in os.listdir(RAW_DIR)
        if f.endswith(".npz")
    )

    print("Total raw files:", len(files))

    print_temperature_stats(files)

    buffer = []

    chunk_files_X = []
    chunk_files_y = []

    skipped_corrupt = 0
    skipped_missing_key = 0

    seq_counter = 0
    total_sequences = 0

    for idx, file in enumerate(tqdm(files)):

        path = os.path.join(RAW_DIR, file)

        try:
            data = np.load(path, mmap_mode="r")
        except:
            skipped_corrupt += 1
            continue

        if "TIR1_TEMP" not in data.files:
            skipped_missing_key += 1
            continue

        try:
            frame_raw = data["TIR1_TEMP"].astype(np.float32)
        except:
            skipped_corrupt += 1
            continue

        frame_norm = normalize(frame_raw)

        patches_raw = extract_patches(frame_raw)
        patches_norm = extract_patches(frame_norm)

        buffer.append((patches_raw, patches_norm))

        if len(buffer) < SEQ_LEN:
            continue

        if idx % TEMPORAL_STRIDE != 0:
            continue

        X_patch = []
        y_patch = []

        for patch_idx in range(len(patches_raw)):

            seq = []

            for t in range(SEQ_LEN):

                seq.append(
                    buffer[-SEQ_LEN + t][1][patch_idx]
                )

            seq = np.stack(seq)

            raw_patch = buffer[-1][0][patch_idx]

            label = classify_patch(raw_patch)

            X_patch.append(seq[..., np.newaxis])
            y_patch.append(label)

        X_patch = np.array(X_patch, dtype=np.float32)
        y_patch = np.array(y_patch, dtype=np.int32)

        xp = os.path.join(TEMP_DIR, f"X_{seq_counter}.npy")
        yp = os.path.join(TEMP_DIR, f"y_{seq_counter}.npy")

        np.save(xp, X_patch)
        np.save(yp, y_patch)

        chunk_files_X.append(xp)
        chunk_files_y.append(yp)

        seq_counter += 1
        total_sequences += len(X_patch)

        del X_patch, y_patch
        gc.collect()

        if len(buffer) > SEQ_LEN:
            buffer = buffer[-SEQ_LEN:]

        if MAX_SEQS and total_sequences >= MAX_SEQS:
            break

    print("\nStreaming merge (memory-safe)...")

    total_samples = sum(
        len(np.load(yp, mmap_mode="r"))
        for yp in chunk_files_y
    )

    sample_shape = np.load(
        chunk_files_X[0],
        mmap_mode="r"
    ).shape[1:]

    X_memmap = np.memmap(
        os.path.join(TEMP_DIR, "X_memmap.dat"),
        dtype=np.float32,
        mode="w+",
        shape=(total_samples,) + sample_shape
    )

    y_memmap = np.memmap(
        os.path.join(TEMP_DIR, "y_memmap.dat"),
        dtype=np.int32,
        mode="w+",
        shape=(total_samples,)
    )

    cursor = 0

    for xp, yp in tqdm(zip(chunk_files_X, chunk_files_y),
                       total=len(chunk_files_X)):

        X_chunk = np.load(xp)
        y_chunk = np.load(yp)

        n = len(X_chunk)

        X_memmap[cursor:cursor+n] = X_chunk
        y_memmap[cursor:cursor+n] = y_chunk

        cursor += n

        del X_chunk, y_chunk

    X_memmap.flush()
    y_memmap.flush()

    print("\nSaving compressed dataset...")

    np.savez_compressed(
        OUTPUT_FILE,
        X=X_memmap,
        y=y_memmap
    )

    print("\nDataset saved successfully.")
    print("Total sequences created:", total_samples)

    unique, counts = np.unique(y_memmap, return_counts=True)

    print("\nClass distribution:")

    for u, c in zip(unique, counts):
        print(f"class {u} → {c}")

    print("\nSkipped corrupted files:", skipped_corrupt)
    print("Skipped missing-key files:", skipped_missing_key)


if __name__ == "__main__":
    main()