import os
import glob
import gc
import numpy as np
import tensorflow as tf
from tqdm import tqdm

MODEL_PATH = "models/cnn/unet_best.keras"
INPUT_DIR  = "data/raw_npz"
OUTPUT_DIR = "data/predicted_masks"

PATCH_SIZE = 128
BATCH_SIZE = 128

# Match exactly to create_cnn_dataset.py
TB_MIN = 180
TB_MAX = 320

os.makedirs(OUTPUT_DIR, exist_ok=True)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU enabled")
else:
    print("Running on CPU")


def extract_patches_tf(image):
    image   = tf.convert_to_tensor(image, dtype=tf.float32)
    image   = tf.expand_dims(tf.expand_dims(image, axis=0), axis=-1)
    patches = tf.image.extract_patches(
        images=image,
        sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
        strides=[1, PATCH_SIZE, PATCH_SIZE, 1],
        rates=[1, 1, 1, 1],
        padding="SAME"
    )
    patches = tf.reshape(patches, [-1, PATCH_SIZE, PATCH_SIZE, 1])
    return patches.numpy()


def reconstruct(patches, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    idx  = 0
    for y in range(0, h, PATCH_SIZE):
        for x in range(0, w, PATCH_SIZE):
            patch = patches[idx]
            mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = patch[:h-y, :w-x]
            idx += 1
    return mask


def main():
    print("Loading CNN model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully")

    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.npz")))
    print(f"Total NPZ files: {len(files)}")

    for file in tqdm(files):
        name     = os.path.basename(file).replace(".npz", "")
        out_path = os.path.join(OUTPUT_DIR, name + "_mask.npy")

        if os.path.exists(out_path):
            continue

        try:
            data = np.load(file)
            tb   = data["TIR1_TEMP"].astype(np.float32)
        except Exception as e:
            print(f"\nSkipping corrupted file: {file} | {e}")
            continue

        h, w = tb.shape

        # Normalize to match training data range
        tb = np.clip(tb, TB_MIN, TB_MAX)
        tb = (tb - TB_MIN) / (TB_MAX - TB_MIN)

        patches = extract_patches_tf(tb)

        preds = model.predict(patches, batch_size=BATCH_SIZE, verbose=0)

        # Explicit axis squeeze — safe for any number of patches
        preds = (preds > 0.5).astype(np.uint8).squeeze(axis=-1)

        mask = reconstruct(preds, h, w)
        np.save(out_path, mask)

        del patches, preds, mask
        gc.collect()

    print("\nMask prediction completed successfully.")


if __name__ == "__main__":
    main()