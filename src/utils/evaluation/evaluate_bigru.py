import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import Layer, Dense


class TemporalAttention(Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = Dense(units, activation="tanh", use_bias=False)
        self.V = Dense(1, use_bias=False)

    def call(self, x):
        score = self.V(self.W(x))
        weights = tf.nn.softmax(score, axis=1)
        weights = tf.cast(weights, x.dtype)
        return tf.reduce_sum(x * weights, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

# ================================
# PATHS
# ================================

MODEL_PATH = "models/bigru/cloud_bigru_best.keras"
DATA_DIR   = "data/lstm_dataset"

# ================================
# CLASS NAMES
# ================================

CLASS_NAMES = [
    "STABLE",
    "GROWING",
    "SHRINKING",
    "INTENSIFYING",
    "SPLITTING",
    "MERGING"
]

# ================================
# LOAD DATA
# ================================

def load_data():
    X_val = np.load(f"{DATA_DIR}/X_test.npy").astype(np.float32)
    y_cls_val = np.load(f"{DATA_DIR}/y_cls_test.npy").astype(np.int32)

    print(f"Validation data: {X_val.shape}")
    print(f"Class dist: {np.unique(y_cls_val, return_counts=True)}")

    return X_val, y_cls_val

# ================================
# EVALUATE
# ================================

def evaluate(model, X_val, y_cls_val):

    print("\n===== RUNNING EVALUATION =====")

    preds = model.predict(X_val)
    y_pred = np.argmax(preds[1], axis=1)

    # -----------------------
    # CONFUSION MATRIX
    # -----------------------
    cm = confusion_matrix(
        y_cls_val,
        y_pred,
        labels=np.arange(len(CLASS_NAMES))
    )

    print("\nConfusion Matrix:\n", cm)

    # -----------------------
    # REPORT
    # -----------------------
    report = classification_report(
        y_cls_val,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    print("\nClassification Report:\n", report)

    # -----------------------
    # PER CLASS ACCURACY
    # -----------------------
    print("\nPer-Class Accuracy:")
    for i, name in enumerate(CLASS_NAMES):
        total = cm[i].sum()
        correct = cm[i][i]
        acc = correct / total if total > 0 else 0

        print(f"{name:15s}: {acc:.4f} ({correct}/{total})")

    # -----------------------
    # SAVE IMAGE
    # -----------------------
    os.makedirs("output/evaluation", exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45)
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    save_path = "output/evaluation/confusion_matrix.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"\nSaved confusion matrix: {save_path}")

# ================================
# MAIN
# ================================

def main():
    print("Loading model...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"TemporalAttention": TemporalAttention},
        compile=False   # 🔥 THIS FIXES EVERYTHING
    )
    X_val, y_cls_val = load_data()

    evaluate(model, X_val, y_cls_val)

# ================================
# RUN
# ================================

if __name__ == "__main__":
    main()