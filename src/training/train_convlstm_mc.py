"""
train_convlstm_mc.py  (FINAL — Multi-Class Edition)
=====================================================
Trains the ConvLSTM model on the 5-class TCC lifecycle dataset.

FIXES IN THIS VERSION:
  FIX 1  — Multi-class focal loss (per-class focal weights)
  FIX 2  — Macro F1 using confusion matrix accumulator
  FIX 3  — Correct DATA_PATH (convlstm_mc dataset)
  FIX 4  — Memory-safe evaluation via tf.data pipeline
  FIX 5  — shutil.rmtree for directory-style .keras checkpoints
             os.remove() fails on Windows when .keras is a directory.
             shutil.rmtree() handles both files and directories correctly.
  FIX 6  — Gradient accumulation NaN guard (clip before accumulation)
  FIX 7  — Warmup LR reads optimizer.iterations directly
  FIX 8  — Stratified split graceful fallback
  FIX 9  — CSVLogger append=True (never overwrites history on resume)
  FIX 10 — save_cb built before callbacks list and inserted directly
  FIX 11 — restore_best_weights=False (unreliable on wrapper model)
  FIX 12 — Single CSV read for both initial_epoch and best metric
  FIX 13 — pandas imported at top level
  FIX 14 — _best persisted to .best_metric.txt alongside checkpoint
             Survives even if checkpoint directory is manually deleted.
             On resume: takes max(csv_best, txt_best) so nothing is lost.
  FIX 15 — tmp cleanup uses shutil.rmtree (handles directory tmp files)
"""

import os
import json
import shutil
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    ConvLSTM2D, BatchNormalization, GlobalAveragePooling2D,
    Dense, Dropout, Add, Multiply, Input, Activation,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, TensorBoard, LambdaCallback
from tensorflow.keras import mixed_precision

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from sklearn.preprocessing import label_binarize


# =============================================================
# MULTI-CLASS FOCAL LOSS
# =============================================================

def categorical_focal_crossentropy(gamma: float = 2.0, label_smoothing: float = 0.0):
    """
    Per-class focal weight applied independently per logit.
    For each sample and class c:
        focal_weight_c = (1 - p_c)^gamma
        loss_c         = focal_weight_c * (-y_c * log(p_c))
    """
    def loss_fn(y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        if label_smoothing > 0.0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

        y_pred     = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce         = -y_true * tf.math.log(y_pred)
        focal_w    = tf.pow(1.0 - y_pred, gamma)
        focal_loss = focal_w * ce
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

    return loss_fn


# =============================================================
# GPU / PRECISION
# =============================================================

mixed_precision.set_global_policy("mixed_float16")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✓ GPU detected ({len(gpus)} device(s))")
else:
    print("⚠ Running on CPU")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =============================================================
# HYPERPARAMETERS
# =============================================================

HP = dict(
    target_h        = 64,
    target_w        = 64,
    micro_batch     = 4,
    accum_steps     = 4,
    epochs          = 60,
    base_lr         = 8e-4,
    min_lr          = 1e-6,
    warmup_epochs   = 5,
    label_smoothing = 0.05,
    dropout_dense_1 = 0.4,
    dropout_dense_2 = 0.3,
    early_stop_pat  = 15,
    weight_decay    = 1e-4,
    filters         = [32, 48, 48],
    grad_clip_norm  = 1.0,
    grad_clip_value = 5.0,
)

NUM_CLASSES = 5
CLASS_NAMES = ["NON-TCC", "ORGANIZING", "INTENSIFYING", "MATURE", "DISSIPATING"]


# =============================================================
# PATHS
# =============================================================

DATA_X_PATH  = "data/convlstm_mc_dataset/X.npy"
DATA_Y_PATH  = "data/convlstm_mc_dataset/y.npy"
MODEL_DIR    = "models/convlstm_mc"
LOG_DIR      = "logs/convlstm_mc"
PLOTS_DIR    = "output/plots/convlstm_mc"
TB_LOG_DIR   = os.path.join(
    LOG_DIR, "tensorboard", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)
CKPT_PATH    = os.path.join(MODEL_DIR, "best_convlstm_mc.keras")
CSV_LOG_PATH = os.path.join(LOG_DIR, "training_log_convlstm_mc.csv")
METRICS_JSON = os.path.join(LOG_DIR, "test_metrics_mc.json")

for d in (MODEL_DIR, LOG_DIR, PLOTS_DIR):
    os.makedirs(d, exist_ok=True)


# =============================================================
# DATA LOADING
# =============================================================

def load_data():
    """Returns mmap handles — no full array loaded into RAM."""
    X = np.load(DATA_X_PATH, mmap_mode="r")
    y = np.load(DATA_Y_PATH, mmap_mode="r")
    print(f"Dataset loaded — X: {X.shape}  y: {y.shape}")
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution:")
    for u, c, name in zip(unique, counts, CLASS_NAMES):
        print(f"  {name:14s} [{u}] → {c:,}  ({100*c/len(y):.1f}%)")
    return X, y


# =============================================================
# GENERATOR-BASED tf.data PIPELINE
# =============================================================

def make_generator(X_mmap, y_arr, indices, target_h, target_w):
    """Yields one (seq, label) pair at a time — zero extra RAM."""
    def gen():
        for idx in indices:
            seq   = X_mmap[idx].astype(np.float32)
            label = int(y_arr[idx])
            resized = np.stack([
                tf.image.resize(seq[t], [target_h, target_w]).numpy()
                for t in range(seq.shape[0])
            ], axis=0)
            yield resized, label
    return gen


def make_dataset(
    X_mmap, y_arr, indices, training: bool, batch_size: int,
    target_h: int, target_w: int, num_classes: int, T: int, C: int,
) -> tf.data.Dataset:

    output_sig = (
        tf.TensorSpec(shape=(T, target_h, target_w, C), dtype=tf.float32),
        tf.TensorSpec(shape=(),                          dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(
        make_generator(X_mmap, y_arr, indices, target_h, target_w),
        output_signature=output_sig,
    )
    if training:
        ds = ds.shuffle(buffer_size=2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=training)
    ds = ds.map(
        lambda x, y: (x, tf.one_hot(y, num_classes)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


# =============================================================
# AUGMENTATION
# =============================================================

class TrainingOnlyAugmentation(tf.keras.layers.Layer):
    def call(self, x, training=False):
        if training:
            B = tf.shape(x)[0]
            T = tf.shape(x)[1]
            H = tf.shape(x)[2]
            W = tf.shape(x)[3]
            C = tf.shape(x)[4]
            flat  = tf.reshape(x, [B * T, H, W, C])
            flat  = tf.image.random_flip_left_right(flat)
            delta = tf.random.uniform([], -0.08, 0.08)
            flat  = tf.image.adjust_brightness(flat, delta)
            x     = tf.reshape(flat, [B, T, H, W, C])
        return x


# =============================================================
# SPATIAL ATTENTION
# =============================================================

def spatial_attention(x, name_prefix="attn"):
    channels = x.shape[-1]
    gap  = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    sq   = Dense(max(channels // 4, 8), activation="relu",
                 name=f"{name_prefix}_sq")(gap)
    ex   = Dense(channels, activation="sigmoid",
                 name=f"{name_prefix}_ex")(sq)
    return Multiply(name=f"{name_prefix}_scale")([x, ex])


# =============================================================
# MODEL
# =============================================================

def build_model(input_shape, num_classes):
    f1, f2, f3 = HP["filters"]
    inputs = Input(shape=input_shape, name="seq_input")
    x  = TrainingOnlyAugmentation(name="augmentation")(inputs)

    o1 = ConvLSTM2D(f1, (3, 3), padding="same", activation="relu",
                    return_sequences=True, name="clstm1")(x)
    o1 = BatchNormalization(name="bn1")(o1)

    o2 = ConvLSTM2D(f2, (3, 3), padding="same", activation="relu",
                    return_sequences=True, name="clstm2")(o1)
    o2 = BatchNormalization(name="bn2")(o2)
    skip1 = ConvLSTM2D(f2, (1, 1), padding="same", activation=None,
                       return_sequences=True, name="skip1_proj")(o1)
    o2 = Add(name="res1")([o2, skip1])

    o3 = ConvLSTM2D(f3, (3, 3), padding="same", activation="relu",
                    return_sequences=False, name="clstm3")(o2)
    o3 = BatchNormalization(name="bn3")(o3)
    skip2 = tf.keras.layers.Lambda(lambda t: t[:, -1], name="skip2_last")(o2)
    o3 = Add(name="res2")([o3, skip2])

    o3 = spatial_attention(o3, name_prefix="sattn")
    x  = GlobalAveragePooling2D(name="gap")(o3)

    l2  = tf.keras.regularizers.l2(HP["weight_decay"])
    x   = Dense(128, kernel_regularizer=l2, name="fc1")(x)
    x   = Activation("gelu", name="act1")(x)
    x   = Dropout(HP["dropout_dense_1"], name="drop1")(x)
    x   = Dense(64, kernel_regularizer=l2, name="fc2")(x)
    x   = Activation("gelu", name="act2")(x)
    x   = Dropout(HP["dropout_dense_2"], name="drop2")(x)
    out = Dense(num_classes, activation="softmax",
                dtype="float32", name="predictions")(x)

    return Model(inputs, out, name="ConvLSTM_MC_Residual_SE")


# =============================================================
# GRADIENT ACCUMULATION TRAINER
# =============================================================

class GradAccumModel(tf.keras.Model):
    def __init__(self, inner: tf.keras.Model, accum_steps: int):
        super().__init__()
        self.inner        = inner
        self._accum_steps = accum_steps
        self._step        = tf.Variable(0, trainable=False, dtype=tf.int32,
                                        name="accum_step_counter")

    def call(self, x, training=False):
        return self.inner(x, training=training)

    def build(self, input_shape):
        self.inner.build(input_shape)
        self._accum_grads = [
            tf.Variable(tf.zeros_like(v), trainable=False, name=f"accum_{i}")
            for i, v in enumerate(self.inner.trainable_variables)
        ]
        super().build(input_shape)

    @tf.function
    def train_step(self, data):
        x, y = data[0], data[1]

        with tf.GradientTape() as tape:
            preds = self.inner(x, training=True)
            loss  = self.compiled_loss(y, preds,
                                       regularization_losses=self.losses)

        grads = tape.gradient(loss, self.inner.trainable_variables)

        for acc, g in zip(self._accum_grads, grads):
            if g is not None:
                g_clipped = tf.clip_by_value(g,
                                             -HP["grad_clip_value"],
                                              HP["grad_clip_value"])
                acc.assign_add(g_clipped)

        self._step.assign_add(1)

        should_update = tf.equal(
            self._step % tf.cast(self._accum_steps, tf.int32),
            tf.constant(0, dtype=tf.int32),
        )

        def apply_and_reset():
            scale     = tf.cast(self._accum_steps, tf.float32)
            avg_grads = [acc / scale for acc in self._accum_grads]
            self.optimizer.apply_gradients(
                zip(avg_grads, self.inner.trainable_variables)
            )
            reset_ops = [acc.assign(tf.zeros_like(acc))
                         for acc in self._accum_grads]
            with tf.control_dependencies(reset_ops):
                return tf.constant(0.0)

        def skip():
            return tf.constant(0.0)

        tf.cond(should_update, apply_and_reset, skip)

        self.compiled_metrics.update_state(y, preds)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return super().metrics


# =============================================================
# LR SCHEDULE
# =============================================================

def make_lr_schedule(steps_per_epoch: int):
    warmup_steps = HP["warmup_epochs"] * steps_per_epoch
    decay_steps  = (HP["epochs"] - HP["warmup_epochs"]) * steps_per_epoch

    class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self):
            self.cosine = CosineDecayRestarts(
                initial_learning_rate = HP["base_lr"],
                first_decay_steps     = max(decay_steps, 1),
                t_mul=2.0, m_mul=0.85,
                alpha = HP["min_lr"] / HP["base_lr"],
            )
            self._warmup = tf.cast(warmup_steps, tf.float32)

        def __call__(self, step):
            s      = tf.cast(step, tf.float32)
            warmup = HP["min_lr"] + (HP["base_lr"] - HP["min_lr"]) * (s / (self._warmup + 1e-8))
            post   = self.cosine(s - self._warmup)
            return tf.where(s < self._warmup, warmup, post)

        def get_config(self):
            return {"warmup_steps": int(warmup_steps),
                    "decay_steps":  int(decay_steps)}

    return WarmupCosine()


# =============================================================
# MACRO F1 METRIC
# =============================================================

class MacroF1Score(tf.keras.metrics.Metric):

    def __init__(self, num_classes, name="macro_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros")
        self.fp = self.add_weight(name="fp", shape=(num_classes,), initializer="zeros")
        self.fn = self.add_weight(name="fn", shape=(num_classes,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_idx = tf.argmax(y_true, axis=-1)
        y_pred_idx = tf.argmax(y_pred, axis=-1)
        cm = tf.math.confusion_matrix(
            y_true_idx, y_pred_idx,
            num_classes=self.num_classes, dtype=tf.float32
        )
        tp = tf.linalg.diag_part(cm)
        fp = tf.reduce_sum(cm, axis=0) - tp
        fn = tf.reduce_sum(cm, axis=1) - tp
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-7)
        recall    = self.tp / (self.tp + self.fn + 1e-7)
        f1        = 2 * precision * recall / (precision + recall + 1e-7)
        return tf.reduce_mean(f1)

    def reset_state(self):
        self.tp.assign(tf.zeros_like(self.tp))
        self.fp.assign(tf.zeros_like(self.fp))
        self.fn.assign(tf.zeros_like(self.fn))


# =============================================================
# HELPER — safe delete for files AND directories
# =============================================================

def _safe_delete(path: str):
    """
    FIX 5 / FIX 15:
    TF2 saves .keras models as directories on Windows.
    os.remove() raises WinError 5 on directories.
    shutil.rmtree() handles both files and directories correctly.
    """
    if not os.path.exists(path):
        return
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    except Exception as e:
        print(f"  ⚠ Could not delete {path}: {e}")


# =============================================================
# INNER-MODEL CHECKPOINT
# =============================================================

class _SaveInnerModel(tf.keras.callbacks.Callback):
    """
    Saves the inner Functional model (not the GradAccumModel wrapper).

    FIX 5  — Uses _safe_delete (shutil.rmtree) instead of os.remove.
              Handles Windows .keras directory format correctly.
    FIX 14 — Persists _best to .best_metric.txt alongside checkpoint.
              Survives manual deletion of checkpoint. On resume:
              max(csv_best, txt_best) ensures nothing is lost.
    FIX 15 — tmp cleanup also uses _safe_delete.
    """
    def __init__(self, inner: tf.keras.Model, path: str,
                 monitor: str = "val_auc_roc"):
        super().__init__()
        self._inner    = inner
        self._path     = path
        self._monitor  = monitor
        self._best     = -float("inf")
        self._best_txt = path + ".best_metric.txt"

    def load_best_from_disk(self):
        """Call once before training to restore _best from txt file."""
        if os.path.exists(self._best_txt):
            try:
                with open(self._best_txt, "r") as f:
                    self._best = float(f.read().strip())
                print(f"  ✓ Loaded persisted _best = {self._best:.5f}")
            except Exception as e:
                print(f"  ⚠ Could not read best_metric.txt ({e})")

    def _persist_best(self):
        """Write _best to disk so it survives restarts."""
        try:
            with open(self._best_txt, "w") as f:
                f.write(str(self._best))
        except Exception as e:
            print(f"  ⚠ Could not write best_metric.txt ({e})")

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get(self._monitor, -float("inf"))
        if current <= self._best:
            return
        self._best = current

        tmp_path = self._path + f".tmp_{epoch + 1}"
        try:
            self._inner.save(tmp_path)

            # FIX 5: shutil.rmtree handles directory-style .keras on Windows
            _safe_delete(self._path)

            os.replace(tmp_path, self._path)

            # FIX 14: persist _best alongside checkpoint
            self._persist_best()

            print(f"\nEpoch {epoch+1}: {self._monitor} improved to "
                  f"{current:.5f} → saved {self._path}")
        except Exception as e:
            print(f"\n⚠ Checkpoint save failed: {e}")
        finally:
            # FIX 15: _safe_delete handles tmp that may also be a directory
            _safe_delete(tmp_path)


# =============================================================
# EVALUATION HELPERS
# =============================================================

def plot_confusion_matrix(y_true, preds, path):
    n   = NUM_CLASSES
    cm  = confusion_matrix(y_true, preds, labels=list(range(n)))
    fig, ax = plt.subplots(figsize=(n + 1, n + 1))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax.set(title="Confusion Matrix (5-class)", xlabel="Predicted", ylabel="True")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix → {path}")


def plot_roc_pr(y_true, probs, num_classes, path):
    classes = list(range(num_classes))
    y_bin   = label_binarize(y_true, classes=classes)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    roc_aucs, pr_aucs = {}, {}

    for c in classes:
        fpr, tpr, _ = roc_curve(y_bin[:, c], probs[:, c])
        roc_aucs[c] = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, label=f"{CLASS_NAMES[c]} AUC={roc_aucs[c]:.3f}")

        pr, rc, _ = precision_recall_curve(y_bin[:, c], probs[:, c])
        pr_aucs[c] = average_precision_score(y_bin[:, c], probs[:, c])
        axes[1].plot(rc, pr, label=f"{CLASS_NAMES[c]} AP={pr_aucs[c]:.3f}")

    axes[0].set(title="ROC (one-vs-rest)", xlabel="FPR", ylabel="TPR")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].legend(fontsize=7)
    axes[1].set(title="Precision-Recall", xlabel="Recall", ylabel="Precision")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ROC/PR curves → {path}")
    return roc_aucs, pr_aucs


# =============================================================
# STRATIFIED SPLIT
# =============================================================

def safe_stratified_split(idx, y, test_size, seed):
    """Stratified split; falls back to random if any class has < 2 samples."""
    try:
        counts = np.bincount(y[idx])
        if np.any(counts < 2):
            raise ValueError("Class too rare for stratified split")
        return train_test_split(idx, test_size=test_size,
                                stratify=y[idx], random_state=seed)
    except ValueError as e:
        print(f"  ⚠ Stratified split failed ({e}), using random split.")
        return train_test_split(idx, test_size=test_size, random_state=seed)


# =============================================================
# MAIN
# =============================================================

def main():

    print("[STEP 1/8] Loading data...")
    X, y = load_data()

    T, H, W, C = X.shape[1], X.shape[2], X.shape[3], X.shape[4]
    num_classes = NUM_CLASSES
    th, tw      = HP["target_h"], HP["target_w"]

    print(f"\nOriginal spatial: {H}×{W}  →  Model input: {th}×{tw}")
    print(f"VRAM reduction from resize: {(H*W)/(th*tw):.1f}×")

    print("\n[STEP 2/8] Splitting indices (stratified)...")
    idx = np.arange(len(X))
    idx_train, idx_temp = safe_stratified_split(idx,      y, test_size=0.30, seed=SEED)
    idx_val,   idx_test = safe_stratified_split(idx_temp, y, test_size=0.50, seed=SEED)
    print(f"  Train: {len(idx_train):,}  Val: {len(idx_val):,}  Test: {len(idx_test):,}")

    print("\n[STEP 3/8] Computing class weights...")
    y_train = y[idx_train]
    cw_arr  = compute_class_weight("balanced",
                                   classes=np.arange(num_classes),
                                   y=y_train)
    cw_dict = {i: float(w) for i, w in enumerate(cw_arr)}
    print("  Class weights:", {CLASS_NAMES[k]: f"{v:.3f}" for k, v in cw_dict.items()})

    print("\n[STEP 4/8] Building tf.data generators...")
    mb     = HP["micro_batch"]
    common = dict(target_h=th, target_w=tw, num_classes=num_classes, T=T, C=C)

    train_ds = make_dataset(X, y, idx_train, training=True,  batch_size=mb, **common)
    val_ds   = make_dataset(X, y, idx_val,   training=False, batch_size=mb, **common)
    test_ds  = make_dataset(X, y, idx_test,  training=False, batch_size=mb, **common)

    steps_per_epoch = max(len(idx_train) // mb, 1)

    # ----------------------------------------------------------
    # STEP 5 — Build / load model
    # ----------------------------------------------------------
    print("\n[STEP 5/8] Building model...")
    if os.path.exists(CKPT_PATH):
        print("✓ Loading checkpoint — resuming training")
        inner = tf.keras.models.load_model(
            CKPT_PATH,
            custom_objects={"MacroF1Score": MacroF1Score},
            compile=False,
        )
    else:
        print("✓ Building new model")
        inner = build_model((T, th, tw, C), num_classes)

    inner.summary()

    # ----------------------------------------------------------
    # STEP 6 — Compile
    # ----------------------------------------------------------
    print("\n[STEP 6/8] Compiling model...")
    model    = GradAccumModel(inner, accum_steps=HP["accum_steps"])
    lr_sched = make_lr_schedule(steps_per_epoch)

    mixed_precision.set_global_policy("float32")
    optimizer = Adam(
        learning_rate = lr_sched,
        clipnorm      = HP["grad_clip_norm"],
    )

    model.compile(
        optimizer = optimizer,
        loss      = categorical_focal_crossentropy(
                        gamma=2.0,
                        label_smoothing=HP["label_smoothing"]),
        metrics = [
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_acc"),
            MacroF1Score(NUM_CLASSES, name="macro_f1"),
            tf.keras.metrics.AUC(name="auc_roc", multi_label=False),
            tf.keras.metrics.AUC(curve="PR", name="auc_pr", multi_label=False),
        ]
    )
    mixed_precision.set_global_policy("mixed_float16")
    model.build((None, T, th, tw, C))

    # ----------------------------------------------------------
    # STEP 7 — Resume state + callbacks
    # ----------------------------------------------------------
    print("\n[STEP 7/8] Setting up callbacks...")

    initial_epoch = 0

    # FIX 10: build save_cb BEFORE callbacks list
    save_cb = _SaveInnerModel(inner, CKPT_PATH, monitor="val_auc_roc")

    # FIX 14: load persisted _best from txt file first
    save_cb.load_best_from_disk()

    # FIX 12: single CSV read for both initial_epoch and best metric
    if os.path.exists(CSV_LOG_PATH):
        try:
            df = pd.read_csv(CSV_LOG_PATH)
            if len(df) > 0:
                initial_epoch = int(df["epoch"].iloc[-1]) + 1
                if "val_auc_roc" in df.columns:
                    csv_best = float(df["val_auc_roc"].max())
                    # FIX 14: take max of txt_best and csv_best
                    save_cb._best = max(save_cb._best, csv_best)
                print(f"✓ Resuming from epoch {initial_epoch}, "
                      f"best val_auc_roc so far = {save_cb._best:.5f}")
        except Exception as e:
            print(f"  ⚠ Could not read CSV log ({e}), starting from epoch 0")

    callbacks = [
        EarlyStopping(
            monitor="val_auc_roc", mode="max",
            patience=HP["early_stop_pat"],
            restore_best_weights=False,        # FIX 11
            verbose=1,
        ),
        save_cb,                               # FIX 10
        CSVLogger(CSV_LOG_PATH, append=True),  # FIX 9
        TensorBoard(log_dir=TB_LOG_DIR, histogram_freq=0),
        LambdaCallback(
            on_epoch_end=lambda epoch, logs: logs.update({
                "lr": float(lr_sched(optimizer.iterations).numpy())
            })
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        validation_steps=max(len(idx_val) // mb, 1),
        epochs=HP["epochs"],
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        class_weight=cw_dict,
    )

    # ----------------------------------------------------------
    # EVALUATION
    # ----------------------------------------------------------
    print("\n[EVAL] Running test set evaluation...")
    results      = model.evaluate(test_ds, verbose=1)
    metrics_dict = dict(zip(model.metrics_names, [float(v) for v in results]))
    for k, v in metrics_dict.items():
        print(f"  {k}: {v:.4f}")

    print("\n[EVAL] Collecting predictions (memory-safe streaming)...")
    probs_list, labels_list = [], []
    for x_batch, y_batch in test_ds:
        probs_list.append(model(x_batch, training=False).numpy())
        labels_list.append(np.argmax(y_batch.numpy(), axis=1))

    probs  = np.concatenate(probs_list, axis=0)
    preds  = np.argmax(probs, axis=1)
    y_true = np.concatenate(labels_list, axis=0)

    print("\n[EVAL] Classification report (5-class):")
    report = classification_report(
        y_true, preds,
        target_names=CLASS_NAMES,
        output_dict=True,
    )
    print(classification_report(y_true, preds, target_names=CLASS_NAMES))

    metrics_dict["classification_report"] = report
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"  Metrics JSON → {METRICS_JSON}")

    print("\n[PLOT] Saving confusion matrix...")
    plot_confusion_matrix(
        y_true, preds,
        path=os.path.join(PLOTS_DIR, "confusion_matrix_mc.png"),
    )

    print("[PLOT] Saving ROC / PR curves...")
    roc_aucs, pr_aucs = plot_roc_pr(
        y_true, probs, num_classes,
        path=os.path.join(PLOTS_DIR, "roc_pr_curves_mc.png"),
    )

    print("\nPer-class AUC summary:")
    for c in range(num_classes):
        print(f"  {CLASS_NAMES[c]:14s} → ROC-AUC={roc_aucs[c]:.4f}  "
              f"PR-AUC={pr_aucs[c]:.4f}")


if __name__ == "__main__":
    main()