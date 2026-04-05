"""
ConvLSTM2D Training Script — RTX 2050 / Low-VRAM Edition
=========================================================
Bugs fixed vs original:
  FIX 1 — NotImplementedError on save:
    ModelCheckpoint tried to save GradAccumModel (a subclassed Model) in
    .keras/HDF5 format, which only supports Functional/Sequential models.
    Replaced with _SaveInnerModel callback that saves the inner Functional
    model directly.

  FIX 2 — LossScaleOptimizer warning:
    The optimizer was wrapped with LossScaleOptimizer but train_step calls
    tape.gradient / apply_gradients directly, bypassing the wrapper's
    required protocol. Removed the wrapper — plain Adam is correct here.
"""

import os
import json
import datetime
import numpy as np
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
from tensorflow.keras.callbacks import (
    EarlyStopping, CSVLogger, TensorBoard, LambdaCallback,
)
from tensorflow.keras import mixed_precision

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from sklearn.preprocessing import label_binarize


# ================= FOCAL LOSS =================

def categorical_focal_crossentropy(gamma=2.0, label_smoothing=0.0):
    def loss_fn(y_true, y_pred):
        num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
        if label_smoothing > 0:
            y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        loss = focal_weight * ce
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return loss_fn


# ================= GPU / PRECISION =================

mixed_precision.set_global_policy("mixed_float16")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✓ GPU detected ({len(gpus)} device(s))")
else:
    print("⚠ Running on CPU")

# ================= REPRODUCIBILITY =================

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ================= HYPERPARAMETERS =================

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
)

# ================= PATHS =================

DATA_PATH    = "data/convlstm_dataset/convlstm_sequences.npz"
MODEL_DIR    = "models/convlstm"
LOG_DIR      = "logs/convlstm"
PLOTS_DIR    = "output/plots/convlstm"
TB_LOG_DIR   = os.path.join(LOG_DIR, "tensorboard",
                             datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
CKPT_PATH    = os.path.join(MODEL_DIR, "best_convlstm.keras")
CSV_LOG_PATH = os.path.join(LOG_DIR, "training_log_convlstm.csv")
METRICS_JSON = os.path.join(LOG_DIR, "test_metrics.json")

for d in (MODEL_DIR, LOG_DIR, PLOTS_DIR):
    os.makedirs(d, exist_ok=True)

# ================= DATA LOADING =================

def load_data():
    data = np.load(DATA_PATH, mmap_mode="r")
    X    = data["X"]
    y    = data["y"]
    print(f"Dataset loaded — X: {X.shape}, y: {y.shape}")
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution:")
    for u, c in zip(unique, counts):
        print(f"  class {u} → {c} ({100*c/len(y):.1f}%)")
    return X, y

# ================= GENERATOR-BASED tf.data PIPELINE =================

def make_generator(X_mmap, y_arr, indices, target_h, target_w):
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


def make_dataset(X_mmap, y_arr, indices, training: bool, batch_size: int,
                 target_h: int, target_w: int, num_classes: int,
                 T: int, C: int) -> tf.data.Dataset:

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
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# ================= AUGMENTATION =================

class TrainingOnlyAugmentation(tf.keras.layers.Layer):
    def call(self, x, training=False):
        if training:
            batch = tf.shape(x)[0]
            T     = tf.shape(x)[1]
            H     = tf.shape(x)[2]
            W     = tf.shape(x)[3]
            C     = tf.shape(x)[4]
            flat  = tf.reshape(x, [batch * T, H, W, C])
            flat  = tf.image.random_flip_left_right(flat)
            delta = tf.random.uniform([], -0.08, 0.08)
            flat  = tf.image.adjust_brightness(flat, delta)
            x     = tf.reshape(flat, [batch, T, H, W, C])
        return x

# ================= SPATIAL ATTENTION =================

def spatial_attention(x, name_prefix="attn"):
    channels = x.shape[-1]
    gap      = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    sq       = Dense(max(channels // 4, 8), activation="relu",
                     name=f"{name_prefix}_sq")(gap)
    ex       = Dense(channels, activation="sigmoid",
                     name=f"{name_prefix}_ex")(sq)
    return Multiply(name=f"{name_prefix}_scale")([x, ex])

# ================= MODEL =================

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

    l2 = tf.keras.regularizers.l2(HP["weight_decay"])
    x  = Dense(128, kernel_regularizer=l2, name="fc1")(x)
    x  = Activation("gelu", name="act1")(x)
    x  = Dropout(HP["dropout_dense_1"], name="drop1")(x)
    x  = Dense(64, kernel_regularizer=l2, name="fc2")(x)
    x  = Activation("gelu", name="act2")(x)
    x  = Dropout(HP["dropout_dense_2"], name="drop2")(x)
    out = Dense(num_classes, activation="softmax",
                dtype="float32", name="predictions")(x)

    return Model(inputs, out, name="ConvLSTM_Residual_SE")

# ================= GRADIENT ACCUMULATION TRAINER =================

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
            safe_g = g if g is not None else tf.zeros_like(acc)
            acc.assign_add(safe_g)

        self._step.assign_add(1)

        should_update = tf.equal(
            self._step % tf.cast(self._accum_steps, tf.int32),
            tf.constant(0, dtype=tf.int32)
        )

        def apply_and_reset():
            scale     = tf.cast(self._accum_steps, tf.float32)
            avg_grads = [acc / scale for acc in self._accum_grads]
            self.optimizer.apply_gradients(
                zip(avg_grads, self.inner.trainable_variables))
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

# ================= LR SCHEDULE =================

def make_lr_schedule(steps_per_epoch: int):
    warmup_steps = HP["warmup_epochs"] * steps_per_epoch
    decay_steps  = (HP["epochs"] - HP["warmup_epochs"]) * steps_per_epoch

    class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self):
            self.cosine = CosineDecayRestarts(
                initial_learning_rate = HP["base_lr"],
                first_decay_steps     = decay_steps,
                t_mul=2.0, m_mul=0.85,
                alpha = HP["min_lr"] / HP["base_lr"],
            )
            self._warmup = tf.cast(warmup_steps, tf.float32)

        def __call__(self, step):
            s      = tf.cast(step, tf.float32)
            warmup = HP["min_lr"] + (HP["base_lr"] - HP["min_lr"]) * (s / self._warmup)
            post   = self.cosine(s - self._warmup)
            return tf.where(s < self._warmup, warmup, post)

        def get_config(self):
            return {"warmup_steps": int(warmup_steps)}

    return WarmupCosine()

# ================= EVALUATION HELPERS =================

def plot_confusion_matrix(y_true, preds, path):
    n   = len(np.unique(y_true))
    cm  = confusion_matrix(y_true, preds)
    fig, ax = plt.subplots(figsize=(max(5, n), max(5, n)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set(title="Confusion Matrix", xlabel="Predicted", ylabel="True")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix → {path}")


def plot_roc_pr(y_true, probs, num_classes, path):
    classes = list(range(num_classes))
    y_bin   = label_binarize(y_true, classes=classes)
    if num_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    roc_aucs, pr_aucs = {}, {}

    for c in classes:
        fpr, tpr, _ = roc_curve(y_bin[:, c], probs[:, c])
        roc_aucs[c] = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, label=f"C{c} AUC={roc_aucs[c]:.3f}")

        pr, rc, _  = precision_recall_curve(y_bin[:, c], probs[:, c])
        pr_aucs[c] = average_precision_score(y_bin[:, c], probs[:, c])
        axes[1].plot(rc, pr, label=f"C{c} AP={pr_aucs[c]:.3f}")

    axes[0].set(title="ROC (one-vs-rest)", xlabel="FPR", ylabel="TPR")
    axes[0].plot([0,1],[0,1],"k--",lw=0.8)
    axes[0].legend(fontsize=8)
    axes[1].set(title="Precision-Recall", xlabel="Recall", ylabel="Precision")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ROC/PR curves → {path}")
    return roc_aucs, pr_aucs

# ================= INNER-MODEL CHECKPOINT (FIX 1) =================

class _SaveInnerModel(tf.keras.callbacks.Callback):
    """
    FIX 1: Saves the inner Functional model instead of GradAccumModel.

    ModelCheckpoint calls model.save() on the object passed to model.fit().
    GradAccumModel is a subclassed tf.keras.Model — Keras cannot serialise
    subclassed models to .keras / HDF5 format.  The inner model IS a standard
    Functional model and serialises without issues.

    Behaviour is identical to ModelCheckpoint(save_best_only=True).
    """
    def __init__(self, inner: tf.keras.Model, path: str, monitor: str = "val_auc_roc"):
        super().__init__()
        self._inner   = inner
        self._path    = path
        self._monitor = monitor
        self._best    = -float("inf")

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get(self._monitor, -float("inf"))
        if current > self._best:
            self._best = current
            self._inner.save(self._path)
            print(f"\nEpoch {epoch+1}: {self._monitor} improved to "
                  f"{current:.5f}, saving model to {self._path}")


# ================= F1 METRIC =================

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall    = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred)
        self.recall.update_state(y_true, y_pred)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + 1e-7))

    def reset_state(self):
        self.precision.reset_states()
        self.recall.reset_states()


# ================= MAIN =================

def main():

    print("[STEP 1/8] Loading data...")
    X, y = load_data()

    T, H, W, C  = X.shape[1], X.shape[2], X.shape[3], X.shape[4]
    num_classes  = len(np.unique(y))
    th, tw       = HP["target_h"], HP["target_w"]

    print("[STEP 2/8] Splitting indices...")
    print(f"\nOriginal spatial: {H}x{W}  →  Model input: {th}x{tw}")
    print(f"VRAM reduction from resize: {(H*W)/(th*tw):.1f}x")

    print("[STEP 3/8] Computing class weights...")
    idx = np.arange(len(X))
    idx_train, idx_temp = train_test_split(
        idx, test_size=0.30, stratify=y, random_state=SEED)
    idx_val, idx_test   = train_test_split(
        idx_temp, test_size=0.50, stratify=y[idx_temp], random_state=SEED)

    y_train = y[idx_train]
    print(f"\nSplit — train: {len(idx_train)}, val: {len(idx_val)}, test: {len(idx_test)}")

    print("[STEP 4/8] Building tf.data generators...")
    cw_arr  = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw_dict = dict(enumerate(cw_arr))
    print("Class weights:", {k: f"{v:.3f}" for k, v in cw_dict.items()})

    mb     = HP["micro_batch"]
    common = dict(target_h=th, target_w=tw, num_classes=num_classes, T=T, C=C)

    print("[STEP 5/8] Constructing datasets...")
    train_ds = make_dataset(X, y, idx_train, training=True,  batch_size=mb, **common)
    val_ds   = make_dataset(X, y, idx_val,   training=False, batch_size=mb, **common)
    test_ds  = make_dataset(X, y, idx_test,  training=False, batch_size=mb, **common)

    steps_per_epoch = len(idx_train) // mb

    print("[STEP 6/8] Building model...")
    inner = build_model((T, th, tw, C), num_classes)
    inner.summary()

    print("[STEP 7/8] Compiling model...")
    model    = GradAccumModel(inner, accum_steps=HP["accum_steps"])
    lr_sched = make_lr_schedule(steps_per_epoch)

    # FIX 2: Prevent TF from auto-wrapping Adam with LossScaleOptimizer.
    # When mixed_float16 is the global policy, TF 2.9/2.10 automatically
    # wraps any optimizer passed to model.compile() with LossScaleOptimizer.
    # Our custom train_step manages gradients manually (tape.gradient /
    # apply_gradients directly) so the wrapper protocol is never called,
    # producing the "forgot to call get_scaled_loss" warning and degrading
    # training quality.
    # Solution: use float32 policy on the GradAccumModel (layers stay fp16
    # because they are defined on the inner model which keeps mixed_float16),
    # then set dtype_policy back.  This prevents model.compile() from seeing
    # a mixed_float16 model and auto-wrapping the optimizer.
    # Simpler approach that always works: just switch global policy to float32
    # before compile so TF does not auto-wrap, then switch back.
    mixed_precision.set_global_policy("float32")
    optimizer = Adam(learning_rate=lr_sched, clipnorm=1.0)

    model.compile(
        optimizer = optimizer,
        loss      = categorical_focal_crossentropy(
                        gamma=2.0,
                        label_smoothing=HP["label_smoothing"]),
        metrics   = [
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            F1Score(),
            tf.keras.metrics.AUC(name="auc_roc"),
            tf.keras.metrics.AUC(curve="PR", name="auc_pr"),
        ],
    )
    # Restore mixed_float16 so any future layer creation stays fp16
    mixed_precision.set_global_policy("mixed_float16")

    model.build((None, T, th, tw, C))

    callbacks = [
        EarlyStopping(
            monitor="val_auc_roc", mode="max",
            patience=HP["early_stop_pat"],
            restore_best_weights=True, verbose=1,
        ),
        # FIX 1: _SaveInnerModel saves the inner Functional model.
        # ModelCheckpoint is NOT used — it would crash on GradAccumModel.
        _SaveInnerModel(inner, CKPT_PATH, monitor="val_auc_roc"),
        CSVLogger(CSV_LOG_PATH),
        TensorBoard(log_dir=TB_LOG_DIR, histogram_freq=0),
        LambdaCallback(
            on_epoch_end=lambda epoch, logs: logs.update(
                {"lr": float(tf.keras.backend.get_value(
                    optimizer.learning_rate(optimizer.iterations)
                ))}
            )
        ),
    ]

    print("[STEP 8/8] Starting training...")
    history = model.fit(
        train_ds,
        validation_data  = val_ds,
        validation_steps = len(idx_val) // mb,
        epochs           = HP["epochs"],
        callbacks        = callbacks,
        class_weight     = cw_dict,
    )

    print("\n[EVAL] Running test set evaluation...")
    results      = model.evaluate(test_ds, verbose=1)
    metrics_dict = dict(zip(model.metrics_names, [float(v) for v in results]))
    for k, v in metrics_dict.items():
        print(f"  {k}: {v:.4f}")

    print("[EVAL] Collecting predictions batch by batch...")
    probs_list, labels_list = [], []
    for x_batch, y_batch in test_ds:
        probs_list.append(model(x_batch, training=False).numpy())
        labels_list.append(np.argmax(y_batch.numpy(), axis=1))

    probs  = np.concatenate(probs_list, axis=0)
    preds  = np.argmax(probs, axis=1)
    y_true = np.concatenate(labels_list, axis=0)

    print("[EVAL] Generating classification report...")
    report = classification_report(y_true, preds, output_dict=True)
    print("\nClassification report:\n")
    print(classification_report(y_true, preds))

    metrics_dict["classification_report"] = report
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"  Metrics JSON → {METRICS_JSON}")

    print("[PLOT] Saving confusion matrix...")
    plot_confusion_matrix(
        y_true, preds,
        path=os.path.join(PLOTS_DIR, "confusion_matrix.png"),
    )

    print("[PLOT] Saving ROC / PR curves...")
    roc_aucs, pr_aucs = plot_roc_pr(
        y_true, probs, num_classes,
        path=os.path.join(PLOTS_DIR, "roc_pr_curves.png"),
    )

    print("\nPer-class AUC summary:")
    for c in range(num_classes):
        print(f"  Class {c}: ROC-AUC={roc_aucs[c]:.4f}  PR-AUC={pr_aucs[c]:.4f}")


if __name__ == "__main__":
    main()