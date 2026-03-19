import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Bidirectional, GRU, Dense,
    Dropout, BatchNormalization, Layer
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, TensorBoard, CSVLogger
)
from tensorflow.keras.regularizers import l2

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR  = "data/lstm_dataset"
MODEL_DIR = "models/bigrustages"
LOG_DIR   = "logs/bigrustages/stage2"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)


# ================================
# GPU
# ================================

def enable_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("GPU enabled + mixed_float16")
    else:
        print("CPU mode")


# ================================
# ATTENTION
# ================================

class TemporalAttention(Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = Dense(units, activation="tanh", use_bias=False)
        self.V = Dense(1, use_bias=False)

    def call(self, x):
        score   = self.V(self.W(x))
        weights = tf.nn.softmax(score, axis=1)
        weights = tf.cast(weights, x.dtype)   # fix mixed precision
        return tf.reduce_sum(x * weights, axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg


# ================================
# FOCAL LOSS
# ================================

def focal_loss(gamma=2.0, alpha=0.4):
    """
    Stage 2 has 2.5:1 imbalance (INTENSIFYING:ORGANIZING).
    alpha=0.4 gives moderate weight to minority ORGANIZING class.
    Less aggressive than stage 1 since imbalance is much smaller.
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        bce     = -(y_true * tf.math.log(y_pred) +
                    (1 - y_true) * tf.math.log(1 - y_pred))
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha  + (1 - y_true) * (1 - alpha)
        focal_w = tf.pow(1.0 - p_t, gamma)

        return tf.reduce_mean(alpha_t * focal_w * bce)
    return loss


# ================================
# MODEL
# ================================

def build_model(input_shape):
    """
    Smaller than Stage 1 — Stage 2 has only ~1176 training samples.
    L2 regularization to prevent overfitting on small dataset.
    """
    inp = Input(shape=input_shape, name="sequence_input")

    x = Bidirectional(GRU(32, return_sequences=True), name="bigru_1")(inp)
    x = BatchNormalization(name="bn_1")(x)
    x = Dropout(0.5, name="drop_1")(x)

    x = Bidirectional(GRU(16, return_sequences=True), name="bigru_2")(x)
    x = BatchNormalization(name="bn_2")(x)
    x = Dropout(0.5, name="drop_2")(x)

    x = TemporalAttention(units=16, name="attention")(x)

    x = Dense(32, activation="relu",
              kernel_regularizer=l2(0.01),
              name="dense_1")(x)
    x = Dropout(0.5, name="drop_3")(x)

    out = Dense(1, activation="sigmoid",
                kernel_regularizer=l2(0.01),
                dtype="float32", name="output")(x)

    model = Model(inp, out)
    model.summary()
    return model


# ================================
# LOAD DATA
# ================================

def load_data():
    print("\nLoading Stage 2 data...")

    X_train = np.load(f"{DATA_DIR}/X_train.npy").astype(np.float32)
    y_train = np.load(f"{DATA_DIR}/y_stage2_train.npy").astype(np.float32)
    X_val   = np.load(f"{DATA_DIR}/X_val.npy").astype(np.float32)
    y_val   = np.load(f"{DATA_DIR}/y_stage2_val.npy").astype(np.float32)

    # Remove NORMAL samples (-1)
    mask_train = y_train != -1
    mask_val   = y_val   != -1
    X_train = X_train[mask_train]
    y_train = y_train[mask_train]
    X_val   = X_val[mask_val]
    y_val   = y_val[mask_val]

    print(f"  Train: {X_train.shape}")
    print(f"    ORGANIZING(0)  : {(y_train==0).sum()}")
    print(f"    INTENSIFYING(1): {(y_train==1).sum()}")

    # Oversample ORGANIZING to match INTENSIFYING count
    org_idx    = np.where(y_train == 0)[0]
    intens_idx = np.where(y_train == 1)[0]
    n_org      = len(org_idx)
    n_intens   = len(intens_idx)

    if n_org < n_intens:
        print(f"\n  Oversampling ORGANIZING {n_org} -> {n_intens}...")
        repeats   = n_intens // n_org
        remainder = n_intens  % n_org
        org_X     = X_train[org_idx]
        noise_std = 0.02

        aug_X = [X_train]
        aug_y = [y_train]

        rep_X = np.tile(org_X, (repeats, 1, 1))
        rep_X += np.random.normal(0, noise_std, rep_X.shape)
        aug_X.append(rep_X.astype(np.float32))
        aug_y.append(np.zeros(len(rep_X), dtype=np.float32))

        if remainder > 0:
            idx   = np.random.choice(n_org, remainder, replace=False)
            rem_X = org_X[idx] + np.random.normal(
                0, noise_std, (remainder, org_X.shape[1], org_X.shape[2])
            )
            aug_X.append(rem_X.astype(np.float32))
            aug_y.append(np.zeros(remainder, dtype=np.float32))

        X_train = np.concatenate(aug_X)
        y_train = np.concatenate(aug_y)
        perm    = np.random.permutation(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]
        print(f"  After: ORGANIZING={( y_train==0).sum()} "
              f"INTENSIFYING={(y_train==1).sum()}")

    classes = np.array([0, 1])
    cw      = compute_class_weight("balanced", classes=classes, y=y_train)
    cw      = np.clip(cw, 1.0, 3.0)
    cw      = cw / cw.mean()
    class_weights = {0: float(cw[0]), 1: float(cw[1])}
    print(f"  Class weights: {class_weights}")

    return X_train, y_train, X_val, y_val, class_weights
# ================================
# TRAIN
# ================================

def train():
    print("\n===== STAGE 2: ORGANIZING vs INTENSIFYING =====")
    print(f"Start: {datetime.now()}")

    enable_gpu()

    X_train, y_train, X_val, y_val, cw = load_data()

    # Safety check — need at least both classes in val
    if len(np.unique(y_val)) < 2:
        print("WARNING: val set missing a class — AUC undefined. Using accuracy.")
        monitor_metric = "val_accuracy"
    else:
        monitor_metric = "val_auc"

    model = build_model((X_train.shape[1], X_train.shape[2]))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=5e-4,
            clipnorm=1.0,
            epsilon=1e-7
        ),
        loss=focal_loss(gamma=2.0, alpha=0.4),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    )

    callbacks = [
        ModelCheckpoint(
            os.path.join(MODEL_DIR, "bigru_stage2.keras"),
            save_best_only=True,
            monitor=monitor_metric,
            mode="max",
            verbose=1
        ),
        EarlyStopping(
            monitor=monitor_metric,
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor=monitor_metric,
            mode="max",
            patience=3,
            factor=0.5,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(log_dir=LOG_DIR, histogram_freq=0),
        CSVLogger(os.path.join(LOG_DIR, "stage2_log.csv"), append=False),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=32,       # small batch — only ~1176 training samples
        class_weight=cw,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\nStage 2 complete: {datetime.now()}")
    print(f"Model saved: {MODEL_DIR}/bigru_stage2.keras")


if __name__ == "__main__":
    train()