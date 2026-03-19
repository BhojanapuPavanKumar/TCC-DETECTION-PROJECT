import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Bidirectional, GRU, Dense,
    Dropout, BatchNormalization, Layer
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, CSVLogger, TensorBoard
)

# ================================
# PATHS
# ================================

DATA_DIR  = "data/lstm_dataset"
MODEL_DIR = "models/bigru"
LOG_DIR   = "logs/bigru"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "logs_training.txt")

# ================================
# LOGGER
# ================================

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

# ================================
# GPU
# ================================

def enable_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        log("GPU enabled")
    else:
        log("CPU mode")

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
        s = self.V(self.W(x))
        w = tf.nn.softmax(s, axis=1)
        w = tf.cast(w, x.dtype)
        return tf.reduce_sum(x * w, axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg

# ================================
# MODEL
# ================================

def build_model(input_shape, n_reg, n_cls):

    inp = Input(shape=input_shape)

    x = Bidirectional(GRU(64, return_sequences=True))(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(GRU(32, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    context = TemporalAttention()(x)

    # Regression
    r = Dense(64, activation="relu")(context)
    r = Dropout(0.2)(r)
    reg_out = Dense(n_reg, dtype="float32", name="regression")(r)

    # Classification
    c = Dense(64, activation="relu")(context)
    c = Dropout(0.3)(c)
    cls_out = Dense(n_cls, activation="softmax", name="classification")(c)

    return Model(inp, [reg_out, cls_out])

# ================================
# LOAD DATA (UPDATED)
# ================================

def load_data():
    log("\nLoading dataset...")

    X_train = np.load(f"{DATA_DIR}/X_train.npy").astype(np.float32)
    y_reg_train = np.load(f"{DATA_DIR}/y_reg_train.npy").astype(np.float32)
    y_cls_train = np.load(f"{DATA_DIR}/y_cls_train.npy").astype(np.int32)

    X_val = np.load(f"{DATA_DIR}/X_test.npy").astype(np.float32)
    y_reg_val = np.load(f"{DATA_DIR}/y_reg_test.npy").astype(np.float32)
    y_cls_val = np.load(f"{DATA_DIR}/y_cls_test.npy").astype(np.int32)

    cw = np.load(f"{DATA_DIR}/class_weights.npy")
    cw = np.clip(cw, 1.0, 4.0)
    cw = cw / np.mean(cw)
    log(f"Train: {X_train.shape}")
    log(f"Val  : {X_val.shape}")
    log(f"Class dist (train): {np.unique(y_cls_train, return_counts=True)}")

    return X_train, y_reg_train, y_cls_train, X_val, y_reg_val, y_cls_val, cw

# ================================
# TF DATASET
# ================================

def make_tf_dataset(X, y_reg, y_cls, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((
        X,
        {
            "regression": y_reg,
            "classification": y_cls,
        }
    ))
    if shuffle:
        ds = ds.shuffle(10000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ================================
# LOSS
# ================================

def get_weighted_loss(class_weights):

    cw = tf.constant(class_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)

        # Standard loss (NO label smoothing)
        base = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        base = tf.cast(base, tf.float32)

        weights = tf.gather(cw, y_true)
        weights = tf.cast(weights, tf.float32)

        # softer normalization (important)
        weights = weights / tf.pow(tf.reduce_mean(weights), 0.05)

        return tf.reduce_mean(base * weights)

    return loss
# ================================
# TRAIN
# ================================

def train():

    log("========== TRAINING START ==========")
    log(f"Start: {datetime.now()}")

    enable_gpu()

    X_train, y_reg_train, y_cls_train, X_val, y_reg_val, y_cls_val, cw = load_data()

    train_ds = make_tf_dataset(X_train, y_reg_train, y_cls_train, 64)
    val_ds   = make_tf_dataset(X_val, y_reg_val, y_cls_val, 64, shuffle=False)

    model = build_model(
        (X_train.shape[1], X_train.shape[2]),
        y_reg_train.shape[1],
        len(cw)
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={
            "regression": "huber",
            "classification": get_weighted_loss(cw)
        },
        loss_weights={
            "regression": 0.3,
            "classification": 1.1
        },
        metrics={
            "regression": [
                "mae",
                tf.keras.metrics.RootMeanSquaredError(name="rmse")
            ],
            "classification": ["accuracy"]
        }
    )

    callbacks = [
        ModelCheckpoint(
            os.path.join(MODEL_DIR, "cloud_bigru_best.keras"),
            save_best_only=True,
            monitor="val_loss",
            verbose=1
        ),
        EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
        CSVLogger(os.path.join(LOG_DIR, "training_log.csv")),
        TensorBoard(log_dir=LOG_DIR)
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=callbacks,
        verbose=1
    )

    # ================================
    # EVALUATION
    # ================================

    preds = model.predict(X_val)
    pred_cls = np.argmax(preds[1], axis=1)

    log("\n===== VALIDATION =====")
    log(f"Pred dist: {np.unique(pred_cls, return_counts=True)}")
    log(f"True dist: {np.unique(y_cls_val, return_counts=True)}")

    # ================================
    # SAVE FINAL
    # ================================

    final_path = os.path.join(MODEL_DIR, "cloud_bigru_final.keras")
    model.save(final_path)

    log(f"Final model saved: {final_path}")
    log(f"End: {datetime.now()}")
    log("========== TRAINING END ==========")


if __name__ == "__main__":
    train()