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

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR  = "data/lstm_dataset"
MODEL_DIR = "models/bigrustages"
LOG_DIR   = "logs/bigrustages/stage1"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

def enable_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("GPU enabled + mixed_float16")
    else:
        print("CPU mode")

class TemporalAttention(Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = Dense(units, activation="tanh", use_bias=False)
        self.V = Dense(1, use_bias=False)
    def call(self, x):
        score   = self.V(self.W(x))
        weights = tf.nn.softmax(score, axis=1)
        weights = tf.cast(weights, x.dtype)
        return tf.reduce_sum(x * weights, axis=1)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg

def focal_loss(gamma=1.5, alpha=0.75):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce     = -(y_true * tf.math.log(y_pred) + (1-y_true)*tf.math.log(1-y_pred))
        p_t     = y_true*y_pred + (1-y_true)*(1-y_pred)
        alpha_t = y_true*alpha  + (1-y_true)*(1-alpha)
        focal_w = tf.pow(1.0 - p_t, gamma)
        return tf.reduce_mean(alpha_t * focal_w * bce)
    return loss

def oversample_events(X_train, y_train, target_ratio=10):
    normal_idx = np.where(y_train == 0)[0]
    event_idx  = np.where(y_train == 1)[0]
    n_normal   = len(normal_idx)
    n_event    = len(event_idx)
    target_n   = n_normal // target_ratio
    print(f"  Original: {n_normal} normal, {n_event} events ({n_normal//n_event}:1)")
    print(f"  Target  : {n_normal} normal, {target_n} events ({target_ratio}:1)")
    if target_n <= n_event:
        print("  No oversampling needed")
        return X_train, y_train
    repeats   = target_n // n_event
    remainder = target_n  % n_event
    event_X   = X_train[event_idx]
    noise_std = 0.02
    aug_X = [X_train]
    aug_y = [y_train]
    repeated_X = np.tile(event_X, (repeats, 1, 1))
    repeated_X += np.random.normal(0, noise_std, repeated_X.shape)
    aug_X.append(repeated_X.astype(np.float32))
    aug_y.append(np.ones(len(repeated_X), dtype=np.float32))
    if remainder > 0:
        idx   = np.random.choice(n_event, remainder, replace=False)
        rem_X = event_X[idx] + np.random.normal(0, noise_std, (remainder, event_X.shape[1], event_X.shape[2]))
        aug_X.append(rem_X.astype(np.float32))
        aug_y.append(np.ones(remainder, dtype=np.float32))
    X_out = np.concatenate(aug_X)
    y_out = np.concatenate(aug_y)
    perm  = np.random.permutation(len(X_out))
    X_out, y_out = X_out[perm], y_out[perm]
    n_ev = int(y_out.sum())
    print(f"  After  : {len(X_out)-n_ev} normal, {n_ev} events ({(len(X_out)-n_ev)//max(n_ev,1)}:1)")
    return X_out, y_out

def build_model(input_shape):
    inp = Input(shape=input_shape, name="sequence_input")
    x = Bidirectional(GRU(64, return_sequences=True), name="bigru_1")(inp)
    x = BatchNormalization(name="bn_1")(x)
    x = Dropout(0.4, name="drop_1")(x)
    x = Bidirectional(GRU(32, return_sequences=True), name="bigru_2")(x)
    x = BatchNormalization(name="bn_2")(x)
    x = Dropout(0.4, name="drop_2")(x)
    x = TemporalAttention(units=32, name="attention")(x)
    x = Dense(64, activation="relu", name="dense_1")(x)
    x = Dropout(0.3, name="drop_3")(x)
    x = Dense(32, activation="relu", name="dense_2")(x)
    out = Dense(1, activation="sigmoid", dtype="float32", name="output")(x)
    model = Model(inp, out)
    model.summary()
    return model

def load_data():
    print("\nLoading Stage 1 data...")
    X_train = np.load(f"{DATA_DIR}/X_train.npy").astype(np.float32)
    y_train = np.load(f"{DATA_DIR}/y_stage1_train.npy").astype(np.float32)
    X_val   = np.load(f"{DATA_DIR}/X_val.npy").astype(np.float32)
    y_val   = np.load(f"{DATA_DIR}/y_stage1_val.npy").astype(np.float32)
    print(f"  Raw train: {X_train.shape}  Events: {y_train.sum():.0f}/{len(y_train)}")
    print(f"  Val      : {X_val.shape}    Events: {y_val.sum():.0f}/{len(y_val)}")
    print("\nOversampling minority EVENT class...")
    X_train, y_train = oversample_events(X_train, y_train, target_ratio=10)
    classes = np.array([0, 1])
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = np.clip(cw, 1.0, 5.0)
    cw = cw / cw.mean()
    class_weights = {0: float(cw[0]), 1: float(cw[1])}
    print(f"  Class weights: {class_weights}")
    return X_train, y_train, X_val, y_val, class_weights

def train():
    print("\n===== STAGE 1: NORMAL vs EVENT (with oversampling) =====")
    print(f"Start: {datetime.now()}")
    enable_gpu()
    X_train, y_train, X_val, y_val, cw = load_data()
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0, epsilon=1e-7),
        loss=focal_loss(gamma=1.5, alpha=0.75),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    )
    callbacks = [
        ModelCheckpoint(os.path.join(MODEL_DIR,"bigru_stage1.keras"),
                        save_best_only=True, monitor="val_recall", mode="max", verbose=1),
        EarlyStopping(monitor="val_recall", mode="max", patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_recall", mode="max", patience=3,
                          factor=0.5, min_lr=1e-6, verbose=1),
        TensorBoard(log_dir=LOG_DIR, histogram_freq=0),
        CSVLogger(os.path.join(LOG_DIR,"stage1_log.csv"), append=False),
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=50, batch_size=128, class_weight=cw,
              callbacks=callbacks, verbose=1)
    print(f"\nStage 1 complete: {datetime.now()}")
    print(f"Model saved: {MODEL_DIR}/bigru_stage1.keras")

if __name__ == "__main__":
    train()