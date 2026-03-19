import os
import numpy as np
import tensorflow as tf
import random

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    ConvLSTM2D,
    BatchNormalization,
    Dropout,
    Dense,
    Flatten
)
from tensorflow.keras.callbacks import EarlyStopping


# ----------------------------------
# SETTINGS
# ----------------------------------

X_PATH = "data/sequences/X_sequences.npy"
Y_PATH = "data/sequences/y_labels.npy"
MODEL_PATH = "models/storm_prediction_convlstm.keras"


# ----------------------------------
# REPRODUCIBILITY
# ----------------------------------

def set_seed():

    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)


# ----------------------------------
# LOAD DATA
# ----------------------------------

def load_dataset():

    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        raise FileNotFoundError("Dataset not found. Run sequence generation first.")

    print("\nLoading dataset...")

    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    print("Original X shape:", X.shape)

    samples, timesteps, features = X.shape

    # pad features to 9 (3x3)
    if features < 9:
        padding = np.zeros((samples, timesteps, 9 - features))
        X = np.concatenate([X, padding], axis=2)

    # reshape to ConvLSTM format
    X = X.reshape(samples, timesteps, 3, 3, 1)

    print("ConvLSTM input shape:", X.shape)
    print("y shape:", y.shape)

    X = X.astype("float32")

    return X, y


# ----------------------------------
# BUILD CONVLSTM MODEL
# ----------------------------------

def build_model(input_shape):

    model = Sequential()

    model.add(tf.keras.layers.Input(shape=input_shape))

    model.add(
        ConvLSTM2D(
            filters=32,
            kernel_size=(3,3),
            activation="relu",
            padding="same",
            return_sequences=True
        )
    )

    model.add(BatchNormalization())

    model.add(
        ConvLSTM2D(
            filters=64,
            kernel_size=(3,3),
            activation="relu",
            padding="same",
            return_sequences=False
        )
    )

    model.add(BatchNormalization())

    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(64, activation="relu"))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ----------------------------------
# TRAIN MODEL
# ----------------------------------

def train_lstm():

    set_seed()

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    if os.path.exists(MODEL_PATH):

        print("\nModel already exists. Loading model...\n")
        model = load_model(MODEL_PATH)

    else:

        print("\nTraining new ConvLSTM model...\n")

        model = build_model(X.shape[1:])

        model.summary()

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=30,
            batch_size=4,
            callbacks=[early_stop],
            verbose=1
        )

        os.makedirs("models", exist_ok=True)

        model.save(MODEL_PATH)

        print("\nModel saved:", MODEL_PATH)

    print("\nEvaluating model...")

    loss, accuracy = model.evaluate(X_test, y_test)

    print("\nTest Loss:", loss)
    print("Test Accuracy:", accuracy)

    print("\nTraining complete")
    print("Model path:", MODEL_PATH)


# ----------------------------------
# MAIN
# ----------------------------------

if __name__ == "__main__":
    train_lstm()