import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def iou_metric(y_true, y_pred):

    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection

    return intersection / (union + 1e-7)


def dice_metric(y_true, y_pred):

    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)

    return (2 * intersection) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-7
    )

def train_cnn():

    IMG_DIR = "data/cnn_dataset/images"
    MASK_DIR = "data/cnn_dataset/masks"
    os.makedirs("models", exist_ok=True)

    BATCH_SIZE = 12
    EPOCHS = 5
    PATCH_SIZE = 128
    image_files = set(os.listdir(IMG_DIR))
    mask_files = set(os.listdir(MASK_DIR))

    common_files = sorted(list(image_files & mask_files))

    print("Images:", len(image_files))
    print("Masks:", len(mask_files))
    print("Matched pairs:", len(common_files))

    image_paths = [os.path.join(IMG_DIR, f) for f in common_files]
    mask_paths = [os.path.join(MASK_DIR, f) for f in common_files]
    print("Total patches:", len(image_paths))
    # -----------------------------
    # TRAIN / VAL / TEST SPLIT
    # -----------------------------

    train_x, test_x, train_y, test_y = train_test_split(
        image_paths, mask_paths,
        test_size=0.2,
        random_state=42
    )

    train_x, val_x, train_y, val_y = train_test_split(
        train_x, train_y,
        test_size=0.2,
        random_state=42
    )

    # -----------------------------
    # DATA LOADER
    # -----------------------------

    def load_sample(img_path, mask_path):

        img = np.load(img_path.numpy().decode())
        mask = np.load(mask_path.numpy().decode())

        return img.astype(np.float32), mask.astype(np.float32)

    def tf_loader(img_path, mask_path):

        img, mask = tf.py_function(
            load_sample,
            [img_path, mask_path],
            [tf.float32, tf.float32]
        )

        img.set_shape((PATCH_SIZE, PATCH_SIZE, 1))
        mask.set_shape((PATCH_SIZE, PATCH_SIZE, 1))

        return img, mask

    def create_dataset(img_paths, mask_paths):

        dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))

        dataset = dataset.map(
            tf_loader,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    train_dataset = create_dataset(train_x, train_y)
    val_dataset = create_dataset(val_x, val_y)
    test_dataset = create_dataset(test_x, test_y)

    # -----------------------------
    # SIMPLE U-NET
    # -----------------------------

    inputs = tf.keras.layers.Input((PATCH_SIZE, PATCH_SIZE, 1))

    c1 = tf.keras.layers.Conv2D(16,3,activation="relu",padding="same")(inputs)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = tf.keras.layers.Conv2D(32,3,activation="relu",padding="same")(p1)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    c3 = tf.keras.layers.Conv2D(64,3,activation="relu",padding="same")(p2)

    u1 = tf.keras.layers.UpSampling2D()(c3)
    m1 = tf.keras.layers.Concatenate()([u1,c2])

    c4 = tf.keras.layers.Conv2D(32,3,activation="relu",padding="same")(m1)

    u2 = tf.keras.layers.UpSampling2D()(c4)
    m2 = tf.keras.layers.Concatenate()([u2,c1])

    outputs = tf.keras.layers.Conv2D(1,1,activation="sigmoid")(m2)

    model = tf.keras.Model(inputs,outputs)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            iou_metric,
            dice_metric,
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )

    model.summary()

    # -----------------------------
    # TRAIN
    # -----------------------------
    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),

        ModelCheckpoint(
            "models/unet_best.keras",
            monitor="val_loss",
            save_best_only=True
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2
        ),
        CSVLogger(
            "training_log.csv",
            append=True
        )
    ]

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    

    # -----------------------------
    # TEST
    # -----------------------------

    results = model.evaluate(test_dataset)

    print("Evaluation Results:")
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value}")

    

    model.save("models/unet_cloud_detection.keras")

    print("Model saved")

