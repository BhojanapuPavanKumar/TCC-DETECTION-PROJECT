import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.utils.class_weight import compute_class_weight

INPUT_FILE      = "data/storm_tracks/storm_tracks.csv"
OUTPUT_DIR      = "data/lstm_dataset"
SEQUENCE_LENGTH = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# FEATURES / TARGETS
# ================================

FEATURE_COLS = [
    "centroid_lat","centroid_lon","area_km2","mean_tb",
    "min_tb","max_tb","std_tb","tb_p10",
    "convective_core","cold_cloud_frac",
    "speed_kmh","growth","dlat","dlon",
]

TARGET_COLS = [
    "centroid_lat","centroid_lon","area_km2",
    "mean_tb","convective_core","growth",
]

# ================================
# LABEL MAPPING
# ================================

BEHAVIOR_MAP = {
    "STABLE": "NORMAL",
    "GROWING": "NORMAL",
    "SHRINKING": "NORMAL",
    "MERGING": "ORGANIZING",
    "SPLITTING": "ORGANIZING",
    "INTENSIFYING": "INTENSIFYING"
}

CLASS_TO_INDEX = {
    "NORMAL": 0,
    "ORGANIZING": 1,
    "INTENSIFYING": 2
}

FINAL_CLASSES = ["NORMAL", "ORGANIZING", "INTENSIFYING"]

# ================================
# HELPERS
# ================================

def apply_log(df):
    df = df.copy()
    for c in ["area_km2","speed_kmh","growth"]:
        df[c] = np.log1p(np.clip(df[c], 0, None))
    return df


def add_velocity(df):
    df = df.sort_values(["cluster_id","time"]).copy()
    df["dlat"] = df.groupby("cluster_id")["centroid_lat"].diff().fillna(0).clip(-2,2)
    df["dlon"] = df.groupby("cluster_id")["centroid_lon"].diff().fillna(0).clip(-2,2)
    return df


def generate_sequences(df):
    X, y_reg, y_cls, y_s1, y_s2 = [], [], [], [], []

    for _, g in tqdm(df.groupby("cluster_id")):
        if len(g) <= SEQUENCE_LENGTH:
            continue

        Xv = g[FEATURE_COLS].values
        Yr = g[TARGET_COLS].values
        Yc = g["behavior_label"].values
        Ys1 = g["stage1_label"].values
        Ys2 = g["stage2_label"].values

        for i in range(len(g) - SEQUENCE_LENGTH):
            X.append(Xv[i:i+SEQUENCE_LENGTH])
            y_reg.append(Yr[i+SEQUENCE_LENGTH])
            y_cls.append(Yc[i+SEQUENCE_LENGTH])
            y_s1.append(Ys1[i+SEQUENCE_LENGTH])
            y_s2.append(Ys2[i+SEQUENCE_LENGTH])

    return (
        np.array(X, np.float32),
        np.array(y_reg, np.float32),
        np.array(y_cls, np.int32),
        np.array(y_s1, np.int32),
        np.array(y_s2, np.int32)
    )

# ================================
# MAIN
# ================================

def main():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_FILE)
    df["time"] = pd.to_datetime(df["time"])

    # ---- Label mapping ----
    df = df[df["behavior"].isin(BEHAVIOR_MAP.keys())].copy()
    df["behavior_grouped"] = df["behavior"].map(BEHAVIOR_MAP)

    print("\nRaw class distribution:")
    print(df["behavior_grouped"].value_counts())

    # ---- Feature engineering ----
    df = add_velocity(df)
    df = apply_log(df)

    # ---- Clean ----
    df = df.dropna().copy()

    # ---- Encode labels ----
    df["behavior_label"] = df["behavior_grouped"].map(CLASS_TO_INDEX).astype(np.int32)

    # ================================
    # STAGE LABELS
    # ================================

    # Stage1 → NORMAL vs EVENT
    df["stage1_label"] = (df["behavior_label"] != 0).astype(np.int32)

    # Stage2 → only for EVENT (NORMAL = -1)
    df["stage2_label"] = -1
    df.loc[df["behavior_label"] == 1, "stage2_label"] = 0
    df.loc[df["behavior_label"] == 2, "stage2_label"] = 1

    joblib.dump(CLASS_TO_INDEX, os.path.join(OUTPUT_DIR, "label_map.save"))

    # ================================
    # TRAIN / VAL / TEST SPLIT
    # ================================

    clusters = df["cluster_id"].unique()
    np.random.seed(42)
    np.random.shuffle(clusters)

    n = len(clusters)
    train_end = int(0.7 * n)
    val_end   = int(0.85 * n)

    train_clusters = set(clusters[:train_end])
    val_clusters   = set(clusters[train_end:val_end])
    test_clusters  = set(clusters[val_end:])

    df_train = df[df["cluster_id"].isin(train_clusters)].copy()
    df_val   = df[df["cluster_id"].isin(val_clusters)].copy()
    df_test  = df[df["cluster_id"].isin(test_clusters)].copy()

    print(f"\nTrain clusters: {len(train_clusters)}")
    print(f"Val clusters  : {len(val_clusters)}")
    print(f"Test clusters : {len(test_clusters)}")

    # ================================
    # SCALING
    # ================================

    scaler = StandardScaler()
    t_scaler = StandardScaler()

    df_train.loc[:, FEATURE_COLS] = scaler.fit_transform(df_train[FEATURE_COLS])
    df_val.loc[:, FEATURE_COLS]   = scaler.transform(df_val[FEATURE_COLS])
    df_test.loc[:, FEATURE_COLS]  = scaler.transform(df_test[FEATURE_COLS])

    df_train.loc[:, TARGET_COLS] = t_scaler.fit_transform(df_train[TARGET_COLS])
    df_val.loc[:, TARGET_COLS]   = t_scaler.transform(df_val[TARGET_COLS])
    df_test.loc[:, TARGET_COLS]  = t_scaler.transform(df_test[TARGET_COLS])

    joblib.dump(scaler, os.path.join(OUTPUT_DIR,"scaler.save"))
    joblib.dump(t_scaler, os.path.join(OUTPUT_DIR,"target_scaler.save"))

    # ================================
    # SEQUENCES
    # ================================

    print("\nGenerating TRAIN sequences...")
    X_train, y_reg_train, y_cls_train, y_s1_train, y_s2_train = generate_sequences(df_train)

    print("\nGenerating VAL sequences...")
    X_val, y_reg_val, y_cls_val, y_s1_val, y_s2_val = generate_sequences(df_val)

    print("\nGenerating TEST sequences...")
    X_test, y_reg_test, y_cls_test, y_s1_test, y_s2_test = generate_sequences(df_test)

    # ================================
    # CLASS WEIGHTS
    # ================================

    classes = np.array([0,1,2])
    class_weights = compute_class_weight("balanced", classes=classes, y=y_cls_train)
    np.save(os.path.join(OUTPUT_DIR,"class_weights.npy"), class_weights)

    # ================================
    # SAVE ALL DATA
    # ================================

    # TRAIN
    np.save(os.path.join(OUTPUT_DIR,"X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR,"y_reg_train.npy"), y_reg_train)
    np.save(os.path.join(OUTPUT_DIR,"y_cls_train.npy"), y_cls_train)
    np.save(os.path.join(OUTPUT_DIR,"y_stage1_train.npy"), y_s1_train)
    np.save(os.path.join(OUTPUT_DIR,"y_stage2_train.npy"), y_s2_train)

    # VAL
    np.save(os.path.join(OUTPUT_DIR,"X_val.npy"), X_val)
    np.save(os.path.join(OUTPUT_DIR,"y_reg_val.npy"), y_reg_val)
    np.save(os.path.join(OUTPUT_DIR,"y_cls_val.npy"), y_cls_val)
    np.save(os.path.join(OUTPUT_DIR,"y_stage1_val.npy"), y_s1_val)
    np.save(os.path.join(OUTPUT_DIR,"y_stage2_val.npy"), y_s2_val)

    # TEST
    np.save(os.path.join(OUTPUT_DIR,"X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR,"y_reg_test.npy"), y_reg_test)
    np.save(os.path.join(OUTPUT_DIR,"y_cls_test.npy"), y_cls_test)
    np.save(os.path.join(OUTPUT_DIR,"y_stage1_test.npy"), y_s1_test)
    np.save(os.path.join(OUTPUT_DIR,"y_stage2_test.npy"), y_s2_test)

    print("\n✅ FINAL DATASET READY")
    print(f"Train: {X_train.shape}")
    print(f"Val  : {X_val.shape}")
    print(f"Test : {X_test.shape}")

    print("\nStage1 distribution (TRAIN):")
    print(np.unique(y_s1_train, return_counts=True))

    print("\nStage2 distribution (TRAIN):")
    print(np.unique(y_s2_train, return_counts=True))


if __name__ == "__main__":
    main()