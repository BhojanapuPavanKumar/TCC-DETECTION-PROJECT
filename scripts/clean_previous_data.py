import os
import glob
import shutil

# ================================
# PATH CONFIGURATION
# ================================

RAW_PATH = "data/raw_npz"

# DATE_BASED_OUTPUT_FOLDERS = [
#     "data/predicted_masks",
#     "data/clusters",
#     "data/storm_tracks",
#     "data/prediction",
#     "data/metrics",
#     "output/scientific",
#     "output/actual",
#     "output/predicted",
#     "output/visualization",
#     "output/pipeline_day",
# ]

# Sequence files regenerated each run
LSTM_FILES_TO_CLEAR = [
    "data/lstm_dataset/X_sequences.npy",
    "data/lstm_dataset/y_labels.npy",
    "data/lstm_dataset/y_behavior.npy",
    "data/lstm_dataset/y_stage1_train.npy",
    "data/lstm_dataset/y_stage1_val.npy",
    "data/lstm_dataset/y_stage1_test.npy",
    "data/lstm_dataset/y_stage2_train.npy",
    "data/lstm_dataset/y_stage2_val.npy",
    "data/lstm_dataset/y_stage2_test.npy",
    "data/lstm_dataset/X_train.npy",
    "data/lstm_dataset/X_val.npy",
    "data/lstm_dataset/X_test.npy",
    "data/lstm_dataset/y_reg_train.npy",
    "data/lstm_dataset/y_reg_val.npy",
    "data/lstm_dataset/y_reg_test.npy",
    "data/lstm_dataset/y_cls_train.npy",
    "data/lstm_dataset/y_cls_val.npy",
    "data/lstm_dataset/y_cls_test.npy",
    "data/lstm_dataset/class_weights.npy",
    "data/lstm_dataset/bob_sequence_idx.npy",
]

# Never delete (important for inference)
LSTM_FILES_TO_KEEP = [
    "data/lstm_dataset/scaler.save",
    "data/lstm_dataset/target_scaler.save",
    "data/lstm_dataset/behavior_encoder.save",
    "data/lstm_dataset/label_map.save",
    "data/lstm_dataset/feature_cols.npy",
    "data/lstm_dataset/target_cols.npy",
    "data/lstm_dataset/log_transform_cols.npy",
]


# ================================
# GET RAW DATA DATES
# ================================

def get_raw_dates():

    if not os.path.exists(RAW_PATH):
        print("Raw NPZ folder missing — skipping selective cleaning")
        return set()

    dates = set()

    for file in os.listdir(RAW_PATH):

        if file.endswith(".npz"):

            date = file[:10]  # assumes YYYY-MM-DD filename format
            dates.add(date)

    return dates


# # ================================
# # REMOVE OUTDATED DATE-BASED FILES
# # ================================

# def remove_outdated_outputs():

#     raw_dates = get_raw_dates()

#     if not raw_dates:
#         return

#     removed_total = 0

#     print("\nChecking outdated pipeline outputs...")

#     for folder in DATE_BASED_OUTPUT_FOLDERS:

#         if not os.path.exists(folder):
#             continue

#         removed = 0

#         for file in os.listdir(folder):

#             filepath = os.path.join(folder, file)

#             if not os.path.isfile(filepath):
#                 continue

#             # Extract date from filename
#             date = file[:10]

#             if date not in raw_dates:

#                 os.remove(filepath)
#                 removed += 1
#                 removed_total += 1

#         if removed > 0:
#             print(f"  Removed {removed} outdated files from {folder}")

#     if removed_total == 0:
#         print("  No outdated files found")


# ================================
# CLEAR LSTM GENERATED DATA ONLY
# ================================

def clear_lstm_sequences():

    print("\nClearing LSTM sequence datasets (keeping scalers)...")

    removed = 0

    for filepath in LSTM_FILES_TO_CLEAR:

        if os.path.exists(filepath):

            os.remove(filepath)
            removed += 1

    print(f"  Removed {removed} sequence files")


# ================================
# VERIFY SCALERS SAFE
# ================================

def verify_scalers():

    print("\nVerifying scaler + encoder safety:")

    for filepath in LSTM_FILES_TO_KEEP:

        status = "OK" if os.path.exists(filepath) else "MISSING"
        print(f"  [{status}] {filepath}")


# ================================
# REMOVE EMPTY DIRECTORIES
# ================================

# def remove_empty_directories():

#     print("\nRemoving empty directories...")

#     removed = 0

#     for folder in DATE_BASED_OUTPUT_FOLDERS:

#         if not os.path.exists(folder):
#             continue

#         if len(os.listdir(folder)) == 0:

#             shutil.rmtree(folder)
#             removed += 1
#             print(f"  Removed empty folder: {folder}")

#     if removed == 0:
#         print("  No empty directories removed")


# ================================
# MAIN CLEAN FUNCTION
# ================================

def clean_pipeline():

    print("\n================================")
    print(" SMART PIPELINE CLEANING STARTED ")
    print("================================")

    raw_dates = get_raw_dates()

    if raw_dates:
        print(f"\nDetected {len(raw_dates)} dataset date(s) in raw_npz")
    else:
        print("\nNo dataset dates detected — skipping selective cleaning")

    # Step 1: remove outdated outputs
    remove_outdated_outputs()

    # Step 2: remove sequence files only
    clear_lstm_sequences()

    # Step 3: verify scalers safe
    verify_scalers()

    # Step 4: remove empty dirs
    remove_empty_directories()

    print("\nCleaning completed successfully")
    print("Models preserved in models/")
    print("Scalers preserved in data/lstm_dataset/")
    print("Raw NPZ preserved in data/raw_npz/")
    print("================================\n")


# ================================
# DIRECT EXECUTION SUPPORT
# ================================

if __name__ == "__main__":
    clean_pipeline()