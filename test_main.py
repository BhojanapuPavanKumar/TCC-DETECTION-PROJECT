# import os
# import numpy as np
# from preprocessing.load_npz import load_npz

# DATA_FOLDER = "data/raw_npz"


# def main():

#     files = sorted(os.listdir(DATA_FOLDER))

#     print("Total files:", len(files))

#     # test only first file
#     file_path = os.path.join(DATA_FOLDER, files[0])

#     print("\nTesting file:", files[0])

#     tb, lat, lon, time = load_npz(file_path)

#     print("\nResults:")

#     print("Time:", time)
#     print("TB shape:", tb.shape)
#     print("Latitude shape:", lat.shape)
#     print("Longitude shape:", lon.shape)

#     print("Temperature range:",np.nanmin(tb), "to", np.nanmax(tb))


# if __name__ == "__main__":
#     main()



# import os
# import glob

# from preprocessing.load_npz_parallel import (
#     group_files_by_day,
#     load_day_parallel
# )


# def main():

#     folder = "data/raw_npz"

#     if not os.path.exists(folder):
#         print("Folder not found:", folder)
#         return

#     files = glob.glob(os.path.join(folder, "*.npz"))

#     if len(files) == 0:
#         print("No NPZ files found")
#         return

#     print("Total NPZ files:", len(files))

#     # ------------------------------------------------
#     # Test grouping by day
#     # ------------------------------------------------

#     days = group_files_by_day(folder)

#     print("\nDays detected:", len(days))

#     for day, files in list(days.items())[:2]:   # test only first 2 days

#         print(f"\nTesting day {day}")
#         print("Files in this day:", len(files))

#         # ------------------------------------------------
#         # Test parallel loading
#         # ------------------------------------------------

#         data = load_day_parallel(files[:5], workers=2)  # test first 5 files only

#         print("Loaded frames:", len(data))

#         if len(data) > 0:

#             tb, lat, lon = data[0]

#             print("TB shape:", tb.shape)
#             print("Latitude shape:", lat.shape)
#             print("Longitude shape:", lon.shape)


# if __name__ == "__main__":
#     main()

# *************************************test_image processing


# import os
# import numpy as np
# from preprocessing.load_npz import load_npz
# from preprocessing.preprocess_images import preprocess_images

# DATA_FOLDER = "data/raw_npz"

# files = sorted(os.listdir(DATA_FOLDER))

# file_path = os.path.join(DATA_FOLDER, files[0])

# tb, lat, lon, time = load_npz(file_path)

# tb_clean = preprocess_images(tb)

# print("Original range:", np.nanmin(tb), np.nanmax(tb))
# print("Processed range:", np.nanmin(tb_clean), np.nanmax(tb_clean))
# print("Shape:", tb_clean.shape)

# import matplotlib.pyplot as plt

# plt.figure(figsize=(12,5))

# plt.subplot(1,2,1)
# plt.imshow(tb, cmap="turbo")
# plt.title("Original Image")
# plt.colorbar()

# plt.subplot(1,2,2)
# plt.imshow(tb_clean, cmap="turbo")
# plt.title("After Preprocessing")
# plt.colorbar()

# plt.show()












#**********************************************test_clusters
# import os

# from preprocessing.load_npz import load_npz
# from preprocessing.preprocess_images import preprocess_images
# from preprocessing.detect_clusters import detect_clusters

# DATA_FOLDER = "data/raw_npz"

# files = sorted(os.listdir(DATA_FOLDER))

# print("Total files:", len(files))

# for file in files:

#     # Only process NPZ files
#     if not file.endswith(".npz"):
#         continue

#     file_path = os.path.join(DATA_FOLDER, file)

#     print("\nProcessing:", file)

#     # -------------------------
#     # Load NPZ
#     # -------------------------

#     tb, lat, lon, time = load_npz(file_path)

#     # Skip if load failed
#     if tb is None:
#         print("Skipping file due to load error")
#         continue

#     # -------------------------
#     # Preprocess
#     # -------------------------

#     tb_clean = preprocess_images(tb)

#     # -------------------------
#     # Detect clusters
#     # -------------------------

#     clusters = detect_clusters(tb_clean, lat, lon, time)

#     # detect_clusters already skips processed files
#     if clusters is None:
#         continue

#     print("Detected clusters:", len(clusters))







# **********************************test_track_cluster
# from tracking.track_clusters import track_clusters

# track_clusters()












#************************************test_storm lifecycle_analysis
# from tracking.storm_lifecycle_analysis import analyze_storm_lifecycle

# analyze_storm_lifecycle()







# *********************************************test_ml_dataset
# from dataset.generate_ml_dataset import generate_ml_dataset

# generate_ml_dataset()
# import pandas as pd

# df = pd.read_csv("data/lifecycle/storm_lifecycle_summary.csv")

# print("Min temp change:", df["mean_temp_change"].min())
# print("Max temp change:", df["mean_temp_change"].max())




# # *********************************test_sequence
# from dataset.generate_sequences import generate_sequences

# generate_sequences()
# import pandas as pd

# df = pd.read_csv("data/ml_dataset/ml_dataset.csv")

# print(df["label"].value_counts())





# **************************************test_train_lstm
# from training.train_lstm import train_lstm


# def main():

#     print("Starting ConvLSTM model training...\n")

#     train_lstm()

#     print("\nTraining complete")


# if __name__ == "__main__":
#     main()













#*****************************test_cnn_dataset

# from dataset.create_cnn_dataset import create_dataset

# def main():

#     print("Creating CNN dataset...\n")

#     create_dataset()

#     print("\nCNN dataset generation finished")


# if __name__ == "__main__":
#     main()










# **********************************test_train_cnn
# from training.train_cnn import train_cnn


# def main():

#     print("Starting CNN model training...\n")

#     train_cnn()

#     print("\nTraining complete")


# if __name__ == "__main__":
#     main()







# import tensorflow as tf
# print(tf.config.list_physical_devices())






# ******************************* validate training
# from utils.evaluation.visualize_predictions import visualize_predictions


# def main():

#     print("visulaizing cnn model...\n")

#     visualize_predictions()

#     print("\nvisualize complete")


# if __name__ == "__main__":
#     main()







#************************single frame evaluation using prediction
# from prediction.validate_single_frame import main as validate_main


# def main():

#     print("visualizing cnn model...\n")

#     validate_main()

#     print("\nvisualize complete")


# if __name__ == "__main__":
#     main()






#**********************************predict mask
# from prediction.predict_masks import main


# def run():
#     main()


# if __name__ == "__main__":
#     run()








# *************************************detect cluster 
# from detection.detect_clusters import main


# def run():
#     main()


# if __name__ == "__main__":
#     run()



# *****************************************************track culuster
# from tracking.track_clusters import main


# def run():
#     main()


# if __name__ == "__main__":
#     run()








#********************************************creating sequence to train gru 
# from lifecycle.create_sequences import main


# def run():
#     main()


# if __name__ == "__main__":
#     run()






#**********************************************train_GRU
# from training.train_gru import main


# def run():
#     main()


# if __name__ == "__main__":
#     run()





#**************************************predict dataset
# from prediction.predict_dataset import main


# def run():
#     main()


# if __name__ == "__main__":
#     run()




# **********************************************evaluate predicyions
# from utils.evaluation.evaluate_predictions import main


# def run():
#     main()


# if __name__ == "__main__":
#     run()




# ************************** visualize  predicted storm and future prediction
# from utils.evaluation.plot_future_predictions import main


# def run():
#     main()


# if __name__ == "__main__":
#     run()





#****************************** visualize predict next frame and future frame
# from prediction.predict_future_frames import main


# def run():
#     main()


# if __name__ == "__main__":
#     run()


# ***********************************to test data which we hVW
import numpy as np
import joblib
import pandas as pd

X     = np.load("data/lstm_dataset/X_sequences.npy")
y_reg = np.load("data/lstm_dataset/y_labels.npy")
y_cls = np.load("data/lstm_dataset/y_behavior.npy")
le    = joblib.load("data/lstm_dataset/behavior_encoder.save")
df    = pd.read_csv("data/storm_tracks/storm_tracks.csv")

print(f"Sequences     : {X.shape}")
print(f"Behavior enc  : {list(le.classes_)}")
print(f"Date range    : {df.time.min()} to {df.time.max()}")
print(f"Lat range     : {df.centroid_lat.min():.1f} to {df.centroid_lat.max():.1f}")
print(f"Lon range     : {df.centroid_lon.min():.1f} to {df.centroid_lon.max():.1f}")
print(f"Total clusters: {df.cluster_id.nunique()}")

# Find interesting clusters — long tracks with varied behaviors
interesting = df[df["behavior"].isin(
    ["GROWING", "SPLITTING", "MERGING", "INTENSIFYING"]
)]
top = interesting.groupby("cluster_id").size().sort_values(
    ascending=False
).head(10)

print("\nMost interesting clusters (most varied behavior events):")
for cid, cnt in top.items():
    s = df[df["cluster_id"] == cid].sort_values("time")
    bhv = s["behavior"].value_counts().to_dict()
    print(f"\n  cluster {cid}: {len(s)} total frames, {cnt} active events")
    print(f"    behaviors : {bhv}")
    print(f"    lat/lon   : {s.centroid_lat.mean():.2f}N, {s.centroid_lon.mean():.2f}E")
    print(f"    area      : {s.area_km2.mean():.0f} km2 avg")
    print(f"    mean_tb   : {s.mean_tb.mean():.1f} K avg")
    print(f"    time      : {s.time.iloc[0]} to {s.time.iloc[-1]}")