import numpy as np
import joblib
import pandas as pd

X      = np.load("data/lstm_dataset/X_sequences.npy")
scaler = joblib.load("data/lstm_dataset/scaler.save")

# Find BoB sequence
best_idx = len(X) - 1
for i in range(len(X) - 1, -1, -1):
    seq = scaler.inverse_transform(X[i])
    lat = seq[-1, 0]
    lon = seq[-1, 1]
    if 5 <= lat <= 25 and 65 <= lon <= 100:
        best_idx = i
        print(f"Sequence index : {i}")
        print(f"Last obs       : {lat:.3f}N, {lon:.3f}E")
        break

# Find in storm_tracks.csv
df = pd.read_csv("data/storm_tracks/storm_tracks.csv")
seq_real   = scaler.inverse_transform(X[best_idx])
target_lat = round(float(seq_real[-1, 0]), 2)
target_lon = round(float(seq_real[-1, 1]), 2)

print(f"\nSearching for lat={target_lat}, lon={target_lon}")

match = df[
    (df["centroid_lat"].round(2) == target_lat) &
    (df["centroid_lon"].round(2) == target_lon)
]

if len(match) > 0:
    row = match.iloc[0]
    print(f"Storm ID  : {row['storm_id']}")
    print(f"Timestamp : {row['time']}")
    print(f"Speed     : {row['speed_kmh']:.1f} km/h")
    print("\nFull sequence timestamps:")
    storm = df[df["storm_id"] == row["storm_id"]].sort_values("time")
    print(storm[["storm_id","time","centroid_lat","centroid_lon","speed_kmh"]].to_string())
else:
    print("No exact match — finding nearest...")
    df["dist"] = ((df["centroid_lat"] - target_lat)**2 +
                  (df["centroid_lon"] - target_lon)**2)**0.5
    nearest = df.nsmallest(5, "dist")
    print(nearest[["storm_id","time","centroid_lat","centroid_lon","speed_kmh"]].to_string())
