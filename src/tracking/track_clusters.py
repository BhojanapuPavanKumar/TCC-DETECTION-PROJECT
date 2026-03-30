import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import cKDTree

CLUSTER_DIR           = "data/clusters"
OUTPUT_DIR            = "data/storm_tracks"
FRAME_INTERVAL_HOURS  = 0.5
MAX_MATCH_DISTANCE_KM = 150
MERGE_DISTANCE_KM     = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# VECTORIZED HAVERSINE
# ================================

def haversine_vec(lat1, lon1, lat2_arr, lon2_arr):
    """
    Vectorized haversine — compute distances from one point
    to an array of points simultaneously. Much faster than
    calling haversine() in a loop.
    """
    R    = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2_arr)
    lon2 = np.radians(lon2_arr)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a    = (np.sin(dlat/2)**2 +
            np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def haversine_scalar(lat1, lon1, lat2, lon2):
    R    = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a    = (np.sin(dlat/2)**2 +
            np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def direction_vec(lat1, lon1, lat2, lon2):
    dlon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    y    = np.sin(dlon) * np.cos(lat2)
    x    = (np.cos(lat1) * np.sin(lat2) -
            np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


# ================================
# FAST FRAME LOADER
# ================================

def load_frames():
    files = sorted(glob.glob(os.path.join(CLUSTER_DIR, "*_clusters.csv")))
    print(f"  Loading {len(files)} cluster files...")

    dtype_map = {
        "centroid_lat"   : np.float32,
        "centroid_lon"   : np.float32,
        "area_km2"       : np.float32,
        "mean_tb"        : np.float32,
        "min_tb"         : np.float32,
        "max_tb"         : np.float32,
        "std_tb"         : np.float32,
        "tb_p10"         : np.float32,
        "tb_p90"         : np.float32,
        "compactness"    : np.float32,
        "convective_core": np.float32,
        "cold_cloud_frac": np.float32,
        "pixel_count"    : np.int32,
        "perimeter_px"   : np.int32,
    }

    frames = []
    for f in tqdm(files, desc="Loading CSVs"):
        try:
            df = pd.read_csv(f, dtype=dtype_map)
            if len(df) == 0:
                continue
            df = df.dropna(subset=["centroid_lat", "centroid_lon",
                                   "area_km2", "mean_tb"])
            if len(df) == 0:
                continue
            frames.append(df.reset_index(drop=True))
        except Exception:
            continue
    return frames

# ================================
# VECTORIZED BEHAVIOR DETECTION
# ================================

def detect_behaviors_vectorized(curr_df, prev_df,
                                 prev_lats, prev_lons,
                                 prev_tree, curr_tree,
                                 matched_prev_idx,
                                 is_new_mask):
    """
    Detect behavior for ALL current clusters at once
    instead of one at a time.
    Returns arrays of behavior strings, merge_src, split_dst.
    """
    n         = len(curr_df)
    behaviors = np.array(["NEW"] * n, dtype=object)
    merge_src = np.full(n, -1, dtype=np.int64)
    split_dst = np.full(n, -1, dtype=np.int64)

    curr_lats = curr_df["centroid_lat"].values
    curr_lons = curr_df["centroid_lon"].values
    curr_areas = curr_df["area_km2"].values
    curr_tbs   = curr_df["mean_tb"].values

    # Only process non-new clusters
    active = np.where(~is_new_mask)[0]
    if len(active) == 0:
        return behaviors, merge_src, split_dst

    prev_areas = prev_df["area_km2"].values
    prev_tbs   = prev_df["mean_tb"].values

    # ---- Batch area/TB change for all active clusters ----
    pidx         = matched_prev_idx[active]
    area_now     = prev_areas[pidx]
    area_next    = curr_areas[active]
    tb_now       = prev_tbs[pidx]
    tb_next      = curr_tbs[active]

    area_change_pct = (area_next - area_now) / (area_now + 1e-6) * 100
    tb_change       = tb_next - tb_now

    # ---- Classify by physics ----
    for i, ci in enumerate(active):
        acp = area_change_pct[i]
        tbc = tb_change[i]

        if   acp >  25 :  behaviors[ci] = "GROWING"
        elif acp < -25 :  behaviors[ci] = "SHRINKING"
        elif tbc < -5  :  behaviors[ci] = "INTENSIFYING"
        else           :  behaviors[ci] = "STABLE"

    # ---- Detect MERGE: find curr clusters with 2+ nearby prev clusters ----
    # Query k=3 neighbours for all curr points at once
    k_prev = min(3, len(prev_df))
    dists_p, idx_p = prev_tree.query(
        np.column_stack([curr_lats[active], curr_lons[active]]),
        k=k_prev, workers=-1   # use all cores
    )
    if k_prev == 1:
        dists_p = dists_p[:, np.newaxis]
        idx_p   = idx_p[:,   np.newaxis]

    for i, ci in enumerate(active):
        # Convert KDTree distances to km for 2nd nearest
        if k_prev >= 2:
            d2_km = haversine_scalar(
                curr_lats[ci], curr_lons[ci],
                prev_lats[idx_p[i, 1]], prev_lons[idx_p[i, 1]]
            )
            if d2_km < MERGE_DISTANCE_KM:
                merge_src[ci]   = int(idx_p[i, 1])
                behaviors[ci]   = "MERGING"

    # ---- Detect SPLIT: find prev clusters with 2+ nearby curr clusters ----
    k_curr = min(3, len(curr_df))
    prev_active_lats = prev_lats[matched_prev_idx[active]]
    prev_active_lons = prev_lons[matched_prev_idx[active]]

    dists_c, idx_c = curr_tree.query(
        np.column_stack([prev_active_lats, prev_active_lons]),
        k=k_curr, workers=-1
    )
    if k_curr == 1:
        dists_c = dists_c[:, np.newaxis]
        idx_c   = idx_c[:,   np.newaxis]

    for i, ci in enumerate(active):
        if behaviors[ci] == "MERGING":
            continue   # already classified
        if k_curr >= 2:
            d2_km = haversine_scalar(
                prev_active_lats[i], prev_active_lons[i],
                curr_lats[idx_c[i, 1]], curr_lons[idx_c[i, 1]]
            )
            if d2_km < MERGE_DISTANCE_KM:
                split_dst[ci]  = int(idx_c[i, 1])
                behaviors[ci]  = "SPLITTING"

    return behaviors, merge_src, split_dst


# ================================
# MAIN TRACKING LOOP
# ================================

def track(frames):
    cluster_counter = 0
    active_tracks   = {}
    all_results     = []

    prev_frame = frames[0]
    prev_lats  = prev_frame["centroid_lat"].values
    prev_lons  = prev_frame["centroid_lon"].values
    prev_tree  = cKDTree(np.column_stack([prev_lats, prev_lons]))

    # Assign IDs to first frame
    for i in range(len(prev_frame)):
        cluster_counter += 1
        active_tracks[(0, i)] = cluster_counter

    for fi in tqdm(range(1, len(frames)), desc="Tracking"):
        curr      = frames[fi]
        curr_time = curr["time"].iloc[0]
        curr_lats = curr["centroid_lat"].values
        curr_lons = curr["centroid_lon"].values
        curr_tree = cKDTree(np.column_stack([curr_lats, curr_lons]))
        n_curr    = len(curr)

        # ---- Batch nearest-neighbour query for ALL curr clusters ----
        dists, prev_indices = prev_tree.query(
            np.column_stack([curr_lats, curr_lons]),
            k=1, workers=-1
        )

        # ---- Vectorized haversine for all matches ----
        matched_prev_lats = prev_lats[prev_indices]
        matched_prev_lons = prev_lons[prev_indices]
        dist_km = haversine_vec(
            curr_lats, curr_lons,
            matched_prev_lats, matched_prev_lons
        )

        # ---- Assign cluster IDs ----
        cluster_ids = np.zeros(n_curr, dtype=np.int64)
        is_new      = dist_km > MAX_MATCH_DISTANCE_KM

        for ci in range(n_curr):
            if is_new[ci]:
                cluster_counter    += 1
                cluster_ids[ci]     = cluster_counter
            else:
                key = (fi - 1, int(prev_indices[ci]))
                cid = active_tracks.get(key, None)
                if cid is None:
                    cluster_counter += 1
                    cid              = cluster_counter
                cluster_ids[ci] = cid
            active_tracks[(fi, ci)] = cluster_ids[ci]

        # ---- Vectorized behavior detection ----
        behaviors, merge_srcs, split_dsts = detect_behaviors_vectorized(
            curr, prev_frame,
            prev_lats, prev_lons,
            prev_tree, curr_tree,
            prev_indices, is_new
        )

        # ---- Vectorized physics ----
        speeds   = dist_km / FRAME_INTERVAL_HOURS
        dir_degs = direction_vec(
            matched_prev_lats, matched_prev_lons,
            curr_lats, curr_lons
        )
        prev_areas = prev_frame["area_km2"].values[prev_indices]
        prev_tbs   = prev_frame["mean_tb"].values[prev_indices]
        curr_areas = curr["area_km2"].values
        curr_tbs   = curr["mean_tb"].values

        area_changes = curr_areas - prev_areas
        temp_changes = curr_tbs   - prev_tbs
        growths      = np.where(
            prev_areas > 0,
            area_changes / prev_areas,
            0.0
        )

        # ---- Build result rows for this frame ----
        frame_results = {
            "cluster_id"     : cluster_ids,
            "time"           : curr_time,
            "centroid_lat"   : curr_lats,
            "centroid_lon"   : curr_lons,
            "area_km2"       : curr_areas,
            "mean_tb"        : curr_tbs,
            "distance_km"    : dist_km,
            "speed_kmh"      : speeds,
            "direction_deg"  : dir_degs,
            "area_change"    : area_changes,
            "temp_change"    : temp_changes,
            "growth"         : growths,
            "behavior"       : behaviors,
            "merge_src"      : merge_srcs,
            "split_dst"      : split_dsts,
            "is_new"         : is_new,

            # ---- Pass through new cluster features ----
            "min_tb"         : curr["min_tb"].values,
            "max_tb"         : curr["max_tb"].values,
            "std_tb"         : curr["std_tb"].values,
            "tb_p10"         : curr["tb_p10"].values,
            "tb_p90"         : curr["tb_p90"].values,
            "pixel_count"    : curr["pixel_count"].values,
            "radius_km"      : curr["radius_km"].values,
            "perimeter_px"   : curr["perimeter_px"].values,
            "compactness"    : curr["compactness"].values,
            "convective_core": curr["convective_core"].values,
            "cold_cloud_frac": curr["cold_cloud_frac"].values,
        }
        all_results.append(pd.DataFrame(frame_results))

        prev_frame = curr
        prev_lats  = curr_lats
        prev_lons  = curr_lons
        prev_tree  = curr_tree

    return pd.concat(all_results, ignore_index=True)


# ================================
# MAIN
# ================================

def main():

    out_path = os.path.join(OUTPUT_DIR, "storm_tracks.csv")

    if os.path.exists(out_path):
        print("Tracking already completed — skipping")
        return
    frames = load_frames()
    print(f"Frames loaded: {len(frames)}")

    df = track(frames)

    print(f"\nCloud monitoring summary:")
    print(f"  Total observations : {len(df)}")
    print(f"  Unique clusters    : {df['cluster_id'].nunique()}")

    lengths = df.groupby("cluster_id").size()
    print(f"\n  Cluster lifetime distribution:")
    print(f"    1-2  frames : {(lengths<=2).sum()}")
    print(f"    3-5  frames : {((lengths>=3)&(lengths<=5)).sum()}")
    print(f"    6-10 frames : {((lengths>=6)&(lengths<=10)).sum()}")
    print(f"    11-20 frames: {((lengths>=11)&(lengths<=20)).sum()}")
    print(f"    20+  frames : {(lengths>20).sum()}")

    print(f"\n  Behavior distribution:")
    for beh, cnt in df["behavior"].value_counts().items():
        print(f"    {beh:12s}: {cnt:6d} ({100*cnt/len(df):.1f}%)")

    out_path = os.path.join(OUTPUT_DIR, "storm_tracks.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()