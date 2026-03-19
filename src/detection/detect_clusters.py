import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import label, center_of_mass, find_objects
from concurrent.futures import ProcessPoolExecutor, as_completed

MASK_DIR   = "data/predicted_masks"
NPZ_DIR    = "data/raw_npz"
OUTPUT_DIR = "data/clusters"

MIN_CLUSTER_SIZE  = 150
PIXEL_AREA_KM2    = 4
DEEP_CONV_THRESH  = 210
COLD_CLOUD_THRESH = 235

os.makedirs(OUTPUT_DIR, exist_ok=True)


def fast_perimeter(binary_mask):
    m      = binary_mask.astype(np.uint8)
    top    = np.pad(m[:-1, :], ((0,1),(0,0)), mode="constant")
    bottom = np.pad(m[1:,  :], ((1,0),(0,0)), mode="constant")
    left   = np.pad(m[:, :-1], ((0,0),(0,1)), mode="constant")
    right  = np.pad(m[:,  1:], ((0,0),(1,0)), mode="constant")
    boundary = m & ((m != top) | (m != bottom) |
                    (m != left) | (m != right))
    return max(int(boundary.sum()), 1)


def compute_compactness(area_px, perimeter_px):
    return (4 * np.pi * area_px) / (perimeter_px ** 2 + 1e-6)


def process_file(mask_file):
    name     = os.path.basename(mask_file).replace("_mask.npy", "")
    npz_path = os.path.join(NPZ_DIR,    name + ".npz")
    out_file = os.path.join(OUTPUT_DIR, name + "_clusters.csv")

    if os.path.exists(out_file):
        return 0
    if not os.path.exists(npz_path):
        return 0

    try:
        mask = np.load(mask_file)
        data = np.load(npz_path)
        tb   = data["TIR1_TEMP"].astype(np.float32)
        lat  = data["Latitude"].astype(np.float32)
        lon  = data["Longitude"].astype(np.float32)
        time = str(data["time"])
    except Exception:
        return 0

    labeled, num_clusters = label(mask)
    if num_clusters == 0:
        return 0

    objects        = find_objects(labeled)
    centroids      = center_of_mass(mask, labeled, range(1, num_clusters+1))
    labeled_flat   = labeled.ravel()
    tb_flat        = tb.ravel()

    rows = []

    for cid, obj in enumerate(objects, start=1):
        if obj is None:
            continue

        local_mask  = (labeled[obj] == cid)
        pixel_count = int(local_mask.sum())

        if pixel_count < MIN_CLUSTER_SIZE:
            continue

        cy, cx = centroids[cid - 1]
        if np.isnan(cy) or np.isnan(cx):
            continue

        cy = int(cy);  cx = int(cx)
        if cy >= lat.shape[0] or cx >= lat.shape[1] or cy < 0 or cx < 0:
            continue

        cluster_tb = tb_flat[labeled_flat == cid]
        if len(cluster_tb) == 0:
            continue

        area_km2    = pixel_count * PIXEL_AREA_KM2
        perim       = fast_perimeter(local_mask)

        rows.append({
            "time"           : time,
            "cluster_id"     : cid,
            "centroid_lat"   : float(lat[cy, cx]),
            "centroid_lon"   : float(lon[cy, cx]),
            "pixel_count"    : pixel_count,
            "area_km2"       : area_km2,
            "radius_km"      : float(np.sqrt(area_km2 / np.pi)),
            "mean_tb"        : float(cluster_tb.mean()),
            "min_tb"         : float(cluster_tb.min()),
            "max_tb"         : float(cluster_tb.max()),
            "std_tb"         : float(cluster_tb.std()),
            "tb_p10"         : float(np.percentile(cluster_tb, 10)),
            "tb_p90"         : float(np.percentile(cluster_tb, 90)),
            "perimeter_px"   : perim,
            "compactness"    : float(compute_compactness(pixel_count, perim)),
            "convective_core": float((cluster_tb < DEEP_CONV_THRESH).sum()  / pixel_count),
            "cold_cloud_frac": float((cluster_tb < COLD_CLOUD_THRESH).sum() / pixel_count),
        })

    if rows:
        pd.DataFrame(rows).to_csv(out_file, index=False)

    return len(rows)


def main():
    mask_files = sorted(glob.glob(os.path.join(MASK_DIR, "*_mask.npy")))
    pending    = [f for f in mask_files
                  if not os.path.exists(
                      os.path.join(OUTPUT_DIR,
                                   os.path.basename(f).replace(
                                       "_mask.npy", "_clusters.csv")))]

    print(f"Total masks : {len(mask_files)}")
    print(f"Pending     : {len(pending)}")

    if not pending:
        print("All done.")
        return

    # ---- Check CPU cores ----
    import multiprocessing
    n_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {n_cores} CPU cores")

    total = 0
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = {executor.submit(process_file, f): f
                   for f in pending}
        for future in tqdm(as_completed(futures), total=len(pending)):
            try:
                total += future.result()
            except Exception as e:
                print(f"Error: {e}")

    print(f"\nCluster detection completed")
    print(f"Total clusters found: {total}")


if __name__ == "__main__":
    main()