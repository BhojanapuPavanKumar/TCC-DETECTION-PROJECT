"""
find_best_day.py
Analyzes storm_tracks.csv to find the most scientifically
interesting days for pipeline analysis.
Ranks days by event richness, cluster diversity and activity.
"""

import pandas as pd
import numpy as np
import os

TRACKS = "data/storm_tracks/storm_tracks.csv"


def main():
    print("Loading storm tracks...")
    df = pd.read_csv(TRACKS)
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date

    dates = sorted(df["date"].unique())
    print(f"Available dates: {len(dates)}")
    print(f"Range: {min(dates)} to {max(dates)}\n")

    # ---- Score each day ----
    records = []

    for date in dates:
        day = df[df["date"] == date]

        n_clusters    = day["cluster_id"].nunique()
        n_frames      = day["time"].nunique()
        n_obs         = len(day)

        # Cluster lengths
        lengths       = day.groupby("cluster_id").size()
        long_clusters = (lengths >= 9).sum()   # enough for prediction

        # Behavior counts
        bev = day["behavior"].value_counts().to_dict()
        n_intensifying = bev.get("INTENSIFYING", 0)
        n_splitting    = bev.get("SPLITTING", 0)
        n_merging      = bev.get("MERGING", 0)
        n_growing      = bev.get("GROWING", 0)
        n_stable       = bev.get("STABLE", 0)
        n_events       = n_intensifying + n_splitting + n_merging

        # Activity ratio — non-stable / total
        activity_ratio = (n_obs - n_stable) / n_obs if n_obs > 0 else 0

        # Score — weighted sum of scientifically interesting features
        score = (
            n_events       * 10 +   # rare events are most valuable
            n_intensifying *  5 +   # deep convection
            n_splitting    *  8 +   # organizational events
            n_merging      *  8 +
            n_growing      *  2 +
            long_clusters  *  3 +   # clusters long enough to predict
            activity_ratio * 50     # overall cloud activity
        )

        records.append({
            "date"           : date,
            "n_frames"       : n_frames,
            "n_clusters"     : n_clusters,
            "long_clusters"  : long_clusters,
            "n_events"       : n_events,
            "n_intensifying" : n_intensifying,
            "n_splitting"    : n_splitting,
            "n_merging"      : n_merging,
            "n_growing"      : n_growing,
            "activity_ratio" : round(activity_ratio, 3),
            "score"          : round(score, 1),
        })

    results = pd.DataFrame(records).sort_values(
        "score", ascending=False
    ).reset_index(drop=True)

    # ---- Print top 15 ----
    print("=" * 90)
    print(f"{'Rank':<5} {'Date':<12} {'Score':<8} {'Events':<8} "
          f"{'Intens':<8} {'Split':<7} {'Merge':<7} "
          f"{'Long':<7} {'Activity':<10} {'Frames'}")
    print("=" * 90)

    for i, row in results.head(15).iterrows():
        marker = " <-- BEST" if i == 0 else (" <-- JAN 5 (current)" if str(row["date"]) == "2025-01-05" else "")
        print(f"{i+1:<5} {str(row['date']):<12} {row['score']:<8.1f} "
              f"{row['n_events']:<8} {row['n_intensifying']:<8} "
              f"{row['n_splitting']:<7} {row['n_merging']:<7} "
              f"{row['long_clusters']:<7} {row['activity_ratio']:<10} "
              f"{row['n_frames']}{marker}")

    print("=" * 90)

    # ---- Best day details ----
    best = results.iloc[0]
    print(f"\nBest day: {best['date']}")
    print(f"  Score          : {best['score']:.1f}")
    print(f"  Total events   : {best['n_events']}")
    print(f"    INTENSIFYING : {best['n_intensifying']}")
    print(f"    SPLITTING    : {best['n_splitting']}")
    print(f"    MERGING      : {best['n_merging']}")
    print(f"  Long clusters  : {best['long_clusters']} (9+ frames)")
    print(f"  Activity ratio : {best['activity_ratio']:.1%}")
    print(f"  Frames         : {best['n_frames']}")

    # ---- Show top cluster on best day ----
    best_day_df = df[df["date"] == best["date"]]
    lengths     = best_day_df.groupby("cluster_id").size()
    events_per  = best_day_df[
        best_day_df["behavior"].isin(["INTENSIFYING","SPLITTING","MERGING"])
    ].groupby("cluster_id").size().sort_values(ascending=False)

    print(f"\nTop 5 clusters on {best['date']}:")
    for cid, cnt in events_per.head(5).items():
        s    = best_day_df[best_day_df["cluster_id"] == cid]
        bhv  = s["behavior"].value_counts().to_dict()
        print(f"  cluster {cid}: {len(s)} frames | {cnt} events | "
              f"lat={s['centroid_lat'].mean():.1f} "
              f"lon={s['centroid_lon'].mean():.1f}")
        print(f"    {bhv}")

    # ---- Run command ----
    print(f"\nTo run pipeline for best day:")
    print(f"  python main_pipeline.py --date {best['date']}")
    print(f"\nTo run pipeline for current day (Jan 5):")
    print(f"  python main_pipeline.py --date 2025-01-05")

    # ---- Save ranking to CSV ----
    out = "output/best_days.csv"
    os.makedirs("output", exist_ok=True)
    results.to_csv(out, index=False)
    print(f"\nFull ranking saved: {out}")


if __name__ == "__main__":
    main()