import pandas as pd

TRACKS = "data/storm_tracks/storm_tracks.csv"
SEQ_LEN = 8

EVENT_TYPES = ["INTENSIFYING", "MERGING", "SPLITTING"]


def cluster_score(track):
    """
    Scientific scoring function for selecting best visualization cluster
    """

    behaviors = track["behavior"].value_counts().to_dict()

    n_frames = len(track)
    n_events = sum(behaviors.get(x, 0) for x in EVENT_TYPES)

    has_intensifying = behaviors.get("INTENSIFYING", 0) > 0
    has_organizing = (
        behaviors.get("MERGING", 0) > 0
        or behaviors.get("SPLITTING", 0) > 0
    )

    diversity = len(behaviors)

    score = (
        n_frames
        + 6 * n_events
        + 10 * has_intensifying
        + 10 * has_organizing
        + 2 * diversity
    )

    return score


def select_best_case():

    df = pd.read_csv(TRACKS)
    df["time"] = pd.to_datetime(df["time"])

    best_cluster = None
    best_date = None
    best_score = -1

    grouped = df.groupby("cluster_id")

    for cid, track in grouped:

        if len(track) < SEQ_LEN + 1:
            continue

        score = cluster_score(track)

        if score > best_score:
            best_score = score
            best_cluster = cid

            # choose dominant date of that cluster
            best_date = (
                track["time"]
                .dt.date
                .mode()
                .iloc[0]
            )

    return int(best_cluster), str(best_date)


def main():

    cid, date = select_best_case()

    print("\nBest automatic case-study selection:")
    print(f"Cluster ID : {cid}")
    print(f"Best date  : {date}")


if __name__ == "__main__":
    main()