import os
import sys
import argparse
import json
from datetime import datetime


# =====================================================
# PATH CONFIGURATION
# =====================================================

STATE_FILE = "output/pipeline_state.json"
RAW_DATA_PATH = "data/raw_npz"
TRACKS_FILE = "data/storm_tracks/storm_tracks.csv"


# =====================================================
# SAFE CASE SELECTION (NO EARLY CRASH)
# =====================================================

def safe_select_best_case():

    if not os.path.exists(TRACKS_FILE):
        return None, None

    try:
        from scripts.select_best_case import select_best_case
        return select_best_case()
    except Exception:
        return None, None


# =====================================================
# DATASET CHANGE DETECTION
# =====================================================

def dataset_changed():

    if not os.path.exists(RAW_DATA_PATH):
        return False

    current_files = sorted([
        f for f in os.listdir(RAW_DATA_PATH)
        if f.endswith(".npz")
    ])

    if not os.path.exists(STATE_FILE):
        return True

    with open(STATE_FILE, "r") as f:
        state = json.load(f)

    previous_files = state.get("file_list", [])

    return current_files != previous_files


def save_pipeline_state():

    if not os.path.exists(RAW_DATA_PATH):
        return

    current_files = sorted([
        f for f in os.listdir(RAW_DATA_PATH)
        if f.endswith(".npz")
    ])

    state = {
        "file_list": current_files,
        "last_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    os.makedirs("output", exist_ok=True)

    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)


# =====================================================
# STEP OUTPUT EXISTENCE CHECK
# =====================================================

def step_output_exists(step_num):

    CLUSTER_ID, DATE = safe_select_best_case()

    checks = {

        1: "data/predicted_masks",

        2: "data/clusters",

        3: TRACKS_FILE,

        # sequences depend on tracking
        4: TRACKS_FILE,

        5: f"output/pipeline_day/pipeline_{DATE}.json"
        if DATE else None,

        6: "output/scientific",

        7: f"output/visualization/cloud_s1s2_cluster_{CLUSTER_ID}.html"
        if CLUSTER_ID else None,

        8: f"output/actual/actual_{DATE}.html"
        if DATE else None,

        9: f"output/predicted/predicted_{DATE}.html"
        if DATE else None,

        10: f"output/scientific/results_{DATE}.json"
        if DATE else None,
    }

    path = checks.get(step_num)

    if not path:
        return False

    return os.path.exists(path)


# =====================================================
# STEP DEFINITIONS
# =====================================================

def step0_clean():
    print("\n Step 0: Cleaning previous outputs...")
    from scripts.clean_previous_data import clean_pipeline
    clean_pipeline()


def step1_predict_masks():
    print("\n Step 1: Predicting cloud masks using CNN...")
    from src.prediction.predict_masks import main
    main()


def step2_detect_clusters():
    print("\n Step 2: Detecting cloud clusters...")
    from src.detection.detect_clusters import main
    main()


def step3_track_clusters():
    print("\n Step 3: Tracking cloud clusters...")
    from src.tracking.track_clusters import main
    main()


def step4_create_sequences():
    print("\n Step 4: Creating sequences for BiGRU...")
    from src.lifecycle.create_sequences import main
    main()


def step5_run_pipeline_day():

    _, DATE = safe_select_best_case()

    if not DATE:
        print("Skipping Step 5 (tracking output not available yet)")
        return

    print(f"\n Step 5: Running full day pipeline for {DATE}...")

    from src.dashboard.cnnbigru.run_pipeline_day import main
    main()


def step6_evaluate():
    print("\n Step 6: Evaluating Stage 1 + Stage 2 combined pipeline...")
    from src.utils.evaluation.evaluate_s1_s2 import main
    main()


def step7_visualize_case_study():

    CLUSTER_ID, _ = safe_select_best_case()

    if not CLUSTER_ID:
        print("Skipping Step 7 (case study cluster unavailable)")
        return

    print("\n Step 7: Building case study validation dashboard...")

    from src.utils.evaluation.visualize_cloud_s1s2 import main
    main()


def step8_build_actual():

    _, DATE = safe_select_best_case()

    if not DATE:
        print("Skipping Step 8 (tracking output not available yet)")
        return

    print("\n Step 8: Building actual behavior dashboard...")

    from src.dashboard.cnnbigru.build_actual_dashboard import main
    main()


def step9_build_predicted():

    _, DATE = safe_select_best_case()

    if not DATE:
        print("Skipping Step 9 (tracking output not available yet)")
        return

    print("\n Step 9: Building predicted behavior dashboard...")

    from src.dashboard.cnnbigru.build_predicted_dashboard import main
    main()


def step10_export_results():

    _, DATE = safe_select_best_case()

    if not DATE:
        print("Skipping Step 10 (tracking output not available yet)")
        return

    print("\n Step 10: Exporting scientific JSON results...")

    from src.dashboard.cnnbigru.export_results import main
    main()


# =====================================================
# STEP REGISTRY
# =====================================================

STEPS = {

    0: ("Clean outputs", step0_clean),

    1: ("CNN predict masks", step1_predict_masks),

    2: ("Detect clusters", step2_detect_clusters),

    3: ("Track clusters", step3_track_clusters),

    4: ("Create sequences", step4_create_sequences),

    5: ("Run day pipeline", step5_run_pipeline_day),

    6: ("Evaluate S1+S2", step6_evaluate),

    7: ("Visualize case study", step7_visualize_case_study),

    8: ("Build actual dashboard", step8_build_actual),

    9: ("Build predicted dashboard", step9_build_predicted),

    10: ("Export scientific results", step10_export_results),
}


# =====================================================
# MAIN PIPELINE EXECUTION
# =====================================================

def main(clean=False, start_step=1, end_step=10, skip_steps=None):

    skip = set(skip_steps or [])

    dataset_modified = dataset_changed()

    # CLEANING CONTROL

    if clean:

        print("\nManual cleaning requested")
        step0_clean()

    elif dataset_modified:

        print("\nNew dataset detected → cleaning pipeline outputs")
        step0_clean()

    else:

        print("\nDataset unchanged → skipping cleaning")

    print("=" * 60)
    print("  INSAT-3D Cloud Monitoring Pipeline")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Steps   : {start_step} to {end_step}")

    if skip:
        skipped_names = [STEPS[s][0] for s in sorted(skip) if s in STEPS]
        print(f"  Skipping: {skipped_names}")

    print("=" * 60)

    failed_step = None

    for step_num in range(start_step, end_step + 1):

        if step_num not in STEPS:
            continue

        if step_num in skip:
            print(f"\n[SKIP] Step {step_num}: {STEPS[step_num][0]}")
            continue

        if not dataset_modified and step_output_exists(step_num):
            print(f"\n[AUTO-SKIP] Step {step_num}: already completed")
            continue

        step_name, step_fn = STEPS[step_num]

        print(f"\n{'='*60}")
        print(f"  STEP {step_num}/{end_step}: {step_name}")
        print(f"{'='*60}")

        t_start = datetime.now()

        try:

            step_fn()

            elapsed = (datetime.now() - t_start).seconds

            print(f"\n  [OK] Step {step_num} completed in {elapsed}s")

        except Exception as e:

            print(f"\n  [ERROR] Step {step_num} failed: {e}")

            failed_step = step_num

            break

    print("\n" + "=" * 60)

    if failed_step:

        print(f"Pipeline FAILED at step {failed_step}: {STEPS[failed_step][0]}")
        print(f"Fix the error and rerun from step {failed_step}:")
        print(f"python main_pipeline.py --start {failed_step}")

    else:

        print("Pipeline COMPLETED successfully")

        save_pipeline_state()

        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print("\nGenerated pipeline outputs:")

        CLUSTER_ID, DATE = safe_select_best_case()

        outputs = [

            ("Predicted masks",
            "data/predicted_masks"),

            ("Cluster CSV files",
            "data/clusters"),

            ("Storm tracking file",
            "data/storm_tracks/storm_tracks.csv"),

            ("Sequence dataset",
            "data/lstm_dataset"),

            ("Daily pipeline JSON",
            f"output/pipeline_day/pipeline_{DATE}.json"
            if DATE else None),

            ("Scientific JSON results",
            f"output/scientific/results_{DATE}.json"
            if DATE else None),

            ("Scientific compact JSON",
            f"output/scientific/results_{DATE}_compact.json"
            if DATE else None),

            ("Case study visualization",
            f"output/visualization/cloud_s1s2_cluster_{CLUSTER_ID}.html"
            if CLUSTER_ID else None),

            ("Actual dashboard",
            f"output/actual/actual_{DATE}.html"
            if DATE else None),

            ("Predicted dashboard",
            f"output/predicted/predicted_{DATE}.html"
            if DATE else None),
        ]

        for name, path in outputs:

            if path and os.path.exists(path):
                print(f"  {name:<28} → {os.path.abspath(path)}")
    print("=" * 60)


# =====================================================
# CLI ENTRY POINT
# =====================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="INSAT-3D Cloud Monitoring Pipeline"
    )

    parser.add_argument("--start", type=int, default=1)

    parser.add_argument("--end", type=int, default=10)

    parser.add_argument("--skip", type=int, nargs="+", default=[])

    parser.add_argument("--clean", action="store_true")

    parser.add_argument("--list", action="store_true")

    args = parser.parse_args()

    if args.list:

        print("\nAvailable steps:")

        for num, (name, _) in STEPS.items():
            print(f"{num:2d}. {name}")

        sys.exit(0)

    main(
        clean=args.clean,
        start_step=args.start,
        end_step=args.end,
        skip_steps=args.skip,
    )