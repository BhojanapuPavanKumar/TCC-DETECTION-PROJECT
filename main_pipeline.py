import os
import sys
import argparse
from datetime import datetime

# ================================
# STEP 0: Clean old outputs
# ================================

def step0_clean():
    print("\n Step 0: Cleaning previous outputs...")
    from scripts.clean_previous_data import clean_pipeline
    clean_pipeline()


# ================================
# STEP 1: CNN - Predict Masks
# ================================

def step1_predict_masks():
    print("\n Step 1: Predicting cloud masks using CNN...")
    from src.prediction.predict_masks import main as predict_masks_main
    predict_masks_main()


# ================================
# STEP 2: Detect Clusters
# ================================

def step2_detect_clusters():
    print("\n Step 2: Detecting cloud clusters...")
    from src.detection.detect_clusters import main as detect_clusters_main
    detect_clusters_main()


# ================================
# STEP 3: Track Clusters
# ================================

def step3_track_clusters():
    print("\n Step 3: Tracking cloud clusters...")
    from src.tracking.track_clusters import main as track_clusters_main
    track_clusters_main()


# ================================
# STEP 4: Create Sequences
# ================================

def step4_create_sequences():
    print("\n Step 4: Creating sequences for BiGRU...")
    from src.lifecycle.create_sequences import main as create_sequences_main
    create_sequences_main()


# ================================
# STEP 5A: Train Stage 1 (NORMAL vs EVENT)
# ================================

# def step5a_train_stage1():
#     print("\n Step 5A: Training Stage 1 — NORMAL vs EVENT...")
#     from src.training.train_gru_stage1 import train as train_stage1
#     train_stage1()


# ================================
# STEP 5B: Train Stage 2 (ORGANIZING vs INTENSIFYING)
# ================================

# def step5b_train_stage2():
#     print("\n Step 5B: Training Stage 2 — ORGANIZING vs INTENSIFYING...")
#     from src.training.train_gru_stage2 import train as train_stage2
#     train_stage2()


# ================================
# STEP 6: Run Day Pipeline (predictions on 1 day)
# ================================

def step6_run_pipeline_day(date="2025-01-05"):
    print(f"\n Step 6: Running full day pipeline for {date}...")
    from src.dashboard.run_pipeline_day import main as pipeline_day_main
    pipeline_day_main()


# ================================
# STEP 7: Evaluate Stage 1 + Stage 2
# ================================

def step7_evaluate():
    print("\n Step 7: Evaluating Stage 1 + Stage 2 combined pipeline...")
    from src.utils.evaluation.evaluate_s1_s2 import main as evaluate_main
    evaluate_main()


# ================================
# STEP 8: Visualize Cloud S1+S2 (case study)
# ================================

def step8_visualize_case_study():
    print("\n Step 8: Building case study validation dashboard...")
    from src.utils.evaluation.visualize_cloud_s1s2 import main as viz_main
    viz_main()


# ================================
# STEP 9: Build Actual Dashboard
# ================================

def step9_build_actual():
    print("\n Step 9: Building actual behavior dashboard...")
    from src.dashboard.build_actual_dashboard import main as actual_main
    actual_main()


# ================================
# STEP 10: Build Predicted Dashboard
# ================================

def step10_build_predicted():
    print("\n Step 10: Building predicted behavior dashboard...")
    from src.dashboard.build_predicted_dashboard import main as predicted_main
    predicted_main()


# ================================
# STEP 11: Export Scientific Results
# ================================

def step11_export_results():
    print("\n Step 11: Exporting scientific JSON results...")
    from src.dashboard.export_results import main as export_main
    export_main()


# ================================
# STEP REGISTRY
# ================================

STEPS = {
    0 : ("Clean outputs",             step0_clean),
    1 : ("CNN predict masks",         step1_predict_masks),
    2 : ("Detect clusters",           step2_detect_clusters),
    3 : ("Track clusters",            step3_track_clusters),
    4 : ("Create sequences",          step4_create_sequences),
    5 : ("Run day pipeline",          step6_run_pipeline_day),
    6 : ("Evaluate S1+S2",            step7_evaluate),
    7 : ("Visualize case study",      step8_visualize_case_study),
    8 : ("Build actual dashboard",    step9_build_actual),
    9 : ("Build predicted dashboard", step10_build_predicted),
    10: ("Export scientific results", step11_export_results),
}

# ================================
# MAIN PIPELINE
# ================================

def main(
    clean=False,
    start_step=1,
    end_step=12,
    skip_steps=None,
):
    skip = set(skip_steps or [])
    if not clean:
        skip.add(0)

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
            print(f"\n  [SKIP] Step {step_num}: {STEPS[step_num][0]}")
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
        print(f"  Pipeline FAILED at step {failed_step}: {STEPS[failed_step][0]}")
        print(f"  Fix the error and rerun from step {failed_step}:")
        print(f"    python main_pipeline.py --start {failed_step}")
    else:
        print(f"  Pipeline COMPLETED successfully")
        print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("  Output files:")
        print("    output/pipeline_day/pipeline_2025-01-05.json")
        print("    output/actual/actual_2025-01-05.html")
        print("    output/predicted/predicted_2025-01-05.html")
        print("    output/scientific/results_2025-01-05.json")
        print("    output/visualization/cloud_s1s2_cluster_6846.html")
    print("=" * 60)


# ================================
# ENTRY POINT WITH CLI ARGS
# ================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="INSAT-3D Cloud Monitoring Pipeline"
    )
    parser.add_argument(
        "--start", type=int, default=1,
        help="Start from step N (default: 1)"
    )
    parser.add_argument(
        "--end", type=int, default=10,      # FIX: default 10 not 12 (only 10 steps now)
        help="End at step N (default: 10)"
    )
    parser.add_argument(
        "--skip", type=int, nargs="+", default=[],
        help="Skip specific step numbers e.g. --skip 2 3"
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Run step 0 to clean previous outputs"
    )
    parser.add_argument(
        "--list", action="store_true",      # FIX: was --end duplicate
        help="List all steps and exit"
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable steps:")
        for num, (name, _) in STEPS.items():
            print(f"  {num:2d}. {name}")
        sys.exit(0)

    main(
        clean=args.clean,
        start_step=args.start,
        end_step=args.end,
        skip_steps=args.skip,
    )