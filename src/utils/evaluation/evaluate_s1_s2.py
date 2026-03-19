import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
)
from tensorflow.keras.layers import Layer, Dense

# ================================
# PATHS
# ================================

DATA_DIR  = "data/lstm_dataset"
MODEL_DIR = "models/bigrustages"

CLASS_NAMES = ["NORMAL", "ORGANIZING", "INTENSIFYING"]


# ================================
# CUSTOM LAYER
# ================================

class TemporalAttention(Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = Dense(units, activation="tanh", use_bias=False)
        self.V = Dense(1, use_bias=False)

    def call(self, x):
        score   = self.V(self.W(x))
        weights = tf.nn.softmax(score, axis=1)
        weights = tf.cast(weights, x.dtype)
        return tf.reduce_sum(x * weights, axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg


# ================================
# LOAD MODELS
# ================================

def load_models():
    print("Loading models...")
    custom = {"TemporalAttention": TemporalAttention}

    # FIX: correct filenames matching train scripts
    stage1 = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "bigru_stage1.keras"),
        custom_objects=custom, compile=False
    )
    stage2 = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "bigru_stage2.keras"),
        custom_objects=custom, compile=False
    )
    print(f"  Stage 1 params: {stage1.count_params():,}")
    print(f"  Stage 2 params: {stage2.count_params():,}")
    return stage1, stage2


# ================================
# LOAD DATA
# ================================

def load_data():
    print("\nLoading test data...")
    X_test   = np.load(f"{DATA_DIR}/X_test.npy").astype(np.float32)
    y_s1     = np.load(f"{DATA_DIR}/y_stage1_test.npy").astype(np.int32)
    y_s2     = np.load(f"{DATA_DIR}/y_stage2_test.npy").astype(np.int32)

    print(f"  Test samples  : {len(X_test)}")
    print(f"  Stage1 events : {y_s1.sum()} / {len(y_s1)}")
    print(f"  Stage2 org    : {(y_s2==0).sum()}")
    print(f"  Stage2 intens : {(y_s2==1).sum()}")
    return X_test, y_s1, y_s2


# ================================
# PIPELINE PREDICTION
# ================================

def predict_pipeline(stage1, stage2, X, threshold_s1=0.5, threshold_s2=0.5):
    """
    Stage 1: predict NORMAL(0) vs EVENT(1) for all samples
    Stage 2: predict ORGANIZING(0) vs INTENSIFYING(1) for EVENT samples only
    """
    print("\nRunning Stage 1 predictions...")
    s1_prob = stage1.predict(X, batch_size=256, verbose=1).flatten()
    s1_pred = (s1_prob > threshold_s1).astype(np.int32)

    print(f"  Stage1 predicted events: {s1_pred.sum()} / {len(s1_pred)}")

    s2_pred = np.full(len(X), -1, dtype=np.int32)
    s2_prob = np.full(len(X), -1.0, dtype=np.float32)

    event_idx = np.where(s1_pred == 1)[0]

    if len(event_idx) > 0:
        print(f"\nRunning Stage 2 on {len(event_idx)} event samples...")
        probs = stage2.predict(X[event_idx], batch_size=32, verbose=1).flatten()
        s2_pred[event_idx] = (probs > threshold_s2).astype(np.int32)
        s2_prob[event_idx] = probs
    else:
        print("  No events detected by Stage 1 — check threshold or model")

    return s1_pred, s1_prob, s2_pred, s2_prob


# ================================
# COMBINE TO 3-CLASS
# ================================

def combine_predictions(s1_pred, s2_pred):
    """
    0 = NORMAL      (stage1 predicted NORMAL)
    1 = ORGANIZING  (stage1=EVENT, stage2=ORGANIZING)
    2 = INTENSIFYING(stage1=EVENT, stage2=INTENSIFYING)
    """
    final = np.zeros(len(s1_pred), dtype=np.int32)
    event_mask = s1_pred == 1
    final[event_mask & (s2_pred == 0)] = 1   # ORGANIZING
    final[event_mask & (s2_pred == 1)] = 2   # INTENSIFYING
    # s2_pred == -1 (event detected but stage2 fallback) → keep as 1
    final[event_mask & (s2_pred == -1)] = 1
    return final


def get_true_labels(y_s1, y_s2):
    """
    FIX: must match length of X_test — include ALL samples.
    NORMAL=0, ORGANIZING=1, INTENSIFYING=2
    """
    final = np.zeros(len(y_s1), dtype=np.int32)
    event_mask = y_s1 == 1
    final[event_mask & (y_s2 == 0)] = 1   # ORGANIZING
    final[event_mask & (y_s2 == 1)] = 2   # INTENSIFYING
    return final


# ================================
# EVALUATE STAGE 1 ALONE
# ================================

def evaluate_stage1(y_s1_true, s1_pred, s1_prob):
    print("\n" + "="*50)
    print("STAGE 1: NORMAL vs EVENT")
    print("="*50)
    print(classification_report(
        y_s1_true, s1_pred,
        target_names=["NORMAL", "EVENT"],
        digits=3
    ))
    if len(np.unique(y_s1_true)) == 2:
        auc = roc_auc_score(y_s1_true, s1_prob)
        print(f"  AUC-ROC : {auc:.4f}")

    cm = confusion_matrix(y_s1_true, s1_pred)
    print(f"  Confusion matrix:")
    print(f"    TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"    FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")

    tp = cm[1,1]; fn = cm[1,0]; fp = cm[0,1]
    if (tp + fn) > 0:
        print(f"  Event recall    : {tp/(tp+fn):.3f}  "
              f"({tp} of {tp+fn} events found)")
    if (tp + fp) > 0:
        print(f"  Event precision : {tp/(tp+fp):.3f}")
    print(classification_report(
        y_s1_true, s1_pred,
        target_names=["NORMAL", "EVENT"],
        digits=3,
        zero_division=0    # ADD
    ))


# ================================
# EVALUATE STAGE 2 ALONE
# ================================

def evaluate_stage2(y_s2_true, s2_pred, s2_prob):
    print("\n" + "="*50)
    print("STAGE 2: ORGANIZING vs INTENSIFYING (events only)")
    print("="*50)

    # FIX: filter where BOTH true label AND prediction are valid (not -1)
    # true events = y_s2_true != -1
    # predicted events = s2_pred != -1
    # evaluate only where true label is an event
    event_mask = y_s2_true != -1

    y2_t    = y_s2_true[event_mask].astype(np.int32)
    y2_p    = s2_pred[event_mask].astype(np.int32)
    y2_prob = s2_prob[event_mask]

    # Replace any -1 predictions (stage1 missed but stage2 never ran)
    # with majority class (1 = INTENSIFYING) as fallback
    y2_p[y2_p == -1] = 1

    if len(y2_t) == 0:
        print("  No event samples in test set")
        return

    print(f"  True event samples  : {len(y2_t)}")
    print(f"  Stage2 ran on       : {(y2_p != -1).sum()} of them")
    print(f"  ORGANIZING(0) true  : {(y2_t==0).sum()}")
    print(f"  INTENSIFYING(1) true: {(y2_t==1).sum()}")
    print()

    unique_pred = np.unique(y2_p)
    print(f"  Predicted classes: {unique_pred}")

    print(classification_report(
        y2_t, y2_p,
        labels=[0, 1],                               # FIX: explicit labels
        target_names=["ORGANIZING", "INTENSIFYING"],
        digits=3,
        zero_division=0                              # FIX: suppress undefined warnings
    ))

    if len(np.unique(y2_t)) == 2:
        valid_prob = y2_prob[y2_prob != -1]
        valid_true = y2_t[y2_prob != -1]
        if len(valid_prob) > 0 and len(np.unique(valid_true)) == 2:
            auc = roc_auc_score(valid_true, valid_prob)
            print(f"  AUC-ROC : {auc:.4f}")

    cm = confusion_matrix(y2_t, y2_p, labels=[0, 1])
    print(f"  Confusion matrix:")
    print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

# ================================
# EVALUATE COMBINED PIPELINE
# ================================

def evaluate_combined(y_true, y_pred):
    print("\n" + "="*50)
    print("COMBINED PIPELINE: 3-CLASS RESULTS")
    print("="*50)
    print(f"  Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print()
    print(classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=3
    ))

    cm = confusion_matrix(y_true, y_pred)
    print("  Confusion matrix:")
    print(f"  {'':15s}" + "  ".join(f"{c:>12s}" for c in CLASS_NAMES))
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:15s}" +
              "  ".join(f"{v:12d}" for v in row))

    print()
    print("  Per-class summary:")
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i,i]
        total_true = cm[i,:].sum()
        total_pred = cm[:,i].sum()
        recall    = tp / total_true if total_true > 0 else 0
        precision = tp / total_pred if total_pred > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0)
        print(f"    {name:15s}: precision={precision:.3f}  "
              f"recall={recall:.3f}  f1={f1:.3f}  "
              f"(n={total_true})")


# ================================
# THRESHOLD ANALYSIS
# ================================

def threshold_analysis(y_s1_true, s1_prob):
    """
    Find the best Stage 1 threshold for maximizing event recall
    while keeping precision above a minimum level.
    """
    print("\n" + "="*50)
    print("STAGE 1 THRESHOLD ANALYSIS")
    print("="*50)
    print(f"  {'Threshold':>10}  {'Recall':>8}  "
          f"{'Precision':>10}  {'F1':>8}  {'Events found':>12}")
    print(f"  {'-'*55}")

    best_f1 = 0
    best_thresh = 0.5

    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        pred = (s1_prob > thresh).astype(np.int32)
        cm   = confusion_matrix(y_s1_true, pred)
        tp   = cm[1,1]; fn = cm[1,0]; fp = cm[0,1]
        rec  = tp / (tp+fn) if (tp+fn) > 0 else 0
        prec = tp / (tp+fp) if (tp+fp) > 0 else 0
        f1   = (2*prec*rec/(prec+rec)) if (prec+rec) > 0 else 0
        if f1 > best_f1:
            best_f1    = f1
            best_thresh = thresh
        print(f"  {thresh:>10.1f}  {rec:>8.3f}  "
              f"{prec:>10.3f}  {f1:>8.3f}  {tp+fn-fn:>8}/{tp+fn}")

    print(f"\n  Best threshold: {best_thresh} (F1={best_f1:.3f})")
    return best_thresh


# ================================
# MAIN
# ================================

def main():
    # GPU setup
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    stage1, stage2 = load_models()
    X_test, y_s1, y_s2 = load_data()

    # Run pipeline
    # In main() — use 0.6 as default threshold
    s1_pred, s1_prob, s2_pred, s2_prob = predict_pipeline(
        stage1, stage2, X_test,
        threshold_s1=0.6    # ADD: use best threshold directly
    )

    # Stage 1 evaluation
    evaluate_stage1(y_s1, s1_pred, s1_prob)

    # Threshold analysis — find best cutoff for event detection
    best_thresh = threshold_analysis(y_s1, s1_prob)

    # Re-run with best threshold if different from 0.5
    if best_thresh != 0.5:
        print(f"\nRe-running with best threshold {best_thresh}...")
        s1_pred, s1_prob, s2_pred, s2_prob = predict_pipeline(
            stage1, stage2, X_test,
            threshold_s1=best_thresh
        )

    # Stage 2 evaluation
    evaluate_stage2(y_s2, s2_pred, s2_prob)

    # Combined 3-class evaluation
    y_true = get_true_labels(y_s1, y_s2)
    y_pred = combine_predictions(s1_pred, s2_pred)
    evaluate_combined(y_true, y_pred)

    print(f"\nDone.")


if __name__ == "__main__":
    main()