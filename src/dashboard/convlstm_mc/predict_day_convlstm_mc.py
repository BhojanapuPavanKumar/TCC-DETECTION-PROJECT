"""
predict_day_convlstm_mc.py  (v2 — all rendering bugs fixed)
============================================================
Multi-Class ConvLSTM (Method 3) — one-day TCC lifecycle detection.

Bugs fixed vs v1
----------------
  BUG 1  smooth_class_blend values too small (same centre-pixel + wide-sigma issue)
         → fills ENTIRE patch area per class before smoothing with narrow sigma
  BUG 2  Class bars computed from smoothed prob → could mismatch badge values
         → all stats computed from pred_map (patch labels) directly
  BUG 3  Lifecycle overlays invisible — class colours blended into BT colours
         → using high-alpha saturated colours that contrast with ALL BT backgrounds
  BUG 4  BT colormap vmin/vmax did not match real INSAT range
         → MOSDAC-matched colormap vmin=175, vmax=345 K
  BUG 5  No temperature reference scale visible
         → BT colorbar with K ticks + 245 K TCC threshold marker added

Classes: 0=NON-TCC  1=ORGANIZING  2=INTENSIFYING  3=MATURE  4=DISSIPATING

Output
------
  output/convlstm_mc/animation_<date>.html

Usage
-----
  python src/dashboard/convlstm_mc/predict_day_convlstm_mc.py
  python src/dashboard/convlstm_mc/predict_day_convlstm_mc.py --date 20250206

Requirements:  numpy  matplotlib  scipy
Optional:      tensorflow >= 2.10
"""

import os, sys, glob, re, json, io, base64, argparse, warnings
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")

try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    TF_OK = True
except ImportError:
    TF_OK = False

# ══════════════════════════════════════════════════════════════════════════
# CONFIG  — mirrors create_seq_convlstm.py
# ══════════════════════════════════════════════════════════════════════════
SEQ_LEN      = 5
PATCH_SIZE   = 64
PATCH_STRIDE = 64
MAX_NAN_FRAC = 0.50

THRESHOLDS = dict(
    non_tcc_min_tb  = 270,
    non_tcc_soft_tb = 255,
    intensify_trend = -6,
    organize_trend  = -2,
    mature_max_tb   = 230,
    dissipate_trend =  2,
)

RAW_DIR    = "data/raw_npz"
MODEL_PATH = "models/convlstm_mc/best_convlstm_mc.keras"
OUT_DIR    = "output/convlstm_mc"
DEF_EXTENT = [40.0, 120.0, -60.0, 60.0]
FRAME_DPI  = 110

N_CLASSES   = 5
CLASS_NAMES = ["NON-TCC","ORGANIZING","INTENSIFYING","MATURE","DISSIPATING"]

# ── FIX BUG 3: high-saturation overlay colours visible on any BT background
# Each colour chosen to contrast strongly with the blue/cyan/orange BT map
CLASS_HEX = [
    "#90CAF9",   # 0 NON-TCC     — pale blue (not shown on map, kept for legend)
    "#00FF88",   # 1 ORGANIZING  — neon spring green
    "#FFFF00",   # 2 INTENSIFYING— bright yellow
    "#FF1493",   # 3 MATURE      — deep pink / hot pink (distinct from all BT cols)
    "#00FFFF",   # 4 DISSIPATING — cyan (contrasts with orange warm and dark blue cold)
]
# RGB float tuples for numpy overlay
CLASS_RGB = [
    None,                   # 0 NON-TCC — no overlay
    (0.00, 1.00, 0.53),     # 1 ORGANIZING  — neon green
    (1.00, 1.00, 0.00),     # 2 INTENSIFYING— yellow
    (1.00, 0.08, 0.58),     # 3 MATURE      — hot pink
    (0.00, 1.00, 1.00),     # 4 DISSIPATING — cyan
]

# ── FIX BUG 4: MOSDAC-matched BT colormap + correct vmin/vmax ────────────
BT_CMAP = LinearSegmentedColormap.from_list("mosdac_ir", [
    "#CC00FF",   # 175 K  bright magenta
    "#7B00D4",   # 191 K  violet
    "#0000FF",   # 208 K  blue
    "#00BFFF",   # 228 K  deep sky blue
    "#00FF80",   # 244 K  spring green   ← TCC threshold ≈ 245 K
    "#AAFF00",   # 258 K  yellow-green
    "#FFFF00",   # 272 K  yellow
    "#FF8C00",   # 287 K  dark orange
    "#FF4500",   # 303 K  orange-red
    "#CC3300",   # 345 K  dark red
], N=256)
BT_VMIN, BT_VMAX = 175, 345


# ══════════════════════════════════════════════════════════════════════════
# PREPROCESSING  (identical to create_seq_convlstm.py)
# ══════════════════════════════════════════════════════════════════════════

def normalize(x):
    # Mask INSAT fill values: outside satellite disk stored as 0K (finite, not NaN)
    x = np.where((x > 100.0) & (x < 400.0), x, np.nan)
    m = np.nanmean(x)
    x = np.nan_to_num(x, nan=float(m) if np.isfinite(m) else 270.0)
    return (x - x.mean()) / (x.std() + 1e-6)


def seq_valid(seq_raw):
    # Check physical validity (not NaN, not fill value 0K outside disk)
    for t in range(seq_raw.shape[0]):
        valid = (seq_raw[t] > 100.0) & (seq_raw[t] < 400.0) & np.isfinite(seq_raw[t])
        if valid.mean() < (1.0 - MAX_NAN_FRAC):
            return False
    valid_all = (seq_raw > 100.0) & (seq_raw < 400.0) & np.isfinite(seq_raw)
    return bool(valid_all.any()) and float(seq_raw[valid_all].std()) > 1e-3


def _valid_mean(frame):
    """Mean of physically valid BT pixels only (excludes 0K fill outside disk)."""
    v = frame[(frame > 100.0) & (frame < 400.0) & np.isfinite(frame)]
    return float(v.mean()) if v.size > 0 else 270.0


def threshold_label(seq_raw):
    """Exact copy of classify_patch_sequence() from create_seq_convlstm.py,
    extended to exclude fill values."""
    T     = THRESHOLDS
    last  = _valid_mean(seq_raw[-1])
    first = _valid_mean(seq_raw[0])
    trend = last - first
    if last  > T["non_tcc_min_tb"]:   return 0
    if trend < T["intensify_trend"]:  return 2
    if trend < T["organize_trend"]:   return 1
    if last  < T["mature_max_tb"]:    return 3
    if trend > T["dissipate_trend"]:  return 4
    if last  > T["non_tcc_soft_tb"]:  return 0
    return 3


def extract_patches(frame):
    H, W, out = *frame.shape, []
    for i in range(0, H - PATCH_SIZE + 1, PATCH_STRIDE):
        for j in range(0, W - PATCH_SIZE + 1, PATCH_STRIDE):
            out.append((frame[i:i+PATCH_SIZE, j:j+PATCH_SIZE], i, j))
    return out


def build_pred_map(H, W, preds):
    m = np.full((H, W), -1, dtype=np.int8)
    for label, r, c in preds:
        m[r:r+PATCH_SIZE, c:c+PATCH_SIZE] = label
    return m


# ── FIX BUG 1: fill entire patch area, narrow sigma for smooth edges ──────
def smooth_class_blend(pred_map, H, W, bt_frame=None):
    """
    Build (H, W, 4) RGBA blend from MC class patch labels.

    Two key improvements over previous version:
    1. bt_frame display guard — only shows lifecycle overlay where BT < 262 K
       (actual cold cloud zone). Suppresses green/yellow blobs over warm clear-sky
       patches and over fill-value pixels outside the satellite disk.
    2. Larger sigma (PATCH_SIZE × 0.50) for smooth boundary blending instead
       of blocky checkerboard appearance.
    """
    blend = np.zeros((H, W, 4), dtype=np.float32)
    sigma = PATCH_SIZE * 0.50   # ≈ 32 px for PATCH_SIZE=64 — smooth boundaries

    # Display gate: only show lifecycle overlay over cold cloud regions
    # Warm clear-sky (>262 K) and fill-value pixels (outside disk, ~0 K) get no overlay
    if bt_frame is not None:
        valid_px   = (bt_frame > 100.0) & (bt_frame < 400.0) & np.isfinite(bt_frame)
        cold_cloud = (valid_px & (bt_frame < 262.0)).astype(np.float32)
    else:
        cold_cloud = np.ones((H, W), dtype=np.float32)

    for cls, rgb in enumerate(CLASS_RGB):
        if rgb is None:
            continue

        raw  = np.zeros((H, W), dtype=np.float32)
        mask = np.zeros((H, W), dtype=np.float32)

        for r in range(0, H - PATCH_SIZE + 1, PATCH_STRIDE):
            for c in range(0, W - PATCH_SIZE + 1, PATCH_STRIDE):
                lbl = int(pred_map[r + PATCH_SIZE//2, c + PATCH_SIZE//2])
                if lbl >= 0:
                    raw [r:r+PATCH_SIZE, c:c+PATCH_SIZE] = 1.0 if lbl == cls else 0.0
                    mask[r:r+PATCH_SIZE, c:c+PATCH_SIZE] = 1.0

        sm = gaussian_filter(raw * mask, sigma=sigma)
        wt = gaussian_filter(mask,       sigma=sigma)
        with np.errstate(invalid="ignore", divide="ignore"):
            prob_cls = np.clip(np.where(wt > 0.01, sm / wt, 0.0), 0.0, 1.0)

        # Apply cold-cloud display gate — zero out warm/fill regions
        prob_cls = prob_cls * cold_cloud

        active = prob_cls >= 0.45
        alpha  = np.where(active,
                          np.clip((prob_cls - 0.45) / 0.55, 0.0, 1.0) * 0.72 + 0.08,
                          0.0)

        for ch in range(3):
            blend[:, :, ch] = np.where(active,
                                        blend[:,:,ch] + alpha * rgb[ch],
                                        blend[:,:,ch])
        blend[:, :, 3] = np.where(active,
                                   np.clip(blend[:,:,3] + alpha, 0.0, 0.82),
                                   blend[:,:,3])

    return blend


# ══════════════════════════════════════════════════════════════════════════
# I/O
# ══════════════════════════════════════════════════════════════════════════

def load_frame(path):
    try:
        d = np.load(path, mmap_mode="r")
    except Exception as e:
        print(f"  [WARN] {os.path.basename(path)}: {e}"); return None, None, None
    if "TIR1_TEMP" not in d.files:
        return None, None, None
    bt  = d["TIR1_TEMP"].astype(np.float32)
    lat = d["LAT"].astype(np.float32)  if "LAT" in d.files else None
    lon = d["LON"].astype(np.float32)  if "LON" in d.files else None
    return bt, lat, lon


def geo_extent(lat, lon):
    if lat is not None and lon is not None:
        lf = lat.flatten(); lf = lf[np.isfinite(lf)]
        lo = lon.flatten(); lo = lo[np.isfinite(lo)]
        if lf.size and lo.size:
            return [float(lo.min()), float(lo.max()),
                    float(lf.min()), float(lf.max())]
    return DEF_EXTENT


def parse_ts(fname):
    m = re.search(r'_(\d{4})(?:_|\.)', fname)
    return f"{m.group(1)[:2]}:{m.group(1)[2:]} UTC" if m else os.path.basename(fname)


def find_files(raw_dir, date_str):
    all_f = sorted(glob.glob(os.path.join(raw_dir, "*.npz")))
    day   = [f for f in all_f if date_str in os.path.basename(f)]
    if not day and all_f:
        m = re.search(r'(\d{8})', os.path.basename(all_f[0]))
        if m:
            date_str = m.group(1)
            day = [f for f in all_f if date_str in os.path.basename(f)]
            print(f"  [INFO] Auto-selected: {date_str} ({len(day)} files)")
    if not day:
        day = all_f[:48]; print(f"  [INFO] Using first {len(day)} files.")
    return day, date_str


# ══════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════

def load_model(path):
    if not TF_OK:
        print("  [INFO] TF not available — threshold fallback."); return None
    if not os.path.exists(path):
        print(f"  [INFO] Model not found — threshold fallback."); return None
    try:
        m = tf.keras.models.load_model(path, compile=False)
        print(f"  [OK]   Model loaded: {path}"); return m
    except Exception as e:
        print(f"  [WARN] {e} — threshold fallback."); return None


# ══════════════════════════════════════════════════════════════════════════
# PREDICTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def run_day(raw_dir, date_str, model):
    day_files, date_str = find_files(raw_dir, date_str)
    if not day_files:
        raise RuntimeError(f"No .npz files in {raw_dir}")
    print(f"  Processing {len(day_files)} frames …")

    buffer, frames_meta = deque(maxlen=SEQ_LEN), []
    batch_X, batch_meta, results = [], [], []

    for fidx, fpath in enumerate(day_files):
        bt, lat, lon = load_frame(fpath)
        if bt is None: continue
        ts    = parse_ts(os.path.basename(fpath))
        pw    = extract_patches(bt)
        pnorm = [normalize(p) for p, _, _ in pw]
        frames_meta.append((bt, lat, lon, ts))
        buffer.append((pw, pnorm))

        if len(buffer) < SEQ_LEN: continue
        buf = list(buffer)
        n_p = min(len(buf[t][0]) for t in range(SEQ_LEN))
        fp  = []

        for pi in range(n_p):
            seq_raw = np.stack([buf[t][0][pi][0] for t in range(SEQ_LEN)])
            if not seq_valid(seq_raw): continue
            _, r, c = buf[-1][0][pi]
            if model is not None:
                sn = np.stack([buf[t][1][pi] for t in range(SEQ_LEN)])[..., np.newaxis].astype(np.float32)
                batch_X.append(sn); batch_meta.append((len(results), r, c))
            else:
                fp.append((threshold_label(seq_raw), r, c))

        anc = frames_meta[min(fidx, len(frames_meta)-1)]
        results.append({"bt": anc[0], "lat": anc[1], "lon": anc[2],
                         "ts": anc[3], "fp": fp, "blend": None,
                         "class_fracs": [0.0] * N_CLASSES})

    if model is not None and batch_X:
        total_seq = len(batch_X)
        print(f"  Model inference on {total_seq:,} sequences …")
        # ── Chunked inference — avoids GPU OOM on 2 GB VRAM ──────────────
        # MC seq: SEQ_LEN × 64 × 64 × 1 × 4 B ≈ 0.08 MB each
        # INFER_BATCH=512 → ~41 MB per chunk, safe for 2 GB GPU
        # Lower to 256 if OOM persists
        INFER_BATCH = 256          # 256 × 0.08 MB ≈ 20 MB/chunk — safe on 2 GB VRAM
        all_lbs = []
        for start in range(0, total_seq, INFER_BATCH):
            chunk = np.array(batch_X[start:start + INFER_BATCH], dtype=np.float32)
            preds = model.predict(chunk, verbose=0)
            all_lbs.append(np.argmax(preds, axis=1))
            del chunk, preds
            if (start // INFER_BATCH) % 10 == 0:
                done = min(start + INFER_BATCH, total_seq)
                print(f"    {done:,}/{total_seq:,}", end="\r", flush=True)
        print(f"    {total_seq:,}/{total_seq:,} — done.   ")
        lbs = np.concatenate(all_lbs).astype(np.int8)
        del all_lbs, batch_X
        for si, (ri, r, c) in enumerate(batch_meta):
            if ri < len(results):
                results[ri]["fp"].append((int(lbs[si]), r, c))

    for res in results:
        H, W  = res["bt"].shape
        pm    = build_pred_map(H, W, res["fp"])
        blend = smooth_class_blend(pm, H, W, res["bt"])
        res["blend"] = blend

        # ── FIX BUG 2: class fracs from pred_map (patch labels) ──────────
        cov = pm >= 0
        if cov.any():
            for c in range(N_CLASSES):
                res["class_fracs"][c] = float((pm[cov] == c).mean() * 100)

    print(f"  {len(results)} prediction frames ready.")
    return results, date_str


# ══════════════════════════════════════════════════════════════════════════
# FRAME RENDERER
# ══════════════════════════════════════════════════════════════════════════

def render_b64(res):
    ext  = geo_extent(res["lat"], res["lon"])
    bt   = res["bt"]

    fig, ax = plt.subplots(figsize=(7.8, 5.2), facecolor="#0D1117")
    ax.set_facecolor("#0D1117")

    # ── FIX BUG 4: MOSDAC-matched BT colormap ─────────────────────────────
    im_bt = ax.imshow(bt, cmap=BT_CMAP, extent=ext, aspect="auto",
                      origin="upper", vmin=BT_VMIN, vmax=BT_VMAX, alpha=0.92)

    # ── FIX BUG 3: high-alpha, high-contrast lifecycle overlay ────────────
    if res["blend"] is not None:
        ax.imshow(res["blend"], extent=ext, aspect="auto", origin="upper")

    ax.axhline(0, color="white", lw=0.7, linestyle="--", alpha=0.30)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel("Longitude (°E)", fontsize=8, color="#8B949E")
    ax.set_ylabel("Latitude (°N)",  fontsize=8, color="#8B949E")
    ax.tick_params(colors="#8B949E", labelsize=7)
    for sp in ax.spines.values(): sp.set_color("#30363D")

    # ── FIX BUG 5: BT colorbar with temperature scale ─────────────────────
    cbar = plt.colorbar(im_bt, ax=ax, orientation="vertical",
                        fraction=0.025, pad=0.02, shrink=0.88)
    cbar.set_label("Brightness Temperature (K)", fontsize=7, color="#8B949E", labelpad=6)
    cbar.set_ticks([175, 200, 225, 245, 275, 300, 325])
    cbar.ax.tick_params(labelsize=6.5, colors="#8B949E", length=3)
    norm_245 = (245 - BT_VMIN) / (BT_VMAX - BT_VMIN)
    cbar.ax.axhline(y=norm_245, color="#FFFF00", linewidth=1.8, alpha=0.9)
    cbar.ax.text(1.08, norm_245, "← 245 K\n  TCC\nthresh.",
                 transform=cbar.ax.transAxes,
                 fontsize=5.5, color="#FFFF00", va="center", ha="left",
                 linespacing=1.4)

    # Dominant active stage annotation
    fracs    = res["class_fracs"]
    conv_sum = sum(fracs[1:])
    dom_c    = int(np.argmax(fracs[1:]) + 1) if conv_sum > 1 else 0
    dom_col  = CLASS_HEX[dom_c]
    dom_name = CLASS_NAMES[dom_c]
    ax.text(0.01, 0.98,
            f"{dom_name}  ({fracs[dom_c]:.1f}% of region)",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=7.5, color=dom_col, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0D111788",
                      edgecolor=dom_col + "55", alpha=0.85))

    plt.tight_layout(pad=0.4)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=FRAME_DPI,
                bbox_inches="tight", facecolor="#0D1117")
    plt.close(); buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ══════════════════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ══════════════════════════════════════════════════════════════════════════

HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>INSAT-3D MC Lifecycle — {date}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0D1117;color:#E6EDF3;font-family:'Segoe UI',system-ui,sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}}
.hdr{{background:#161B22;border-bottom:1px solid #30363D;padding:10px 20px;display:flex;align-items:center;justify-content:space-between;flex-shrink:0}}
.ht{{font-size:15px;font-weight:700;color:#58A6FF}}
.hs{{font-size:11px;color:#8B949E;margin-top:2px}}
.lg{{display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
.li{{display:flex;align-items:center;gap:5px;font-size:11px;color:#C9D1D9}}
.dot{{width:10px;height:10px;border-radius:50%;display:inline-block}}
.main{{display:flex;flex:1;overflow:hidden}}
.fa{{flex:1;display:flex;align-items:center;justify-content:center;padding:10px;background:#0D1117}}
#fi{{max-width:100%;max-height:100%;border-radius:6px;border:1px solid #21262D;display:block;transition:opacity .15s}}
#fi.fade{{opacity:0.15}}
.ip{{width:230px;background:#161B22;border-left:1px solid #30363D;padding:16px 14px;display:flex;flex-direction:column;gap:14px;flex-shrink:0;overflow-y:auto}}
.il{{font-size:9px;text-transform:uppercase;letter-spacing:1px;color:#8B949E;margin-bottom:6px;font-weight:600}}
.ts{{font-size:24px;font-weight:700;color:#58A6FF;letter-spacing:1px}}
.dt{{font-size:11px;color:#8B949E;margin-top:2px}}
.br{{margin-bottom:7px}}
.bn{{font-size:10px;color:#C9D1D9;margin-bottom:2px;display:flex;justify-content:space-between}}
.bt{{height:7px;background:#21262D;border-radius:4px;overflow:hidden}}
.bf{{height:100%;border-radius:4px;transition:width .3s}}
.dm{{padding:7px 10px;border-radius:6px;font-size:12px;font-weight:700;text-align:center;border:1px solid;letter-spacing:.5px}}
.fc{{font-size:11px;color:#8B949E;text-align:center}}
.cb-note{{font-size:9px;color:#8B949E;line-height:1.5;border-top:1px solid #21262D;padding-top:8px}}
.cb-row{{display:flex;align-items:center;gap:5px;margin-bottom:3px}}
.cb-swatch{{width:14px;height:10px;border-radius:2px;flex-shrink:0}}
.ctrl{{background:#161B22;border-top:1px solid #30363D;padding:10px 20px 12px;flex-shrink:0}}
.pr{{display:flex;align-items:center;gap:10px;margin-bottom:8px}}
.tl{{font-size:10px;color:#8B949E;white-space:nowrap;min-width:55px}}
.pb{{flex:1;height:4px;background:#21262D;border-radius:2px;cursor:pointer;position:relative;transition:height .15s}}
.pb:hover{{height:7px}}
.pf{{height:100%;background:linear-gradient(90deg,#1F6FEB,#58A6FF);border-radius:2px;transition:width .2s}}
.pt{{position:absolute;top:50%;transform:translate(-50%,-50%);width:13px;height:13px;border-radius:50%;background:#58A6FF;border:2px solid #0D1117;box-shadow:0 0 6px #58A6FF88;transition:left .2s}}
.brow{{display:flex;align-items:center;justify-content:center;gap:12px}}
.btn{{background:#21262D;border:1px solid #30363D;color:#E6EDF3;border-radius:6px;padding:6px 14px;font-size:12px;cursor:pointer;transition:all .15s;display:flex;align-items:center;gap:5px}}
.btn:hover{{background:#30363D;border-color:#58A6FF;color:#58A6FF}}
.bpl{{background:#1F6FEB;border-color:#1F6FEB;padding:8px 30px;font-size:14px;font-weight:700;border-radius:8px;min-width:130px;justify-content:center;color:white}}
.bpl:hover{{background:#388BFF;border-color:#388BFF}}
.bpl.on{{background:#B91C1C;border-color:#B91C1C}}
.bpl.on:hover{{background:#DC2626}}
select{{background:#21262D;border:1px solid #30363D;color:#E6EDF3;border-radius:4px;padding:3px 6px;font-size:11px;cursor:pointer}}
.sl{{font-size:11px;color:#8B949E}}
</style></head><body>
<div class="hdr">
  <div>
    <div class="ht">🛰️ INSAT-3D TIR-1 — TCC Lifecycle Classification &nbsp;|&nbsp; Method 3: ConvLSTM MC</div>
    <div class="hs">Indian Ocean &nbsp;|&nbsp; {date} &nbsp;|&nbsp; 30-min intervals &nbsp;|&nbsp; {method}</div>
  </div>
  <div class="lg">
    <div class="li"><span class="dot" style="background:#90CAF9"></span>NON-TCC</div>
    <div class="li"><span class="dot" style="background:#00FF88"></span>ORGANIZING</div>
    <div class="li"><span class="dot" style="background:#FFFF00"></span>INTENSIFYING</div>
    <div class="li"><span class="dot" style="background:#FF1493"></span>MATURE</div>
    <div class="li"><span class="dot" style="background:#00FFFF"></span>DISSIPATING</div>
  </div>
</div>
<div class="main">
  <div class="fa"><img id="fi" src="" alt="frame"/></div>
  <div class="ip">
    <div><div class="il">Timestamp</div><div class="ts" id="ts">--:--</div><div class="dt">{date}</div></div>
    <div><div class="il">Stage Distribution</div><div id="bs"></div></div>
    <div><div class="il">Dominant Active Stage</div><div class="dm" id="dm">—</div></div>
    <div>
      <div class="il">BT Colour Guide</div>
      <div class="cb-note">
        <div class="cb-row"><div class="cb-swatch" style="background:#CC00FF"></div>≤ 190 K  Very cold tops</div>
        <div class="cb-row"><div class="cb-swatch" style="background:#0000FF"></div>190–225 K  Cold cloud</div>
        <div class="cb-row"><div class="cb-swatch" style="background:#00FF80"></div>225–250 K  TCC zone ✓</div>
        <div class="cb-row"><div class="cb-swatch" style="background:#FFFF00"></div>250–275 K  Mid level</div>
        <div class="cb-row"><div class="cb-swatch" style="background:#FF8C00"></div>275–310 K  Warm/clear</div>
        <div class="cb-row"><div class="cb-swatch" style="background:#CC3300"></div>310+ K  Land/sea sfc</div>
      </div>
    </div>
    <div class="fc" id="fc">Frame 1 / {n}</div>
  </div>
</div>
<div class="ctrl">
  <div class="pr">
    <span class="tl" id="tL">--:--</span>
    <div class="pb" id="pb" onclick="seek(event)">
      <div class="pf" id="pf" style="width:0%"></div>
      <div class="pt" id="pt" style="left:0%"></div>
    </div>
    <span class="tl" style="text-align:right" id="tR">--:--</span>
  </div>
  <div class="brow">
    <button class="btn" onclick="step(-5)">⏮ -5</button>
    <button class="btn" onclick="step(-1)">◀ Prev</button>
    <button class="btn bpl" id="pb2" onclick="tog()">▶ Play</button>
    <button class="btn" onclick="step(1)">Next ▶</button>
    <button class="btn" onclick="step(5)">+5 ⏭</button>
    <span class="sl">Speed:</span>
    <select id="sp" onchange="setSpd()">
      <option value="1400">0.5×</option>
      <option value="800" selected>1×</option>
      <option value="400">2×</option>
      <option value="180">4×</option>
    </select>
  </div>
</div>
<script>
const F={frames_json};
const N=F.length;
const NM={class_names_json};
const HX={class_hex_json};
let cur=0,playing=false,iv=null,spd=800;
function show(i){{
  cur=((i%N)+N)%N;const f=F[cur];
  const img=document.getElementById('fi');
  img.classList.add('fade');
  setTimeout(()=>{{img.src='data:image/png;base64,'+f.img;img.classList.remove('fade');}},80);
  document.getElementById('ts').textContent=f.ts;
  document.getElementById('fc').textContent='Frame '+(cur+1)+' / '+N;
  document.getElementById('tL').textContent=f.ts;
  document.getElementById('tR').textContent=F[N-1].ts;
  const p=N>1?(cur/(N-1))*100:0;
  document.getElementById('pf').style.width=p+'%';
  document.getElementById('pt').style.left=p+'%';
  // Stage bars (all 5 classes)
  let h='',domC=0,domV=0;
  for(let c=0;c<5;c++){{
    const v=(f.cf[c]||0);
    if(c>0&&v>domV){{domV=v;domC=c;}}
    const vs=v.toFixed(1);
    h+=`<div class="br"><div class="bn"><span style="color:${{HX[c]}}">${{NM[c]}}</span><span style="color:#8B949E">${{vs}}%</span></div><div class="bt"><div class="bf" style="width:${{Math.min(vs,100)}}%;background:${{HX[c]}}"></div></div></div>`;
  }}
  document.getElementById('bs').innerHTML=h;
  const dc=HX[domC],dn=NM[domC];
  const d=document.getElementById('dm');
  d.textContent=dn;d.style.background=dc+'22';d.style.color=dc;d.style.borderColor=dc+'55';
}}
function tog(){{
  playing=!playing;const b=document.getElementById('pb2');
  if(playing){{b.textContent='⏸ Pause';b.classList.add('on');
    iv=setInterval(()=>{{if(cur>=N-1){{tog();return;}}show(cur+1);}},spd);
  }}else{{b.textContent='▶ Play';b.classList.remove('on');clearInterval(iv);}}
}}
function step(d){{if(playing)tog();show(cur+d);}}
function seek(e){{const r=document.getElementById('pb').getBoundingClientRect();
  show(Math.round(Math.max(0,Math.min(1,(e.clientX-r.left)/r.width))*(N-1)));}}
function setSpd(){{spd=parseInt(document.getElementById('sp').value);
  if(playing){{clearInterval(iv);tog();tog();}}}}
document.addEventListener('keydown',e=>{{
  if(e.code==='Space'){{e.preventDefault();tog();}}
  if(e.code==='ArrowRight')step(1);if(e.code==='ArrowLeft')step(-1);
}});
show(0);
</script></body></html>"""


def build_html(frames_data, date_str, method_str):
    return HTML.format(
        date=date_str, n=len(frames_data), method=method_str,
        frames_json=json.dumps(frames_data),
        class_names_json=json.dumps(CLASS_NAMES),
        class_hex_json=json.dumps(CLASS_HEX),
    )


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Multi-Class ConvLSTM — one-day TCC lifecycle + HTML animation"
    )
    ap.add_argument("--date",    default="20250206")
    ap.add_argument("--raw_dir", default=RAW_DIR)
    ap.add_argument("--model",   default=MODEL_PATH)
    ap.add_argument("--out",     default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("=" * 62)
    print("  Multi-Class ConvLSTM — One-Day Prediction  (Method 3)  v2")
    print("=" * 62)
    print(f"  Date    : {args.date}")
    print(f"  Raw dir : {args.raw_dir}")
    print(f"  Model   : {args.model}")
    print(f"  Output  : {args.out}")
    print(f"  TF      : {'available' if TF_OK else 'not available — threshold fallback'}")
    print("=" * 62)

    model  = load_model(args.model)
    method = "ConvLSTM MC Model" if model else "Threshold Lifecycle Rules"

    results, date_used = run_day(args.raw_dir, args.date, model)
    if not results:
        print("[ERROR] No results — check raw_dir and date."); sys.exit(1)

    print(f"\n  Rendering {len(results)} frames …")
    frames_data = []
    for i, res in enumerate(results):
        print(f"    {i+1}/{len(results)}", end="\r", flush=True)
        b64 = render_b64(res)
        frames_data.append({
            "img": b64,
            "ts":  res["ts"],
            "cf":  [round(v, 1) for v in res["class_fracs"]],
        })
    print(f"    {len(results)}/{len(results)} — done.   ")

    out_path = os.path.join(args.out, f"animation_{date_used}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(build_html(frames_data, date_used, method))

    mb = os.path.getsize(out_path) / 1024**2
    print(f"\n✓  Saved : {out_path}  ({mb:.1f} MB)")
    print("   Open in any browser — fully self-contained, no internet needed.")
    print("\n   Controls: Space=Play/Pause  ←→=step  click bar=seek  speed=0.5-4×")


if __name__ == "__main__":
    main()