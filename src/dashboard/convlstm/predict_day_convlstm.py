"""
predict_day_convlstm.py  (v2 — all rendering bugs fixed)
=========================================================
Binary ConvLSTM (Method 2) — one-day TCC detection on INSAT-3D TIR-1 data.

Bugs fixed vs v1
----------------
  BUG 1  smooth_prob returned values ≤ 0.35 (sigma too wide, only centre pixel set)
         → now fills ENTIRE patch area before smoothing with narrow sigma
  BUG 2  Class-distribution bars showed 0% while TCC-coverage badge showed real value
         → both now computed from the same source (pred_map patch labels)
  BUG 3  TCC overlay was invisible — red on dark-blue BT background ≈ black
         → changed to bright YELLOW which contrasts on any BT background colour
  BUG 4  BT colormap vmin/vmax (200–310K) clipped real INSAT range (175–345K)
         → now uses MOSDAC-matched colormap with vmin=175, vmax=345
  BUG 5  No temperature scale visible for the viewer
         → BT colorbar with K labels + 245K threshold marker added to every frame

Output
------
  output/convlstm/animation_<date>.html   — self-contained HTML movie player

Usage
-----
  python src/dashboard/convlstm/predict_day_convlstm.py
  python src/dashboard/convlstm/predict_day_convlstm.py --date 20250206

Requirements:  numpy  matplotlib  scipy
Optional:      tensorflow >= 2.10
"""

import os, sys, glob, re, json, io, base64, argparse, warnings
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
# CONFIG  — mirrors create_sequence_convlstm.py
# ══════════════════════════════════════════════════════════════════════════
SEQ_LEN      = 5
PATCH_SIZE   = 128
PATCH_STRIDE = 256
TB_THRESHOLD = 245          # K — cold cloud threshold used during training

RAW_DIR    = "data/raw_npz"
MODEL_PATH = "models/convlstm/best_convlstm.keras"
OUT_DIR    = "output/convlstm"
DEF_EXTENT = [40.0, 120.0, -60.0, 60.0]   # [lon_min, lon_max, lat_min, lat_max]
FRAME_DPI  = 110

# ── BT colormap: matches MOSDAC INSAT-3D display (175 K → 345 K) ─────────
# cold (purple) → blue → cyan → green → yellow → orange → red (warm)
BT_CMAP = LinearSegmentedColormap.from_list("mosdac_ir", [
    "#CC00FF",   # 175 K  bright magenta   (very cold cloud tops)
    "#7B00D4",   # 191 K  violet
    "#0000FF",   # 208 K  blue
    "#00BFFF",   # 228 K  deep sky blue
    "#00FF80",   # 244 K  spring green    ← TCC threshold ≈ 245 K
    "#AAFF00",   # 258 K  yellow-green
    "#FFFF00",   # 272 K  yellow
    "#FF8C00",   # 287 K  dark orange
    "#FF4500",   # 303 K  orange-red
    "#CC3300",   # 345 K  dark red
], N=256)

BT_VMIN, BT_VMAX = 175, 345   # full INSAT-3D brightness temperature range


# ══════════════════════════════════════════════════════════════════════════
# PREPROCESSING  (identical to create_sequence_convlstm.py)
# ══════════════════════════════════════════════════════════════════════════

def normalize(x):
    # Mask INSAT fill values: outside satellite disk stored as 0K (finite, not NaN)
    x = np.where((x > 100.0) & (x < 400.0), x, np.nan)
    m = np.nanmean(x)
    x = np.nan_to_num(x, nan=float(m) if np.isfinite(m) else 270.0)
    return (x - x.mean()) / (x.std() + 1e-6)


def extract_patches(frame):
    H, W, out = *frame.shape, []
    for i in range(0, H - PATCH_SIZE + 1, PATCH_STRIDE):
        for j in range(0, W - PATCH_SIZE + 1, PATCH_STRIDE):
            out.append((frame[i:i+PATCH_SIZE, j:j+PATCH_SIZE], i, j))
    return out


def _valid_bt(patch):
    """Return only physically valid BT pixels (100–400 K, finite)."""
    return patch[(patch > 100.0) & (patch < 400.0) & np.isfinite(patch)]


def threshold_label(patch):
    # Use physical BT range — fill values (0K outside disk) are excluded
    v = _valid_bt(patch)
    # Skip patch if < 60% pixels are valid (patch straddles disk boundary)
    if v.size < 0.60 * patch.size:
        return 0
    return 1 if float(v.mean()) <= TB_THRESHOLD else 0


def build_pred_map(H, W, preds):
    m = np.full((H, W), -1, dtype=np.int8)
    for label, r, c in preds:
        m[r:r+PATCH_SIZE, c:c+PATCH_SIZE] = label
    return m


# ── FIX BUG 1: fill entire patch area, use narrow sigma for smooth edges ──
def smooth_prob(pred_map):
    """
    Convert patch-label map to smooth probability float map.

    OLD (broken): set only the centre PIXEL of each patch → sigma=140 px
    spread caused severe dilution; max prob ≤ 0.35.

    NEW (fixed):  fill the entire PATCH_SIZE×PATCH_SIZE area with the label
    (0.0 or 1.0), then apply a narrow Gaussian (sigma = PATCH_SIZE/4)
    purely for smooth boundary blending. TCC regions reach prob ≈ 0.85–1.0.
    """
    H, W = pred_map.shape
    raw  = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.float32)

    for r in range(0, H - PATCH_SIZE + 1, PATCH_STRIDE):
        for c in range(0, W - PATCH_SIZE + 1, PATCH_STRIDE):
            lbl = int(pred_map[r + PATCH_SIZE//2, c + PATCH_SIZE//2])
            if lbl >= 0:
                raw [r:r+PATCH_SIZE, c:c+PATCH_SIZE] = float(lbl)
                mask[r:r+PATCH_SIZE, c:c+PATCH_SIZE] = 1.0

    # Narrow sigma: only blends at patch boundaries, not across whole image
    sigma = PATCH_SIZE * 0.28      # ≈ 36 px for PATCH_SIZE=128
    sm = gaussian_filter(raw  * mask, sigma=sigma)
    wt = gaussian_filter(mask,        sigma=sigma)

    with np.errstate(invalid="ignore", divide="ignore"):
        prob = np.where(wt > 0.01, sm / wt, 0.0)
    return np.clip(prob, 0.0, 1.0)


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
    # Try HH MM from 4-digit run in filename, e.g. _0030_ or _1245_
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
            print(f"  [INFO] Auto-selected date: {date_str} ({len(day)} files)")
    if not day:
        day = all_f[:48]
        print(f"  [INFO] Using first {len(day)} files as proxy day.")
    return day, date_str


# ══════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════

def load_model(path):
    if not TF_OK:
        print("  [INFO] TF not available — threshold fallback."); return None
    if not os.path.exists(path):
        print(f"  [INFO] Model not found at {path} — threshold fallback."); return None
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
            sn = np.stack([buf[t][1][pi] for t in range(SEQ_LEN)])[..., np.newaxis].astype(np.float32)
            _, r, c = buf[-1][0][pi]
            if model is not None:
                batch_X.append(sn); batch_meta.append((len(results), r, c))
            else:
                raw_last = buf[-1][0][pi][0]
                fp.append((threshold_label(raw_last), r, c))

        anc = frames_meta[min(fidx, len(frames_meta)-1)]
        results.append({"bt": anc[0], "lat": anc[1], "lon": anc[2],
                         "ts": anc[3], "fp": fp, "prob": None,
                         "tcc_pct": 0.0, "ntcc_pct": 100.0})

    if model is not None and batch_X:
        total_seq = len(batch_X)
        print(f"  Model inference on {total_seq:,} sequences …")
        # ── Chunked inference — avoids GPU OOM on 2 GB VRAM ──────────────
        # Binary seq: SEQ_LEN × 128 × 128 × 1 × 4 B ≈ 0.33 MB each
        # INFER_BATCH=256 → ~85 MB per chunk, safe for 2 GB GPU
        INFER_BATCH = 256
        all_lbs = []
        for start in range(0, total_seq, INFER_BATCH):
            chunk = np.array(batch_X[start:start + INFER_BATCH], dtype=np.float32)
            preds = model.predict(chunk, verbose=0)
            all_lbs.append(np.argmax(preds, axis=1))
            del chunk, preds
            if (start // INFER_BATCH) % 20 == 0:
                done = min(start + INFER_BATCH, total_seq)
                print(f"    {done:,}/{total_seq:,}", end="\r", flush=True)
        print(f"    {total_seq:,}/{total_seq:,} — done.   ")
        lbs = np.concatenate(all_lbs).astype(np.int8)
        del all_lbs, batch_X
        for si, (ri, r, c) in enumerate(batch_meta):
            if ri < len(results):
                results[ri]["fp"].append((int(lbs[si]), r, c))

    for res in results:
        H, W   = res["bt"].shape
        pm     = build_pred_map(H, W, res["fp"])
        prob   = smooth_prob(pm)
        res["prob"] = prob

        # ── FIX BUG 2: both tcc_pct and cf bars use same source (pred_map) ──
        cov = pm >= 0
        if cov.any():
            tcc_frac        = float((pm[cov] == 1).mean() * 100)
            res["tcc_pct"]  = tcc_frac
            res["ntcc_pct"] = 100.0 - tcc_frac
        else:
            res["tcc_pct"]  = 0.0
            res["ntcc_pct"] = 100.0

    print(f"  {len(results)} prediction frames ready.")
    return results, date_str


# ══════════════════════════════════════════════════════════════════════════
# FRAME RENDERER  → base64 PNG
# ══════════════════════════════════════════════════════════════════════════

def render_b64(res):
    ext = geo_extent(res["lat"], res["lon"])
    bt  = res["bt"]
    H, W = bt.shape

    fig, ax = plt.subplots(figsize=(7.8, 5.2), facecolor="#0D1117")
    ax.set_facecolor("#0D1117")

    # ── FIX BUG 4: MOSDAC-matched colormap + correct vmin/vmax ───────────
    im_bt = ax.imshow(bt, cmap=BT_CMAP, extent=ext, aspect="auto",
                      origin="upper", vmin=BT_VMIN, vmax=BT_VMAX, alpha=0.95)

    # ── Layer 1: pixel-level cold indicator — ALL pixels with BT < 245 K ────
    # Shows EVERY individual cold pixel regardless of patch-level classification.
    # Reason: the 128×128 patch mean can exceed 245 K even when isolated deep
    # convective cores (175–220 K, blue/purple in the image) are present —
    # surrounding warm pixels dilute the mean above the threshold.
    # This faint layer lets the viewer see ALL cold areas; Layer 2 shows model detections.
    valid_cold = (bt > 100.0) & (bt < float(TB_THRESHOLD)) & np.isfinite(bt)
    cold_layer = np.zeros((H, W, 4), dtype=np.float32)
    cold_layer[valid_cold, :3] = 1.0    # white tint
    cold_layer[valid_cold,  3] = 0.20   # faint — background hint only
    ax.imshow(cold_layer, extent=ext, aspect="auto", origin="upper")

    # ── Layer 2: patch-level TCC classification (model / threshold output) ─
    # Colour: PURE WHITE — completely absent from the MOSDAC BT colormap
    # (every BT colour is chromatic: purple→blue→green→yellow→orange→red).
    # No risk of visual confusion with any satellite temperature colour.
    if res["prob"] is not None:
        prob    = res["prob"]
        overlay = np.zeros((H, W, 4), dtype=np.float32)

        # Solid white where patch is classified TCC; gradient at boundaries
        alpha_layer = np.where(
            prob >= 0.50,
            np.clip((prob - 0.50) / 0.50, 0.0, 1.0) * 0.72 + 0.12,   # 0.12 → 0.84
            np.where(
                prob >= 0.30,
                np.clip((prob - 0.30) / 0.20, 0.0, 1.0) * 0.12,
                0.0
            )
        ).astype(np.float32)

        tcc_active = prob >= 0.30
        overlay[tcc_active, 0] = 1.0    # R  }
        overlay[tcc_active, 1] = 1.0    # G  }  white = R + G + B
        overlay[tcc_active, 2] = 1.0    # B  }
        overlay[:, :, 3] = alpha_layer
        ax.imshow(overlay, extent=ext, aspect="auto", origin="upper")

    # Equator line
    ax.axhline(0, color="white", lw=0.7, linestyle="--", alpha=0.35,
               label="Equator (0°)")

    # ── FIX BUG 5: BT colorbar with temperature scale ─────────────────────
    cbar = plt.colorbar(im_bt, ax=ax, orientation="vertical",
                        fraction=0.025, pad=0.02, shrink=0.88)
    cbar.set_label("Brightness Temperature (K)", fontsize=7, color="#8B949E",
                   labelpad=6)
    cbar.set_ticks([175, 200, 225, 245, 275, 300, 325])
    cbar.ax.tick_params(labelsize=6.5, colors="#8B949E", length=3)

    # Mark the 245 K TCC detection threshold on the colorbar
    norm_245 = (245 - BT_VMIN) / (BT_VMAX - BT_VMIN)
    cbar.ax.axhline(y=norm_245, color="white", linewidth=2.0, alpha=0.95)
    cbar.ax.text(1.08, norm_245, "← 245 K\n  TCC\nthresh.",
                 transform=cbar.ax.transAxes,
                 fontsize=5.5, color="white", va="center", ha="left",
                 linespacing=1.4)

    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel("Longitude (°E)", fontsize=8, color="#8B949E")
    ax.set_ylabel("Latitude (°N)",  fontsize=8, color="#8B949E")
    ax.tick_params(colors="#8B949E", labelsize=7)
    for sp in ax.spines.values(): sp.set_color("#30363D")

    # TCC percentage annotation on map
    tcc_pct = res.get("tcc_pct", 0.0)
    color   = "#FFFFFF" if tcc_pct > 5 else "#8B949E"
    ax.text(0.01, 0.98,
            f"TCC detected: {tcc_pct:.1f}% of region",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=7.5, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0D111788",
                      edgecolor=color + "55", alpha=0.85))

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
<title>INSAT-3D Binary TCC — {date}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0D1117;color:#E6EDF3;font-family:'Segoe UI',system-ui,sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}}
.hdr{{background:#161B22;border-bottom:1px solid #30363D;padding:10px 20px;display:flex;align-items:center;justify-content:space-between;flex-shrink:0}}
.ht{{font-size:15px;font-weight:700;color:#58A6FF}}
.hs{{font-size:11px;color:#8B949E;margin-top:2px}}
.lg{{display:flex;gap:14px;align-items:center}}
.li{{display:flex;align-items:center;gap:5px;font-size:11px;color:#C9D1D9}}
.dot{{width:10px;height:10px;border-radius:50%;display:inline-block}}
.main{{display:flex;flex:1;overflow:hidden}}
.fa{{flex:1;display:flex;align-items:center;justify-content:center;padding:10px;background:#0D1117}}
#fi{{max-width:100%;max-height:100%;border-radius:6px;border:1px solid #21262D;display:block;transition:opacity .15s}}
#fi.fade{{opacity:0.15}}
.ip{{width:220px;background:#161B22;border-left:1px solid #30363D;padding:16px 14px;display:flex;flex-direction:column;gap:14px;flex-shrink:0;overflow-y:auto}}
.il{{font-size:9px;text-transform:uppercase;letter-spacing:1px;color:#8B949E;margin-bottom:6px;font-weight:600}}
.ts{{font-size:24px;font-weight:700;color:#58A6FF;letter-spacing:1px}}
.dt{{font-size:11px;color:#8B949E;margin-top:2px}}
.tb{{display:inline-block;padding:5px 12px;border-radius:20px;font-size:14px;font-weight:700;background:#21262D;color:#FFFFFF;border:1px solid #FFFFFF60}}
.br{{margin-bottom:7px}}
.bn{{font-size:10px;color:#C9D1D9;margin-bottom:2px;display:flex;justify-content:space-between}}
.bt{{height:7px;background:#21262D;border-radius:4px;overflow:hidden}}
.bf{{height:100%;border-radius:4px;transition:width .3s}}
.dm{{padding:7px 10px;border-radius:6px;font-size:12px;font-weight:700;text-align:center;border:1px solid}}
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
    <div class="ht">🛰️ INSAT-3D TIR-1 — Binary TCC Detection &nbsp;|&nbsp; Method 2: ConvLSTM</div>
    <div class="hs">Indian Ocean &nbsp;|&nbsp; {date} &nbsp;|&nbsp; 30-min intervals &nbsp;|&nbsp; {method}</div>
  </div>
  <div class="lg">
    <div class="li"><span class="dot" style="background:#FFFFFF"></span>TCC detected — white patch overlay (model/threshold)</div>
    <div class="li"><span class="dot" style="background:#FFFFFF55;border:1px solid #FFFFFF66"></span>Faint white = all pixels BT &lt; 245 K (cold layer)</div>
    <div class="li"><span class="dot" style="background:#555"></span>Background = BT colormap (see colorbar in image)</div>
  </div>
</div>
<div class="main">
  <div class="fa"><img id="fi" src="" alt="frame"/></div>
  <div class="ip">
    <div><div class="il">Timestamp</div><div class="ts" id="ts">--:--</div><div class="dt">{date}</div></div>
    <div><div class="il">TCC Coverage</div><div class="tb" id="tc">--%</div></div>
    <div>
      <div class="il">Class Distribution</div>
      <div id="bs"></div>
    </div>
    <div><div class="il">Status</div><div class="dm" id="dm">—</div></div>
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
let cur=0,playing=false,iv=null,spd=800;
function show(i){{
  cur=((i%N)+N)%N;const f=F[cur];
  const img=document.getElementById('fi');
  img.classList.add('fade');
  setTimeout(()=>{{img.src='data:image/png;base64,'+f.img;img.classList.remove('fade');}},80);
  document.getElementById('ts').textContent=f.ts;
  document.getElementById('tc').textContent=f.tcc.toFixed(1)+'%';
  document.getElementById('fc').textContent='Frame '+(cur+1)+' / '+N;
  document.getElementById('tL').textContent=f.ts;
  document.getElementById('tR').textContent=F[N-1].ts;
  const p=N>1?(cur/(N-1))*100:0;
  document.getElementById('pf').style.width=p+'%';
  document.getElementById('pt').style.left=p+'%';
  // Class bars — same source as tcc badge (patch labels)
  const nm=['NON-TCC','TCC'],hx=['#2196F3','#FFFFFF'];
  let h='';
  for(let c=0;c<2;c++){{const v=(f.cf[c]||0).toFixed(1);
    h+=`<div class="br"><div class="bn"><span style="color:${{hx[c]}}">${{nm[c]}}</span><span style="color:#8B949E">${{v}}%</span></div><div class="bt"><div class="bf" style="width:${{Math.min(v,100)}}%;background:${{hx[c]}}"></div></div></div>`;}}
  document.getElementById('bs').innerHTML=h;
  // Status badge
  const tcc=f.tcc;
  let dc,dn;
  if(tcc>20){{dc='#FF4444';dn='HIGH TCC ACTIVITY';}}
  else if(tcc>5){{dc='#FFFFFF';dn='TCC PRESENT';}}
  else{{dc='#2196F3';dn='CLEAR / NON-TCC';}}
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
        date=date_str, n=len(frames_data),
        method=method_str, frames_json=json.dumps(frames_data)
    )


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Binary ConvLSTM — one-day TCC detection + HTML animation"
    )
    ap.add_argument("--date",    default="20250206")
    ap.add_argument("--raw_dir", default=RAW_DIR)
    ap.add_argument("--model",   default=MODEL_PATH)
    ap.add_argument("--out",     default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("=" * 62)
    print("  Binary ConvLSTM — One-Day Prediction  (Method 2)  v2")
    print("=" * 62)
    print(f"  Date    : {args.date}")
    print(f"  Raw dir : {args.raw_dir}")
    print(f"  Model   : {args.model}")
    print(f"  Output  : {args.out}")
    print(f"  TF      : {'available' if TF_OK else 'not available — threshold fallback'}")
    print("=" * 62)

    model  = load_model(args.model)
    method = "ConvLSTM Model" if model else "Threshold Fallback (mean BT ≤ 245 K)"

    results, date_used = run_day(args.raw_dir, args.date, model)
    if not results:
        print("[ERROR] No results — check raw_dir and date."); sys.exit(1)

    print(f"\n  Rendering {len(results)} frames …")
    frames_data = []
    for i, res in enumerate(results):
        print(f"    {i+1}/{len(results)}", end="\r", flush=True)
        b64 = render_b64(res)
        # ── FIX BUG 2: cf uses same patch-label source as tcc badge ────────
        frames_data.append({
            "img": b64,
            "ts":  res["ts"],
            "tcc": round(res["tcc_pct"],  1),    # fraction of patches = TCC
            "cf":  [round(res["ntcc_pct"], 1),   # NON-TCC patch fraction
                    round(res["tcc_pct"],  1)],   # TCC patch fraction
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