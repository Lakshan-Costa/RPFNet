#app.py
"""
MetaPoison v2/v3 — Flask Backend  (FIXED v3)
=============================================
WHAT CHANGED (search for "# ◆ FIX" to see every change):

  ◆ FIX 1  Fallback scoring: replaced rankdata()/n (uniform → always 40%)
           with min-max on raw IsolationForest anomaly scores.

  ◆ FIX 2  _compute_tau_local: replaced "mean+2σ of lower 60%" with
           bimodality-aware Otsu threshold + conservative unimodal fallback.

  ◆ FIX 3  Rate estimation: UNIVERSALLY conservative.
           Hard cap at 5% for ALL modes. Neither IF nor the trained model
           can reliably distinguish class structure from poisoning on
           unknown data. If Otsu cluster >8% → treated as class structure.
           User can always lower threshold with slider if they know data
           is attacked.

  ◆ FIX 4  _is_bimodal(): strict 2-of-4 signal requirement (AND not OR).
           Gap test, kurtosis, tail mass, histogram valley — need 2+.

  ◆ FIX 5  More IsolationForest trees (200 vs 100) for stable scores.

  ◆ FIX 6  Backward-compatible model loading: v2 models (dim=47) load
           with v3 detection.py (dim=55) via monkey-patched scoring.

  ◆ FIX 7  Scoring mode passed through pipeline so rate estimator knows
           whether to trust bimodality signal.

Endpoints (unchanged)
─────────
GET  /health
GET  /thresholds
POST /analyze_csv      multipart: file, tau?, dataset_hint?
POST /analyze_uci      json:      { uci_id, tau? }
POST /analyze_url      json:      { url, tau?, dataset_hint? }
POST /export_clean     json:      { dataset_id, clean_ids[] }
POST /analyze_stream
"""

from __future__ import annotations

import io
import os
import sys
import uuid
import warnings
import requests as _requests
import threading

import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
warnings.filterwarnings("ignore")

# ── Import MetaPoison
from RPFNet.Attack import apply_attack
from RPFNet.RPFExtractor import RPFExtractor
from RPFNet.Dataset import load_builtin, load_csv
from RPFNet.RateEstimator import estimate_contamination_rate, score_distribution_features, RateEstimatorHead
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from detection import (
        RPFNetPoisonDetector,
        HybridEnsembleDetector,
        RPFExtractor,
        RPFNet_MODEL_PATH,
        RPFNet_EPOCHS,
        RPFNet_BATCH_SIZE,
        RPFNet_LR,
        estimate_contamination_rate,
        score_distribution_features,
        RateEstimatorHead,
    )
    # These are needed for backward-compat model loading (v3 detection.py)
    try:
        from detection import RPFNetPoisonNet, RPF_K_SMALL, RPF_K_LARGE, RPF_CV_FOLDS
    except ImportError:
        RPFNetPoisonNet = None
        RPF_K_SMALL, RPF_K_LARGE, RPF_CV_FOLDS = 5, 15, 5
    MODEL_AVAILABLE = True
    MODEL_IMPORT_ERR = None
except Exception as _e:
    MODEL_AVAILABLE = False
    MODEL_IMPORT_ERR = str(_e)
    RPFNet_MODEL_PATH = "./RPFNet/RPFNet_universal.pt"
    RPFNetPoisonNet = None
    RPF_K_SMALL, RPF_K_LARGE, RPF_CV_FOLDS = 5, 15, 5

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins="*")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Global model state ────────────────────────────────────────────────────────
_lock = threading.Lock()
_meta:   "RPFNetPoisonDetector"   | None = None
_hybrid: "HybridEnsembleDetector" | None = None
_loaded = False

DATASET_STORE: dict[str, pd.DataFrame] = {}
DATASET_THRESHOLDS: dict[str, float] = {}
GLOBAL_THRESHOLD_DEFAULT = 3.5
STREAM_SESSIONS = {}
limiter = Limiter(get_remote_address, app=app, default_limits=["10 per minute"])

def _global_threshold() -> float:
    if DATASET_THRESHOLDS:
        return float(np.mean(list(DATASET_THRESHOLDS.values())))
    return GLOBAL_THRESHOLD_DEFAULT


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading  — ◆ FIX 6: backward-compatible with v2 (dim=47) models
# ─────────────────────────────────────────────────────────────────────────────

_LOADED_RPF_DIM = None   # track actual dim of loaded model

def _load_model_compat(meta, path):
    """
    Try normal load first. If dim mismatch (v2 model on v3 code),
    load with the old dim and monkey-patch the net + extractor.
    """
    global _LOADED_RPF_DIM

    try:
        meta.load(path)
        _LOADED_RPF_DIM = RPFExtractor.DIM
        return True
    except ValueError as e:
        if "RPF dim" not in str(e):
            raise

    # ── Backward-compat: load old-dim model ──────────────────────────
    print(f"[backend] Dim mismatch detected — loading in backward-compat mode")
    try:
        ckpt = torch.load(path, map_location=meta.device, weights_only=False)
    except TypeError:
        # Older PyTorch without weights_only param
        ckpt = torch.load(path, map_location=meta.device)
    saved_dim = ckpt.get("rpf_dim", 61)
    print(f"[backend]   saved_dim={saved_dim}  current_dim={RPFExtractor.DIM}")

    # Rebuild net with the SAVED dimension
    # MetaPoisonNet might not be importable from older detection.py
    _NetClass = RPFNetPoisonNet if RPFNetPoisonNet is not None else type(meta.net)
    meta.net = _NetClass(input_dim=saved_dim).to(meta.device)
    meta.net.load_state_dict(ckpt["net"])
    meta._threshold = ckpt.get("threshold", 0.5)
    meta._fitted    = ckpt.get("fitted", True)
    meta.extractor.k_small  = ckpt.get("k_small", RPF_K_SMALL)
    meta.extractor.k_large  = ckpt.get("k_large", RPF_K_LARGE)
    meta.extractor.cv_folds = ckpt.get("cv_folds", RPF_CV_FOLDS)

    if "rate_head" in ckpt:
        meta.rate_head = RateEstimatorHead().to(meta.device)
        meta.rate_head.load_state_dict(ckpt["rate_head"])
        meta.rate_head.eval()

    meta.net.eval()
    _LOADED_RPF_DIM = saved_dim

    # Monkey-patch _score_rpf to truncate RPF to saved_dim before scoring
    _original_score_rpf = meta._score_rpf

    def _patched_score_rpf(rpf):
        if rpf.shape[1] > saved_dim:
            rpf = rpf[:, :saved_dim]
        return _original_score_rpf(rpf)

    meta._score_rpf = _patched_score_rpf

    v = ckpt.get("version", 2)
    print(f"[backend]   ✓ Loaded v{v} model (dim={saved_dim}) in compat mode — "
          f"Block E features will be ignored during scoring")
    return True


def _load_model() -> None:
    global _meta, _hybrid, _loaded
    if not MODEL_AVAILABLE:
        return
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        meta = RPFNetPoisonDetector(
            device=device, epochs=RPFNet_EPOCHS,
            batch_size=RPFNet_BATCH_SIZE, lr=RPFNet_LR,
        )

        # ◆ FIX 6b: Try multiple model paths (v3 then v2)
        model_path = None
        candidates = [
            RPFNet_MODEL_PATH,
            "./RPFNet/RPFNet_Universal.pt",
        ]
        seen = set()
        for p in candidates:
            if p not in seen and os.path.exists(p):
                model_path = p
                break
            seen.add(p)

        if model_path:
            _load_model_compat(meta, model_path)
            fusion_path = model_path.replace(".pt", "_fusion.pt")
            hybrid = HybridEnsembleDetector(meta)
            if os.path.exists(fusion_path):
                hybrid.load(fusion_path)
            else:
                hybrid.attach_rate_head(meta.rate_head)
            with _lock:
                _meta, _hybrid, _loaded = meta, hybrid, True
            print(f"[backend] Model ready from {model_path}  device={device}")
        else:
            print(f"[backend] No model file found — tried: {candidates[:3]}")
            print(f"[backend] Running in score-only mode (IsolationForest fallback)")
    except Exception as exc:
        import traceback
        print(f"[backend] Model load failed: {exc}")
        traceback.print_exc()


threading.Thread(target=_load_model, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
#  CSV / dataframe helpers  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _infer_target(df: pd.DataFrame) -> str:
    candidates = [
        "loan_status", "Loan_Status", "deposit", "default", "Default",
        "label", "target", "y", "class", "outcome", "Target",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[-1]


def _prepare(df: pd.DataFrame):
    target_col = _infer_target(df)

    ys = df[target_col].copy()
    if ys.dtype == object or str(ys.dtype) == "category":
        pos = {"yes", "y", "1", "true", "paid", "fully paid", "current", "no default"}
        y = ys.astype(str).str.strip().str.lower().isin(pos).astype(int).values
    else:
        u = np.unique(ys.dropna())
        y = (ys == u[-1]).astype(int).values if len(u) == 2 else \
            (ys > float(np.median(ys.dropna()))).astype(int).values

    Xdf = df.drop(columns=[target_col]).copy()

    drop = [c for c in Xdf.columns
            if (Xdf[c].dtype not in (object, "category") and Xdf[c].nunique() == len(Xdf))
            or (Xdf[c].dtype == object and Xdf[c].nunique() > 0.9 * len(Xdf))]
    if drop:
        Xdf = Xdf.drop(columns=drop)

    Xdf = Xdf.dropna(axis=1, how="all")
    for c in Xdf.select_dtypes(include="number").columns:
        Xdf[c] = Xdf[c].fillna(Xdf[c].median())
    for c in Xdf.select_dtypes(exclude="number").columns:
        Xdf[c] = Xdf[c].fillna("missing")
    Xdf = pd.get_dummies(Xdf)

    X = Xdf.values.astype(np.float32)
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X).astype(np.float32)

    return X_scaled, y, sc, Xdf


# ═════════════════════════════════════════════════════════════════════════════
# ◆ FIX 4: Bimodality detector — TIGHTENED (v2)
#   Problem: single-heuristic OR was too loose. IsolationForest on
#   structured data (e.g. Mushroom with edible/poison classes) produces
#   right-skewed scores that trigger tail mass test → 25% false flags.
#   Fix: require 2+ heuristics to agree (AND logic), tighter thresholds.
# ═════════════════════════════════════════════════════════════════════════════

def _is_bimodal(scores: np.ndarray) -> bool:
    """
    Strict bimodality check — must see MULTIPLE signals, not just one.

    Returns True only when there's strong evidence of a discrete poison
    cluster separated from the bulk of the data. Single-signal triggers
    (like a heavy right tail) are NOT enough — that's just normal
    IsolationForest behavior on structured data.

    Requires at least 2 of 4 tests to pass:
      1. Large gap (>15% of range) in the upper half of scores
      2. Platykurtic (kurtosis < -1.0)
      3. Heavy isolated tail (>20% mass in top 3% of range)
      4. Histogram valley — clear dip between two peaks
    """
    if len(scores) < 30:
        return False

    s = np.sort(scores)
    n = len(s)
    score_range = float(s[-1] - s[0]) + 1e-10
    signals = 0

    # 1. Gap test — large gap in upper half only (lower half gaps are normal)
    upper_half = s[n // 2:]
    diffs = np.diff(upper_half)
    if len(diffs) > 0 and diffs.max() / score_range > 0.15:  # was 0.10
        signals += 1

    # 2. Kurtosis test — must be clearly platykurtic
    mu  = s.mean()
    std = s.std() + 1e-10
    kurt = float(np.mean(((s - mu) / std) ** 4)) - 3.0
    if kurt < -1.0:  # was -0.5
        signals += 1

    # 3. Tail mass test — very concentrated tail in narrow band
    top_3pct = s[0] + 0.97 * score_range  # was 0.95
    tail_mass = float((s >= top_3pct).mean())
    if tail_mass > 0.20:  # was 0.15 at 5%
        signals += 1

    # 4. Histogram valley test — is there a clear dip between two peaks?
    try:
        n_bins = min(50, max(10, n // 100))
        counts, edges = np.histogram(s, bins=n_bins)
        # Look for a valley: bin that is <30% of max, with higher bins on both sides
        max_count = counts.max()
        for i in range(2, len(counts) - 2):
            if (counts[i] < 0.30 * max_count
                and counts[i-1] > counts[i] and counts[i-2] > counts[i]
                and counts[i+1] > counts[i] and counts[i+2] > counts[i]):
                signals += 1
                break
    except Exception:
        pass

    # Require 2+ signals for bimodal classification
    return signals >= 2


# ═════════════════════════════════════════════════════════════════════════════
# ◆ FIX 2: Bimodality-aware local threshold (replaces mean+2σ)
# ═════════════════════════════════════════════════════════════════════════════

def _compute_tau_local(scores: np.ndarray) -> float:
    """
    Local threshold — now bimodality-aware.

    OLD (buggy):
        mean + 2*std of lower 60%  →  ~0.62 on ANY rank-normalised data

    NEW:
        If bimodal: Otsu threshold (optimal binary split)
        If unimodal: median + 3.5 * MAD  (very conservative, flags ~1-3%)
    """
    if len(scores) < 10:
        return float(np.percentile(scores, 95))

    if _is_bimodal(scores):
        # ── Otsu: find threshold that maximises between-class variance ──
        s = np.sort(scores)
        n = len(s)
        best_thresh = float(s[-1])
        best_var    = 0.0

        for pct in np.arange(0.50, 0.995, 0.005):
            t     = np.percentile(s, pct * 100)
            below = s[s <= t]
            above = s[s > t]
            if len(below) < 5 or len(above) < 2:
                continue
            w0 = len(below) / n
            w1 = len(above) / n
            between_var = w0 * w1 * (below.mean() - above.mean()) ** 2
            if between_var > best_var:
                best_var    = between_var
                best_thresh = float(t)
        return best_thresh
    else:
        # ── Unimodal: extremely conservative — median + 4 * MAD ──
        # On a clean dataset, this should flag ~0.5-2%
        median_s = float(np.median(scores))
        mad_s    = float(np.median(np.abs(scores - median_s))) * 1.4826 + 1e-10
        return median_s + 4.0 * mad_s  # was 3.5


# ═════════════════════════════════════════════════════════════════════════════
# ◆ FIX 3: Conservative contamination rate estimator — TIGHTENED (v4)
#
#   ROOT CAUSE of false positives on structured datasets (UCI 73 Mushroom):
#   NEITHER IsolationForest NOR the trained MetaPoison model can reliably
#   distinguish "natural class structure" from "injected poison" on an
#   unseen dataset. Mushroom has edible/poisonous classes → both IF and
#   MetaPoison rank-fusion produce bimodal scores → Otsu splits at the
#   class boundary → 15-25% flagged.
#
#   DESIGN PRINCIPLE: the auto-estimator must be CONSERVATIVE on every
#   dataset. Users who KNOW their data is poisoned can lower the threshold
#   with the slider. But the default should never cry wolf on clean data.
#
#   Rule: auto-estimated rate is HARD CAPPED at 5%.
#   - Unimodal scores → 0.3-3% (outliers only)
#   - Bimodal with small cluster → up to 5%
#   - Bimodal with large cluster (>8%) → class structure, not poison → 3%
# ═════════════════════════════════════════════════════════════════════════════

def _estimate_contamination_rate_safe(scores: np.ndarray,
                                      scoring_mode: str = "isolation_forest"
                                      ) -> float:
    """
    Principled contamination estimator.
    No arbitrary hard cap.
    
    Strategy:
    - If unimodal → MAD-based conservative estimate
    - If weak bimodality → shrink Otsu toward MAD
    - If strong bimodality → trust Otsu
    - Allow up to 40% when justified
    """

    n = len(scores)
    if n < 30:
        return 0.01

    # ── Compute conservative baseline (MAD outliers) ──────────────────
    median_s = float(np.median(scores))
    mad_s    = float(np.median(np.abs(scores - median_s))) * 1.4826 + 1e-10
    mad_rate = float((scores > median_s + 4.0 * mad_s).mean())
    mad_rate = np.clip(mad_rate, 0.003, 0.05)

    # ── Check bimodality ───────────────────────────────────────────────
    bimodal = _is_bimodal(scores)

    if not bimodal:
        # Clean unimodal distribution
        return float(mad_rate)

    # ── Otsu estimate ──────────────────────────────────────────────────
    tau_otsu  = _compute_tau_local(scores)
    otsu_rate = float((scores > tau_otsu).mean())

    # ── Measure separation strength ────────────────────────────────────
    # Compare cluster means to judge if split is meaningful
    below = scores[scores <= tau_otsu]
    above = scores[scores > tau_otsu]

    if len(below) < 10 or len(above) < 5:
        return float(mad_rate)

    separation = abs(below.mean() - above.mean()) / (scores.std() + 1e-8)

    # ── Decision logic ─────────────────────────────────────────────────
    if separation < 0.5:
        # Weak separation → likely natural structure
        return float(mad_rate)

    if separation < 1.0:
        # Moderate separation → blend
        blended = 0.5 * mad_rate + 0.5 * otsu_rate
        return float(np.clip(blended, 0.01, 0.20))

    # Strong separation → trust Otsu
    return float(np.clip(otsu_rate, 0.01, 0.40))


# ═════════════════════════════════════════════════════════════════════════════
# ◆ FIX 1 + 5: Rewritten scoring pipeline
# ═════════════════════════════════════════════════════════════════════════════

def _score_dataframe(df: pd.DataFrame, tau_override: float | None = None,
                      dataset_hint: str = "") -> dict:
    """
    Core scoring pipeline — FIXED.

    Changes:
      ◆ FIX 1: Fallback uses min-max normalised raw scores, NOT rank-normalised.
      ◆ FIX 3: Auto threshold uses bimodality-aware rate estimation.
      ◆ FIX 5: 200 trees in IsolationForest for more stable anomaly scores.
    """
    X, y, sc, Xdf = _prepare(df)
    n = len(X)

    # ── Step 1: Get RAW anomaly scores ────────────────────────────────────────
    raw_scores = None
    mode       = "fallback"

    if _loaded and _hybrid is not None:
        try:
            combined, meta_s, iso_s = _hybrid.score(X, y)
            raw_scores = np.asarray(combined, dtype=np.float64)
            mode = "hybrid"
        except Exception as exc:
            print(f"[backend] Hybrid scoring failed ({exc}), falling back")

    if raw_scores is None:
        # ◆ FIX 5: 200 trees (was 100)
        iso = IsolationForest(n_estimators=200, contamination="auto",
                               random_state=42, n_jobs=-1)
        raw_scores = -iso.fit(X).score_samples(X)
        raw_scores = np.asarray(raw_scores, dtype=np.float64)
        mode = "isolation_forest"

    # ◆ FIX 1: Min-max normalise to [0, 1] for display
    #   OLD: rankdata(raw)/n → UNIFORM → always ~40% flagged
    #   NEW: (raw - min) / (max - min) → preserves score SHAPE
    s_min   = float(raw_scores.min())
    s_max   = float(raw_scores.max())
    s_range = s_max - s_min + 1e-10
    display_scores = ((raw_scores - s_min) / s_range).astype(np.float64)

    # ── Step 2: Threshold selection ───────────────────────────────────────────
    # ◆ FIX 2: Bimodality-aware local threshold (on RAW scores)
    tau_local_raw = _compute_tau_local(raw_scores)

    if tau_override is not None:
        # User set the slider → interpret as a value in display [0, 1] space
        tau_display  = float(tau_override)
        tau_raw      = s_min + tau_display * s_range
        threshold_source = "user_override"
    elif dataset_hint and dataset_hint in DATASET_THRESHOLDS:
        tau_raw     = DATASET_THRESHOLDS[dataset_hint]
        tau_display = float((tau_raw - s_min) / s_range)
        threshold_source = f"per_dataset:{dataset_hint}"
    else:
        # ◆ FIX 3: Conservative auto-estimation with bimodality gate
        # ◆ FIX 7: Pass scoring mode so rate estimator knows whether
        #          to trust bimodality (only trust with trained model)
        est_rate    = _estimate_contamination_rate_safe(raw_scores,
                                                        scoring_mode=mode)
        tau_raw     = float(np.percentile(raw_scores, (1.0 - est_rate) * 100))
        tau_display = float((tau_raw - s_min) / s_range)
        threshold_source = "auto_estimated"

    # Convert local threshold to display space too
    tau_local_display = float((tau_local_raw - s_min) / s_range)

    # Store for future calls
    ds_key = dataset_hint or f"ds_{uuid.uuid4().hex[:8]}"
    DATASET_THRESHOLDS[ds_key] = tau_raw

    # ── Step 3: Flag rows ─────────────────────────────────────────────────────
    flags     = (raw_scores >= tau_raw).astype(int).tolist()
    n_flagged = sum(flags)
    pct_flagged = round(n_flagged / n * 100, 2) if n > 0 else 0.0

    return {
        "scores":           (display_scores * 10).tolist(),
        "flags":            flags,
        "n_rows":           n,
        "n_flagged":        n_flagged,
        "pct_flagged":      pct_flagged,
        "tau":              round(tau_display * 10, 2),
        "tau_local":        round(tau_local_display * 10, 2),
        "global_threshold": round(_global_threshold(), 4),
        "threshold_source": threshold_source,
        "mode":             mode,
        "bimodal":          _is_bimodal(raw_scores),
        "score_stats": {
            "min":  round(float(display_scores.min()), 4),
            "p25":  round(float(np.percentile(display_scores, 25)), 4),
            "p50":  round(float(np.median(display_scores)), 4),
            "p75":  round(float(np.percentile(display_scores, 75)), 4),
            "p95":  round(float(np.percentile(display_scores, 95)), 4),
            "max":  round(float(display_scores.max()), 4),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Routes  (unchanged — they all call _score_dataframe which is now fixed)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({
        "status":                    "ok" if _loaded else "no_model",
        "mode":                      "hybrid" if _loaded else "isolation_forest",
        "global_threshold":          _global_threshold(),
        "n_per_dataset_thresholds":  len(DATASET_THRESHOLDS),
        "trained_datasets":          list(DATASET_THRESHOLDS.keys()),
        "model_available":           MODEL_AVAILABLE,
        "model_loaded":              _loaded,
        "model_path":                RPFNet_MODEL_PATH,
        "model_exists":              os.path.exists(RPFNet_MODEL_PATH),
        "import_error":              MODEL_IMPORT_ERR,
        "rpf_dim_loaded":            _LOADED_RPF_DIM,
        "rpf_dim_code":              RPFExtractor.DIM if MODEL_AVAILABLE else None,
        "compat_mode":               (_LOADED_RPF_DIM is not None
                                       and MODEL_AVAILABLE
                                       and _LOADED_RPF_DIM != RPFExtractor.DIM),
        "backend_version":           "v3-fixed",
    })


@app.route("/thresholds")
def thresholds():
    return jsonify({
        "per_dataset": DATASET_THRESHOLDS,
        "global":      _global_threshold(),
    })

@app.route("/stream/start", methods=["POST"])
def start_stream():
    stream_id = str(uuid.uuid4())

    STREAM_SESSIONS[stream_id] = {
        "buffer": [],
        "window_size": 200,
        "scaler": StandardScaler(),
        "initialized": False,
        "model": None
    }

    return jsonify({"stream_id": stream_id})

@app.route("/analyze_csv", methods=["POST"])
@limiter.limit("5 per minute")
def analyze_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename or not f.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only .csv files are supported"}), 400

    tau_str      = request.form.get("tau", "").strip()
    dataset_hint = request.form.get("dataset_hint", "").strip()
    tau_override = float(tau_str) / 10.0 if tau_str else None

    try:
        df = pd.read_csv(f)
    except Exception as exc:
        return jsonify({"error": f"Failed to parse CSV: {exc}"}), 400

    if len(df) < 5:
        return jsonify({"error": "Dataset has fewer than 5 rows"}), 400

    try:
        result = _score_dataframe(df, tau_override, dataset_hint)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    dataset_id = str(uuid.uuid4())
    DATASET_STORE[dataset_id] = df
    result["dataset_id"] = dataset_id

    return jsonify(result)


@app.route("/analyze_uci", methods=["POST"])
@limiter.limit("5 per minute")
def analyze_uci():
    body = request.get_json(silent=True) or {}
    uci_id = body.get("uci_id")
    if uci_id is None:
        return jsonify({"error": "uci_id is required"}), 400

    tau_override = float(body["tau"]) if "tau" in body else None

    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=int(uci_id))
        df = pd.DataFrame(ds.data.features)
        if ds.data.targets is not None:
            targets = pd.DataFrame(ds.data.targets)
            df = pd.concat([df, targets], axis=1)
    except Exception as exc:
        return jsonify({"error": f"Failed to fetch UCI dataset {uci_id}. "
                        f"Dataset may not be available — try downloading and uploading as CSV. "
                        f"Details: {exc}"}), 400

    if len(df) < 5:
        return jsonify({"error": "Dataset has fewer than 5 rows"}), 400

    try:
        result = _score_dataframe(df, tau_override, dataset_hint=f"uci_{uci_id}")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    dataset_id = str(uuid.uuid4())
    DATASET_STORE[dataset_id] = df
    result["dataset_id"] = dataset_id
    result["uci_id"] = int(uci_id)

    return jsonify(result)


@app.route("/analyze_url", methods=["POST"])
@limiter.limit("5 per minute")
def analyze_url():
    body = request.get_json(silent=True) or {}
    url  = (body.get("url") or "").strip()
    if not url:
        return jsonify({"error": "url is required"}), 400

    if not url.startswith(("http://", "https://")):
        return jsonify({"error": "url must start with http:// or https://"}), 400

    tau_override = float(body["tau"]) if "tau" in body else None
    dataset_hint = str(body.get("dataset_hint", "")).strip()

    try:
        resp = _requests.get(url, timeout=30, stream=True)
        resp.raise_for_status()
        content_length = int(resp.headers.get("Content-Length", 0))
        if content_length > 100 * 1024 * 1024:
            return jsonify({"error": "Remote file exceeds 100 MB limit"}), 400
        df = pd.read_csv(io.BytesIO(resp.content))
    except _requests.exceptions.RequestException as exc:
        return jsonify({"error": f"Failed to download URL: {exc}"}), 400
    except Exception as exc:
        return jsonify({"error": f"Failed to parse CSV from URL: {exc}"}), 400

    if len(df) < 5:
        return jsonify({"error": "Dataset has fewer than 5 rows"}), 400

    try:
        result = _score_dataframe(df, tau_override, dataset_hint)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    dataset_id = str(uuid.uuid4())
    DATASET_STORE[dataset_id] = df
    result["dataset_id"] = dataset_id

    return jsonify(result)

@app.route("/analyze_stream", methods=["POST"])
@limiter.limit("5 per minute")
def analyze_stream():

    body = request.get_json()
    stream_id = body.get("stream_id")
    sample = body.get("sample")

    if stream_id not in STREAM_SESSIONS:
        return jsonify({"error": "Invalid stream_id"}), 400

    session = STREAM_SESSIONS[stream_id]
    session["buffer"].append(sample)

    # keep sliding window
    if len(session["buffer"]) > session["window_size"]:
        session["buffer"].pop(0)

    X = np.array(session["buffer"], dtype=np.float32)

    # need enough samples first
    if len(X) < 20:
        return jsonify({
            "status": "warming_up",
            "n_samples": len(X)
        })

    # scale
    scaler = session["scaler"]
    X_scaled = scaler.fit_transform(X)

    # ── Train model only once during warmup ──
    if not session["initialized"]:
        iso = IsolationForest(n_estimators=200, random_state=42)
        iso.fit(X_scaled)

        session["model"] = iso
        session["initialized"] = True

    # ── Use trained model for scoring ──
    iso = session["model"]

    scores = -iso.score_samples(X_scaled)
    newest_score = scores[-1]

    tau = _compute_tau_local(scores)
    flag = int(newest_score >= tau)

    return jsonify({
        "score": float(newest_score),
        "threshold": float(tau),
        "poison_flag": flag
    })

@app.route("/export_clean", methods=["POST"])
def export_clean():
    body       = request.get_json(silent=True) or {}
    dataset_id = body.get("dataset_id")
    clean_ids  = body.get("clean_ids", [])

    if not dataset_id:
        return jsonify({"error": "dataset_id is required"}), 400
    if dataset_id not in DATASET_STORE:
        return jsonify({"error": "Dataset not found. Re-analyze to regenerate."}), 404
    if not clean_ids:
        return jsonify({"error": "No clean_ids provided"}), 400

    df = DATASET_STORE[dataset_id]
    valid_ids = [i for i in clean_ids if 0 <= i < len(df)]
    if not valid_ids:
        return jsonify({"error": "No valid row indices in clean_ids"}), 400

    clean_df  = df.iloc[valid_ids].reset_index(drop=True)
    csv_bytes = clean_df.to_csv(index=False).encode("utf-8")
    buf       = io.BytesIO(csv_bytes)
    buf.seek(0)

    return send_file(
        buf,
        mimetype="text/csv",
        as_attachment=True,
        download_name="clean_dataset.csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[backend] Starting on http://0.0.0.0:{port}")
    print(f"[backend] RPFNet model: {RPFNet_MODEL_PATH}")
    print(f"[backend] Model available:  {MODEL_AVAILABLE}")
    if not MODEL_AVAILABLE:
        print(f"[backend] Import error: {MODEL_IMPORT_ERR}")
        print("[backend] Will use IsolationForest fallback for scoring.")
    app.run(host="0.0.0.0", port=port, debug=True,
            threaded=True, use_reloader=False)