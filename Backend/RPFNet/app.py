"""
Endpoints
GET  /health
GET  /thresholds
POST /analyze_csv
POST /analyze_uci
POST /analyze_url
POST /export_clean
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
from RPFNet.InvariantAnalyzer import InvariantAnalyzer
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

app = Flask(__name__)
CORS(app, origins="*")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

_lock = threading.Lock()
_meta:   "RPFNetPoisonDetector"   | None = None
_hybrid: "HybridEnsembleDetector" | None = None
_loaded = False
_invariant_analyzer = InvariantAnalyzer(alpha=0.05)

DATASET_STORE: dict[str, pd.DataFrame] = {}
DATASET_THRESHOLDS: dict[str, float] = {}
GLOBAL_THRESHOLD_DEFAULT = 3.5
STREAM_SESSIONS = {}
limiter = Limiter(get_remote_address, app=app, default_limits=["10 per minute"])

def _global_threshold() -> float:
    if DATASET_THRESHOLDS:
        return float(np.mean(list(DATASET_THRESHOLDS.values())))
    return GLOBAL_THRESHOLD_DEFAULT

_LOADED_RPF_DIM = None

def _load_model_compat(meta, path):
    global _LOADED_RPF_DIM

    try:
        meta.load(path)
        _LOADED_RPF_DIM = RPFExtractor.DIM
        return True
    except ValueError as e:
        if "RPF dim" not in str(e):
            raise

    print(f"[backend] Dim mismatch detected — loading in backward-compat mode")
    try:
        ckpt = torch.load(path, map_location=meta.device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=meta.device)
    saved_dim = ckpt.get("rpf_dim", 61)
    print(f"[backend]   saved_dim={saved_dim}  current_dim={RPFExtractor.DIM}")

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

    _original_score_rpf = meta._score_rpf

    def _patched_score_rpf(rpf):
        if rpf.shape[1] > saved_dim:
            rpf = rpf[:, :saved_dim]
        return _original_score_rpf(rpf)

    meta._score_rpf = _patched_score_rpf

    v = ckpt.get("version", 2)
    print(f"[backend] Loaded v{v} model (dim={saved_dim}) in compat mode - "
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
    MAX_CARDINALITY = 20
    for c in Xdf.select_dtypes(include="number").columns:
        Xdf[c] = Xdf[c].fillna(Xdf[c].median())
    for c in Xdf.select_dtypes(exclude="number").columns:
        Xdf[c] = Xdf[c].fillna("missing")
        if Xdf[c].nunique() > MAX_CARDINALITY:
            freq = Xdf[c].value_counts(normalize=True)
            Xdf[c] = Xdf[c].map(freq).astype(np.float32)
    Xdf = pd.get_dummies(Xdf)

    X = Xdf.values.astype(np.float32)
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X).astype(np.float32)

    return X_scaled, y, sc, Xdf

def _is_bimodal(scores: np.ndarray) -> bool:
    if len(scores) < 30:
        return False

    s = np.sort(scores)
    n = len(s)
    score_range = float(s[-1] - s[0]) + 1e-10
    signals = 0

    upper_half = s[n // 2:]
    diffs = np.diff(upper_half)
    if len(diffs) > 0 and diffs.max() / score_range > 0.15:
        signals += 1

    mu  = s.mean()
    std = s.std() + 1e-10
    kurt = float(np.mean(((s - mu) / std) ** 4)) - 3.0
    if kurt < -1.0:
        signals += 1

    top_3pct = s[0] + 0.97 * score_range
    tail_mass = float((s >= top_3pct).mean())
    if tail_mass > 0.20:
        signals += 1

    try:
        n_bins = min(50, max(10, n // 100))
        counts, edges = np.histogram(s, bins=n_bins)
        max_count = counts.max()
        for i in range(2, len(counts) - 2):
            if (counts[i] < 0.30 * max_count
                and counts[i-1] > counts[i] and counts[i-2] > counts[i]
                and counts[i+1] > counts[i] and counts[i+2] > counts[i]):
                signals += 1
                break
    except Exception:
        pass

    return signals >= 2

def _compute_tau_local(scores: np.ndarray) -> float:
    if len(scores) < 10:
        return float(np.percentile(scores, 95))

    if _is_bimodal(scores):
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
        median_s = float(np.median(scores))
        mad_s    = float(np.median(np.abs(scores - median_s))) * 1.4826 + 1e-10
        return median_s + 4.0 * mad_s

def _estimate_contamination_rate_safe(scores: np.ndarray, scoring_mode: str = "isolation_forest") -> float:

    n = len(scores)
    if n < 30:
        return 0.01

    median_s = float(np.median(scores))
    mad_s    = float(np.median(np.abs(scores - median_s))) * 1.4826 + 1e-10
    mad_rate = float((scores > median_s + 4.0 * mad_s).mean())
    mad_rate = np.clip(mad_rate, 0.0, 0.05)

    bimodal = _is_bimodal(scores)

    if not bimodal:
        return float(mad_rate)

    tau_otsu  = _compute_tau_local(scores)
    otsu_rate = float((scores > tau_otsu).mean())

    below = scores[scores <= tau_otsu]
    above = scores[scores > tau_otsu]

    if len(below) < 10 or len(above) < 5:
        return float(mad_rate)

    separation = abs(below.mean() - above.mean()) / (scores.std() + 1e-8)

    if separation < 0.5:
        return float(mad_rate)

    if separation < 1.0:
        blended = 0.5 * mad_rate + 0.5 * otsu_rate
        return float(np.clip(blended, 0.01, 0.20))

    return float(np.clip(otsu_rate, 0.01, 0.40))

def _score_dataframe(df: pd.DataFrame, tau_override: float | None = None,
                dataset_hint: str = "") -> dict:

    DATASET_THRESHOLDS.clear()

    X, y, sc, Xdf = _prepare(df)
    n = len(X)

    raw_scores = None
    mode = "fallback"

    if _loaded and _hybrid is not None:
        try:
            combined, meta_s, iso_s = _hybrid.score(X, y)
            raw_scores = np.asarray(combined, dtype=np.float64)
            mode = "hybrid"
        except Exception as exc:
            print(f"[backend] Hybrid scoring failed ({exc}), falling back")

    if raw_scores is None:
        iso = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=42,
            n_jobs=-1
        )
        raw_scores = -iso.fit(X).score_samples(X)
        raw_scores = np.asarray(raw_scores, dtype=np.float64)
        mode = "isolation_forest"

    if raw_scores.std() < 1e-4:
        return {
            "scores": ((raw_scores - raw_scores.min()) /
                       (raw_scores.ptp() + 1e-10) * 10).tolist(),
            "flags": [0] * n,
            "invariant_violations": [[] for _ in range(n)],
            "violation_details": [[] for _ in range(n)],
            "n_rows": n,
            "n_flagged": 0,
            "pct_flagged": 0.0,
            "tau": 0,
            "tau_local": 0,
            "global_threshold": _global_threshold(),
            "threshold_source": "clean_detected",
            "mode": mode,
            "bimodal": False,
            "score_stats": {}
        }

    s_min = float(raw_scores.min())
    s_max = float(raw_scores.max())
    s_range = s_max - s_min + 1e-10

    display_scores = ((raw_scores - s_min) / s_range).astype(np.float64)

    tau_local_raw = _compute_tau_local(raw_scores)

    if tau_override is not None:
        tau_display  = float(tau_override)
        tau_raw      = s_min + tau_display * s_range
        threshold_source = "user_override"

    elif dataset_hint and dataset_hint in DATASET_THRESHOLDS:
        tau_raw = DATASET_THRESHOLDS[dataset_hint]
        tau_display = float((tau_raw - s_min) / s_range)
        threshold_source = f"per_dataset:{dataset_hint}"

    else:
        est_rate = _estimate_contamination_rate_safe(
            raw_scores, scoring_mode=mode
        )

        if est_rate <= 0.0:
            tau_raw = float("inf")
        else:
            tau_raw = float(
                np.percentile(raw_scores, (1.0 - est_rate) * 100)
            )

        tau_display = float((tau_raw - s_min) / s_range) \
            if np.isfinite(tau_raw) else 1.0

        threshold_source = "auto_estimated"

    tau_local_display = float((tau_local_raw - s_min) / s_range)

    ds_key = dataset_hint or f"ds_{uuid.uuid4().hex[:8]}"
    DATASET_THRESHOLDS[ds_key] = tau_raw

    invariant_violations = []
    violation_details = []

    try:
        if _loaded and _meta is not None:
            rpf = _meta.extractor.extract(X, y)
        else:
            extractor = RPFExtractor()
            rpf = extractor.extract(X, y)

        _invariant_analyzer.fit_clean_bounds(X, y)
        legacy, details = _invariant_analyzer.compute_row_violation_details(
            X, y, feature_names=list(Xdf.columns)
        )

        invariant_violations = legacy
        violation_details = details

    except Exception as e:
        print(f"[backend] Invariant analysis failed: {e}")
        invariant_violations = [[] for _ in range(len(X))]
        violation_details = [[] for _ in range(len(X))]

    bimodal = _is_bimodal(raw_scores)

    clean_distribution = False
    if not bimodal and n >= 30:
        mu = raw_scores.mean()
        std = raw_scores.std() + 1e-10
        skewness = float(np.mean(((raw_scores - mu) / std) ** 3))
        p95 = float(np.percentile(raw_scores, 95))
        tail_ratio = (s_max - p95) / s_range
        cv = std / (abs(mu) + 1e-10)
        if skewness <= 2.0 and (tail_ratio < 0.15 or (cv < 0.3 and skewness < 1.0)):
            clean_distribution = True

    if tau_override is not None:
        flags = (raw_scores >= tau_raw).astype(int).tolist()
    elif clean_distribution:
        flags = []
        for i in range(n):
            has_violation = bool(invariant_violations and len(invariant_violations[i]) > 0)
            score_above = raw_scores[i] >= tau_raw
            flags.append(int(score_above and has_violation))
        threshold_source = "auto_suppressed:clean_distribution"
    else:
        flags = []
        for i in range(n):
            has_violation = bool(invariant_violations and len(invariant_violations[i]) > 0)
            score_above = raw_scores[i] >= tau_raw
            flags.append(int(score_above and has_violation))

    n_flagged = sum(flags)
    pct_flagged = round(n_flagged / n * 100, 2) if n > 0 else 0.0

    return {
        "scores": (display_scores * 10).tolist(),
        "flags": flags,
        "invariant_violations": invariant_violations,
        "violation_details": violation_details,
        "n_rows": n,
        "n_flagged": n_flagged,
        "pct_flagged": pct_flagged,
        "tau": round(tau_display * 10, 2),
        "tau_local": round(tau_local_display * 10, 2),
        "global_threshold": round(_global_threshold(), 4),
        "threshold_source": threshold_source,
        "mode":             mode,
        "bimodal":          bimodal,
        "clean_distribution": clean_distribution,
        "score_stats": {
            "min": round(float(display_scores.min()), 4),
            "p25": round(float(np.percentile(display_scores, 25)), 4),
            "p50": round(float(np.median(display_scores)), 4),
            "p75": round(float(np.percentile(display_scores, 75)), 4),
            "p95": round(float(np.percentile(display_scores, 95)), 4),
            "max": round(float(display_scores.max()), 4),
        },
    }

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

    if len(session["buffer"]) > session["window_size"]:
        session["buffer"].pop(0)

    X = np.array(session["buffer"], dtype=np.float32)

    if len(X) < 20:
        return jsonify({
            "status": "warming_up",
            "n_samples": len(X)
        })

    scaler = session["scaler"]
    X_scaled = scaler.fit_transform(X)

    if not session["initialized"]:
        iso = IsolationForest(n_estimators=200, random_state=42)
        iso.fit(X_scaled)

        session["model"] = iso
        session["initialized"] = True

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