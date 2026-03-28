from __future__ import annotations

import importlib
import importlib.resources as _pkg_resources
import io
import os
import warnings
from typing import Literal, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Package-relative base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_bundled_model() -> str:
    local = os.path.join(BASE_DIR, "RPFNet_universal.pt")
    if os.path.exists(local):
        return local

    try:
        ref = _pkg_resources.files("RPFNet").joinpath("RPFNet_universal.pt")
        with _pkg_resources.as_file(ref) as p:
            if os.path.exists(str(p)):
                return str(p)
    except Exception:
        pass

    try:
        import pkg_resources as _pr
        path = _pr.resource_filename("RPFNet", "RPFNet_universal.pt")
        if os.path.exists(path):
            return path
    except Exception:
        pass

    return local

# Global model state
_meta = None
_hybrid = None
_model_loaded = False
_model_init_attempted = False

# Populated by _try_import_backend()
_MetaPoisonDetector = None
_HybridEnsembleDetector = None
_RateEstimatorHead = None
_MetaPoisonNet = None
_RPFExtractor_cls = None

_META_MODEL_PATH = _resolve_bundled_model()   # robust absolute path — set BEFORE backend import
_META_EPOCHS = 30
_META_BATCH_SIZE = 512
_META_LR = 1e-3
_RPF_K_SMALL = 5
_RPF_K_LARGE = 15
_RPF_CV_FOLDS = 3
_BACKEND_AVAILABLE = False
_LOADED_RPF_DIM = None   # actual dim of the loaded checkpoint

# Backend import
def _try_import_backend() -> bool:
    global _MetaPoisonDetector, _HybridEnsembleDetector
    global _RateEstimatorHead, _MetaPoisonNet, _RPFExtractor_cls
    global _META_MODEL_PATH, _META_EPOCHS, _META_BATCH_SIZE, _META_LR
    global _RPF_K_SMALL, _RPF_K_LARGE, _RPF_CV_FOLDS, _BACKEND_AVAILABLE

    def _import(mod_name: str):
        """Try relative then absolute import, return module or raise."""
        last_err = None
        for pkg, rel in (("RPFNet", True), (None, False)):
            try:
                if rel:
                    return importlib.import_module(f".{mod_name}", package="RPFNet")
                else:
                    return importlib.import_module(f"RPFNet.{mod_name}")
            except Exception as e:
                last_err = e
        raise ImportError(f"Cannot import {mod_name}: {last_err}") from last_err

    # Core submodules
    try:
        rpf_mod = _import("RPFExtractor")
        rate_mod = _import("RateEstimator")
        _RPFExtractor_cls  = rpf_mod.RPFExtractor
        _RateEstimatorHead = rate_mod.RateEstimatorHead
    except Exception as e:
        print(f"[rpfnet] Core submodule import failed: {e}")
        return False

    # detection.py
    try:
        det = _import("detection")
    except Exception as e:
        print(f"[rpfnet] detection.py import failed: {e}")
        return False

    if not hasattr(det, "RPFNetPoisonDetector"):
        print("[rpfnet] detection.py missing RPFNetPoisonDetector")
        return False

    _MetaPoisonDetector = det.RPFNetPoisonDetector
    _HybridEnsembleDetector = det.HybridEnsembleDetector
    _MetaPoisonNet = det.RPFNetPoisonDetector

    detected_path = getattr(det, "RPFNet_MODEL_PATH", None)
    if detected_path and os.path.isabs(detected_path) and os.path.exists(detected_path):
        _META_MODEL_PATH = detected_path

    _META_EPOCHS = getattr(det, "RPFNet_EPOCHS",     _META_EPOCHS)
    _META_BATCH_SIZE = getattr(det, "RPFNet_BATCH_SIZE", _META_BATCH_SIZE)
    _META_LR = getattr(det, "RPFNet_LR",         _META_LR)
    # k / cv values from detection.py if present, else keep defaults
    _RPF_K_SMALL = getattr(det, "RPF_K_SMALL",  _RPF_K_SMALL)
    _RPF_K_LARGE = getattr(det, "RPF_K_LARGE",  _RPF_K_LARGE)
    _RPF_CV_FOLDS = getattr(det, "RPF_CV_FOLDS", _RPF_CV_FOLDS)

    _BACKEND_AVAILABLE = True
    return True


_try_import_backend()


# Lazy model loader
def _load_model_compat(meta, path: str) -> bool:
    global _LOADED_RPF_DIM

    import torch

    # Normal load
    try:
        meta.load(path)
        _LOADED_RPF_DIM = getattr(_RPFExtractor_cls, "DIM", None)
        return True
    except ValueError as e:
        if "RPF dim" not in str(e):
            raise

    # Dim-mismatch: backward-compat load
    print("[rpfnet] Dim mismatch — loading checkpoint in backward-compat mode")
    try:
        ckpt = torch.load(path, map_location=meta.device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=meta.device)

    saved_dim = ckpt.get("rpf_dim", 47)
    current_dim = getattr(_RPFExtractor_cls, "DIM", "?")
    print(f"[rpfnet]   saved_dim={saved_dim}  current_dim={current_dim}")

    _NetClass = _MetaPoisonNet if _MetaPoisonNet is not None else type(meta.net)
    meta.net = _NetClass(input_dim=saved_dim).to(meta.device)
    meta.net.load_state_dict(ckpt["net"])
    meta._threshold = ckpt.get("threshold", 0.5)
    meta._fitted    = ckpt.get("fitted", True)

    if hasattr(meta, "extractor"):
        meta.extractor.k_small = ckpt.get("k_small",  _RPF_K_SMALL)
        meta.extractor.k_large = ckpt.get("k_large",  _RPF_K_LARGE)
        meta.extractor.cv_folds = ckpt.get("cv_folds", _RPF_CV_FOLDS)

    if "rate_head" in ckpt:
        meta.rate_head = _RateEstimatorHead().to(meta.device)
        meta.rate_head.load_state_dict(ckpt["rate_head"])
        meta.rate_head.eval()

    meta.net.eval()
    _LOADED_RPF_DIM = saved_dim

    # Monkey-patch scorer to truncate RPF features to saved_dim
    _orig = meta._score_rpf
    def _patched(rpf, _o=_orig, _d=saved_dim):
        return _o(rpf[:, :_d] if rpf.shape[1] > _d else rpf)
    meta._score_rpf = _patched

    v = ckpt.get("version", 2)
    print(f"[rpfnet] Loaded v{v} checkpoint (dim={saved_dim}) in compat mode")
    return True


def _ensure_model():
    """Lazy-load the trained RPFNet model on first use."""
    global _meta, _hybrid, _model_loaded, _model_init_attempted

    if _model_loaded or _model_init_attempted:
        return
    _model_init_attempted = True

    if not _BACKEND_AVAILABLE:
        print("[rpfnet] Backend not available — using IsolationForest fallback")
        return

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        meta = _MetaPoisonDetector(
            device=device,
            epochs=_META_EPOCHS,
            batch_size=_META_BATCH_SIZE,
            lr=_META_LR,
        )

        # Candidate paths — absolute resolved path first, then legacy fallbacks
        candidates = [
            _META_MODEL_PATH,                               # robustly resolved absolute path
            os.path.join(BASE_DIR, "RPFNet_universal.pt"), # same dir as api.py
            os.path.join(BASE_DIR, "..", "RPFNet_universal.pt"),
        ]

        model_path = None
        seen: set[str] = set()
        for p in candidates:
            p = os.path.normpath(p)
            if p not in seen:
                seen.add(p)
                if os.path.exists(p):
                    model_path = p
                    break

        if model_path is None:
            print("[rpfnet] No checkpoint found — using IsolationForest fallback")
            print(f"[rpfnet] Searched: {sorted(seen)}")
            return

        _load_model_compat(meta, model_path)

        hybrid = _HybridEnsembleDetector(meta)
        fusion_path = model_path.replace(".pt", "_fusion.pt")
        if os.path.exists(fusion_path):
            hybrid.load(fusion_path)
        else:
            hybrid.attach_rate_head(meta.rate_head)

        _meta   = meta
        _hybrid = hybrid
        _model_loaded = True
        print(f"[rpfnet] Model loaded from {model_path}  device={device}")

    except Exception as e:
        print(f"[rpfnet] Model load failed: {e} — using IsolationForest fallback")

# DataFrame preparation  (identical to app.py)
_TARGET_CANDIDATES = [
    "loan_status", "Loan_Status", "deposit", "default", "Default",
    "label", "target", "y", "class", "outcome", "Target",
    "is_fraud", "fraud", "Class",]


def _infer_target(df: pd.DataFrame) -> str:
    for c in _TARGET_CANDIDATES:
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

# Bimodality & threshold helpers
def _is_bimodal(scores: np.ndarray) -> bool:
    if len(scores) < 30:
        return False

    s = np.sort(scores)
    n = len(s)
    score_range = float(s[-1] - s[0]) + 1e-10
    signals = 0

    # Large gap in the upper half of scores
    upper_half = s[n // 2:]
    diffs = np.diff(upper_half)
    if len(diffs) > 0 and diffs.max() / score_range > 0.15:
        signals += 1

    # Clearly platykurtic distribution
    mu  = s.mean()
    std = s.std() + 1e-10
    kurt = float(np.mean(((s - mu) / std) ** 4)) - 3.0
    if kurt < -1.0:
        signals += 1

    # Concentrated tail mass in the top 3% of the score range
    top_3pct = s[0] + 0.97 * score_range
    if float((s >= top_3pct).mean()) > 0.20:
        signals += 1

    # Histogram valley between two peaks
    try:
        n_bins = min(50, max(10, n // 100))
        counts, _ = np.histogram(s, bins=n_bins)
        max_count = counts.max()
        for i in range(2, len(counts) - 2):
            if (counts[i] < 0.30 * max_count
                    and counts[i - 1] > counts[i] and counts[i - 2] > counts[i]
                    and counts[i + 1] > counts[i] and counts[i + 2] > counts[i]):
                signals += 1
                break
    except Exception:
        pass

    return signals >= 2


def _compute_tau(scores: np.ndarray) -> float:
    if len(scores) < 10:
        return float(np.percentile(scores, 95))

    if _is_bimodal(scores):
        s = np.sort(scores)
        n = len(s)
        best_thresh = float(s[-1])
        best_var = 0.0
        for pct in np.arange(0.50, 0.995, 0.005):
            t = np.percentile(s, pct * 100)
            below = s[s <= t]
            above = s[s > t]
            if len(below) < 5 or len(above) < 2:
                continue
            w0 = len(below) / n
            w1 = len(above) / n
            bv = w0 * w1 * (below.mean() - above.mean()) ** 2
            if bv > best_var:
                best_var    = bv
                best_thresh = float(t)
        return best_thresh
    else:
        med = float(np.median(scores))
        mad = float(np.median(np.abs(scores - med))) * 1.4826 + 1e-10
        return med + 4.0 * mad


def _estimate_rate(scores: np.ndarray) -> float:
    n = len(scores)
    if n < 30:
        return 0.01

    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med))) * 1.4826 + 1e-10
    mad_rate = float(np.clip((scores > med + 4.0 * mad).mean(), 0.003, 0.05))

    if not _is_bimodal(scores):
        return mad_rate

    tau_otsu = _compute_tau(scores)
    otsu_rate = float((scores > tau_otsu).mean())
    below = scores[scores <= tau_otsu]
    above = scores[scores > tau_otsu]

    if len(below) < 10 or len(above) < 5:
        return mad_rate

    sep = abs(below.mean() - above.mean()) / (scores.std() + 1e-8)

    if sep < 0.5:
        return mad_rate
    if sep < 1.0:
        return float(np.clip(0.5 * mad_rate + 0.5 * otsu_rate, 0.01, 0.20))
    return float(np.clip(otsu_rate, 0.01, 0.40))

# Core scoring pipeline
def _score_dataframe(df: pd.DataFrame) -> dict:
    _ensure_model()
    X, y, sc, Xdf = _prepare(df)
    n = len(X)

    raw_scores: np.ndarray | None = None
    mode = "isolation_forest"

    # Try trained RPFNet hybrid scorer first
    if _model_loaded and _hybrid is not None:
        try:
            combined, _, _ = _hybrid.score(X, y)
            raw_scores = np.asarray(combined, dtype=np.float64)
            mode = "rpfnet_hybrid"
        except Exception as exc:
            print(f"[rpfnet] Hybrid scoring failed ({exc}), falling back")

    # IsolationForest fallback
    if raw_scores is None:
        iso = IsolationForest(n_estimators=200, contamination="auto",
                              random_state=42, n_jobs=-1)
        raw_scores = -iso.fit(X).score_samples(X)
        raw_scores = np.asarray(raw_scores, dtype=np.float64)

    # Min-max normalise for display
    s_min = float(raw_scores.min())
    s_range = float(raw_scores.max()) - s_min + 1e-10
    display = ((raw_scores - s_min) / s_range).astype(np.float64)

    # Threshold & flags
    est_rate = _estimate_rate(raw_scores)
    tau_raw = float(np.percentile(raw_scores, (1.0 - est_rate) * 100))
    flags = (raw_scores >= tau_raw).astype(int)

    n_flagged = int(flags.sum())
    flagged_idx = np.where(flags == 1)[0].tolist()
    clean_idx = np.where(flags == 0)[0].tolist()

    annotated = df.copy()
    annotated["_poison_score"] = display
    annotated["_poison_flag"]  = flags

    return {
        "n_rows": n,
        "n_flagged": n_flagged,
        "n_clean": n - n_flagged,
        "pct_flagged": round(n_flagged / n * 100, 2) if n > 0 else 0.0,
        "estimated_rate": round(est_rate, 4),
        "mode": mode,
        "bimodal": _is_bimodal(raw_scores),
        "scores": display.tolist(),
        "flags": flags.tolist(),
        "flagged_indices": flagged_idx,
        "clean_indices": clean_idx,
        "dataframe": annotated,
        "score_stats": {
            "min": round(float(display.min()), 4),
            "p25": round(float(np.percentile(display, 25)), 4),
            "p50": round(float(np.median(display)), 4),
            "p75": round(float(np.percentile(display, 75)), 4),
            "p95": round(float(np.percentile(display, 95)), 4),
            "max": round(float(display.max()), 4),
        },
    }

# Data loaders
def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def _load_uci(uci_id: int) -> pd.DataFrame:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        raise ImportError(
            "ucimlrepo required for UCI datasets.  "
            "Install with: pip install ucimlrepo"
        )
    ds = fetch_ucirepo(id=uci_id)
    if ds is None:
        raise ValueError(f"UCI dataset {uci_id} not found.")
    df = pd.DataFrame(ds.data.features)
    if ds.data.targets is not None:
        df = pd.concat([df, pd.DataFrame(ds.data.targets)], axis=1)
    return df


def _load_url(url: str) -> pd.DataFrame:
    from urllib.parse import urlparse
    if urlparse(url).scheme not in ("http", "https"):
        raise ValueError("URL must start with http:// or https://")
    try:
        import requests as _req
        resp = _req.get(url, timeout=60)
        resp.raise_for_status()
        return pd.read_csv(io.BytesIO(resp.content))
    except ImportError:
        return pd.read_csv(url)

# Stream state 
class _StreamState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.buffer: list = []
        self.buffer_df: pd.DataFrame | None = None
        self.scores: list = []
        self.flags: list = []
        self.scaler: StandardScaler = StandardScaler()
        self.model: IsolationForest | None = None
        self.initialized: bool = False
        self.window_size: int = 500
        self.warmup: int = 20
        self.columns: list | None = None


_stream = _StreamState()


def _stream_init_baseline(df: pd.DataFrame) -> dict:
    """Initialise stream with a full dataset as baseline."""
    _stream.reset()
    _stream.buffer_df = df.copy()
    _stream.columns = list(df.columns)

    X, y, sc, Xdf = _prepare(df)
    _stream.scaler = StandardScaler()
    X_scaled = _stream.scaler.fit_transform(X)
    _stream.n_features = X_scaled.shape[1]

    _stream.buffer.extend([row.copy() for row in X_scaled])

    _stream.model = IsolationForest(n_estimators=200, random_state=42, n_jobs=-1)
    _stream.model.fit(X_scaled)
    _stream.initialized = True

    raw = -_stream.model.score_samples(X_scaled)
    tau = _compute_tau(raw)

    _stream.scores = [float(s) for s in raw]
    _stream.flags = [bool(s >= tau) for s in raw]

    n_flagged = sum(_stream.flags)
    display = (raw - raw.min()) / (raw.max() - raw.min() + 1e-10)

    annotated = df.copy()
    annotated["_poison_score"] = display
    annotated["_poison_flag"] = [int(f) for f in _stream.flags]

    return {
        "status": "initialized",
        "n_rows": len(df),
        "n_flagged": n_flagged,
        "n_clean": len(df) - n_flagged,
        "pct_flagged": round(n_flagged / len(df) * 100, 2),
        "mode": "stream",
        "scores": display.tolist(),
        "flags": [int(f) for f in _stream.flags],
        "flagged_indices": [i for i, f in enumerate(_stream.flags) if f],
        "clean_indices": [i for i, f in enumerate(_stream.flags) if not f],
        "dataframe": annotated,
    }


def _stream_score_row(row) -> dict:
    """Score a single new sample against the stream model."""
    if isinstance(row, dict):
        row = list(row.values())
    if isinstance(row, pd.Series):
        row = row.values
    sample = np.asarray(row, dtype=np.float32).ravel()
    if _stream.n_features is not None:
        if len(sample) > _stream.n_features:
            sample = sample[:_stream.n_features]          # trim extra cols
        elif len(sample) < _stream.n_features:
            sample = np.pad(sample, (0, _stream.n_features - len(sample)))
    _stream.buffer.append(sample)

    # Sliding window — drop oldest sample + corresponding score/flag
    if len(_stream.buffer) > _stream.window_size:
        _stream.buffer.pop(0)
        if _stream.scores:
            _stream.scores.pop(0)
        if _stream.flags:
            _stream.flags.pop(0)

    X = np.vstack([np.asarray(r, dtype=np.float32).ravel() for r in _stream.buffer])

    # Warmup phase — not enough data yet
    if len(X) < _stream.warmup:
        return {
            "status": "warming_up",
            "n_samples": len(X),
            "warmup": _stream.warmup,
            "score": None,
            "threshold": None,
            "poison_flag": None,
        }

    X_scaled = _stream.scaler.fit_transform(X)

    # First time we have enough data — train model and backfill scores
    if not _stream.initialized:
        _stream.model = IsolationForest(n_estimators=200, random_state=42, n_jobs=-1)
        _stream.model.fit(X_scaled)
        _stream.initialized = True

        raw = -_stream.model.score_samples(X_scaled)
        tau = _compute_tau(raw)
        _stream.scores = [float(s) for s in raw]
        _stream.flags = [bool(s >= tau) for s in raw]

        newest_score = float(raw[-1])
        flag = bool(raw[-1] >= tau)
    else:
        raw_all = -_stream.model.score_samples(X_scaled)
        newest_score = float(raw_all[-1])
        tau = _compute_tau(raw_all)
        flag = newest_score >= tau

        _stream.scores.append(newest_score)
        _stream.flags.append(bool(flag))

    n_flagged = sum(_stream.flags)
    n_total = len(_stream.buffer)

    return {
        "status": "active",
        "score": round(newest_score, 6),
        "threshold": round(float(tau), 6),
        "poison_flag": bool(flag),
        "n_samples": n_total,
        "n_flagged": n_flagged,
        "n_clean": n_total - n_flagged,
        "buffer_pct_flagged": round(n_flagged / n_total * 100, 2),
    }


def _stream_status() -> dict:
    """Return current stream status without consuming any data."""
    n = len(_stream.buffer)
    n_flagged = sum(_stream.flags) if _stream.flags else 0
    return {
        "status":      "active"     if _stream.initialized else
                       "warming_up" if n > 0                else "empty",
        "n_samples": n,
        "n_flagged": n_flagged,
        "n_clean": n - n_flagged,
        "initialized": _stream.initialized,
        "window_size": _stream.window_size,
    }

class AnalysisReport:

    def __init__(self, data: dict):
        self._data = data

    def summary(self):
        s = self._data

        print("\nRPFNet Poison Analysis")
        print("──────────────────────")
        print(f"Rows analyzed     : {s['n_rows']}")
        print(f"Flagged samples   : {s['n_flagged']} ({s['pct_flagged']}%)")
        print(f"Estimated poison  : {s['estimated_rate']*100:.2f}%")
        print(f"Mode              : {s['mode']}")
        print(f"Bimodal           : {s['bimodal']}")
        print()

        stats = s["score_stats"]
        print("Score statistics")
        print(
            f"min={stats['min']}  "
            f"p25={stats['p25']}  "
            f"median={stats['p50']}  "
            f"p75={stats['p75']}  "
            f"p95={stats['p95']}  "
            f"max={stats['max']}"
        )

        if s["flagged_indices"]:
            print("\nFlagged indices:", s["flagged_indices"])
        else:
            print("\nNo poisoned samples detected.")

    def dataframe(self):
        return self._data["dataframe"]

    def clean(self):
        df = self._data["dataframe"]
        return df[df["_poison_flag"] == 0].drop(columns=["_poison_flag","_poison_score"])

    def flagged(self):
        df = self._data["dataframe"]
        return df[df["_poison_flag"] == 1]

    def raw(self):
        return self._data

    def __repr__(self):
        s = self._data
        return (
            f"<RPFNet Report rows={s['n_rows']} "
            f"flagged={s['n_flagged']} "
            f"({s['pct_flagged']}%) "
            f"mode={s['mode']}>"
        )

# PUBLIC API
def analyze(
    source: Literal["uci", "csv", "url", "stream"],
    dataset: Union[int, str, list, dict, np.ndarray, pd.DataFrame,
                   pd.Series, None] = None,) -> dict:
    """
    Analyze data for poisoning.

    Parameters
    ----------
    source : 'uci' | 'csv' | 'url' | 'stream'
    dataset :
        'csv'    → file path (str)
        'uci'    → UCI dataset ID (int)
        'url'    → URL string (str)
        'stream' → DataFrame  — initialise/re-initialise baseline
                   list / dict / ndarray / Series — score one new row
                   None — return current stream status

    Returns
    -------
    dict
        Batch mode  : n_rows, n_flagged, n_clean, pct_flagged,
                      estimated_rate, mode, bimodal, scores, flags,
                      flagged_indices, clean_indices, dataframe, score_stats
        Stream init : same shape as batch + status='initialized'
        Stream row  : score, threshold, poison_flag, n_samples, …
        Stream status: status, n_samples, n_flagged, initialized, …

    Examples
    --------
    >>> report = api.analyze('uci', 73)
    >>> print(report['pct_flagged'])

    >>> report = api.analyze('stream', training_df)
    >>> result = api.analyze('stream', [1.2, 3.4, 5.6])
    >>> print(result['poison_flag'])

    >>> status = api.analyze('stream')
    >>> print(status['n_samples'])
    """
    warnings.warn(
        "rpfnet is in BETA — APIs may change and results should be validated.",
        UserWarning,
        stacklevel=2,
    )

    # Batch sources
    if source == "csv":
        if not isinstance(dataset, str):
            raise TypeError("source='csv' requires dataset as a file path (str)")
        df = _load_csv(dataset)
        if len(df) < 5:
            raise ValueError(f"Dataset too small ({len(df)} rows). Need >= 5.")
        return AnalysisReport(_score_dataframe(df))

    if source == "uci":
        if not isinstance(dataset, int):
            raise TypeError("source='uci' requires dataset as a UCI ID (int)")
        df = _load_uci(dataset)
        if len(df) < 5:
            raise ValueError(f"Dataset too small ({len(df)} rows). Need >= 5.")
        return AnalysisReport(_score_dataframe(df))

    if source == "url":
        if not isinstance(dataset, str):
            raise TypeError("source='url' requires dataset as a URL (str)")
        df = _load_url(dataset)
        if len(df) < 5:
            raise ValueError(f"Dataset too small ({len(df)} rows). Need >= 5.")
        return AnalysisReport(_score_dataframe(df))

    # Stream
    if source == "stream":
        if dataset is None:
            return _stream_status()
        if isinstance(dataset, pd.DataFrame):
            if len(dataset) < 5:
                raise ValueError(f"Baseline too small ({len(dataset)} rows). Need >= 5.")
            return _stream_init_baseline(dataset)
        if isinstance(dataset, (list, dict, np.ndarray, pd.Series)):
            return _stream_score_row(dataset)
        raise TypeError(
            f"source='stream' accepts DataFrame, list, dict, ndarray, "
            f"Series, or None — got {type(dataset).__name__}"
        )

    raise ValueError(
        f"Invalid source '{source}'. Choose from: 'csv', 'uci', 'url', 'stream'."
    )


def clean(
    source: Literal["uci", "csv", "url", "stream"],
    dataset: Union[int, str, pd.DataFrame, None] = None,) -> pd.DataFrame:
    """
    Return only the clean (non-poisoned) rows.

    Parameters
    ----------
    source : 'uci' | 'csv' | 'url' | 'stream'
    dataset :
        'csv'    → file path (str)
        'uci'    → UCI dataset ID (int)
        'url'    → URL string (str)
        'stream' → None  — filter current buffer (call analyze first)
                   DataFrame — initialise baseline then filter

    Returns
    -------
    pd.DataFrame
        Original data with all flagged rows removed.

    Examples
    --------
    >>> clean_df = api.clean('uci', 73)
    >>> print(f"Kept {len(clean_df)} rows")

    >>> api.analyze('stream', training_df)
    >>> clean_df = api.clean('stream')
    """
    # Stream
    if source == "stream":
        if isinstance(dataset, pd.DataFrame):
            _stream_init_baseline(dataset)

        if not _stream.flags:
            raise RuntimeError(
                "Stream has no data yet.  "
                "Call analyze('stream', data) first."
            )

        if (_stream.buffer_df is not None
                and len(_stream.flags) == len(_stream.buffer_df)):
            clean_idx = [i for i, f in enumerate(_stream.flags) if not f]
            return _stream.buffer_df.iloc[clean_idx].reset_index(drop=True)

        # Fallback: reconstruct from numeric buffer
        clean_rows = [
            _stream.buffer[i]
            for i in range(len(_stream.buffer))
            if i < len(_stream.flags) and not _stream.flags[i]
        ]
        return pd.DataFrame(clean_rows) if clean_rows else pd.DataFrame()

    # Batch sources
    if dataset is None:
        raise TypeError(f"dataset is required for source='{source}'")

    result = analyze(source, dataset)

    if isinstance(result, AnalysisReport):
        return result.clean()

    # fallback (if dict returned in future)
    original = result["dataframe"].drop(
        columns=["_poison_score", "_poison_flag"], errors="ignore"
    )
    return original.iloc[result["clean_indices"]].reset_index(drop=True)


def stream_reset() -> None:
    """Reset the stream detector, clearing all buffered data and scores."""
    _stream.reset()


def stream_retrain() -> None:
    """
    Force retrain the stream model on the current buffer.

    Use this after concept drift or when the baseline becomes stale.
    Raises RuntimeError if the buffer is smaller than the warmup threshold.
    """
    if len(_stream.buffer) < _stream.warmup:
        raise RuntimeError(
            f"Need >= {_stream.warmup} samples to retrain.  "
            f"Buffer currently has {len(_stream.buffer)}."
        )
    X = np.vstack([np.asarray(r, dtype=np.float32).ravel() for r in _stream.buffer])
    X_scaled = _stream.scaler.fit_transform(X)


    _stream.model = IsolationForest(n_estimators=200, random_state=42, n_jobs=-1)
    _stream.model.fit(X_scaled)
    _stream.initialized = True

    raw = -_stream.model.score_samples(X_scaled)
    tau = _compute_tau(raw)
    _stream.scores = [float(s) for s in raw]
    _stream.flags = [bool(s >= tau) for s in raw]