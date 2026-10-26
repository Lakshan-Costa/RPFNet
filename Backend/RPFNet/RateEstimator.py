#Rateestimator.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.stats import rankdata
from scipy.stats import ttest_rel, wilcoxon
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import OneClassSVM

def _otsu_rate(scores: np.ndarray) -> float:
    sorted_s = np.sort(scores.astype(np.float64))
    n = len(sorted_s)
    if n < 10:
        return 0.10

    # Efficient Otsu: compute running stats
    cum_sum = np.cumsum(sorted_s)
    cum_sq = np.cumsum(sorted_s ** 2)
    total = cum_sum[-1]

    best_var = np.inf
    best_k = n // 2

    # Only search in plausible range [50%, 98%] of sorted scores
    lo_idx = max(1, int(n * 0.50))
    hi_idx = min(n - 1, int(n * 0.98))

    for k in range(lo_idx, hi_idx):
        w0 = k / n
        w1 = 1.0 - w0
        if w0 <= 0 or w1 <= 0:
            continue

        m0 = cum_sum[k - 1] / k
        m1 = (total - cum_sum[k - 1]) / (n - k)

        v0 = cum_sq[k - 1] / k - m0 ** 2
        v1 = (cum_sq[-1] - cum_sq[k - 1]) / (n - k) - m1 ** 2

        within_var = w0 * max(v0, 0) + w1 * max(v1, 0)

        if within_var < best_var:
            best_var = within_var
            best_k = k

    return (n - best_k) / n


def _bimodality_coefficient(scores: np.ndarray) -> float:
    from scipy.stats import skew, kurtosis
    n = len(scores)
    if n < 4:
        return 0.0
    g = float(skew(scores))
    k = float(kurtosis(scores, fisher=True))
    bc = (g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    return float(np.clip(bc, 0, 1))


def score_distribution_features(scores: np.ndarray) -> np.ndarray:
    """Summary features of a score vector."""
    s = np.sort(scores.astype(np.float64))
    n = len(s)

    pcts = np.percentile(s, [50, 60, 70, 75, 80, 85, 90, 92, 95, 97, 99])
    diffs = np.diff(s[max(0, n // 2):])
    top_gap = float(diffs.max()) if len(diffs) > 0 else 0.0

    return np.array([
        *pcts,
        float(s.mean()),
        float(s.std()),
        float((s > s.mean() + 2 * s.std()).mean()),
        top_gap,
        float(s[-1] - s[n // 2]),
    ], dtype=np.float32)


class RateEstimatorHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32), nn.LayerNorm(32), nn.GELU(),
            nn.Linear(32, 16), nn.GELU(),
            nn.Linear(16, 1), nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1) * 0.40


def _weighted_median(estimates: np.ndarray, weights: np.ndarray) -> float:
    sorted_idx = np.argsort(estimates)
    cum_w = np.cumsum(weights[sorted_idx])
    mid = cum_w[-1] / 2.0
    median_idx = int(np.searchsorted(cum_w, mid))
    return float(estimates[sorted_idx[min(median_idx, len(estimates) - 1)]])


RATE_THRESHOLD = 0.60
RATE_SCALE = 1.0


def _ks_to_uniform(scores: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance to Uniform(0,1)."""
    ss = np.sort(np.asarray(scores, dtype=np.float64))
    return float(np.max(np.abs(np.arange(1, len(ss) + 1) / len(ss) - ss)))


def _is_rank_like(scores: np.ndarray) -> bool:
    s = np.asarray(scores, dtype=np.float64)
    return bool(s.min() >= 0.0 and s.max() <= 1.0 and _ks_to_uniform(s) <= 0.10)


def _is_probability_like(scores: np.ndarray) -> bool:
    """True if `scores` are model probabilities rather than ranks / raw scores."""
    s = np.asarray(scores, dtype=np.float64)
    if s.min() < 0.0 or s.max() > 1.0:
        return False
    # KS distance to Uniform(0,1): ~0 for rank-transformed input, >= 0.21 for
    # every RPFNet score vector measured (min over 486 runs).
    return _ks_to_uniform(s) > 0.10


def estimate_contamination_rate(scores: np.ndarray, lo: float = 0.001,
                                hi: float = 0.40, thr: float = None,
                                scale: float = None, probabilities: bool = None) -> float:
    """Estimate the poisoned fraction from a 1-D array of scores."""
    n = len(scores)
    if n == 0:
        return lo
    scores = np.asarray(scores, dtype=np.float64)
    if float(np.ptp(scores)) < 1e-9:
        # No variation in scores -> nothing is separable -> report the floor.
        return lo

    # probabilities=True/False states what the scores are; None guesses, and
    # the guess can mistake a model's probabilities for ranks.
    is_prob = _is_probability_like(scores) if probabilities is None else probabilities
    if is_prob:
        thr = RATE_THRESHOLD if thr is None else thr
        scale = RATE_SCALE if scale is None else scale
        return float(np.clip(scale * float((scores > thr).mean()), lo, hi))

    if probabilities is None and _is_rank_like(scores):
        # Every rank vector is uniform on (0,1]: no estimator can read a rate
        # off it.  Say so instead of returning a confident-looking number.
        print("[RateEstimator] scores look rank-transformed — no contamination "
              "signal is recoverable from ranks; pass the raw model scores. "
              f"Returning the floor ({lo:.3%}).")
        return lo

    return _shape_based_rate(scores, lo=lo, hi=hi)


def _shape_based_rate(scores: np.ndarray, lo: float = 0.001, hi: float = 0.40) -> float:
    """Fallback for score vectors that are NOT calibrated probabilities."""
    n = len(scores)

    # --- robust tail estimator (honest low-end anchor) ---
    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med))) * 1.4826 + 1e-8
    est_tail = float((scores > med + 3.0 * mad).mean())

    # --- splitting methods + a measure of genuine mode separation ---
    gate = 0.0
    split_estimates = []
    split_weights = []
    try:
        gmm = GaussianMixture(
            n_components=2, random_state=42, max_iter=200,
            covariance_type="full", reg_covar=1e-4
        ).fit(scores.reshape(-1, 1))
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        hi_c = int(np.argmax(means))
        lo_c = 1 - hi_c
        # standardized distance between the two component means: large only when
        # a real, separated poison mode exists (not when GMM bisects one blob).
        mode_sep = abs(means[hi_c] - means[lo_c]) / (stds[hi_c] + stds[lo_c] + 1e-8)
        w_hi = float(gmm.weights_[hi_c])
        g_sep = float(np.clip((mode_sep - 1.8) / 1.2, 0.0, 1.0))
        g_size = float(np.clip((0.35 - w_hi) / 0.10, 0.0, 1.0))
        gate = g_sep * g_size
        split_estimates.append(w_hi); split_weights.append(2.0)
    except Exception as e:
        print(f"[RateEstimator] GaussianMixture estimation failed: {type(e).__name__}: {e}")

    # Otsu's threshold
    split_estimates.append(_otsu_rate(scores)); split_weights.append(2.0)

    # Score-gap (largest jump in the upper portion of the sorted scores)
    sorted_s = np.sort(scores)
    start = max(1, int(n * 0.40))
    diffs = np.diff(sorted_s[start:])
    if len(diffs) > 0:
        gap_pos = int(np.argmax(diffs))
        split_estimates.append((len(diffs) - gap_pos) / n); split_weights.append(1.0)

    est_split = _weighted_median(np.array(split_estimates), np.array(split_weights)) \
        if split_estimates else est_tail

    # Trust the (over-eager) splitting estimate only when modes are separated;
    # otherwise fall back to the robust tail anchor.
    rate = max(est_tail, gate * est_split)
    return float(np.clip(rate, lo, hi))


def calibrate_rate_threshold(score_vectors, true_rates,
                             thr_grid=None, scale_grid=None, verbose=True):
    """Re-fit (RATE_THRESHOLD, RATE_SCALE) for a retrained detector."""
    if thr_grid is None:
        thr_grid = np.arange(0.40, 0.951, 0.025)
    if scale_grid is None:
        scale_grid = np.arange(0.6, 3.01, 0.1)

    rates = np.asarray(true_rates, dtype=np.float64)
    keep = rates > 0
    if keep.sum() < 8:
        if verbose:
            print("[RateEstimator] too few poisoned runs to calibrate; keeping defaults")
        return RATE_THRESHOLD, RATE_SCALE
    vecs = [np.asarray(v, dtype=np.float64) for v, k in zip(score_vectors, keep) if k]
    rates = rates[keep]

    best = (np.inf, RATE_THRESHOLD, RATE_SCALE)
    for t in thr_grid:
        fr = np.array([float((v > t).mean()) for v in vecs])
        for a in scale_grid:
            err = float(np.mean(np.abs(np.log10(np.maximum(a * fr, 1e-4) / rates))))
            if err < best[0]:
                best = (err, float(t), float(a))
    if verbose:
        print(f"  [rate-cal] pi_hat = {best[2]:.2f} * frac(s > {best[1]:.3f})  "
              f"mean|log10 err|={best[0]:.3f}  (n={len(vecs)})")
    return best[1], best[2]

# P-value route to thresholds (FDR control instead of top-k at an estimate)
def empirical_pvalues(scores: np.ndarray, null_scores: np.ndarray) -> np.ndarray:
    """Conformal p-values against clean-data scores."""
    null = np.sort(np.asarray(null_scores, dtype=np.float64))
    m = len(null)
    n_ge = m - np.searchsorted(null, np.asarray(scores, dtype=np.float64), side="left")
    return (1.0 + n_ge) / (m + 1.0)


def bh_select(pvals: np.ndarray, q: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg: boolean mask of the points rejected at FDR level q."""
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    ok = p[order] <= q * np.arange(1, m + 1) / m
    k = ok.nonzero()[0].max() + 1 if ok.any() else 0
    flags = np.zeros(m, dtype=bool)
    flags[order[:k]] = True
    return flags


def storey_rate(pvals: np.ndarray, lam: float = 0.5) -> float:
    """Contamination estimate 1 - pi0, with Storey's (2002) conservative
    pi0 = #{p > lam} / (m (1 - lam))."""
    p = np.asarray(pvals, dtype=np.float64)
    return float(1.0 - min(1.0, (p > lam).mean() / (1.0 - lam)))
