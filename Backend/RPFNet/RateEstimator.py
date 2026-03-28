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


def estimate_contamination_rate(scores: np.ndarray, lo: float = 0.01, hi: float = 0.40) -> float:
    n = len(scores)
    sorted_s = np.sort(scores)
    estimates = []
    weights = []

    bc = _bimodality_coefficient(scores)

    # GMM
    try:
        gmm = GaussianMixture(
            n_components=2, random_state=42, max_iter=200,
            covariance_type="full", reg_covar=1e-4
        ).fit(scores.reshape(-1, 1))
        poison_comp = int(np.argmax(gmm.means_.flatten()))
        est_gmm = float(gmm.weights_[poison_comp])
        estimates.append(est_gmm)
        # GMM is more reliable when scores are bimodal
        weights.append(2.0 if bc > 0.555 else 0.5)
    except Exception as e:
        print(f"[RateEstimator] GaussianMixture estimation failed: {type(e).__name__}: {e}")

    # Score-gap
    start = max(1, int(n * 0.40))
    diffs = np.diff(sorted_s[start:])
    if len(diffs) > 0:
        gap_pos = int(np.argmax(diffs))
        n_above = len(diffs) - gap_pos
        est_gap = n_above / n
        estimates.append(est_gap)
        weights.append(1.5 if bc > 0.555 else 0.5)

    # Otsu's method
    est_otsu = _otsu_rate(scores)
    estimates.append(est_otsu)
    weights.append(2.0)  # Otsu is generally well-calibrated

    # Excess-mass
    half = sorted_s[:max(10, n // 2)]
    mu = float(half.mean())
    sigma = float(half.std()) + 1e-8
    est_em = float((scores > mu + 2.0 * sigma).mean())
    estimates.append(est_em)
    weights.append(1.0)

    if not estimates:
        return 0.10

    # Weighted median
    # When bimodality is weak, shrink toward conservative estimate
    estimates = np.array(estimates)
    weights = np.array(weights)

    if bc < 0.40:
        # Weak separation: poison signal is subtle, shrink estimates down
        # This fixes the overestimation at 5% for boundary flip etc.
        shrink_factor = 0.6
        estimates = estimates * shrink_factor

    # Weighted median
    sorted_idx = np.argsort(estimates)
    cum_w = np.cumsum(weights[sorted_idx])
    mid = cum_w[-1] / 2.0
    median_idx = np.searchsorted(cum_w, mid)
    rate = float(estimates[sorted_idx[min(median_idx, len(estimates) - 1)]])

    rate = float(np.clip(rate, lo, hi))
    return rate