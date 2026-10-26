"""Combining several suspicion scores into one (plan item D1a)."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def robust_scale(v):
    """Median/IQR scaling within one dataset, the rule block I uses."""
    v = np.asarray(v, dtype=np.float64)
    q75, q25 = np.percentile(v, [75, 25])
    scale = (q75 - q25) / 1.349
    scale = scale if scale > 1e-8 else v.std()
    return (v - np.median(v)) / (scale + 1e-8)


class MinPCombiner:
    """Tippett's combination with Fisher's combination as the tie-break."""

    def __init__(self, null):
        self.null = np.sort(np.asarray(null, dtype=np.float64), axis=0)
        n, k = self.null.shape
        # distinct values of max_j -log p_j differ by at least log(1 + 1/(N+1)) > 0.5/(N+1),
        # and t * sum_j -log p_j <= t * K * log(N+1) = 0.5/(N+1)
        self.t = 0.5 / ((n + 1) * k * np.log(n + 1))

    def pvalues(self, S):
        S = np.asarray(S, dtype=np.float64)
        n = len(self.null)
        ge = np.column_stack([n - np.searchsorted(self.null[:, j], S[:, j], side="left")
                              for j in range(S.shape[1])])
        return (1.0 + ge) / (n + 1.0)

    def score(self, S):
        lp = -np.log(self.pvalues(S))
        return lp.max(1) + self.t * lp.sum(1)


class RateModel:
    """Learned contamination-rate estimate."""

    THRESHOLDS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.2, 0.5)

    def __init__(self, combiner, clean, lo=0.001, hi=0.4, random_state=0):
        """clean: the clean reference rows (unsorted; the combiner keeps each
        column sorted separately, which no longer forms rows)."""
        self.comb, self.lo, self.hi = combiner, lo, hi
        self.ref = np.sort(combiner.score(clean))
        # a few dozen to a few hundred training datasets: leaves of 5, not the default 20,
        # or the estimate saturates (10% and 20% both read 14% in a synthetic check)
        self.model = HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.05,
                                                   min_samples_leaf=5, random_state=random_state)

    def profile(self, S):
        s = self.comb.score(S)
        p = (1.0 + len(self.ref) - np.searchsorted(self.ref, s, side="left")) / (len(self.ref) + 1.0)
        return np.log(np.array([np.mean(p <= t) for t in self.THRESHOLDS]) + 1e-4)

    def fit(self, datasets):
        """datasets: iterable of (S, rate), S the signal columns of one poisoned dataset."""
        P, r = zip(*[(self.profile(S), rate) for S, rate in datasets])
        self.model.fit(np.array(P), np.log(np.clip(r, self.lo, self.hi)))
        return self

    def predict(self, S):
        return float(np.clip(np.exp(self.model.predict(self.profile(S)[None])[0]), self.lo, self.hi))


EVIDENCE_T = (1e-3, 1e-2, 5e-2)


def dataset_evidence(P):
    """Dataset-level evidence per score: log of the share of rows with p <= t,
    relative to the share t expected on clean data, for each t in EVIDENCE_T."""
    return np.concatenate([np.log((P <= t).mean(0) / t + 1e-3) for t in EVIDENCE_T])


class LearnedCombiner:
    """Attack-aware combination."""

    def __init__(self, combiner, max_clean=200, random_state=0):
        self.comb, self.max_clean, self.rs = combiner, max_clean, random_state
        self.model = HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05,
                                                    random_state=random_state)

    def features(self, S):
        P = self.comb.pvalues(S)
        D = dataset_evidence(P)
        return np.hstack([-np.log(P), np.broadcast_to(D, (len(P), len(D)))])

    def fit(self, datasets):
        """datasets."""
        rng = np.random.default_rng(self.rs)
        X, Y, W = [], [], []
        for S, y in datasets:
            y = np.asarray(y, bool)
            pos, neg = np.where(y)[0], np.where(~y)[0]
            if len(pos) == 0 or len(neg) == 0:
                continue
            keep = np.concatenate([pos, rng.choice(neg, min(self.max_clean, len(neg)), replace=False)])
            X.append(self.features(S)[keep]); Y.append(y[keep])
            W.append(np.where(y[keep], 0.5 / len(pos), 0.5 / (len(keep) - len(pos))))
        W = np.concatenate(W)
        self.model.fit(np.vstack(X), np.concatenate(Y), sample_weight=W * len(W) / W.sum())
        return self

    def score(self, S):
        return self.model.predict_proba(self.features(S))[:, 1]
