"""Fingerprint version 3."""
import time
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GroupKFold

from RPFNet.RPFExtractor import RPFExtractor
from RPFNet.baselines import ecod

EFFECT_NAMES = ["I_ecod", "I_ecod_class", "I_trim_resid", "I_huber_resid",
                "I_cluster_oof", "I_aum", "I_forget", "I_knn_shapley"]


def _robust_scale(M):
    med = np.median(M, 0, keepdims=True)
    q75, q25 = np.percentile(M, [75, 25], axis=0, keepdims=True)
    scale = (q75 - q25) / 1.349
    scale = np.where(scale > 1e-8, scale, M.std(0, keepdims=True)) + 1e-8
    return (M - med) / scale


def _zscore(v):
    return (v - v.mean()) / (v.std() + 1e-12)


class EffectBlock:
    """The block-I columns. `extract` takes the same arguments as RPFExtractor."""

    names = EFFECT_NAMES

    def __init__(self, random_state=42, trim_budget=0.2, knn_k=10, n_valuation=300,
                 boost_iters=30):
        self.random_state = random_state
        self.trim_budget = trim_budget
        self.knn_k = knn_k
        self.n_valuation = n_valuation
        self.boost_iters = boost_iters
        self.dim = len(self.names)
        self.last_timings = {}

    @property
    def cache_tag(self):
        return (f"effect_rs{self.random_state}_trim{self.trim_budget}_k{self.knn_k}"
                f"_v{self.n_valuation}_b{self.boost_iters}")

    @property
    def source_files(self):
        import RPFNet.baselines as B
        return [__file__, B.__file__]

    # ------------------------------------------------------------- signals
    def _ecod_class(self, X, y, glob):
        out = np.zeros(len(X))
        for c in np.unique(y):
            m = y == c
            s = ecod(X[m], y[m])[0] if m.sum() >= 10 else glob[m]
            q75, q25 = np.percentile(s, [75, 25])
            out[m] = (s - np.median(s)) / ((q75 - q25) / 1.349 + 1e-8)
        return out

    def _targets(self, y, y_cont):
        if y_cont is not None:
            return _zscore(np.asarray(y_cont, dtype=np.float64))[:, None]
        classes = np.unique(y)
        return (y[:, None] == classes[None, :]).astype(np.float64)

    def _trim_resid(self, X, T):
        n = len(X)
        m = n - max(1, int(n * self.trim_budget))
        keep = np.arange(n)
        for _ in range(20):
            r = np.linalg.norm(T - Ridge(alpha=1.0).fit(X[keep], T[keep]).predict(X), axis=1)
            new = np.sort(np.argsort(r)[:m])
            if np.array_equal(new, keep):
                break
            keep = new
        return r

    def _huber_resid(self, X, y_cont):
        if y_cont is None:
            return np.zeros(len(X))
        z = _zscore(np.asarray(y_cont, dtype=np.float64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h = HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=300).fit(X, z)
        return np.abs(z - h.predict(X)) / (h.scale_ + 1e-8)

    def _cluster_oof(self, X, y, y_cont):
        n = len(X)
        k = int(np.clip(n // 100, 5, 20))
        groups = KMeans(n_clusters=k, n_init=3, random_state=self.random_state + 7).fit_predict(X)
        out = np.zeros(n)
        if y_cont is not None:
            z = _zscore(np.asarray(y_cont, dtype=np.float64))
            for tr, te in GroupKFold(n_splits=min(5, k)).split(X, z, groups):
                out[te] = np.abs(z[te] - Ridge(alpha=1.0).fit(X[tr], z[tr]).predict(X[te]))
            return out
        classes = np.unique(y)
        for tr, te in GroupKFold(n_splits=min(5, k)).split(X, y, groups):
            if len(np.unique(y[tr])) < 2:
                out[te] = 1.0 - (y[te] == y[tr][0])
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lr = LogisticRegression(max_iter=300, random_state=self.random_state).fit(X[tr], y[tr])
            P = np.zeros((len(te), len(classes)))
            P[:, np.searchsorted(classes, lr.classes_)] = lr.predict_proba(X[te])
            out[te] = 1.0 - P[np.arange(len(te)), np.searchsorted(classes, y[te])]
        return out

    def _dynamics(self, X, y, y_cont):
        """(negative area under the margin, forgetting rate) over boosting stages."""
        n = len(X)
        if y_cont is not None:
            z = _zscore(np.asarray(y_cont, dtype=np.float64))
            g = HistGradientBoostingRegressor(max_iter=self.boost_iters, early_stopping=False,
                                              random_state=self.random_state).fit(X, z)
            res = np.stack([np.abs(z - p) for p in g.staged_predict(X)])
            return res.mean(0), np.zeros(n)
        classes = np.unique(y)
        if len(classes) < 2:
            return np.zeros(n), np.zeros(n)
        g = HistGradientBoostingClassifier(max_iter=self.boost_iters, early_stopping=False,
                                           random_state=self.random_state).fit(X, y)
        own = np.searchsorted(classes, y)
        margins = []
        for d in g.staged_decision_function(X):
            if d.ndim == 1:                                   # binary: log-odds of classes[1]
                margins.append(np.where(own == 1, d, -d))
            else:
                other = d.copy()
                other[np.arange(n), own] = -np.inf
                margins.append(d[np.arange(n), own] - other.max(1))
        M = np.stack(margins)
        correct = M > 0
        forget = (correct[:-1] & ~correct[1:]).sum(0) / max(1, len(M) - 1)
        return -M.mean(0), forget

    def _knn_shapley(self, X, y):
        """Exact KNN-Shapley (Jia et al."""
        n = len(X)
        if n < self.knn_k + 2:
            return np.zeros(n)
        rng = np.random.default_rng(self.random_state)
        val = rng.choice(n, min(n, self.n_valuation), replace=False)
        D = pairwise_distances(X[val], X)
        D[np.arange(len(val)), val] = np.inf
        order = np.argsort(D, axis=1, kind="stable")[:, :n - 1]
        N, K = n - 1, self.knn_k
        match = (y[order] == y[val][:, None]).astype(np.float64)
        i = np.arange(1, N)                                  # 1-based ranks 1..N-1
        d = (match[:, :-1] - match[:, 1:]) * (np.minimum(K, i) / i / K)[None, :]
        s = np.empty_like(match)
        s[:, -1] = match[:, -1] / N
        s[:, :-1] = s[:, -1:] + np.cumsum(d[:, ::-1], axis=1)[:, ::-1]
        value = np.zeros(n)
        np.add.at(value, order, s)
        return -value / len(val)

    # ------------------------------------------------------------ extract
    def extract(self, X, y, y_cont=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        t = {}
        cols = {}

        def run(name, fn):
            t0 = time.perf_counter()
            try:
                out = fn()
            except Exception as e:           # same policy as RPFExtractor: report, keep going
                print(f"[EffectBlock] {name} failed: {type(e).__name__}: {e}")
                out = np.zeros(len(X))
            t[name] = time.perf_counter() - t0
            return out

        cols["I_ecod"] = run("ecod", lambda: ecod(X, y)[0])
        cols["I_ecod_class"] = run("ecod_class", lambda: self._ecod_class(X, y, cols["I_ecod"]))
        T = self._targets(y, y_cont)
        cols["I_trim_resid"] = run("trim", lambda: self._trim_resid(X, T))
        cols["I_huber_resid"] = run("huber", lambda: self._huber_resid(X, y_cont))
        cols["I_cluster_oof"] = run("cluster_oof", lambda: self._cluster_oof(X, y, y_cont))
        dyn = run("dynamics", lambda: np.stack(self._dynamics(X, y, y_cont), axis=1))
        dyn = dyn if dyn.ndim == 2 else np.zeros((len(X), 2))
        cols["I_aum"], cols["I_forget"] = dyn[:, 0], dyn[:, 1]
        cols["I_knn_shapley"] = run("knn_shapley", lambda: self._knn_shapley(X, y))
        self.last_timings = t
        M = np.column_stack([np.nan_to_num(cols[k], nan=0.0, posinf=0.0, neginf=0.0) for k in self.names])
        return _robust_scale(M).astype(np.float32)


class RPFExtractorV3:
    """Fingerprint version 3 = version 2 columns, then the effect block."""

    feature_version = 3

    def __init__(self, k_small=5, k_large=15, cv_folds=5, random_state=42):
        self.base = RPFExtractor(k_small=k_small, k_large=k_large, cv_folds=cv_folds,
                                 feature_version=2, random_state=random_state)
        self.effect = EffectBlock(random_state=random_state)
        self.parts = [self.base, self.effect]
        self.k_small, self.k_large, self.cv_folds = k_small, k_large, cv_folds
        self.feature_names = list(self.base.feature_names) + list(self.effect.names)
        self.block_names = dict(self.base.block_names)
        self.block_names["I"] = ("Effect and robust signals",
                                 slice(self.base.dim, self.base.dim + self.effect.dim))
        self.dim = len(self.feature_names)
        self.last_timings = {}

    @property
    def random_state(self):
        return self.base.random_state

    @random_state.setter
    def random_state(self, value):
        self.base.random_state = value
        self.effect.random_state = value

    @property
    def _last_neighbors(self):
        return self.base._last_neighbors

    def extract(self, X, y, y_cont=None):
        a = self.base.extract(X, y, y_cont=y_cont)
        b = self.effect.extract(X, y, y_cont=y_cont)
        self.last_timings = {**self.base.last_timings,
                             **{f"I_{k}": v for k, v in self.effect.last_timings.items()}}
        return np.hstack([a, b]).astype(np.float32)


def make_extractor(feature_version, k_small=5, k_large=15, cv_folds=5, random_state=42):
    """The extractor for a fingerprint version (3 = version 2 + block I)."""
    if feature_version == 3:
        return RPFExtractorV3(k_small, k_large, cv_folds, random_state)
    return RPFExtractor(k_small=k_small, k_large=k_large, cv_folds=cv_folds,
                        feature_version=feature_version, random_state=random_state)
