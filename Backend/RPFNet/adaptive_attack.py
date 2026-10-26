"""White-box adaptive attacker against a row-scoring detector (Section 7.5)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neighbors import NearestNeighbors

GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


def _flip(yi, classes, rng):
    others = classes[classes != yi]
    return yi if len(others) == 0 else rng.choice(others)


class CamouflageAttack:
    def __init__(self, X, y, fraction, seed=0, k=5, val_frac=0.25):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y)
        n = len(X)
        n_p = max(1, int(n * fraction))
        self.pidx = np.sort(rng.choice(n, n_p, replace=False))
        classes = np.unique(y)
        self.yp = self.y.copy()
        for i in self.pidx:
            self.yp[i] = _flip(self.y[i], classes, rng)
        clean = np.setdiff1d(np.arange(n), self.pidx)
        self.val = rng.choice(clean, min(len(clean), max(50, int(len(clean) * val_frac))), replace=False)
        self.fit_rows = np.setdiff1d(np.arange(n), self.val)   # the attacker's surrogate never sees val
        # camouflage targets: mean of the k nearest clean rows of the flipped label
        self.target = np.empty((n_p, X.shape[1]))
        for c in np.unique(self.yp[self.pidx]):
            rows = np.where(self.yp[self.pidx] == c)[0]
            pool = clean[self.y[clean] == c]
            nn = NearestNeighbors(n_neighbors=min(k, len(pool))).fit(self.X[pool])
            idx = nn.kneighbors(self.X[self.pidx[rows]], return_distance=False)
            self.target[rows] = self.X[pool][idx].mean(1)
        self.src = self.X[self.pidx].copy()

    def dataset(self, a):
        Xp = self.X.copy()
        Xp[self.pidx] = (1 - a)[:, None] * self.src + a[:, None] * self.target
        return Xp.astype(np.float32), self.yp

    def damage(self, Xp):
        m = LogisticRegression(C=1.0, max_iter=1000).fit(Xp[self.fit_rows], self.yp[self.fit_rows])
        return log_loss(self.y[self.val], m.predict_proba(Xp[self.val]), labels=m.classes_)

    def run(self, score_fn, lam, iters=20, block_frac=0.2, init=None):
        """score_fn(Xp, yp) -> one score per row (higher = more suspicious).
        Returns (a, history); history holds one dict per query."""
        truth = np.zeros(len(self.X), bool)
        truth[self.pidx] = True
        hist = []

        def evaluate(a, kind):
            Xp, yp = self.dataset(a)
            dmg = self.damage(Xp)
            auc = roc_auc_score(truth, np.asarray(score_fn(Xp, yp), dtype=np.float64))
            hist.append(dict(kind=kind, mean_a=float(a.mean()), damage=dmg, auroc=auc, J=dmg - lam * auc))
            return dmg - lam * auc

        n_p = len(self.pidx)
        if init is None:
            best_a, best = None, -np.inf
            for g in GRID:
                a = np.full(n_p, g)
                j = evaluate(a, "uniform")
                if j > best:
                    best_a, best = a, j
        else:
            best_a = np.asarray(init, dtype=float)
            best = evaluate(best_a, "init")
        a = best_a.copy()
        for _ in range(iters):
            cand = a.copy()
            blk = self.rng.choice(n_p, max(1, int(n_p * block_frac)), replace=False)
            step = self.rng.choice([-0.25, 0.25], len(blk))
            cand[blk] = np.clip(cand[blk] + step, 0.0, 1.0)
            j = evaluate(cand, "search")
            if j > best:
                a, best = cand, j
                hist[-1]["accepted"] = True
        return a, hist
