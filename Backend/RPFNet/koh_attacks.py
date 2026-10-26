"""Sanitiser-aware poisoning attacks (Koh, Steinhardt and Liang, 2022)."""
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.linear_model import LogisticRegression

Q_FEASIBLE = 0.90      # quantile of clean rows defining sphere and slab
N_LOCATIONS = 10       # distinct poison locations (copies are concentrated there)
JITTER = 0.01          # copy jitter, in feature standard deviations


def _onehot(y, classes):
    return (y[:, None] == classes[None, :]).astype(np.float64)


def _softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


class FeasibleSets:
    """Sphere, slab and box constraints per class, from the clean data."""

    def __init__(self, X, y, q=Q_FEASIBLE):
        X = np.asarray(X, dtype=np.float64)
        self.classes = np.unique(y)
        self.lo, self.hi = np.percentile(X, [1, 99], axis=0)
        mus = np.stack([X[y == c].mean(0) for c in self.classes])
        self.sets = {}
        for k, c in enumerate(self.classes):
            Xc = X[y == c]
            mu = np.clip(mus[k], self.lo, self.hi)
            other = np.delete(np.arange(len(self.classes)), k)
            j = other[np.argmin(np.linalg.norm(mus[other] - mus[k], axis=1))]
            u = mus[j] - mus[k]
            u = u / (np.linalg.norm(u) + 1e-12)
            D = Xc - mu
            r = np.quantile(np.linalg.norm(D, axis=1), q)
            s = np.quantile(np.abs(D @ u), q)
            self.sets[c] = (mu, max(r, 1e-6), u, max(s, 1e-6))

    def project(self, x, c, iters=25):
        """Project onto the sphere, slab and box (Dykstra), then shrink toward the centroid."""
        mu, r, u, s = self.sets[c]
        x = np.asarray(x, dtype=np.float64).copy()
        p = np.zeros((3, len(x)))
        for _ in range(iters):
            for k in range(3):
                z = x + p[k]
                if k == 0:
                    d = z - mu
                    nd = np.linalg.norm(d)
                    y_ = mu + d * min(1.0, r / (nd + 1e-12))
                elif k == 1:
                    t = (z - mu) @ u
                    y_ = z - (t - np.clip(t, -s, s)) * u
                else:
                    y_ = np.clip(z, self.lo, self.hi)
                p[k] = z - y_
                x = y_
        x = np.clip(x, self.lo, self.hi)
        d = x - mu
        f = min(1.0, r / (np.linalg.norm(d) + 1e-12), s / (abs(d @ u) + 1e-12))
        return mu + f * d

    def contains(self, x, c, tol=1e-6):
        mu, r, u, s = self.sets[c]
        d = x - mu
        return (np.linalg.norm(d) <= r * (1 + tol) and abs(d @ u) <= s * (1 + tol)
                and np.all(x >= self.lo - tol) and np.all(x <= self.hi + tol))


def _fit(X, y, classes, C=1.0, warm=None):
    """Multinomial logistic regression as a K x (d+1) matrix (binary: two rows,
    the first zero, so one softmax formula serves every K)."""
    lr = LogisticRegression(C=C, max_iter=300, tol=1e-4)
    lr.fit(X, y)
    if not np.array_equal(lr.classes_, classes):
        return None
    W = np.hstack([lr.coef_, lr.intercept_[:, None]])
    return np.vstack([np.zeros_like(W), W]) if len(classes) == 2 else W


def _loss_max_point(Theta, F, c, k_c, starts, steps=60):
    """Projected gradient ascent on -log p_c(x) over the feasible set of c."""
    Wd = Theta[:, :-1]
    best, best_val = None, -np.inf
    mu, r, u, s = F.sets[c]
    lr = 0.1 * r
    for x in starts:
        x = F.project(x, c)
        for _ in range(steps):
            p = _softmax((np.append(x, 1.0) @ Theta.T)[None])[0]
            g = Wd.T @ p - Wd[k_c]                     # gradient of -log p_c
            ng = np.linalg.norm(g)
            if ng < 1e-10:
                break
            x = F.project(x + lr * g / ng, c)
        p = _softmax((np.append(x, 1.0) @ Theta.T)[None])[0]
        val = -np.log(p[k_c] + 1e-12)
        if val > best_val:
            best, best_val = x, val
    return best, best_val


def _place_copies(F, loc, c, m, sd, rng):
    pts = loc[None, :] + rng.normal(0, JITTER, (m, len(loc))) * sd
    return np.stack([F.project(p, c, iters=5) for p in pts])


def koh_minmax(X, y, pidx, rng, n_loc=N_LOCATIONS, C=1.0):
    """Returns (X_poison_rows, y_poison_rows, locations, labels_of_locations)."""
    X = np.asarray(X, dtype=np.float64)
    classes = np.unique(y)
    F = FeasibleSets(X, y)
    sd = X.std(0) + 1e-8
    keep = np.setdiff1d(np.arange(len(X)), pidx)
    n_p = len(pidx)
    T = max(1, min(n_loc, n_p))
    blocks = np.array_split(np.arange(n_p), T)
    Xq, yq = np.empty((0, X.shape[1])), np.empty(0, dtype=y.dtype)
    locs, labs = [], []
    for blk in blocks:
        Theta = _fit(np.vstack([X[keep], Xq]), np.concatenate([y[keep], yq]), classes, C)
        if Theta is None:
            break
        best = (None, None, -np.inf)
        for k_c, c in enumerate(classes):
            mu, r, u, s = F.sets[c]
            g0 = Theta[:, :-1].T @ _softmax((np.append(mu, 1.0) @ Theta.T)[None])[0] - Theta[k_c, :-1]
            starts = [mu, mu + r * g0 / (np.linalg.norm(g0) + 1e-12),
                      mu + r * rng.normal(size=len(mu)) / np.sqrt(len(mu))]
            x, v = _loss_max_point(Theta, F, c, k_c, starts)
            if v > best[2]:
                best = (x, c, v)
        x, c, _ = best
        locs.append(x)
        labs.append(c)
        Xq = np.vstack([Xq, _place_copies(F, x, c, len(blk), sd, rng)])
        yq = np.concatenate([yq, np.full(len(blk), c, dtype=y.dtype)])
    if len(yq) < n_p:
        m = n_p - len(yq)
        c = labs[-1] if labs else classes[0]
        x = locs[-1] if locs else F.sets[c][0]
        Xq = np.vstack([Xq, _place_copies(F, x, c, m, sd, rng)])
        yq = np.concatenate([yq, np.full(m, c, dtype=y.dtype)])
    return Xq, yq, np.array(locs), np.array(labs)


def koh_influence(X, y, pidx, rng, steps=15, C=1.0):
    """Influence attack started from the min-max placement."""
    X = np.asarray(X, dtype=np.float64)
    classes = np.unique(y)
    K = len(classes)
    F = FeasibleSets(X, y)
    sd = X.std(0) + 1e-8
    Xq, yq, locs, labs = koh_minmax(X, y, pidx, rng, C=C)
    n_p, d = len(pidx), X.shape[1]
    T = len(locs)
    owner = np.concatenate([np.full(len(b), t) for t, b in enumerate(np.array_split(np.arange(n_p), T))])
    offsets = Xq - locs[owner]          # fixed jitter of each copy
    keep = np.setdiff1d(np.arange(len(X)), pidx)
    val = rng.choice(keep, min(len(keep), max(50, len(keep) // 4)), replace=False)
    Xv = np.hstack([X[val], np.ones((len(val), 1))])
    Ev = _onehot(y[val], classes)
    n = len(X)
    lam = 1.0 / (C * n)

    def val_loss(Theta):
        Pv = _softmax(Xv @ Theta.T)
        return -np.mean(np.log((Pv * Ev).sum(1) + 1e-12))

    Theta = _fit(np.vstack([X[keep], Xq]), np.concatenate([y[keep], yq]), classes, C)
    if Theta is None:
        return Xq, yq, locs, labs
    cur = val_loss(Theta)
    eta = 0.05          # step, as a fraction of each class's sphere radius
    for _ in range(steps):
        Xall = np.vstack([X[keep], Xq])
        yall = np.concatenate([y[keep], yq])
        Xa = np.hstack([Xall, np.ones((n, 1))])
        P = _softmax(Xa @ Theta.T)
        g_val = (_softmax(Xv @ Theta.T) - Ev).T @ Xv / len(val)

        def hvp(vec):
            V = vec.reshape(K, d + 1)
            A = Xa @ V.T
            B = P * A - P * (P * A).sum(1, keepdims=True)
            return ((B.T @ Xa) / n + lam * V).ravel()

        V = cg(LinearOperator((K * (d + 1),) * 2, matvec=hvp), g_val.ravel(),
               maxiter=100)[0].reshape(K, d + 1)
        q0 = len(keep)
        Pi, Ei, Xi = P[q0:], _onehot(yq, classes), Xa[q0:]
        APi = (Xi @ V.T) * Pi
        Wd = Theta[:, :d]
        ds = (Pi - Ei) @ V[:, :d] + APi @ Wd - APi.sum(1, keepdims=True) * (Pi @ Wd)
        new = locs.copy()
        for t in range(T):
            g = -ds[owner == t].sum(0)      # ascent direction on the validation loss
            ng = np.linalg.norm(g)
            if ng > 1e-12:
                new[t] = F.project(locs[t] + eta * F.sets[labs[t]][1] * g / ng, labs[t])
        Xn = np.stack([F.project(new[owner[i]] + offsets[i], yq[i], iters=5) for i in range(n_p)])
        Th = _fit(np.vstack([X[keep], Xn]), yall, classes, C)
        v = val_loss(Th) if Th is not None else -np.inf
        if v > cur:
            locs, Xq, Theta, cur = new, Xn, Th, v
        else:
            eta /= 2
            if eta < 1e-3:
                break
    return Xq, yq, locs, labs
