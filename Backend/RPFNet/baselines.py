"""Baseline poisoning / outlier detectors for the RPFNet evaluation."""
import numpy as np
from scipy.stats import rankdata, skew
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA

ASSUMED_RATE = 0.10   # budget for methods that need one (as in the notebook)


def _grads(X, y, seed):
    """Per-sample multinomial logistic-loss gradients (n x K(d+1))."""
    lr = LogisticRegression(max_iter=500, random_state=seed).fit(X, y)
    P = lr.predict_proba(X)
    E = (y[:, None] == lr.classes_[None, :]).astype(np.float64)
    Xa = np.hstack([X, np.ones((len(X), 1))])
    return ((P - E)[:, :, None] * Xa[:, None, :]).reshape(len(X), -1), P, E


def isoforest(X, y, seed=42):
    iso = IsolationForest(contamination="auto", random_state=seed).fit(X)
    return -iso.score_samples(X), iso.predict(X) == -1


def lof(X, y, seed=42):
    m = LocalOutlierFactor(n_neighbors=min(20, len(X) - 1), contamination="auto")
    flags = m.fit_predict(X) == -1
    return -m.negative_outlier_factor_, flags


def ocsvm(X, y, seed=42):
    m = OneClassSVM(nu=ASSUMED_RATE, kernel="rbf", gamma="scale").fit(X)
    return -m.decision_function(X), m.predict(X) == -1


def spectral_signature(X, y, seed=42):
    """Tran, Li & Madry (2018), on the input features, per class."""
    s = np.zeros(len(X))
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) < 5:
            continue
        Xc = X[idx] - X[idx].mean(0)
        s[idx] = (Xc @ np.linalg.svd(Xc, full_matrices=False)[2][0]) ** 2
    return s, None


def grad_norm(X, y, seed=42):
    G, _, _ = _grads(X, y, seed)
    return np.linalg.norm(G, axis=1), None


def sever(X, y, seed=42, n_iters=4, budget=ASSUMED_RATE):
    """SEVER (Diakonikolas et al."""
    n = len(X)
    remaining = np.arange(n)
    score = np.zeros(n)
    flags = np.zeros(n, dtype=bool)
    per_round = max(1, int(n * budget) // n_iters)
    for r in range(n_iters + 1):
        if len(np.unique(y[remaining])) < 2:
            break
        G, _, _ = _grads(X[remaining], y[remaining], seed)
        G -= G.mean(0)
        proj = (G @ np.linalg.svd(G, full_matrices=False)[2][0]) ** 2
        proj = proj / (proj.max() + 1e-12)
        if r == n_iters:                      # survivors: final projection
            score[remaining] = proj
            break
        top = np.argsort(proj)[-per_round:]
        score[remaining[top]] = (n_iters - r) + proj[top]
        flags[remaining[top]] = True
        remaining = np.delete(remaining, top)
    return score, flags


def grad_spectral(X, y, seed=42):
    """One-shot SEVER score: squared projection on the top singular vector."""
    G, _, _ = _grads(X, y, seed)
    G -= G.mean(0)
    return (G @ np.linalg.svd(G, full_matrices=False)[2][0]) ** 2, None


def knn_label(X, y, seed=42, k=10):
    """Fraction of the k nearest neighbours with a different label; flag when
    the neighbourhood plurality label differs from the point's own."""
    k = min(k, len(X) - 1)
    I = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)[1][:, 1:]
    NL = y[I]
    classes = np.unique(y)
    counts = (NL[:, :, None] == classes[None, None, :]).sum(1)
    plurality = classes[counts.argmax(1)]
    return (NL != y[:, None]).mean(1), plurality != y


def knn_dist(X, y, seed=42):
    """Mean distance to 10 nearest neighbours in a PCA(10) space (the
    notebook's former "DeepKNN")."""
    Z = PCA(n_components=min(10, X.shape[1]), random_state=seed).fit_transform(X)
    D = NearestNeighbors(n_neighbors=min(10, len(X) - 1)).fit(Z).kneighbors(Z)[0]
    return D.mean(1), None


def confident_learning(X, y, seed=42, folds=5):
    """Confident learning (Northcutt, Jiang & Chuang 2021) with out-of-fold logistic-regression probabilities."""
    classes = np.unique(y)
    folds = int(max(2, min(folds, np.unique(y, return_counts=True)[1].min())))
    P = cross_val_predict(LogisticRegression(max_iter=1000, random_state=seed), X, y,
                          cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
                          method="predict_proba")
    own = np.searchsorted(classes, y)
    t = np.array([P[own == j, j].mean() if (own == j).any() else 1.0
                  for j in range(len(classes))])
    above = P >= t[None, :]
    conf = np.where(above.any(1), np.where(above, P, -1).argmax(1), own)
    return 1.0 - P[np.arange(len(y)), own], conf != own


def ecod(X, y, seed=42):
    """ECOD (Li et al."""
    n = len(X)
    left = rankdata(X, method="max", axis=0) / n          # P(X_j <= x_j)
    right = rankdata(-X, method="max", axis=0) / n        # P(X_j >= x_j)
    o_l, o_r = -np.log(left).sum(1), -np.log(right).sum(1)
    auto = np.where(skew(X, axis=0)[None, :] < 0, -np.log(left), -np.log(right)).sum(1)
    return np.maximum.reduce([o_l, o_r, auto]), None


def iforest_cl(X, y, seed=42):
    """Rank average of Isolation Forest and confident learning."""
    a, _ = isoforest(X, y, seed)
    b, _ = confident_learning(X, y, seed)
    return (rankdata(a) + rankdata(b)) / (2 * len(X)), None


def ecod_cl(X, y, seed=42):
    """Rank average of ECOD and confident learning: the simplest combination of
    the strongest feature-space and label-space detectors."""
    a, _ = ecod(X, y, seed)
    b, _ = confident_learning(X, y, seed)
    return (rankdata(a) + rankdata(b)) / (2 * len(X)), None


def autoencoder(X, y, seed=42, epochs=50):
    """Reconstruction error of a small MLP autoencoder (from the notebook)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    n, d = X.shape
    torch.manual_seed(seed)
    h1, h2, b = max(8, d), max(4, d // 2), max(2, d // 4)
    ae = nn.Sequential(nn.Linear(d, h1), nn.LayerNorm(h1), nn.GELU(),
                       nn.Linear(h1, h2), nn.LayerNorm(h2), nn.GELU(), nn.Linear(h2, b),
                       nn.Linear(b, h2), nn.LayerNorm(h2), nn.GELU(),
                       nn.Linear(h2, h1), nn.LayerNorm(h1), nn.GELU(), nn.Linear(h1, d))
    Xt = torch.tensor(X, dtype=torch.float32)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    bs = min(256, n)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            xb = Xt[perm[i:i + bs]]
            loss = F.mse_loss(ae(xb), xb)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return ((Xt - ae(Xt)) ** 2).mean(1).numpy(), None


def trim(X, y_cont, seed=42, budget=ASSUMED_RATE, iters=20):
    """TRIM (Jagielski et al."""
    from sklearn.linear_model import Ridge
    n = len(X)
    keep = np.arange(n)
    m = n - max(1, int(n * budget))
    for _ in range(iters):
        r = np.abs(y_cont - Ridge(alpha=1.0).fit(X[keep], y_cont[keep]).predict(X))
        new = np.sort(np.argsort(r)[:m])
        if np.array_equal(new, keep):
            break
        keep = new
    flags = np.ones(n, dtype=bool)
    flags[keep] = False
    return r, flags


def _trim_fixed(X, z, eps, rng, tol=1e-4, iters=20):
    """TRIM as run by iTRIM."""
    from sklearn.linear_model import LinearRegression
    n_keep = int(np.floor(len(X) / (1 + eps)))
    keep = rng.choice(len(X), n_keep, replace=False)
    err = (LinearRegression().fit(X[keep], z[keep]).predict(X) - z) ** 2
    if eps == 0:
        return np.arange(len(X)), err, float(err.mean())
    for _ in range(iters):
        keep = np.argsort(err)[:n_keep]
        new = (LinearRegression().fit(X[keep], z[keep]).predict(X) - z) ** 2
        delta = float(np.mean(np.abs(err - new)))
        err = new
        if delta < tol:
            break
    return keep, err, float(err[keep].mean())


def itrim(X, y_cont, seed=42, eps_max=0.20, step=0.02, threshold=1e-3):
    """iTRIM (Mueller, Kowatsch & Boettinger, PRDC 2020), re-implemented from the algorithm of their paper and code."""
    rng = np.random.default_rng(seed)
    z = (y_cont - y_cont.mean()) / (y_cont.std() + 1e-12)
    grid = np.round(np.arange(0.0, eps_max + step / 2, step), 10)
    runs = [_trim_fixed(X, z, e, rng) for e in grid]
    mse = np.array([r[2] for r in runs])
    below = np.where(np.abs(mse[:-1] - mse[1:]) < threshold)[0]
    k = int(below[0]) if len(below) else len(grid) - 1
    keep, err, _ = runs[k]
    flags = np.ones(len(X), dtype=bool)
    flags[keep] = False
    return err, flags


REGRESSION_BASELINES = {"TRIM": trim, "iTRIM": itrim}   # called with the continuous target

BASELINES = {
    "IsoForest": isoforest, "LOF": lof, "OCSVM": ocsvm, "SpecSig": spectral_signature,
    "GradNorm": grad_norm, "SEVER": sever, "GradSpectral": grad_spectral,
    "kNNLabel": knn_label, "kNNDist": knn_dist, "ConfidentLearning": confident_learning,
    "ECOD": ecod, "IsoForest+CL": iforest_cl, "ECOD+CL": ecod_cl, "AE-Recon": autoencoder,
}
