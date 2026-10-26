"""Simulation of Proposition 4(a) (Appendix B.2)."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.neighbors import KNeighborsClassifier

EPS, C, R = 0.05, -0.85, 0.0037
eta = lambda x: np.where((x > 0) | (x < -0.7), 0.9, 0.1)

# clean population optimum of logistic regression (intercept, slope), lambda = 0
gx = np.linspace(-1, 1, 200001)
def pop_loss(t):
    z = t[0] + t[1] * gx
    return np.mean(eta(gx) * np.logaddexp(0, -z) + (1 - eta(gx)) * np.logaddexp(0, z))
theta = minimize(pop_loss, [0.0, 1.0], method="BFGS", options={"gtol": 1e-12}).x
pstar = lambda x: expit(theta[0] + theta[1] * x)
lo, hi = C - R, C + R
inf_gap = min(0.5 - pstar(lo), 0.5 - pstar(hi))
rhs = EPS * inf_gap / (2 * (1 - EPS) * 0.5)
bound = 0.8 * (2 * R) / 2            # int_B (2 eta - 1) dP_X with eta = 0.9, density 1/2
print(f"theta* = {theta.round(3)}, p*(c) = {pstar(C):.4f}, inf_B(1/2 - p*) = {inf_gap:.4f}")
print(f"vol(B) = {2 * R:.5f} <= {rhs:.5f}: {2 * R <= rhs};  bound on Delta = {bound:.5f}")

def knn_err(Xtr, ytr, k, grid):
    m = KNeighborsClassifier(n_neighbors=k).fit(Xtr[:, None], ytr)
    pred = m.predict(grid[:, None])
    return np.mean(np.where(pred == 1, 1 - eta(grid), eta(grid)))

def harm(X, y, clean):
    Xa = np.column_stack([np.ones_like(X), X])
    nll = lambda t: np.mean(np.logaddexp(0, -(2 * y - 1) * (Xa @ t)))
    th = minimize(nll, theta, method="BFGS", options={"gtol": 1e-10}).x
    g = (expit(Xa[clean] @ th) - y[clean])[:, None] * Xa[clean]
    return np.linalg.norm(g.mean(0))

rng = np.random.default_rng(0)
grid = np.linspace(-1, 1, 40001)
for n, reps in [(20_000, 20), (100_000, 8), (400_000, 3)]:
    k = int(np.sqrt(n)) | 1
    d, hs = [], []
    for _ in range(reps):
        X = rng.uniform(-1, 1, n)
        y = (rng.random(n) < eta(X)).astype(int)
        pois = rng.random(n) < EPS
        X[pois] = rng.uniform(lo, hi, pois.sum())
        y[pois] = (rng.random(pois.sum()) < pstar(X[pois])).astype(int)
        d.append(knn_err(X, y, k, grid) - knn_err(X[~pois], y[~pois], k, grid))
        hs.append(harm(X, y, ~pois))
    print(f"n={n:>7} k={k:>3} reps={reps:>2}  Delta = {np.mean(d):.4f} (s.e. {np.std(d, ddof=1) / np.sqrt(reps):.4f})"
          f"  proxy harm h = {np.mean(hs):.5f}")
