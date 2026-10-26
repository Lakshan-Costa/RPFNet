"""Numerical checks of the theory section."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


def _fit(X, y, C):
    Xa = np.hstack([X, np.ones((len(X), 1))])
    m = LogisticRegression(C=C, fit_intercept=False, tol=1e-10, max_iter=20000).fit(Xa, y)
    return m.coef_, m.classes_


def _grads(theta, classes, X, y):
    """Per-sample loss gradients in sklearn's parameterisation (binary: one
    weight vector; multiclass: one row per class), flattened."""
    Xa = np.hstack([X, np.ones((len(X), 1))])
    Z = Xa @ theta.T
    if theta.shape[0] == 1:                       # binary, sigmoid
        p = 1.0 / (1.0 + np.exp(-Z[:, 0]))
        return (p - (y == classes[1]))[:, None] * Xa
    P = np.exp(Z - Z.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    E = (y[:, None] == classes[None, :]).astype(np.float64)
    return ((P - E)[:, :, None] * Xa[:, None, :]).reshape(len(X), -1)


def _predict(theta, classes, X):
    Z = np.hstack([X, np.ones((len(X), 1))]) @ theta.T
    return classes[(Z[:, 0] > 0).astype(int)] if theta.shape[0] == 1 else classes[Z.argmax(1)]


def gradient_diagnostics(Xp, yp, pidx, C=1.0, test=None, original=None, seed=0):
    """Measure the quantities of the theory section on one poisoned dataset."""
    Xp = np.asarray(Xp, dtype=np.float64)
    n = len(Xp)
    pidx = np.asarray(pidx, dtype=int)
    clean = np.setdiff1d(np.arange(n), pidx)
    eps = len(pidx) / n
    lam = 1.0 / (C * n)
    th_eps, classes = _fit(Xp, yp, C)
    th_P, classes_P = _fit(Xp[clean], yp[clean], 1.0 / (lam * len(clean)))
    if not np.array_equal(classes, classes_P):
        return {"error": "a class is missing from the clean rows"}

    G = _grads(th_eps, classes, Xp, yp)
    mu_P, mu_Q = G[clean].mean(0), G[pidx].mean(0)
    reg = lam * th_eps.ravel()
    r_all = float(np.linalg.norm(G.mean(0) + reg))                    # ~0 at the optimum
    r_P = float(np.linalg.norm(_grads(th_P, classes, Xp[clean], yp[clean]).mean(0) + lam * th_P.ravel()))
    delta = float(np.linalg.norm(th_eps.ravel() - th_P.ravel()))
    gap = float(np.linalg.norm(mu_Q - mu_P))
    Xa = np.hstack([Xp, np.ones((n, 1))])
    c = 0.25 if th_eps.shape[0] == 1 else 0.5
    beta = lam + c * float(np.linalg.eigvalsh(Xa[clean].T @ Xa[clean] / len(clean))[-1])
    lower, upper = lam * delta / eps, beta * delta / eps
    dlogit = np.linalg.norm(Xa @ (th_eps - th_P).T, axis=1).max()
    dlogit_bound = np.linalg.norm(Xa, axis=1).max() * (eps * gap + r_all + r_P) / lam

    cov = lambda A: np.cov(A.T, bias=True)
    C_P = cov(G[clean])
    W = (1 - eps) * C_P + eps * cov(G[pidx])
    w = float(np.linalg.eigvalsh(W)[-1])
    D = eps * (1 - eps) * gap ** 2 / w
    lam_S, V = np.linalg.eigh(cov(G))
    lam_S, v = float(lam_S[-1]), V[:, -1]
    lam_P = float(np.linalg.eigvalsh(C_P)[-1])
    harm = float(np.linalg.norm(mu_P + reg))                           # ||grad F_P(theta_eps)||
    cert = float(np.sqrt(eps * lam_S / (1 - eps))) + r_all
    u = (mu_Q - mu_P) / gap
    sin = float(np.sqrt(max(0.0, 1 - (v @ u) ** 2)))
    G_v = abs(float(v @ (mu_Q - mu_P)))
    t = (G - G.mean(0)) @ v
    s_P, s_Q = float(t[clean].var()), float(t[pidx].var())
    ytrue = np.zeros(n, bool)
    ytrue[pidx] = True
    auroc_h = float(roc_auc_score(ytrue, np.abs(t)))
    auroc_bound = 1 - (s_P + s_Q) / (G_v ** 2 * (0.5 - eps) ** 2) if eps < 0.5 else float("nan")
    out = {
        "eps": eps, "lam": lam, "n": n, "residual_all": r_all, "residual_P": r_P,
        "delta": delta, "gap": gap, "thm1_lower": lower, "thm1_upper": upper,
        "thm1_holds": bool(eps * gap >= lam * delta - r_P - r_all - 1e-12
                           and eps * gap <= beta * delta + r_P + r_all + 1e-12),
        "gap_over_lower": gap / lower if lower > 0 else float("nan"),
        "max_logit_change": float(dlogit), "logit_change_bound": float(dlogit_bound),
        "detectability_D": D, "sin_theta": sin, "sin_bound": min(1.0, 2 / D),
        "thm2_direction_holds": bool(sin <= min(1.0, 2 / D) + 1e-9),
        "auroc_blockH": auroc_h, "auroc_bound": float(max(0.0, auroc_bound)),
        "thm2_auroc_holds": bool(not auroc_bound > auroc_h + 1e-9),
        "auprc_blockH": float(average_precision_score(ytrue, np.abs(t))),
        "lam_max_S": lam_S, "lam_max_P": lam_P, "harm": harm, "cert": cert,
        "cert_holds": bool(harm <= cert * (1 + 1e-9) + 1e-12),
        "harm_ratio": (1 - eps) * harm ** 2 / (eps * lam_P), "spike_ratio": lam_S / lam_P,
    }
    if test is not None:
        Xt, yt = np.asarray(test[0], dtype=np.float64), np.asarray(test[1])
        acc = lambda th: float((_predict(th, classes, Xt) == yt).mean())
        rf = lambda X, y: RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed) \
            .fit(X, y).score(Xt, yt)
        out["damage_proxy"] = acc(th_P) - acc(th_eps)
        Xo, yo = original if original is not None else (Xp[clean], yp[clean])
        out["damage_rf"] = float(rf(Xo, yo) - rf(Xp, yp))
    return out
