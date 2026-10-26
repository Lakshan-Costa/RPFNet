
import warnings
warnings.filterwarnings("ignore")
import hashlib
import zlib
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

try:
    from iTrim.regressiondatapoisoning.src.defenses import defense_both_trim_and_itrim
    from sklearn.linear_model import LinearRegression
    HAS_ITRIM = True
except ImportError:
    HAS_ITRIM = False

def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


N_JOBS = -1          # forest threads; the result does not depend on it
_RF_IMPORTANCE = {}  # (data hash) -> feature importances, see _rf_importances


def _rf_importances(X, y):
    """Feature importances of the surrogate forest that tree_aware and null_feature fit on the clean data."""
    X, y = np.ascontiguousarray(X), np.ascontiguousarray(y)
    h = hashlib.sha1(f"{X.dtype.str}{X.shape}{y.dtype.str}".encode())
    h.update(X.tobytes()); h.update(y.tobytes())
    key = h.hexdigest()
    if key not in _RF_IMPORTANCE:
        rf = RandomForestClassifier(random_state=42, n_jobs=N_JOBS).fit(X, y)
        _RF_IMPORTANCE[key] = rf.feature_importances_
    return _RF_IMPORTANCE[key]

#  ATTACK LIBRARY
def _flip_label(yi, classes, rng):
    others = classes[classes != yi]
    if len(others) == 0: return yi
    return int(rng.choice(others))


def _trigger_features(X, k=3):
    """The k highest-variance features."""
    v = np.asarray(X, dtype=np.float64).var(axis=0)
    return np.argsort(-np.round(v / (v.max() + 1e-12), 5), kind="stable")[:k]


def _attack(name, X, y, fraction, seed, y_cont):
    rng = np.random.default_rng(seed)
    n = max(1, int(len(X) * fraction))
    Xp, yp = X.copy(), y.copy()
    y_cont_p = y_cont.copy() if y_cont is not None else None
    pidx = rng.choice(len(X), n, replace=False)
    classes = np.unique(y)
    is_bin = len(classes) == 2

    if name == "target_shift":
        # Shift continuous targets
        if y_cont is None:
            # treat as feat_perturb if no continuous target
            name = "feat_perturb"
        else:
            y_std = y_cont.std() + 1e-8
            for i in pidx:
                y_cont_p[i] += 3.0 * y_std * rng.choice([-1., 1.])
            # Re-binarize
            yp = (y_cont_p > 0).astype(int)
            return Xp, yp, pidx, y_cont_p

    if name == "leverage_attack":
        if y_cont is None:
            name = "feat_perturb"
        else:
            stds = X.std(axis=0) + 1e-8
            y_std = y_cont.std() + 1e-8
            for i in pidx:
                # Push features to extreme positions
                Xp[i] += 3.0 * stds * rng.choice([-1., 1.], X.shape[1])
                # Flip target direction
                y_cont_p[i] = -y_cont_p[i] + rng.normal(0, 0.5 * y_std)
            yp = (y_cont_p > 0).astype(int)
            return Xp, yp, pidx, y_cont_p

    if name == "target_flip_extreme":
        # Flip targets to opposite extreme
        if y_cont is None:
            name = "label_flip"
        else:
            y_std = y_cont.std() + 1e-8
            for i in pidx:
                if y_cont[i] > 0:
                    y_cont_p[i] = -abs(y_cont[i]) - 2.0 * y_std
                else:
                    y_cont_p[i] = abs(y_cont[i]) + 2.0 * y_std
            yp = (y_cont_p > 0).astype(int)
            return Xp, yp, pidx, y_cont_p


    # All existing classification attacks
    if name == "label_flip":
        if is_bin:
            yp[pidx] = 1 - yp[pidx]
        else:
            for i in pidx: yp[i] = _flip_label(yp[i], classes, rng)

    elif name == "tree_aware":
        top = np.argsort(-_rf_importances(X, y))[:5]
        for i in pidx:
            for f in top: Xp[i, f] += 0.15 * rng.choice([-1, 1])
        if is_bin:
            yp[pidx] = 1 - yp[pidx]
        else:
            for i in pidx: yp[i] = _flip_label(yp[i], classes, rng)

    elif name == "catback":
        # Simple surrogate-based clean-label backdoor
        sur = LogisticRegression(max_iter=1000, random_state=42).fit(X, y)
        w, b = sur.coef_[0], sur.intercept_[0]

        trigger = np.zeros(X.shape[1])
        top_feats = np.argsort(-np.abs(w))[:3]
        trigger[top_feats] = 2.5  # subtle but consistent pattern

        for i in pidx:
            xi = Xp[i].copy()

            # Keep label SAME
            target = int(y[i])

            # Move toward decision boundary but keep class
            for _ in range(10):
                grad = (_sig(xi @ w + b) - target) * w
                xi -= 0.8 * np.sign(grad)

            # Add small trigger
            xi += 0.5 * trigger

            Xp[i] = xi

    elif name == "gradient_matching":
        sur = LogisticRegression(max_iter=1000, random_state=42).fit(X, y)
        w, b = sur.coef_[0], sur.intercept_[0]

        for i in pidx:
            xi = Xp[i].copy()

            # choose adversarial target
            yi = 1 - int(y[i]) if is_bin else _flip_label(y[i], classes, rng)

            # optimize toward adversarial gradient
            for _ in range(25):  # stronger
                grad = (_sig(xi @ w + b) - yi) * w
                xi -= 1.0 * grad / (np.linalg.norm(grad) + 1e-8)

            # ensure noticeable change
            xi += 0.5 * np.sign(w)

            Xp[i] = xi
            yp[i] = yi

    elif name == "feature_collision":
        # pick target class
        if is_bin:
            tgt_class = 1
        else:
            tgt_class = rng.choice(classes)

        tgt_idx = np.where(y == tgt_class)[0]
        if len(tgt_idx) == 0:
            return X, y, np.array([]), y_cont

        tgt_centroid = X[tgt_idx].mean(axis=0)

        for i in pidx:
            xi = Xp[i].copy()

            # move strongly toward target centroid
            alpha = rng.uniform(0.8, 0.95)
            xi = alpha * tgt_centroid + (1 - alpha) * xi

            # add small noise to avoid perfect clustering
            xi += rng.normal(0, 0.1, size=xi.shape)

            Xp[i] = xi

            # IMPORTANT: flip label to create conflict
            if is_bin:
                yp[i] = 1 - yp[i]
            else:
                yp[i] = tgt_class

    elif name == "clean_label":
        if is_bin:
            sur = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(X, y)
            w, b = sur.coef_[0], sur.intercept_[0]
            for i in pidx:
                xi = Xp[i].copy()
                tgt = 1 - int(y[i])
                for _ in range(10):
                    xi -= 1.5 * np.sign((_sig(xi @ w + b) - tgt) * w)
                Xp[i] = xi
        else:
            for i in pidx:
                tgt = _flip_label(yp[i], classes, rng)
                tgt_idx = np.where(y == tgt)[0]
                if len(tgt_idx) == 0: continue
                tgt_mu = X[tgt_idx].mean(0)
                direction = tgt_mu - Xp[i]
                norm = np.linalg.norm(direction) + 1e-8
                Xp[i] += 2.0 * direction / norm

    elif name == "backdoor":
        # Stamps the trigger without relabelling, so no backdoor is implanted;
        # kept only so earlier runs reproduce. Use "badnets".
        trig = _trigger_features(X)
        for i in pidx: Xp[i, trig] = 4.0

    elif name in ("badnets", "badnets_heavy"):
        trig = _trigger_features(X)
        val = 4.0 if name == "badnets" else 10.0
        if y_cont is None:
            target = classes[int(rng.integers(len(classes)))]
            cand = np.where(y != target)[0]
            pidx = rng.choice(cand, min(n, len(cand)), replace=False)
            Xp[np.ix_(pidx, trig)] = val
            yp[pidx] = target
        else:
            Xp[np.ix_(pidx, trig)] = val
            y_cont_p[pidx] = y_cont.mean() + 3.0 * y_cont.std()
            yp = (y_cont_p > 0).astype(int)

    elif name == "grad_flip":
        if is_bin:
            sur = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(X, y)
            w, b = sur.coef_[0], sur.intercept_[0]
            for i in pidx:
                xi = Xp[i].copy()
                yi = 1 - int(y[i])
                for _ in range(5):
                    xi += 0.5 * np.sign((_sig(xi @ w + b) - yi) * w)
                Xp[i] = xi
                yp[i] = yi
        else:
            for i in pidx:
                tgt = _flip_label(yp[i], classes, rng)
                tgt_idx = np.where(y == tgt)[0]
                if len(tgt_idx) == 0: continue
                Xp[i] += 0.3 * (X[tgt_idx].mean(0) - Xp[i])
                yp[i] = tgt

    elif name == "boundary_flip":
        sur = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(X, y)
        if is_bin:
            conf = np.abs(sur.predict_proba(X)[:, 1] - 0.5)
            cands = np.argsort(conf)[:int(n * 3)]
            pidx = rng.choice(cands, min(n, len(cands)), replace=False)
            yp[pidx] = 1 - yp[pidx]
        else:
            conf = sur.predict_proba(X).max(1)
            cands = np.argsort(conf)[:int(n * 3)]
            pidx = rng.choice(cands, min(n, len(cands)), replace=False)
            for i in pidx: yp[i] = _flip_label(yp[i], classes, rng)

    elif name == "gauss_noise":
        for i in pidx: Xp[i] += rng.normal(0, 0.3, X.shape[1])
        if is_bin:
            yp[pidx] = 1 - yp[pidx]
        else:
            for i in pidx: yp[i] = _flip_label(yp[i], classes, rng)

    elif name == "interpolation":
        for i in pidx:
            if is_bin:
                c0 = np.where(y == 0)[0]
                c1 = np.where(y == 1)[0]
                a = rng.uniform(0.3, 0.7)
                Xp[i] = a * X[rng.choice(c0)] + (1 - a) * X[rng.choice(c1)]
                yp[i] = 1 if a > 0.5 else 0
            else:
                c_a, c_b = rng.choice(classes, size=2, replace=False)
                ia = np.where(y == c_a)[0]
                ib = np.where(y == c_b)[0]
                if len(ia) == 0 or len(ib) == 0: continue
                a = rng.uniform(0.3, 0.7)
                Xp[i] = a * X[rng.choice(ia)] + (1 - a) * X[rng.choice(ib)]
                yp[i] = c_a if a > 0.5 else c_b

    elif name == "null_feature":
        top = np.argsort(-_rf_importances(X, y))[:8]
        for i in pidx: Xp[i, top] = 0.0

    elif name == "targeted_class":
        if is_bin:
            c1 = np.where(y == 1)[0]
            pidx = rng.choice(c1, min(n, len(c1)), replace=False)
            yp[pidx] = 0
        else:
            counts = np.bincount(y.astype(int))
            src_cls = int(np.argmax(counts))
            tgt_cls = int(np.argsort(counts)[-2])
            src_idx = np.where(y == src_cls)[0]
            pidx = rng.choice(src_idx, min(n, len(src_idx)), replace=False)
            yp[pidx] = tgt_cls

    elif name == "feat_perturb":
        stds = X.std(axis=0) + 1e-8
        for i in pidx:
            Xp[i] += 4.0 * stds * rng.choice([-1., 1.], X.shape[1])

    elif name == "backdoor_heavy":
        trig = _trigger_features(X)
        for i in pidx: Xp[i, trig] = 10.0

    elif name == "repr_inversion":
        for i in pidx: Xp[i] = -Xp[i]

    elif name == "dist_shift":
        for i in pidx: Xp[i] = 2.0 * Xp[i]

    elif name == "outlier_inject":
        means = X.mean(0)
        stds = X.std(0) + 1e-8
        nd = max(1, int(X.shape[1] * 0.2))
        for i in pidx:
            fs = rng.choice(X.shape[1], nd, replace=False)
            for j, f in enumerate(fs):
                Xp[i, f] = means[f] + (1 if j % 2 == 0 else -1) * 5.0 * stds[f]

    elif name == "feat_dropout":
        nd = max(1, int(X.shape[1] * 0.3))
        for i in pidx:
            Xp[i, rng.choice(X.shape[1], nd, replace=False)] = 0.0

    elif name == "combo_flip_perturb":
        stds = X.std(axis=0) + 1e-8
        for i in pidx:
            Xp[i] += 2.0 * stds * rng.choice([-1., 1.], X.shape[1])
        if is_bin:
            yp[pidx] = 1 - yp[pidx]
        else:
            for i in pidx: yp[i] = _flip_label(yp[i], classes, rng)

    elif name == "combo_flip_noise":
        for i in pidx:
            Xp[i] += rng.normal(0, 0.5, X.shape[1])
        if is_bin:
            yp[pidx] = 1 - yp[pidx]
        else:
            for i in pidx: yp[i] = _flip_label(yp[i], classes, rng)

    elif name == "adaptive_blend":
        for i in pidx:
            tgt = _flip_label(yp[i], classes, rng)
            tgt_idx = np.where(y == tgt)[0]
            if len(tgt_idx) == 0: continue
            tgt_mu = X[tgt_idx].mean(0)
            alpha = rng.uniform(0.4, 0.6)
            Xp[i] = ((1 - alpha) * Xp[i] + alpha * tgt_mu
                       + rng.normal(0, 0.1, X.shape[1]))
            yp[i] = tgt

    elif name == "influence_lr":
        # Optimised availability attack: wrong labels plus features moved by
        # projected gradient ascent on a retrained model's validation loss.
        if y_cont is not None:
            raise ValueError("influence_lr is a classification attack")
        for i in pidx: yp[i] = _flip_label(yp[i], classes, rng)
        Xp[pidx] = _influence_features(Xp, yp, pidx, X, y, rng)

    elif name == "subpopulation":
        # Jagielski et al. (2021): near-copies of one subpopulation (a k-means
        # cluster) labelled with a single wrong class.
        if y_cont is not None:
            raise ValueError("subpopulation is a classification attack")
        km = KMeans(n_clusters=max(2, min(10, len(X) // 5)), n_init=4,
                    random_state=seed).fit(X)
        sizes = np.bincount(km.labels_)
        members = np.where(km.labels_ == rng.choice(np.where(sizes >= 5)[0]))[0]
        majority = np.bincount(np.searchsorted(classes, y[members])).argmax()
        target = classes[(majority + 1 + rng.integers(len(classes) - 1)) % len(classes)]
        src = rng.choice(members, n, replace=True)
        Xp[pidx] = X[src] + rng.normal(0, 0.01, (n, X.shape[1])) * (X.std(0) + 1e-8)
        yp[pidx] = target

    elif name == "adaptive_dup":
        if y_cont is not None:
            raise ValueError("adaptive_dup is a classification attack")
        sur = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(X, y)
        c = rng.choice(classes)
        members = np.where(y == c)[0]
        p_own = sur.predict_proba(X[members])[:, np.searchsorted(sur.classes_, c)]
        hardest = members[np.argsort(p_own)[:max(1, min(len(members), n))]]
        src = rng.choice(hardest, n, replace=True)
        Xp[pidx] = X[src] + rng.normal(0, 0.01, (n, X.shape[1])) * (X.std(0) + 1e-8)
        yp[pidx] = c

    elif name in ("koh_minmax", "koh_influence"):
        if y_cont is not None:
            raise ValueError(f"{name} is a classification attack")
        from RPFNet.koh_attacks import koh_influence, koh_minmax
        fn = koh_minmax if name == "koh_minmax" else koh_influence
        Xq, yq = fn(X, y, pidx, rng)[:2]
        Xp[pidx] = Xq.astype(Xp.dtype)
        yp[pidx] = yq

    else:
        raise ValueError(f"Unknown attack '{name}'.")

    return Xp, yp, pidx, y_cont_p


def attack_seed(name: str, seed: int) -> int:
    """Stable per-attack seed. Python's hash() of a string changes between
    processes, so seeds built from it are not reproducible."""
    return 1000 * seed + zlib.crc32(name.encode()) % 997


def apply_attack(name, X, y, fraction, seed=42, y_cont=None, return_y_cont=False):
    """Poison a fraction of rows."""
    Xp, yp, pidx, y_cont_p = _attack(name, X, y, fraction, seed, y_cont)
    return (Xp, yp, pidx, y_cont_p) if return_y_cont else (Xp, yp, pidx)


def _onehot(y, classes):
    return (y[:, None] == classes[None, :]).astype(np.float64)


def _softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def _influence_features(Xp, yp, pidx, X_clean, y_clean, rng, C=1.0, steps=20, step=0.5):
    """Projected gradient ascent on the clean validation loss of a logistic regression retrained on the poisoned data."""
    n, d = Xp.shape
    classes = np.unique(y_clean)
    K = len(classes)
    lo, hi = np.percentile(X_clean, [1, 99], axis=0)
    rest = np.setdiff1d(np.arange(n), pidx)
    val = rng.choice(rest, min(len(rest), max(50, len(rest) // 4)), replace=False)
    Xv = np.hstack([X_clean[val], np.ones((len(val), 1))])
    Ev = _onehot(y_clean[val], classes)
    lam = 1.0 / (C * n)
    Xp = Xp.astype(np.float64).copy()
    for _ in range(steps):
        lr = LogisticRegression(C=C, max_iter=500).fit(Xp, yp)
        if not np.array_equal(lr.classes_, classes):
            break
        W = np.hstack([lr.coef_, lr.intercept_[:, None]])
        Theta = np.vstack([np.zeros_like(W), W]) if K == 2 else W   # softmax form
        Xa = np.hstack([Xp, np.ones((n, 1))])
        P = _softmax(Xa @ Theta.T)
        g_val = (_softmax(Xv @ Theta.T) - Ev).T @ Xv / len(val)

        def hvp(vec):
            V = vec.reshape(K, d + 1)
            A = Xa @ V.T
            B = P * A - P * (P * A).sum(1, keepdims=True)
            return ((B.T @ Xa) / n + lam * V).ravel()

        V = cg(LinearOperator((K * (d + 1),) * 2, matvec=hvp), g_val.ravel(),
               maxiter=100)[0].reshape(K, d + 1)
        Pi, Ei, Xi = P[pidx], _onehot(yp[pidx], classes), Xa[pidx]
        APi = (Xi @ V.T) * Pi
        Wd = Theta[:, :d]
        ds = (Pi - Ei) @ V[:, :d] + APi @ Wd - APi.sum(1, keepdims=True) * (Pi @ Wd)
        direction = -ds / (np.linalg.norm(ds, axis=1, keepdims=True) + 1e-12)
        Xp[pidx] = np.clip(Xp[pidx] + step * direction, lo, hi)
    return Xp[pidx]


def evasive_attack(X, y, fraction, score_fn, base="label_flip", lam=1.0,
                   iters=30, step=0.3, seed=42):
    """Adaptive attacker, level 2 (black-box, classification)."""
    rng = np.random.default_rng(seed)
    Xp, yp, pidx, _ = _attack(base, X, y, fraction, seed, None)
    rest = np.setdiff1d(np.arange(len(X)), pidx)
    val = rng.choice(rest, min(len(rest), max(50, len(rest) // 4)), replace=False)
    lo, hi = np.percentile(X, [1, 99], axis=0)
    sd = X.std(0) + 1e-8

    def evaluate(Xc):
        m = LogisticRegression(C=1.0, max_iter=500).fit(Xc, yp)
        dmg = log_loss(y[val], m.predict_proba(X[val]), labels=m.classes_)
        s = float(np.mean(np.asarray(score_fn(Xc, yp))[pidx]))
        return dmg - lam * s, dmg, s

    # Clean rows by label: moving a poison row toward clean rows that carry its
    # label makes it look consistent (stealthier, usually less damaging).
    by_label = {c: rest[yp[rest] == c] for c in np.unique(yp)}
    best, dmg, s = evaluate(Xp)
    history = [(dmg, s)]
    for _ in range(iters):
        cand = Xp.copy()
        sub = rng.choice(pidx, max(1, len(pidx) // 5), replace=False)
        toward = np.array([X[rng.choice(by_label[yp[i]])] for i in sub]) - cand[sub]
        noise = rng.normal(0, step, (len(sub), X.shape[1])) * sd
        use_toward = (rng.random(len(sub)) < 0.5)[:, None]
        move = np.where(use_toward, rng.uniform(0, step, (len(sub), 1)) * toward, noise)
        cand[sub] = np.clip(cand[sub] + move, lo, hi)
        j, dmg_c, s_c = evaluate(cand)
        if j > best:
            Xp, best, dmg, s = cand, j, dmg_c, s_c
        history.append((dmg, s))
    return Xp, yp, pidx, history
