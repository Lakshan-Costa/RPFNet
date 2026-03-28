
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    from iTrim.regressiondatapoisoning.src.defenses import defense_both_trim_and_itrim
    from sklearn.linear_model import LinearRegression
    HAS_ITRIM = True
except ImportError:
    HAS_ITRIM = False

def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

#  ATTACK LIBRARY
def _flip_label(yi, classes, rng):
    others = classes[classes != yi]
    if len(others) == 0: return yi
    return int(rng.choice(others))


def apply_attack(name, X, y, fraction, seed=42, y_cont=None):
    rng     = np.random.default_rng(seed)
    n       = max(1, int(len(X) * fraction))
    Xp, yp  = X.copy(), y.copy()
    y_cont_p = y_cont.copy() if y_cont is not None else None
    pidx    = rng.choice(len(X), n, replace=False)
    classes = np.unique(y)
    is_bin  = len(classes) == 2

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
            return Xp, yp, pidx

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
            return Xp, yp, pidx

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
            return Xp, yp, pidx


    # All existing classification attacks
    if name == "label_flip":
        if is_bin:
            yp[pidx] = 1 - yp[pidx]
        else:
            for i in pidx: yp[i] = _flip_label(yp[i], classes, rng)

    elif name == "tree_aware":
        rf  = RandomForestClassifier(random_state=42, n_jobs=-1).fit(X, y)
        top = np.argsort(-rf.feature_importances_)[:5]
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
            return X, y, np.array([])

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
                xi = Xp[i].copy(); tgt = 1 - int(y[i])
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
        trig = np.argsort(-X.var(axis=0))[:3]
        for i in pidx: Xp[i, trig] = 4.0

    elif name == "grad_flip":
        if is_bin:
            sur = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(X, y)
            w, b = sur.coef_[0], sur.intercept_[0]
            for i in pidx:
                xi = Xp[i].copy(); yi = 1 - int(y[i])
                for _ in range(5):
                    xi += 0.5 * np.sign((_sig(xi @ w + b) - yi) * w)
                Xp[i] = xi; yp[i] = yi
        else:
            for i in pidx:
                tgt = _flip_label(yp[i], classes, rng)
                tgt_idx = np.where(y == tgt)[0]
                if len(tgt_idx) == 0: continue
                Xp[i] += 0.3 * (X[tgt_idx].mean(0) - Xp[i])
                yp[i] = tgt

    elif name == "boundary_flip":
        sur  = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(X, y)
        if is_bin:
            conf  = np.abs(sur.predict_proba(X)[:, 1] - 0.5)
            cands = np.argsort(conf)[:int(n * 3)]
            pidx  = rng.choice(cands, min(n, len(cands)), replace=False)
            yp[pidx] = 1 - yp[pidx]
        else:
            conf  = sur.predict_proba(X).max(1)
            cands = np.argsort(conf)[:int(n * 3)]
            pidx  = rng.choice(cands, min(n, len(cands)), replace=False)
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
                c0 = np.where(y == 0)[0]; c1 = np.where(y == 1)[0]
                a = rng.uniform(0.3, 0.7)
                Xp[i] = a * X[rng.choice(c0)] + (1 - a) * X[rng.choice(c1)]
                yp[i] = 1 if a > 0.5 else 0
            else:
                c_a, c_b = rng.choice(classes, size=2, replace=False)
                ia = np.where(y == c_a)[0]; ib = np.where(y == c_b)[0]
                if len(ia) == 0 or len(ib) == 0: continue
                a = rng.uniform(0.3, 0.7)
                Xp[i] = a * X[rng.choice(ia)] + (1 - a) * X[rng.choice(ib)]
                yp[i] = c_a if a > 0.5 else c_b

    elif name == "null_feature":
        sur = RandomForestClassifier(random_state=42, n_jobs=-1).fit(X, y)
        top = np.argsort(-sur.feature_importances_)[:8]
        for i in pidx: Xp[i, top] = 0.0

    elif name == "targeted_class":
        if is_bin:
            c1   = np.where(y == 1)[0]
            pidx = rng.choice(c1, min(n, len(c1)), replace=False)
            yp[pidx] = 0
        else:
            counts  = np.bincount(y.astype(int))
            src_cls = int(np.argmax(counts))
            tgt_cls = int(np.argsort(counts)[-2])
            src_idx = np.where(y == src_cls)[0]
            pidx    = rng.choice(src_idx, min(n, len(src_idx)), replace=False)
            yp[pidx] = tgt_cls

    elif name == "feat_perturb":
        stds = X.std(axis=0) + 1e-8
        for i in pidx:
            Xp[i] += 4.0 * stds * rng.choice([-1., 1.], X.shape[1])

    elif name == "backdoor_heavy":
        trig = np.argsort(-X.var(axis=0))[:3]
        for i in pidx: Xp[i, trig] = 10.0

    elif name == "repr_inversion":
        for i in pidx: Xp[i] = -Xp[i]

    elif name == "dist_shift":
        for i in pidx: Xp[i] = 2.0 * Xp[i]

    elif name == "outlier_inject":
        means = X.mean(0); stds = X.std(0) + 1e-8
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
            tgt_mu  = X[tgt_idx].mean(0)
            alpha   = rng.uniform(0.4, 0.6)
            Xp[i]   = ((1 - alpha) * Xp[i] + alpha * tgt_mu
                       + rng.normal(0, 0.1, X.shape[1]))
            yp[i]   = tgt

    else:
        raise ValueError(f"Unknown attack '{name}'.")

    return Xp, yp, pidx