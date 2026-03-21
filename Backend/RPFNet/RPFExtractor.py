#RPFextractor.py
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import rankdata
from sklearn.ensemble import (RandomForestClassifier, IsolationForest,
                               GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

class RPFExtractor:
    DIM = 61

    BLOCK_NAMES = {
        "A"  : ("kNN Label Consistency",    slice(0,  8)),
        "B"  : ("Class-Cond. Geometry",     slice(8,  17)),
        "C"  : ("Scale Anomaly",            slice(17, 21)),
        "D"  : ("Cross-Val Influence",      slice(21, 34)),
        "E"  : ("Local Anomaly Extended",   slice(34, 47)),
        "F"  : ("Regression/Influence",     slice(47, 55)),
        "G"  : ("Structural Echo",          slice(55, 61)),
    }

    def __init__(self, k_small: int = 5, k_large: int = 15,
                 cv_folds: int = 5):
        self.k_small  = k_small
        self.k_large  = k_large
        self.cv_folds = cv_folds

    def extract(self, X: np.ndarray, y: np.ndarray,
                y_cont: np.ndarray = None) -> np.ndarray:
        n, d = X.shape
        rpf  = np.zeros((n, self.DIM), dtype=np.float32)

        k1    = max(2, min(self.k_small, n - 2))
        k2    = max(2, min(self.k_large, n - 2))
        k_max = k2 + 1

        nbrs = NearestNeighbors(n_neighbors=k_max, algorithm="ball_tree").fit(X)
        D_all, I_all = nbrs.kneighbors(X)
        D_all = D_all[:, 1:]
        I_all = I_all[:, 1:]

        self._block_a(X, y, rpf, k1, k2, D_all, I_all)
        self._block_b(X, y, rpf)
        self._block_c(X, y, rpf)
        self._block_d_oof(X, y, rpf)
        self._block_e(X, y, rpf, k2, D_all, I_all)
        self._block_f(X, y, rpf, y_cont=y_cont)
        self._block_g(X, y, rpf, k2, D_all, I_all)

        # Z-score normalise per-feature
        mu  = rpf.mean(0, keepdims=True)
        std = rpf.std(0,  keepdims=True) + 1e-8
        rpf = (rpf - mu) / std

        return rpf.astype(np.float32)

    # Block A
    def _block_a(self, X, y, rpf, k1, k2, D_all, I_all):
        for slot, k in enumerate([k1, k2]):
            D  = D_all[:, :k]
            I  = I_all[:, :k]
            NL = y[I]
            same = (NL == y[:, np.newaxis])

            frac_same = same.mean(1)
            d_same    = np.where(same, D, np.nan)
            d_diff    = np.where(~same, D, np.nan)
            max_d     = D.max() + 1.0
            ms = np.nanmean(d_same, 1); ms = np.where(np.isnan(ms), max_d,     ms)
            md = np.nanmean(d_diff, 1); md = np.where(np.isnan(md), max_d * 2, md)
            dist_ratio = np.log1p(ms) - np.log1p(md + 1e-8)

            p       = np.clip(frac_same, 1e-8, 1 - 1e-8)
            entropy = -(p * np.log(p) + (1-p) * np.log(1-p))

            near_same = np.where(same, D, np.inf).min(1)
            near_diff = np.where(~same, D, np.inf).min(1)
            near_same = np.where(np.isinf(near_same), max_d,     near_same)
            near_diff = np.where(np.isinf(near_diff), max_d * 2, near_diff)
            near_ratio = np.log1p(near_diff / (near_same + 1e-8))

            b = slot * 4
            rpf[:, b + 0] = frac_same
            rpf[:, b + 1] = dist_ratio
            rpf[:, b + 2] = entropy
            rpf[:, b + 3] = near_ratio

    # Block B
    def _block_b(self, X, y, rpf):
        n       = len(X)
        classes = np.unique(y)
        K       = len(classes)
        if K < 2:
            return

        mus  = np.zeros((K, X.shape[1]), np.float32)
        stds = np.ones( (K, X.shape[1]), np.float32)
        valid = np.ones(K, bool)
        for ki, c in enumerate(classes):
            idx = np.where(y == c)[0]
            if len(idx) < 2: valid[ki] = False; continue
            mus[ki]  = X[idx].mean(0)
            stds[ki] = X[idx].std(0) + 1e-8

        c2ki   = {c: ki for ki, c in enumerate(classes)}
        own_ki = np.array([c2ki[yi] for yi in y])
        own_mu = mus[own_ki]
        own_st = stds[own_ki]

        m_own = np.sqrt(((X - own_mu) / own_st) ** 2).mean(1).astype(np.float32)

        pca = PCA(n_components=min(5, X.shape[1]))
        Xpca = pca.fit_transform(X)
        mu_pca = np.zeros((K, Xpca.shape[1]))
        for ki, c in enumerate(classes):
            idx = np.where(y == c)[0]
            if len(idx) >= 2:
                mu_pca[ki] = Xpca[idx].mean(0)

        own_mu_pca = mu_pca[own_ki]
        pca_dist = np.linalg.norm(Xpca - own_mu_pca, axis=1)

        mahal_all = np.zeros((n, K), np.float32)
        for ki in range(K):
            if not valid[ki]: mahal_all[:, ki] = 1e9; continue
            mahal_all[:, ki] = np.sqrt(
                ((X - mus[ki]) / stds[ki]) ** 2).mean(1)

        same_mask   = (own_ki[:, None] == np.arange(K)[None, :])
        mahal_other = np.where(same_mask, np.inf, mahal_all)
        near_ki     = np.argmin(mahal_other, axis=1)
        m_other     = mahal_other[np.arange(n), near_ki].astype(np.float32)
        m_other     = np.where(np.isinf(m_other), m_own * 2, m_other)
        near_mu     = mus[near_ki]

        line     = near_mu - own_mu
        line_len = np.linalg.norm(line, axis=1, keepdims=True) + 1e-8
        proj     = ((X - own_mu) * (line / line_len)).sum(1).astype(np.float32)

        pct_own = np.zeros(n, np.float32)
        for ki, c in enumerate(classes):
            idx = np.where(y == c)[0]
            if len(idx) == 0: continue
            pct_own[idx] = rankdata(m_own[idx]).astype(np.float32) / len(idx)

        norm_x  = np.linalg.norm(X,      axis=1) + 1e-8
        norm_m  = np.linalg.norm(own_mu, axis=1) + 1e-8
        cos_own = (X * own_mu).sum(1) / (norm_x * norm_m)

        mid = (own_mu + near_mu) / 2
        bd  = ((X - mid) * (line / line_len)).sum(1).astype(np.float32)

        rpf[:, 8]  = m_own
        rpf[:, 9]  = m_other
        rpf[:, 10] = np.log1p(m_own + 1e-8) - np.log1p(m_other + 1e-8)
        rpf[:, 11] = proj / (line_len.squeeze() + 1e-8)
        rpf[:, 12] = pct_own
        rpf[:, 13] = m_other / (m_own + 1e-8)
        rpf[:, 14] = cos_own.astype(np.float32)
        rpf[:, 15] = bd
        rpf[:, 16] = pca_dist

    # Block C

    def _block_c(self, X, y, rpf):
        n = len(X)
        classes = np.unique(y)
        if len(classes) < 2:
            return

        l2 = np.linalg.norm(X, axis=1)

        l2_z = np.zeros(n, np.float32)
        for c in classes:
            idx = np.where(y == c)[0]
            if len(idx) < 2: continue
            mu_n = l2[idx].mean(); sd_n = l2[idx].std() + 1e-8
            l2_z[idx] = (l2[idx] - mu_n) / sd_n

        mus_c = {c: X[y == c].mean(0) for c in classes if (y == c).sum() >= 2}
        own_mu = np.array([mus_c.get(yi, X.mean(0)) for yi in y], np.float32)
        sign_agr = (np.sign(X) == np.sign(own_mu)).mean(1).astype(np.float32)

        mean_l2 = {c: l2[y == c].mean() + 1e-8 for c in classes
                   if (y == c).sum() >= 1}
        rel_scale = np.array([l2[i] / mean_l2.get(y[i], 1.0)
                               for i in range(n)], np.float32)

        mad = np.zeros(n, np.float32)
        for c in classes:
            idx = np.where(y == c)[0]
            if len(idx) < 2: continue
            med     = np.median(X[idx], axis=0)
            abs_dev = np.abs(X[idx] - med)
            nd      = max(1, int(X.shape[1] * 0.10))
            trimmed = np.sort(abs_dev, axis=1)[:, nd: -nd if nd else None]
            mad[idx] = (trimmed.mean(1) if trimmed.shape[1] > 0
                        else abs_dev.mean(1))

        rpf[:, 17] = l2_z
        rpf[:, 18] = sign_agr
        rpf[:, 19] = rel_scale
        rpf[:, 20] = mad

    # Block D
    def _block_d_oof(self, X: np.ndarray, y: np.ndarray, rpf: np.ndarray):
        n       = len(X)
        classes = np.unique(y)
        K       = len(classes)
        n_folds = min(self.cv_folds, int(min(np.bincount(y.astype(int)))))
        n_folds = max(2, n_folds)

        plr_true = np.full(n, 1.0 / K, np.float32)
        prf_true = np.full(n, 1.0 / K, np.float32)
        pgb_true = np.full(n, 1.0 / K, np.float32)
        prf_fold = np.zeros((n, n_folds), np.float32)
        prf_max_other = np.full(n, 1.0 / K, np.float32)

        rf_pred_class = np.zeros((n, n_folds), np.float32)

        skf      = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_idx = 0

        for train_idx, val_idx in skf.split(X, y):
            Xtr, Xvl = X[train_idx], X[val_idx]
            ytr       = y[train_idx]
            yvl       = y[val_idx]
            if len(np.unique(ytr)) < 2:
                fold_idx += 1; continue

            try:
                lr = LogisticRegression(C=1.0, max_iter=300,
                                        random_state=42).fit(Xtr, ytr)
                proba_lr   = lr.predict_proba(Xvl)
                lr_cls     = lr.classes_.tolist()
                lr_col     = np.array([lr_cls.index(c) if c in lr_cls else 0
                                        for c in yvl])
                plr_true[val_idx] = proba_lr[np.arange(len(val_idx)), lr_col]
            except Exception:
                pass

            try:
                rf = RandomForestClassifier(
                    n_estimators=30, max_depth=4, min_samples_leaf=3,
                    class_weight="balanced", random_state=42, n_jobs=-1
                ).fit(Xtr, ytr)

                proba_rf = rf.predict_proba(Xvl)
                rf_cls   = rf.classes_.tolist()

                rf_col = np.array([rf_cls.index(c) if c in rf_cls else 0
                                for c in yvl])

                p_true = proba_rf[np.arange(len(val_idx)), rf_col]

                prf_true[val_idx] = p_true
                prf_fold[val_idx, fold_idx] = p_true

                proba_rf_cp = proba_rf.copy()
                proba_rf_cp[np.arange(len(val_idx)), rf_col] = 0.0
                prf_max_other[val_idx] = proba_rf_cp.max(1)

            except Exception:
                pass
            try:
                gb = GradientBoostingClassifier(
                    n_estimators=30, max_depth=3, learning_rate=0.1,
                    random_state=42
                ).fit(Xtr, ytr)
                proba_gb = gb.predict_proba(Xvl)
                gb_cls   = gb.classes_.tolist()
                gb_col   = np.array([gb_cls.index(c) if c in gb_cls else 0
                                      for c in yvl])
                pgb_true[val_idx] = proba_gb[np.arange(len(val_idx)), gb_col]
            except Exception:
                pass

            fold_idx += 1

        wrong_lr = (1 - plr_true).astype(np.float32)
        wrong_rf = (1 - prf_true).astype(np.float32)

        ce_lr    = -np.log(np.clip(plr_true, 1e-8, 1.0)).astype(np.float32)
        ce_rf    = -np.log(np.clip(prf_true, 1e-8, 1.0)).astype(np.float32)

        disagree  = np.abs(plr_true - prf_true).astype(np.float32)
        margin_rf = (prf_true - prf_max_other).astype(np.float32)

        rk_lr = rankdata(wrong_lr).astype(np.float32) / n
        rk_rf = rankdata(wrong_rf).astype(np.float32) / n

        fold_std = prf_fold.std(1).astype(np.float32)

        class_instability = (rf_pred_class.std(1) > 0).astype(np.float32)

        rand_thresh = 1.0 / K
        correct_lr  = (plr_true > rand_thresh).astype(np.float32)
        correct_rf  = (prf_true > rand_thresh).astype(np.float32)
        both_agree  = correct_lr * correct_rf

        confidence = prf_true.copy()
        flip_ind = (prf_true < rand_thresh).astype(np.float32)

        # 3-model disagreement score
        disagree_3 = (np.abs(plr_true - pgb_true)
                      + np.abs(prf_true - pgb_true)
                      + np.abs(plr_true - prf_true)).astype(np.float32) / 3.0

        rpf[:, 21] = wrong_lr
        rpf[:, 22] = wrong_rf
        rpf[:, 23] = ce_lr
        rpf[:, 24] = ce_rf
        rpf[:, 25] = disagree
        rpf[:, 26] = margin_rf
        rpf[:, 27] = rk_lr
        rpf[:, 28] = rk_rf
        rpf[:, 29] = fold_std
        rpf[:, 30] = both_agree
        rpf[:, 31] = confidence
        rpf[:, 32] = flip_ind
        rpf[:, 33] = disagree_3

    # Block E
    def _block_e(self, X, y, rpf, k2, D_all, I_all):
        n = len(X)
        k = min(k2, n - 2)
        D = D_all[:, :k]
        I = I_all[:, :k]
        NL   = y[I]
        same = (NL == y[:, np.newaxis])

        try:
            iso     = IsolationForest(n_estimators=100, random_state=42, n_jobs=-1)
            iso_raw = -iso.fit(X).score_samples(X)
        except Exception:
            iso_raw = np.zeros(n, np.float32)

        local_purity = same.mean(1)
        mean_knn     = D.mean(1)
        nb_mean      = mean_knn[I].mean(1)
        lof_approx   = mean_knn / (nb_mean + 1e-8)

        d_same = np.where(same, D, np.nan)
        d_diff = np.where(~same, D, np.nan)
        max_d  = D.max() + 1.0
        ms = np.nanmean(d_same, 1); ms = np.where(np.isnan(ms), max_d,     ms)
        md = np.nanmean(d_diff, 1); md = np.where(np.isnan(md), max_d * 2, md)

        local_geom   = D.std(1)
        iso_impurity = iso_raw * (1.0 - local_purity + 1e-3)
        rk_iso       = rankdata(iso_raw).astype(np.float32) / n

        rpf[:, 34] = iso_raw
        rpf[:, 35] = local_purity
        rpf[:, 36] = lof_approx
        rpf[:, 37] = ms
        rpf[:, 38] = md
        rpf[:, 39] = local_geom
        rpf[:, 40] = iso_impurity
        rpf[:, 41] = rk_iso

        classes   = np.unique(y)
        any_valid = any(len(np.where(y == c)[0]) >= 2 for c in classes)
        if not any_valid:
            return

        local_density = 1.0 / (mean_knn + 1e-8)
        density_ratio = np.zeros(n, np.float32)
        for c in classes:
            idx = np.where(y == c)[0]
            if len(idx) == 0: continue
            density_ratio[idx] = local_density[idx] / (local_density[idx].mean() + 1e-8)
        rpf[:, 42] = density_ratio

        if D.shape[1] > 4:
            trim    = max(1, int(D.shape[1] * 0.1))
            D_trim  = np.sort(D, axis=1)[:, trim: -trim]
            nb_trim = np.sort(D[I], axis=2)[:, :, trim: -trim].mean(2).mean(1)
            lof_trim = D_trim.mean(1) / (nb_trim + 1e-8)
        else:
            lof_trim = lof_approx
        rpf[:, 43] = lof_trim

        K       = len(classes)
        mus_d   = np.zeros((K, X.shape[1]), np.float32)
        for ki, c in enumerate(classes):
            idx = np.where(y == c)[0]
            if len(idx) >= 2: mus_d[ki] = X[idx].mean(0)
        c2ki_d  = {c: ki for ki, c in enumerate(classes)}
        own_ki  = np.array([c2ki_d[yi] for yi in y])
        own_mu  = mus_d[own_ki]
        dists_c = np.stack([np.linalg.norm(X - mus_d[ki], axis=1)
                             for ki in range(K)], axis=1)
        same_m  = (own_ki[:, None] == np.arange(K)[None, :])
        dists_c[same_m] = np.inf
        near_ki = np.argmin(dists_c, axis=1)
        near_mu = mus_d[near_ki]
        normal  = near_mu - own_mu
        nlen    = np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8
        rpf[:, 44] = ((X - own_mu) * (normal / nlen)).sum(1).astype(np.float32)

        med_same = np.where(same, D, np.inf).min(1)
        med_same = np.where(np.isinf(med_same), max_d, med_same)
        rpf[:, 45] = np.where(~same, D < med_same[:, None], False).mean(1)

        try:
            sur      = LogisticRegression(C=1.0, max_iter=300,
                                          random_state=42).fit(X, y)
            ambiguity = (1.0 - sur.predict_proba(X).max(1)).astype(np.float32)
            amb_rank  = np.zeros(n, np.float32)
            for c in classes:
                idx = np.where(y == c)[0]
                if len(idx) == 0: continue
                amb_rank[idx] = (rankdata(ambiguity[idx]).astype(np.float32)
                                  / len(idx))
            rpf[:, 46] = amb_rank
        except Exception:
            rpf[:, 46] = 0.0

    #  Block F: Regression/Influence Features 
    def _block_f(self, X: np.ndarray, y: np.ndarray, rpf: np.ndarray,
                 y_cont: np.ndarray = None):
        n, d = X.shape

        # Use continuous target if available, else cast labels to float
        y_reg = y_cont.astype(np.float64) if y_cont is not None \
                else y.astype(np.float64)

        # Fit ridge regression
        try:
            ridge = Ridge(alpha=1.0).fit(X.astype(np.float64), y_reg)
            y_hat = ridge.predict(X.astype(np.float64))
            resid = y_reg - y_hat
        except Exception:
            rpf[:, 47:55] = 0.0
            return

        # Leverage (hat matrix diagonal)
        try:
            # H = X(X'X + αI)^{-1}X'  — diagonal only
            XtX_inv = np.linalg.inv(
                X.astype(np.float64).T @ X.astype(np.float64)
                + 1.0 * np.eye(d, dtype=np.float64))
            H_diag = (X.astype(np.float64) @ XtX_inv
                      * X.astype(np.float64)).sum(1)
        except np.linalg.LinAlgError:
            H_diag = np.full(n, 1.0 / n, dtype=np.float64)

        rpf[:, 47] = rankdata(H_diag).astype(np.float32) / n

        # Studentized residual 
        mse = (resid ** 2).mean() + 1e-8
        stud_resid = resid / np.sqrt(mse * (1 - np.clip(H_diag, 0, 0.999) + 1e-8))
        rpf[:, 48] = rankdata(np.abs(stud_resid)).astype(np.float32) / n

        # Cook's distance
        p = min(d, n - 1)
        cooks_d = (stud_resid ** 2 / max(p, 1)) * (H_diag / (1 - H_diag + 1e-8))
        rpf[:, 49] = rankdata(cooks_d).astype(np.float32) / n

        # DFFITS 
        dffits = stud_resid * np.sqrt(H_diag / (1 - H_diag + 1e-8))
        rpf[:, 50] = rankdata(np.abs(dffits)).astype(np.float32) / n

        try:
            k = max(2, min(10, n - 2))
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(X)
            _, I_local = nbrs.kneighbors(X)
            I_local = I_local[:, 1:]  # exclude self
            nb_resid = resid[I_local]
            sign_agree = (np.sign(nb_resid) == np.sign(resid[:, np.newaxis])).mean(1)
            rpf[:, 51] = sign_agree.astype(np.float32)
        except Exception:
            rpf[:, 51] = 0.5

        pred_rank = np.zeros(n, np.float32)
        classes = np.unique(y)
        for c in classes:
            idx = np.where(y == c)[0]
            if len(idx) < 2: continue
            pred_rank[idx] = rankdata(y_hat[idx]).astype(np.float32) / len(idx)
        rpf[:, 52] = pred_rank
        rpf[:, 53] = rankdata(np.abs(resid) * H_diag).astype(np.float32) / n

        loo_influence = np.abs(resid * H_diag / (1 - H_diag + 1e-8))
        rpf[:, 54] = rankdata(loo_influence).astype(np.float32) / n

    def _block_g(self, X: np.ndarray, y: np.ndarray, rpf: np.ndarray,
                 k2: int, D_all: np.ndarray, I_all: np.ndarray):
        n = len(X)
        k = min(k2, D_all.shape[1])
        I = I_all[:, :k]
        D = D_all[:, :k]
        rev_counts = np.zeros(n, dtype=np.float32)

        rev_mean = rev_counts.mean() + 1e-8
        rpf[:, 55] = rev_counts / rev_mean

        pos_sum = np.zeros(n, dtype=np.float64)
        pos_cnt = np.zeros(n, dtype=np.float64)
        for p in range(k):
            targets = I[:, p]
            np.add.at(pos_sum, targets, float(p))
            np.add.at(pos_cnt, targets, 1.0)

        avg_position = pos_sum / (pos_cnt + 1e-8)
        embedding_depth = (1.0 - avg_position / (k + 1e-8)).astype(np.float32)

        rpf[:, 56] = embedding_depth

        rev_adj = [[] for _ in range(n)]
        for j in range(n):
            for p in range(k):
                rev_adj[I[j, p]].append(j)

        hop1_sizes = np.array([len(rev_adj[i]) for i in range(n)],
                               dtype=np.float32)
        hop2_sizes = np.zeros(n, dtype=np.float32)

        # For large datasets
        max_hop1_expand = 200  # cap inner loop breadth

        for i in range(n):
            hop1 = rev_adj[i]
            if len(hop1) == 0:
                continue
            hop2_set = set()
            hop1_set = set(hop1)
            expand = hop1 if len(hop1) <= max_hop1_expand else \
                     [hop1[idx] for idx in
                      np.random.default_rng(i).choice(
                          len(hop1), max_hop1_expand, replace=False)]
            for j in expand:
                for m in rev_adj[j]:
                    if m != i and m not in hop1_set:
                        hop2_set.add(m)
            # Scale up if we subsampled
            scale = len(hop1) / len(expand) if len(expand) > 0 else 1.0
            hop2_sizes[i] = len(hop2_set) * scale

        # 2-hop echo magnitude
        hop2_mean = hop2_sizes.mean() + 1e-8
        rpf[:, 57] = hop2_sizes / hop2_mean

        NL = y[I]
        label_agree = (NL == y[:, np.newaxis]).mean(1).astype(np.float32)
        centrality = rev_counts / rev_mean
        rpf[:, 58] = centrality * (1.0 - label_agree + 0.01)
        rpf[:, 59] = hop2_sizes / (hop1_sizes + 1e-8)

        disp_sum = np.zeros(n, dtype=np.float64)
        disp_cnt = np.zeros(n, dtype=np.float64)
        boundary = D[:, -1]  # each sample's kth-neighbor distance

        for p in range(k):
            targets = I[:, p]
            gap = np.maximum(boundary - D[:, p], 0.0)
            np.add.at(disp_sum, targets, gap)
            np.add.at(disp_cnt, targets, 1.0)

        displacement = (disp_sum / (disp_cnt + 1e-8)).astype(np.float32)

        # G5: Displacement cost
        rpf[:, 60] = rankdata(displacement).astype(np.float32) / n