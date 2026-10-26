import warnings
warnings.filterwarnings("ignore")

import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.decomposition import PCA
from scipy.stats import rankdata
from sklearn.ensemble import (RandomForestClassifier, IsolationForest,
                               GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Threads for the forests below. Their output does not depend on it; the
# parallel precompute (evaluation.precompute) sets 1 inside each worker.
N_JOBS = -1

# One name per column of the version-1 fingerprint (see the paper's appendix table).
FEATURE_NAMES_V1 = (
    "A_same_frac_ks", "A_log_dist_ratio_ks", "A_label_entropy_ks", "A_near_diff_same_ks",
    "A_same_frac_kl", "A_log_dist_ratio_kl", "A_label_entropy_kl", "A_near_diff_same_kl",
    "B_own_centroid_dist", "B_other_centroid_dist", "B_log_own_other", "B_proj_to_other",
    "B_own_dist_pct", "B_other_own_ratio", "B_cos_own_mean", "B_midpoint_proj", "B_pca_own_dist",
    "C_norm_z", "C_sign_agree", "C_rel_norm", "C_trimmed_mad",
    "D_lr_err", "D_rf_err", "D_lr_ce", "D_rf_ce", "D_lr_rf_diff", "D_rf_margin",
    "D_lr_err_rank", "D_rf_err_rank", "D_rf_fold_std", "D_lr_rf_both_correct",
    "D_rf_true_prob", "D_rf_below_chance", "D_3model_disagree",
    "E_iforest", "E_local_purity", "E_lof_approx", "E_mean_same_dist", "E_mean_diff_dist",
    "E_knn_dist_std", "E_iforest_x_impurity", "E_iforest_rank", "E_density_ratio",
    "E_lof_trimmed", "E_proj_to_other", "E_diff_closer_frac", "E_ambiguity_rank",
    "F_leverage_rank", "F_stud_resid_rank", "F_cooks_rank", "F_dffits_rank",
    "F_nbr_resid_sign_agree", "F_pred_rank_in_class", "F_resid_x_lev_rank", "F_loo_influence_rank",
    "G_rev_nbr_count", "G_embedding_depth", "G_two_hop_reach", "G_centrality_x_disagree",
    "G_two_hop_one_hop_ratio", "G_displacement_rank",
)


class RPFExtractor:
    """Relational Poison Fingerprint."""
    DIM = 61   # version-1 layout; use `self.dim` for the active version

    BLOCK_NAMES = {
        "A" : ("kNN Label Consistency", slice(0, 8)),
        "B" : ("Class-Cond. Geometry", slice(8, 17)),
        "C" : ("Scale Anomaly", slice(17, 21)),
        "D" : ("Cross-Val Influence", slice(21, 34)),
        "E" : ("Local Anomaly Extended", slice(34, 47)),
        "F" : ("Regression/Influence", slice(47, 55)),
        "G" : ("Structural Echo", slice(55, 61)),
    }

    # v2 drops columns that equal another column by construction on every
    # dataset: 31 = 1 - col 22, 35 = col 4, 50 has the same ranking as col 49.
    V2_DROP = (31, 35, 50)

    def __init__(self, k_small: int = 5, k_large: int = 15,
                 cv_folds: int = 5, feature_version: int = 1,
                 random_state: int = 42):
        if feature_version not in (1, 2):
            raise ValueError(f"feature_version must be 1 or 2, got {feature_version}")
        self.k_small = k_small
        self.k_large = k_large
        self.cv_folds = cv_folds
        self.feature_version = feature_version
        self.random_state = random_state
        self._last_neighbors = None
        self.last_timings = {}

        if feature_version == 1:
            self._keep = np.arange(self.DIM)
            self.feature_names = list(FEATURE_NAMES_V1)
            self.block_names = dict(self.BLOCK_NAMES)
        else:
            self._keep = np.array([i for i in range(self.DIM) if i not in self.V2_DROP])
            names = [FEATURE_NAMES_V1[i] for i in self._keep]
            names[names.index("D_rf_fold_std")] = "D_rf_tree_std"
            self.feature_names = names + ["H_grad_spectral"]
            self.block_names = {}
            for key, (desc, sl) in self.BLOCK_NAMES.items():
                start = int((self._keep < sl.start).sum())
                width = int(((self._keep >= sl.start) & (self._keep < sl.stop)).sum())
                self.block_names[key] = (desc, slice(start, start + width))
            self.block_names["H"] = ("Gradient Spectral", slice(len(self._keep), len(self._keep) + 1))
        self.dim = len(self.feature_names)

    def _compute_neighbors(self, X, k):
        X_f32 = np.ascontiguousarray(X, dtype=np.float32)
        n, d = X_f32.shape
        v1 = self.feature_version == 1
        if FAISS_AVAILABLE and n > 500:
            if n > 50000:
                nlist = min(256, n // 100)
                quantizer = faiss.IndexFlatL2(d)
                index = faiss.IndexIVFFlat(quantizer, d, nlist)
                index.train(X_f32)
                index.add(X_f32)
                index.nprobe = min(32, nlist)
            else:
                index = faiss.IndexFlatL2(d)
                index.add(X_f32)
            distances, indices = index.search(X_f32, k + 1)
            if v1:
                return distances[:, 1:], indices[:, 1:]
            # FAISS returns squared L2; v2 uses true distances, like sklearn.
            D, I_neighbors = np.sqrt(np.maximum(distances, 0.0)), indices
        else:
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(X)
            D, I_neighbors = nbrs.kneighbors(X)
            if v1:
                return D[:, 1:], I_neighbors[:, 1:]
        # v2: drop the query point wherever it appears. With duplicate rows the
        # point itself is not guaranteed to be returned first.
        is_self = I_neighbors == np.arange(n)[:, None]
        keep = ~is_self
        keep[~is_self.any(1), -1] = False
        return D[keep].reshape(n, k), I_neighbors[keep].reshape(n, k)

    def extract(self, X: np.ndarray, y: np.ndarray,
                y_cont: np.ndarray = None) -> np.ndarray:
        n, d = X.shape
        rpf = np.zeros((n, self.DIM), dtype=np.float32)
        timings = {}

        k1 = max(2, min(self.k_small, n - 2))
        k2 = max(2, min(self.k_large, n - 2))
        k_max = k2 + 1

        self._full_lr = None
        t0 = time.perf_counter()
        D_all, I_all = self._compute_neighbors(X, k_max)
        timings["neighbours"] = time.perf_counter() - t0
        self._last_neighbors = (I_all, D_all)

        steps = [
            ("A", lambda: self._block_a(X, y, rpf, k1, k2, D_all, I_all)),
            ("B", lambda: self._block_b(X, y, rpf)),
            ("C", lambda: self._block_c(X, y, rpf)),
            ("D", lambda: self._block_d_oof(X, y, rpf)),
            ("E", lambda: self._block_e(X, y, rpf, k2, D_all, I_all)),
            ("F", lambda: self._block_f(X, y, rpf, D_all, I_all, y_cont=y_cont)),
            ("G", lambda: self._block_g(X, y, rpf, k2, D_all, I_all)),
        ]
        for name, step in steps:
            t0 = time.perf_counter()
            step()
            timings[name] = time.perf_counter() - t0

        if self.feature_version == 1:
            mu = rpf.mean(0, keepdims=True)
            std = rpf.std(0, keepdims=True) + 1e-8
            rpf = (rpf - mu) / std
        else:
            t0 = time.perf_counter()
            h = self._block_h(X, y)
            timings["H"] = time.perf_counter() - t0
            rpf = np.column_stack([rpf[:, self._keep], h]).astype(np.float32)
            # Median/IQR scaling: mean and std are pulled by the poison itself.
            # Columns with zero IQR (rare indicators) fall back to the std.
            med = np.median(rpf, 0, keepdims=True)
            q75, q25 = np.percentile(rpf, [75, 25], axis=0, keepdims=True)
            scale = (q75 - q25) / 1.349
            scale = np.where(scale > 1e-8, scale, rpf.std(0, keepdims=True)) + 1e-8
            rpf = (rpf - med) / scale

        self._full_lr = None
        self.last_timings = timings
        return rpf.astype(np.float32)

    def _block_a(self, X, y, rpf, k1, k2, D_all, I_all):
        for slot, k in enumerate([k1, k2]):
            D = D_all[:, :k]
            I = I_all[:, :k]
            NL = y[I]
            same = (NL == y[:, np.newaxis])

            frac_same = same.mean(1)
            d_same = np.where(same, D, np.nan)
            d_diff = np.where(~same, D, np.nan)
            max_d = D.max() + 1.0
            ms = np.nanmean(d_same, 1)
            ms = np.where(np.isnan(ms), max_d, ms)
            md = np.nanmean(d_diff, 1)
            md = np.where(np.isnan(md), max_d * 2, md)
            dist_ratio = np.log1p(ms) - np.log1p(md + 1e-8)

            p = np.clip(frac_same, 1e-8, 1 - 1e-8)
            entropy = -(p * np.log(p) + (1-p) * np.log(1-p))

            near_same = np.where(same, D, np.inf).min(1)
            near_diff = np.where(~same, D, np.inf).min(1)
            near_same = np.where(np.isinf(near_same), max_d, near_same)
            near_diff = np.where(np.isinf(near_diff), max_d * 2, near_diff)
            near_ratio = np.log1p(near_diff / (near_same + 1e-8))

            b = slot * 4
            rpf[:, b + 0] = frac_same
            rpf[:, b + 1] = dist_ratio
            rpf[:, b + 2] = entropy
            rpf[:, b + 3] = near_ratio

    def _block_b(self, X, y, rpf):
        n = len(X)
        classes = np.unique(y)
        K = len(classes)
        if K < 2:
            return

        mus = np.zeros((K, X.shape[1]), np.float32)
        stds = np.ones( (K, X.shape[1]), np.float32)
        valid = np.ones(K, bool)
        for ki, c in enumerate(classes):
            idx = np.where(y == c)[0]
            if len(idx) < 2: 
                valid[ki] = False
                continue
            mus[ki] = X[idx].mean(0)
            stds[ki] = X[idx].std(0) + 1e-8

        c2ki = {c: ki for ki, c in enumerate(classes)}
        own_ki = np.array([c2ki[yi] for yi in y])
        own_mu = mus[own_ki]
        own_st = stds[own_ki]

        m_own = np.sqrt(((X - own_mu) / own_st) ** 2).mean(1).astype(np.float32)

        pca = PCA(n_components=min(5, X.shape[1]), random_state=self.random_state)
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
            if not valid[ki]: 
                mahal_all[:, ki] = 1e9
                continue
            mahal_all[:, ki] = np.sqrt(
                ((X - mus[ki]) / stds[ki]) ** 2).mean(1)

        same_mask = (own_ki[:, None] == np.arange(K)[None, :])
        mahal_other = np.where(same_mask, np.inf, mahal_all)
        near_ki = np.argmin(mahal_other, axis=1)
        m_other = mahal_other[np.arange(n), near_ki].astype(np.float32)
        m_other = np.where(np.isinf(m_other), m_own * 2, m_other)
        near_mu = mus[near_ki]

        line = near_mu - own_mu
        line_len = np.linalg.norm(line, axis=1, keepdims=True) + 1e-8
        proj = ((X - own_mu) * (line / line_len)).sum(1).astype(np.float32)

        pct_own = np.zeros(n, np.float32)
        for ki, c in enumerate(classes):
            idx = np.where(y == c)[0]
            if len(idx) == 0: continue
            pct_own[idx] = rankdata(m_own[idx]).astype(np.float32) / len(idx)

        norm_x = np.linalg.norm(X, axis=1) + 1e-8
        norm_m = np.linalg.norm(own_mu, axis=1) + 1e-8
        cos_own = (X * own_mu).sum(1) / (norm_x * norm_m)

        mid = (own_mu + near_mu) / 2
        bd = ((X - mid) * (line / line_len)).sum(1).astype(np.float32)

        rpf[:, 8] = m_own
        rpf[:, 9] = m_other
        rpf[:, 10] = np.log1p(m_own + 1e-8) - np.log1p(m_other + 1e-8)
        rpf[:, 11] = proj / (line_len.squeeze() + 1e-8)
        rpf[:, 12] = pct_own
        rpf[:, 13] = m_other / (m_own + 1e-8)
        rpf[:, 14] = cos_own.astype(np.float32)
        rpf[:, 15] = bd
        rpf[:, 16] = pca_dist

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
            mu_n = l2[idx].mean()
            sd_n = l2[idx].std() + 1e-8
            l2_z[idx] = (l2[idx] - mu_n) / sd_n

        # Vectorised; gives the same values as the former per-row loops.
        cls_idx = np.searchsorted(classes, y)
        fallback = X.mean(0)
        mus_c = np.array([X[y == c].mean(0) if (y == c).sum() >= 2 else fallback
                          for c in classes], np.float32)
        own_mu = mus_c[cls_idx]
        sign_agr = (np.sign(X) == np.sign(own_mu)).mean(1).astype(np.float32)

        mean_l2 = np.array([l2[y == c].mean() + 1e-8 for c in classes])
        rel_scale = (l2 / mean_l2[cls_idx]).astype(np.float32)

        mad = np.zeros(n, np.float32)
        for c in classes:
            idx = np.where(y == c)[0]
            if len(idx) < 2: continue
            med = np.median(X[idx], axis=0)
            abs_dev = np.abs(X[idx] - med)
            nd = max(1, int(X.shape[1] * 0.10))
            trimmed = np.sort(abs_dev, axis=1)[:, nd: -nd if nd else None]
            mad[idx] = (trimmed.mean(1) if trimmed.shape[1] > 0
                        else abs_dev.mean(1))

        rpf[:, 17] = l2_z
        rpf[:, 18] = sign_agr
        rpf[:, 19] = rel_scale
        rpf[:, 20] = mad

    def _block_d_oof(self, X: np.ndarray, y: np.ndarray, rpf: np.ndarray):
        n = len(X)
        classes = np.unique(y)
        K = len(classes)
        v1 = self.feature_version == 1
        rs = self.random_state
        # v1 counts with bincount, which reports 0 for absent labels below the
        # maximum (e.g. labels {1, 2}); v2 counts only the classes present.
        counts = np.bincount(y.astype(int)) if v1 else np.unique(y, return_counts=True)[1]
        n_folds = min(self.cv_folds, int(min(counts)))
        n_folds = max(2, n_folds)

        plr_true = np.full(n, 1.0 / K, np.float32)
        prf_true = np.full(n, 1.0 / K, np.float32)
        pgb_true = np.full(n, 1.0 / K, np.float32)
        prf_fold = np.zeros((n, n_folds), np.float32)
        prf_tree_std = np.zeros(n, np.float32)
        prf_max_other = np.full(n, 1.0 / K, np.float32)

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rs)
        fold_idx = 0

        for train_idx, val_idx in skf.split(X, y):
            Xtr, Xvl = X[train_idx], X[val_idx]
            ytr = y[train_idx]
            yvl = y[val_idx]
            if len(np.unique(ytr)) < 2:
                fold_idx += 1
                continue

            # v1 skips the rest of a fold (and its fold counter) after any model
            # fails; v2 lets each model fail on its own.
            ok = True
            try:
                lr = LogisticRegression(C=1.0, max_iter=300,
                                        random_state=rs).fit(Xtr, ytr)
                proba_lr = lr.predict_proba(Xvl)
                lr_cls = lr.classes_.tolist()
                lr_col = np.array([lr_cls.index(c) if c in lr_cls else 0
                                        for c in yvl])
                plr_true[val_idx] = proba_lr[np.arange(len(val_idx)), lr_col]
            except Exception as e:
                print(f"[RPFExtractor] _block_d_oof logistic regression failed: {type(e).__name__}: {e}")
                ok = False

            if ok or not v1:
                try:
                    rf = RandomForestClassifier(
                        n_estimators=30, max_depth=4, min_samples_leaf=3,
                        class_weight="balanced", random_state=rs, n_jobs=N_JOBS
                    ).fit(Xtr, ytr)

                    proba_rf = rf.predict_proba(Xvl)
                    rf_cls = rf.classes_.tolist()

                    rf_col = np.array([rf_cls.index(c) if c in rf_cls else 0
                                    for c in yvl])

                    rows = np.arange(len(val_idx))
                    p_true = proba_rf[rows, rf_col]

                    prf_true[val_idx] = p_true
                    prf_fold[val_idx, fold_idx] = p_true
                    if not v1:
                        # Spread of the true-class probability across the forest's
                        # bootstrap trees: variability under training-set resampling.
                        per_tree = np.stack([t.predict_proba(Xvl)[rows, rf_col]
                                             for t in rf.estimators_])
                        prf_tree_std[val_idx] = per_tree.std(0)

                    proba_rf_cp = proba_rf.copy()
                    proba_rf_cp[rows, rf_col] = 0.0
                    prf_max_other[val_idx] = proba_rf_cp.max(1)

                except Exception as e:
                    print(f"[RPFExtractor] _block_d_oof random forest failed: {type(e).__name__}: {e}")
                    ok = False
            if ok or not v1:
                try:
                    gb = (GradientBoostingClassifier(
                        n_estimators=30, max_depth=3, learning_rate=0.1,
                        random_state=rs) if v1 else
                          HistGradientBoostingClassifier(
                        max_iter=30, max_depth=3, learning_rate=0.1, early_stopping=False,
                        min_samples_leaf=1, max_leaf_nodes=None, random_state=rs)
                          ).fit(Xtr, ytr)
                    proba_gb = gb.predict_proba(Xvl)
                    gb_cls = gb.classes_.tolist()
                    gb_col = np.array([gb_cls.index(c) if c in gb_cls else 0
                                          for c in yvl])
                    pgb_true[val_idx] = proba_gb[np.arange(len(val_idx)), gb_col]
                except Exception as e:
                    print(f"[RPFExtractor] _block_d_oof gradient boosting failed: {type(e).__name__}: {e}")
                    ok = False

            if ok or not v1:
                fold_idx += 1

        wrong_lr = (1 - plr_true).astype(np.float32)
        wrong_rf = (1 - prf_true).astype(np.float32)

        ce_lr = -np.log(np.clip(plr_true, 1e-8, 1.0)).astype(np.float32)
        ce_rf = -np.log(np.clip(prf_true, 1e-8, 1.0)).astype(np.float32)

        disagree = np.abs(plr_true - prf_true).astype(np.float32)
        margin_rf = (prf_true - prf_max_other).astype(np.float32)

        rk_lr = rankdata(wrong_lr).astype(np.float32) / n
        rk_rf = rankdata(wrong_rf).astype(np.float32) / n

        # v1's prf_fold holds one non-zero entry per row, so its std is a fixed
        # multiple of the RF probability; v2 replaces it with the tree spread.
        fold_std = (prf_fold.std(1) if v1 else prf_tree_std).astype(np.float32)

        rand_thresh = 1.0 / K
        correct_lr = (plr_true > rand_thresh).astype(np.float32)
        correct_rf = (prf_true > rand_thresh).astype(np.float32)
        both_agree = correct_lr * correct_rf

        confidence = prf_true.copy()
        flip_ind = (prf_true < rand_thresh).astype(np.float32)

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

    def _block_e(self, X, y, rpf, k2, D_all, I_all):
        n = len(X)
        k = min(k2, n - 2)
        D = D_all[:, :k]
        I = I_all[:, :k]
        NL = y[I]
        same = (NL == y[:, np.newaxis])

        try:
            iso = IsolationForest(n_estimators=100, random_state=self.random_state, n_jobs=N_JOBS)
            iso_raw = -iso.fit(X).score_samples(X)
        except Exception:
            iso_raw = np.zeros(n, np.float32)

        local_purity = same.mean(1)
        mean_knn = D.mean(1)
        nb_mean = mean_knn[I].mean(1)
        lof_approx = mean_knn / (nb_mean + 1e-8)

        d_same = np.where(same, D, np.nan)
        d_diff = np.where(~same, D, np.nan)
        max_d = D.max() + 1.0
        ms = np.nanmean(d_same, 1)
        ms = np.where(np.isnan(ms), max_d, ms)
        md = np.nanmean(d_diff, 1)
        md = np.where(np.isnan(md), max_d * 2, md)

        local_geom = D.std(1)
        iso_impurity = iso_raw * (1.0 - local_purity + 1e-3)
        rk_iso = rankdata(iso_raw).astype(np.float32) / n

        rpf[:, 34] = iso_raw
        rpf[:, 35] = local_purity
        rpf[:, 36] = lof_approx
        rpf[:, 37] = ms
        rpf[:, 38] = md
        rpf[:, 39] = local_geom
        rpf[:, 40] = iso_impurity
        rpf[:, 41] = rk_iso

        classes = np.unique(y)
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
            trim = max(1, int(D.shape[1] * 0.1))
            D_trim = np.sort(D, axis=1)[:, trim: -trim]
            nb_trim = np.sort(D[I], axis=2)[:, :, trim: -trim].mean(2).mean(1)
            lof_trim = D_trim.mean(1) / (nb_trim + 1e-8)
        else:
            lof_trim = lof_approx
        rpf[:, 43] = lof_trim

        K = len(classes)
        mus_d = np.zeros((K, X.shape[1]), np.float32)
        for ki, c in enumerate(classes):
            idx = np.where(y == c)[0]
            if len(idx) >= 2: mus_d[ki] = X[idx].mean(0)
        c2ki_d = {c: ki for ki, c in enumerate(classes)}
        own_ki = np.array([c2ki_d[yi] for yi in y])
        own_mu = mus_d[own_ki]
        dists_c = np.stack([np.linalg.norm(X - mus_d[ki], axis=1)
                             for ki in range(K)], axis=1)
        same_m = (own_ki[:, None] == np.arange(K)[None, :])
        dists_c[same_m] = np.inf
        near_ki = np.argmin(dists_c, axis=1)
        near_mu = mus_d[near_ki]
        normal = near_mu - own_mu
        nlen = np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8
        rpf[:, 44] = ((X - own_mu) * (normal / nlen)).sum(1).astype(np.float32)

        med_same = np.where(same, D, np.inf).min(1)
        med_same = np.where(np.isinf(med_same), max_d, med_same)
        rpf[:, 45] = np.where(~same, D < med_same[:, None], False).mean(1)

        try:
            sur = LogisticRegression(C=1.0, max_iter=300,
                                          random_state=self.random_state).fit(X, y)
            self._full_lr = sur
            ambiguity = (1.0 - sur.predict_proba(X).max(1)).astype(np.float32)
            amb_rank = np.zeros(n, np.float32)
            for c in classes:
                idx = np.where(y == c)[0]
                if len(idx) == 0: continue
                amb_rank[idx] = (rankdata(ambiguity[idx]).astype(np.float32)
                                  / len(idx))
            rpf[:, 46] = amb_rank
        except Exception:
            rpf[:, 46] = 0.0

    def _block_f(self, X: np.ndarray, y: np.ndarray, rpf: np.ndarray,
                 D_all: np.ndarray, I_all: np.ndarray,
                 y_cont: np.ndarray = None):
        n, d = X.shape
        v1 = self.feature_version == 1
        Xf = X.astype(np.float64)

        y_reg = y_cont.astype(np.float64) if y_cont is not None \
                else y.astype(np.float64)
        classes = np.unique(y)
        onehot = (not v1) and y_cont is None and len(classes) > 2

        try:
            if onehot:
                own = np.searchsorted(classes, y)
                Y = (y[:, None] == classes[None, :]).astype(np.float64)
                y_hat = Ridge(alpha=1.0).fit(Xf, Y).predict(Xf)[np.arange(n), own]
                resid = 1.0 - y_hat
            else:
                ridge = Ridge(alpha=1.0).fit(Xf, y_reg)
                y_hat = ridge.predict(Xf)
                resid = y_reg - y_hat
        except Exception:
            rpf[:, 47:55] = 0.0
            return

        try:
            # Ridge fits an intercept, so its hat matrix is 1/n + the leverage of
            # the centred data; v1 omits both, v2 includes them.
            Xh = Xf if v1 else Xf - Xf.mean(0)
            XtX_inv = np.linalg.inv(Xh.T @ Xh + 1.0 * np.eye(d, dtype=np.float64))
            H_diag = (Xh @ XtX_inv * Xh).sum(1) + (0.0 if v1 else 1.0 / n)
        except np.linalg.LinAlgError:
            H_diag = np.full(n, 1.0 / n, dtype=np.float64)

        rpf[:, 47] = rankdata(H_diag).astype(np.float32) / n

        mse = (resid ** 2).mean() + 1e-8
        stud_resid = resid / np.sqrt(mse * (1 - np.clip(H_diag, 0, 0.999) + 1e-8))
        rpf[:, 48] = rankdata(np.abs(stud_resid)).astype(np.float32) / n

        p = min(d, n - 1)
        cooks_d = (stud_resid ** 2 / max(p, 1)) * (H_diag / (1 - H_diag + 1e-8))
        rpf[:, 49] = rankdata(cooks_d).astype(np.float32) / n

        dffits = stud_resid * np.sqrt(H_diag / (1 - H_diag + 1e-8))
        rpf[:, 50] = rankdata(np.abs(dffits)).astype(np.float32) / n

        k = max(2, min(10, n - 2))
        I_local = I_all[:, :k]
        nb_resid = resid[I_local]
        sign_agree = (np.sign(nb_resid) == np.sign(resid[:, np.newaxis])).mean(1)
        rpf[:, 51] = sign_agree.astype(np.float32)

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

        I_flat = I.ravel()
        rev_counts = np.bincount(I_flat, minlength=n).astype(np.float32)

        rev_mean = rev_counts.mean() + 1e-8
        rpf[:, 55] = rev_counts / rev_mean

        pos_vals = np.tile(np.arange(k, dtype=np.float64), n)
        pos_sum = np.bincount(I_flat, weights=pos_vals, minlength=n)
        pos_cnt = np.bincount(I_flat, minlength=n).astype(np.float64)

        avg_position = pos_sum / (pos_cnt + 1e-8)
        embedding_depth = (1.0 - avg_position / (k + 1e-8)).astype(np.float32)

        rpf[:, 56] = embedding_depth

        # Reverse kNN graph in CSR form: row i lists, in ascending order, the
        # points that have i among their k neighbours.
        rev_adj_starts = np.concatenate(
            [[0], np.cumsum(np.bincount(I_flat, minlength=n))]).astype(np.int64)
        rev_adj_flat = (np.argsort(I_flat, kind="stable") // k).astype(np.int64)
        hop1_sizes = np.diff(rev_adj_starts).astype(np.float32)

        # Two-hop reach: |(union of rev(j) for j in rev(i)) minus i and rev(i)|,
        # computed with one sparse product instead of Python loops.
        R = sp.csr_matrix((np.ones(len(rev_adj_flat), np.int32), rev_adj_flat,
                           rev_adj_starts), shape=(n, n))
        R2 = (R @ R).tocsr()
        R2.data[:] = 1
        overlap = np.asarray(R2.multiply(R).sum(1)).ravel()
        diag2, diag1 = R2.diagonal(), R.diagonal()
        hop2_sizes = (R2.getnnz(axis=1) - overlap - diag2 + diag2 * diag1).astype(np.float32)

        if self.feature_version == 1:
            # v1 estimated hubs (> 200 reverse neighbours) from a sample of 200;
            # recomputed exactly as before so v1 output does not change.
            max_hop1_expand = 200
            rng = np.random.default_rng(self.random_state)
            for i in np.where(hop1_sizes > max_hop1_expand)[0]:
                hop1 = rev_adj_flat[rev_adj_starts[i]:rev_adj_starts[i + 1]]
                hop1_set = set(hop1)
                expand = rng.choice(hop1, max_hop1_expand, replace=False)
                hop2_set = set()
                for j in expand:
                    for m in rev_adj_flat[rev_adj_starts[j]:rev_adj_starts[j + 1]]:
                        if m != i and m not in hop1_set:
                            hop2_set.add(m)
                hop2_sizes[i] = len(hop2_set) * (len(hop1) / max_hop1_expand)

        hop2_mean = hop2_sizes.mean() + 1e-8
        rpf[:, 57] = hop2_sizes / hop2_mean

        NL = y[I]
        label_agree = (NL == y[:, np.newaxis]).mean(1).astype(np.float32)
        centrality = rev_counts / rev_mean
        rpf[:, 58] = centrality * (1.0 - label_agree + 0.01)
        rpf[:, 59] = hop2_sizes / (hop1_sizes + 1e-8)

        disp_sum = np.zeros(n, dtype=np.float64)
        disp_cnt = np.zeros(n, dtype=np.float64)
        boundary = D[:, -1]

        for p in range(k):
            targets = I[:, p]
            gap = np.maximum(boundary - D[:, p], 0.0)
            np.add.at(disp_sum, targets, gap)
            np.add.at(disp_cnt, targets, 1.0)

        displacement = (disp_sum / (disp_cnt + 1e-8)).astype(np.float32)

        rpf[:, 60] = rankdata(displacement).astype(np.float32) / n

    def _block_h(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """v2 only."""
        n = len(X)
        if len(np.unique(y)) < 2:
            return np.zeros(n, np.float32)
        try:
            # the same model block E fitted on the same rows, if it did
            lr = getattr(self, "_full_lr", None)
            if lr is None:
                lr = LogisticRegression(C=1.0, max_iter=300,
                                        random_state=self.random_state).fit(X, y)
            P = lr.predict_proba(X)
            Yh = (y[:, None] == lr.classes_[None, :]).astype(np.float64)
            Xa = np.hstack([X.astype(np.float64), np.ones((n, 1))])
            G = ((P - Yh)[:, :, None] * Xa[:, None, :]).reshape(n, -1)
            G -= G.mean(0)
            # ARPACK needs a start vector of length min(n, p); with many classes
            # and features p = K (d + 1) can exceed n (e.g. mfeat_factors).
            v0 = np.random.default_rng(self.random_state).standard_normal(min(G.shape))
            _, _, vt = svds(G, k=1, v0=v0)
            return ((G @ vt[0]) ** 2).astype(np.float32)
        except Exception as e:
            print(f"[RPFExtractor] _block_h failed: {type(e).__name__}: {e}")
            return np.zeros(n, np.float32)
