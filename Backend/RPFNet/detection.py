#detection.py
RPFNet_TRAIN_DATASETS = [
    "breast_cancer", "wine", "digits",
    "california_large", "adult", "credit_g",
    # Regression datasets for RPFNet-training
    "california_reg", "diabetes_reg", "boston_reg",
    "ionosphere",          # 34 feat, binary, small n=351
    "spambase",            # 57 feat, binary, n=4601
    "vehicle",             # 18 feat, 4 classes
    "segment",             # 19 feat, 7 classes
    "pendigits",           # 16 feat, 10 classes, large n=10992
    "satimage",            # 36 feat, 6 classes
    "waveform",            # 40 feat, 3 classes
    "glass",               # 9 feat, 6 classes, very small n=214
    "yeast",               # 8 feat, 10 classes, imbalanced
    "page_blocks",         # 10 feat, 5 classes, highly imbalanced
    "vowel",               # 10 feat, 11 classes
    "optdigits",           # 64 feat, 10 classes
    "steel_plates",        # 27 feat, 7 classes
    "mfeat_factors",       # 216 feat, 10 classes, high-dim
    "cardiotocography",    # 21 feat, 10 classes, medical
    # Additional regression for diversity
    "abalone_reg",         # 8 feat, regression, n=4177
    "energy_reg",          # 8 feat, regression, synthetic stand-in
]

EVAL_BUILTIN = [
    "breast_cancer",   # seen
    "wine",            # seen
    "digits",          # seen
    "moons",           # zero-shot
    "syn_hd",          # zero-shot
    # Regression eval
    "friedman_reg",    # zero-shot regression
    "nigerian_fraud",
]

CSV_DATASETS = [
    ("Bank Marketing", "datasets/csv/bank.csv", "deposit"),
    ("Loan Data", "datasets/csv/loan_data.csv", None),
]

ATTACKS = [
    "label_flip",
    "clean_label",
    "backdoor",
    "feat_perturb",
    "repr_inversion",
    "dist_shift",
    "boundary_flip",
    "feat_dropout",
]

# Regression-specific attacks — only applied to regression datasets
REGRESSION_ATTACKS = [
    "target_shift", "leverage_attack", "target_flip_extreme",
    "feat_perturb", "backdoor", "repr_inversion",
    "dist_shift", "feat_dropout",
]

# RATES = [0.01, 0.05, 0.10, 0.15, 0.2, 0.25]
RATES = [0.15]
ASSUMED_RATE = 0.10

RPFNet_MODEL_PATH = "./RPFNet_universal.pt"
FORCE_RETRAIN = True

RPFNet_TRAIN_ATTACKS = [
    "label_flip", "tree_aware", "grad_flip", "boundary_flip",
]

# Regression attacks added to RPFNet-training
RPFNet_TRAIN_ATTACKS_REG = [
]

RPFNet_TRAIN_RATES = [ 0.15]
RPFNet_TRAIN_SEEDS = 1
RPFNet_EPOCHS = 30
RPFNet_BATCH_SIZE = 512
RPFNet_LR = 1e-3
EVAL_SEEDS = 2

TREES_DEFAULTS = {"max_iter": 200, "max_depth": 6, "learning_rate": 0.1, "max_clean": 200}

RPF_K_SMALL = 5
RPF_K_LARGE = 15
RPF_CV_FOLDS = 3   # ← CHANGED (was 3, TODO fulfilled)

#  IMPORTS
import os, time, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.stats import rankdata
from scipy.stats import ttest_rel, wilcoxon
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import OneClassSVM
from RPFNet.Attack import apply_attack, attack_seed
from RPFNet.RPFExtractor import RPFExtractor
from RPFNet.effect_blocks import make_extractor
from RPFNet.Dataset import load_builtin, load_csv
from RPFNet.RateEstimator import estimate_contamination_rate, score_distribution_features, RateEstimatorHead
from RPFNet.Figures import generate_all_figures
try:
    from iTrim.regressiondatapoisoning.src.defenses import defense_both_trim_and_itrim
    from sklearn.linear_model import LinearRegression
    HAS_ITRIM = True
except ImportError:
    HAS_ITRIM = False

#  RPF CACHE
RPF_CACHE = {}

def get_rpf_cached(extractor, X, y, key, y_cont=None):
    """y_cont: optional continuous target for regression features."""
    if key not in RPF_CACHE:
        RPF_CACHE[key] = extractor.extract(X, y, y_cont=y_cont)
    return RPF_CACHE[key]

#Utils
def focal_loss(logits, targets, gamma: float = 2.0, pos_weight: float = 4.0, weight=None):
    pw = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pw, reduction="none")
    pt = torch.exp(-F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"))
    loss = (1 - pt) ** gamma * bce
    return loss.mean() if weight is None else (loss * weight).mean()
#  RPFNET (input_dim=61)
class RPFNetPoisonNet(nn.Module):
    def __init__(self, input_dim: int = 61,
                 hidden=(96, 192, 96), dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h),
                       nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)

def rpf_context(rpf):
    """Dataset-level profile of invariant violations."""
    return 10.0 * np.concatenate([(rpf > 3).mean(0), (rpf < -3).mean(0)]).astype(np.float32)


#  RPFNetPoisonDetector
class RPFNetPoisonDetector:

    def __init__(self, device: str = "cpu", epochs: int = 400,
                 batch_size: int = 512, lr: float = 1e-3,
                 feature_version: int = 1, drop_features=None, seed=None, context=False,
                 balance=False, scorer: str = "net", trees=None):
        """feature_version picks the fingerprint (see RPFExtractor."""
        self.device = torch.device(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.extractor = make_extractor(feature_version, RPF_K_SMALL, RPF_K_LARGE, RPF_CV_FOLDS)
        self._set_drop(drop_features)
        self.context = bool(context)
        self.balance = bool(balance)
        if scorer not in ("net", "trees"):
            raise ValueError(f"scorer must be 'net' or 'trees', got {scorer!r}")
        self.scorer = scorer
        self.trees_params = {**TREES_DEFAULTS, **(trees or {})}
        self.trees = None
        if seed is not None:
            torch.manual_seed(seed)
        self.net = RPFNetPoisonNet(input_dim=self._input_dim()).to(self.device)
        # Kept so existing checkpoints and callers still load it; it is not
        # used for scoring or for estimating the contamination rate.
        self.rate_head = RateEstimatorHead().to(self.device)
        self._fitted = False
        self._threshold = 0.5
        self.null_scores = None   # sorted clean-data scores, for p-values

    def _set_drop(self, names):
        names = list(names or [])
        unknown = [f for f in names if f not in self.extractor.feature_names]
        if unknown:
            raise ValueError(f"Unknown fingerprint features: {unknown}")
        self.drop_features = names
        self._drop_idx = np.array(
            [self.extractor.feature_names.index(f) for f in names], dtype=int)

    def _mask(self, rpf):
        if len(self._drop_idx) == 0:
            return rpf
        rpf = rpf.copy()
        rpf[:, self._drop_idx] = 0.0
        return rpf

    def _input_dim(self):
        return self.extractor.dim * (3 if self.context else 1)

    def _features(self, rpf):
        """Network input for the fingerprints of ONE dataset."""
        rpf = self._mask(rpf)
        if not self.context:
            return rpf
        c = rpf_context(rpf)
        return np.hstack([rpf, np.broadcast_to(c, (len(rpf), len(c)))]).astype(np.float32)

    def build_pool(self, datasets: dict, attacks=None, rates=None,
                   seeds: int = 3, verbose: bool = True,
                   attacks_reg=None, rpf_fn=None):
        """Poison every training dataset with every attack, rate and seed and extract fingerprints."""
        if attacks is None: attacks = RPFNet_TRAIN_ATTACKS
        if attacks_reg is None: attacks_reg = RPFNet_TRAIN_ATTACKS_REG
        if rates is None: rates = RPFNet_TRAIN_RATES
        fv = self.extractor.feature_version

        n_combos = len(datasets) * len(attacks) * len(rates) * seeds
        if verbose:
            print(f"\n[RPFNet-fit] {len(datasets)} datasets × {len(attacks)} attacks × "
                  f"{len(rates)} rates × {seeds} seeds = {n_combos} combos")

        rpf_pool, label_pool, meta = [], [], []
        t0 = time.time()
        ok = 0

        for ds_name, ds_val in datasets.items():
            # Unpack: support both (X, y) and (X, y, y_cont) tuples
            if len(ds_val) == 3:
                X, y, y_cont = ds_val
            else:
                X, y = ds_val
                y_cont = None

            ds_attacks = attacks_reg if y_cont is not None else attacks

            for atk in ds_attacks:
                for rate in rates:
                    for seed in range(seeds):
                        try:
                            Xp, yp, pidx, y_cont_p = apply_attack(
                                atk, X, y, fraction=rate,
                                seed=attack_seed(atk, seed),
                                y_cont=y_cont, return_y_cont=True)
                            if len(pidx) < 3: continue
                            ytrue = np.zeros(len(Xp), np.float32)
                            ytrue[pidx] = 1.0
                            key = ("RPFNet", fv, ds_name, atk, rate, seed)
                            rpf = (rpf_fn(key, Xp, yp, y_cont_p) if rpf_fn else
                                   get_rpf_cached(self.extractor, Xp, yp, key,
                                                  y_cont=y_cont_p))
                            rpf_pool.append(rpf)
                            label_pool.append(ytrue)
                            meta.append((ds_name, atk, rate, seed, y_cont is not None))
                            ok += 1
                        except Exception as e:
                            if verbose and "Unknown attack" in str(e):
                                pass  # skip silently
                            elif verbose:
                                print(f"[RPFNet-FIT FAIL] {ds_name}|{atk}|{rate}|{seed}: {e}")

        if not rpf_pool:
            raise RuntimeError("No RPF batches generated.")

        X_pool = np.concatenate(rpf_pool, axis=0).astype(np.float32)
        y_pool = np.concatenate(label_pool, axis=0).astype(np.float32)
        self._pool_groups = [len(r) for r in rpf_pool]   # rows per poisoned dataset
        self._pool_meta = meta                             # (dataset, attack, rate, seed, regression)
        if verbose:
            print(f"  → {ok} combos  |  "
                  f"pool: {len(X_pool):,} samples  |  "
                  f"poison frac: {y_pool.mean():.3f}  |  {time.time() - t0:.1f}s")
        return X_pool, y_pool

    def _train_trees(self, X_pool, y_pool, groups=None):
        """Gradient-boosted trees on the fingerprint (scorer "trees")."""
        from sklearn.ensemble import HistGradientBoostingClassifier
        groups = groups if groups is not None else getattr(self, "_pool_groups", None)
        if groups is None or sum(groups) != len(X_pool):
            raise ValueError("the tree scorer needs the rows per dataset of the pool (from build_pool)")
        p = self.trees_params
        seed = 0 if self.seed is None else self.seed
        rng = np.random.default_rng(seed)
        bounds = np.cumsum([0] + list(groups))
        idx, w = [], []
        for a, b in zip(bounds[:-1], bounds[1:]):
            yy = y_pool[a:b] > 0.5
            pos, neg = np.where(yy)[0], np.where(~yy)[0]
            neg = rng.choice(neg, min(p["max_clean"], len(neg)), replace=False)
            for part in (pos, neg):
                if len(part):
                    idx.append(a + part)
                    w.append(np.full(len(part), 0.5 / len(part)))
        idx, w = np.concatenate(idx), np.concatenate(w)
        self.trees = HistGradientBoostingClassifier(
            max_iter=p["max_iter"], max_depth=p["max_depth"], learning_rate=p["learning_rate"],
            random_state=seed).fit(self._mask(X_pool[idx]), y_pool[idx] > 0.5,
                                   sample_weight=w * len(w) / w.sum())
        self._fitted = True
        self._best_loss = float("nan")
        return []

    def train_net(self, X_pool: np.ndarray, y_pool: np.ndarray, verbose: bool = True, groups=None):
        """Train the scorer on a fingerprint pool."""
        if self.scorer == "trees":
            return self._train_trees(X_pool, y_pool, groups)
        if verbose:
            print(f"  Phase 2 — Training RPFNetPoisonNet ({self.epochs} epochs)...")
        if self.seed is not None:
            torch.manual_seed(self.seed)
        X_pool = self._mask(X_pool)
        if self.context or self.balance:
            groups = groups if groups is not None else getattr(self, "_pool_groups", None)
            if groups is None or sum(groups) != len(X_pool):
                raise ValueError("context and balanced training need the rows per dataset "
                                 "of the pool (from build_pool)")
            gid = np.repeat(np.arange(len(groups)), groups)
        if self.context:
            bounds = np.cumsum([0] + list(groups))
            ctx = np.stack([rpf_context(X_pool[a:b]) for a, b in zip(bounds[:-1], bounds[1:])])
        if self.balance:   # mean weight 1; every poisoned dataset contributes equally
            w = (len(X_pool) / len(groups)) / np.asarray(groups, dtype=np.float64)[gid]

        opt = torch.optim.AdamW(self.net.parameters(),
                                lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=self.epochs, eta_min=self.lr * 0.01)
        rng = np.random.default_rng(42)
        n = len(X_pool)
        best = float("inf")
        hist = []

        dev = None   # GPU copies of the pool, when it fits
        if self.device.type == "cuda":
            try:
                dev = {"X": torch.as_tensor(X_pool, device=self.device),
                       "y": torch.as_tensor(y_pool, device=self.device)}
                if self.context:
                    dev["ctx"] = torch.as_tensor(ctx, device=self.device)
                    dev["gid"] = torch.as_tensor(gid, device=self.device)
                if self.balance:
                    dev["w"] = torch.as_tensor(w, dtype=torch.float32, device=self.device)
            except RuntimeError:   # not enough GPU memory: cut batches on the host as before
                dev = None
                torch.cuda.empty_cache()

        self.net.train()
        for ep in range(self.epochs):
            perm = rng.permutation(n)
            pt = torch.as_tensor(perm, device=self.device) if dev is not None else None
            ep_sum, ep_n = 0.0, 0
            for i in range(0, n, self.batch_size):
                idx = perm[i: i + self.batch_size]
                if len(idx) < 2: continue
                if dev is not None:
                    it = pt[i: i + self.batch_size]
                    xb = dev["X"][it] if not self.context else torch.cat([dev["X"][it], dev["ctx"][dev["gid"][it]]], 1)
                    yb = dev["y"][it]
                else:
                    xb = X_pool[idx] if not self.context else np.hstack([X_pool[idx], ctx[gid[idx]]])
                    xb = torch.tensor(xb, device=self.device)
                    yb = torch.tensor(y_pool[idx], device=self.device)
                logits = self.net(xb)
                if not self.balance:
                    wb = None
                elif dev is not None:
                    wb = dev["w"][it]
                else:
                    wb = torch.tensor(w[idx], dtype=logits.dtype, device=self.device)
                loss = focal_loss(logits, yb, weight=wb)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                ep_sum = ep_sum + loss.detach().double()   # summed on the device: no sync per step
                ep_n += 1
            sched.step()
            ml = float(ep_sum / ep_n) if ep_n else 0.
            hist.append(ml)
            if ml < best: best = ml
            if verbose and (ep + 1) % 80 == 0:
                print(f"  [ep {ep+1:>4}/{self.epochs}]  "
                      f"loss={ml:.4f}  best={best:.4f}")

        self._fitted = True
        self._best_loss = best
        return hist

    def reset_net(self):
        """Fresh, re-seeded network (e.g. before retraining on a larger pool)."""
        if self.seed is not None:
            torch.manual_seed(self.seed)
        self.net = RPFNetPoisonNet(input_dim=self._input_dim()).to(self.device)
        self.trees = None

    def adversarial_pool(self, datasets: dict, rates, iters=10, lam=1.0, base="label_flip",
                         seed=0, max_n=3000, rpf_fn=None):
        """Extra training data against adaptive attackers."""
        from RPFNet.Attack import evasive_attack
        fv = self.extractor.feature_version
        rpfs, labels = [], []
        for name, val in datasets.items():
            if len(val) == 3 or len(val[0]) > max_n:          # regression, or too large
                continue
            X, y = val[0], val[1]
            for rate in rates:
                Xp, yp, pidx, _ = evasive_attack(
                    X, y, rate, score_fn=lambda Xc, yc: self.score(Xc, yc)[0],
                    base=base, lam=lam, iters=iters, seed=seed)
                if len(pidx) < 3:
                    continue
                key = ("adversarial", fv, name, base, rate, seed)
                rpfs.append(rpf_fn(key, Xp, yp, None) if rpf_fn else self.extractor.extract(Xp, yp))
                t = np.zeros(len(Xp), np.float32)
                t[pidx] = 1.0
                labels.append(t)
        if not rpfs:
            return None
        return (np.concatenate(rpfs).astype(np.float32), np.concatenate(labels),
                [len(r) for r in rpfs])

    def RPFNet_fit(self, datasets: dict, attacks=None, rates=None,
                 seeds: int = 3, verbose: bool = True,
                 attacks_reg=None, rpf_fn=None):
        """Build the fingerprint pool, train the net, then calibrate."""
        X_pool, y_pool = self.build_pool(datasets, attacks, rates, seeds, verbose,
                                         attacks_reg=attacks_reg, rpf_fn=rpf_fn)
        hist = self.train_net(X_pool, y_pool, verbose)

        if verbose:
            print("  Phase 3 — Calibrating threshold and clean-score null...")
        self._calibrate(datasets, rpf_fn=rpf_fn)

        if verbose:
            n_params = sum(p.numel() for p in self.net.parameters())
            print(f"\n  ✓ Done.  best_loss={self._best_loss:.4f}  "
                  f"threshold={self._threshold:.4f}  params={n_params:,}")
        return hist

    def _calibrate(self, datasets: dict, rpf_fn=None):
        """Score the unpoisoned training datasets. Their 95th percentile is the
        fallback threshold; their distribution is kept as a null for p-values."""
        all_s = []
        fv = self.extractor.feature_version
        for ds_name, ds_val in datasets.items():
            try:
                X, y = ds_val[0], ds_val[1]
                y_cont = ds_val[2] if len(ds_val) == 3 else None
                rpf = (rpf_fn(("clean", fv, ds_name), X, y, y_cont) if rpf_fn else
                       self.extractor.extract(X, y, y_cont=y_cont))
                all_s.append(self._score_rpf(rpf))
            except Exception as e:
                print(f"[RPFNet-CALIBRATE] skipped dataset due to error {type(e).__name__}: {e}")
                continue
        if all_s:
            s = np.concatenate(all_s)
            self._threshold = float(np.percentile(s, 95))
            self.null_scores = np.sort(
                np.quantile(s, np.linspace(0, 1, min(len(s), 20001)))).astype(np.float32)

    def score(self, X: np.ndarray, y: np.ndarray, y_cont=None):
        rpf = self.extractor.extract(X, y, y_cont=y_cont)
        return self._score_rpf(rpf), rpf

    def _score_rpf(self, rpf: np.ndarray) -> np.ndarray:
        if self.scorer == "trees":
            if self.trees is None:
                raise RuntimeError("the tree scorer has not been trained")
            return self.trees.predict_proba(self._mask(rpf))[:, 1].astype(np.float32)
        self.net.eval()
        with torch.no_grad():
            t = torch.tensor(self._features(rpf), dtype=torch.float32, device=self.device)
            s = torch.sigmoid(self.net(t)).cpu().numpy()
        return s.astype(np.float32)

    def predict(self, X, y, threshold=None, y_cont=None):
        scores, _ = self.score(X, y, y_cont=y_cont)
        if threshold is None:
            return (rankdata(scores) / len(scores) > 0.85).astype(int)
        return (scores > threshold).astype(int)

    def predict_adaptive(self, X, y, assumed_rate=ASSUMED_RATE, y_cont=None):
        scores, _ = self.score(X, y, y_cont=y_cont)
        est_rate = estimate_contamination_rate(scores)
        n = len(scores)
        k = max(1, int(n * est_rate))
        pred = np.zeros(n, dtype=int)
        pred[np.argsort(scores)[-k:]] = 1
        return pred, scores

    def predict_topk(self, X, y, rate, y_cont=None):
        scores, _ = self.score(X, y, y_cont=y_cont)
        k = max(1, int(len(scores) * rate))
        pred = np.zeros(len(scores), dtype=int)
        pred[np.argsort(scores)[-k:]] = 1
        return pred, scores

    def feature_importance(self, X, y, y_cont=None):
        if self.scorer != "net":
            raise NotImplementedError("feature_importance uses the network's input gradients")
        rpf = self._features(self.extractor.extract(X, y, y_cont=y_cont))
        t = torch.tensor(rpf, dtype=torch.float32,
                            device=self.device, requires_grad=True)
        self.net.eval()
        self.net(t).sum().backward()
        return t.grad.abs().mean(0).cpu().numpy()[:self.extractor.dim]

    def ablate_block(self, rpf, block):
        _, sl = self.extractor.block_names[block]
        r = rpf.copy()
        r[:, sl] = 0.0
        return r

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "net": self.net.state_dict(),
            "threshold": self._threshold,
            "fitted": self._fitted,
            "k_small": self.extractor.k_small,
            "k_large": self.extractor.k_large,
            "cv_folds": self.extractor.cv_folds,
            "rpf_dim": self.extractor.dim,
            "feature_version": self.extractor.feature_version,
            "drop_features": list(self.drop_features),
            "context": self.context,
            "balance": self.balance,
            "null_scores": (None if self.null_scores is None
                            else torch.tensor(self.null_scores)),
            "version": 5,
            "rate_head": self.rate_head.state_dict(),
            "scorer": self.scorer,
            "trees_params": dict(self.trees_params),
        }, path)
        if self.scorer == "trees":
            import joblib
            joblib.dump(self.trees, path + ".trees")
        n = sum(p.numel() for p in self.net.parameters())
        print(f"\n[save] RPFNetPoison (features v{self.extractor.feature_version}) → {path}  params={n:,}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        # Checkpoints saved before feature versions existed are version 1.
        extractor = make_extractor(ckpt.get("feature_version", 1),
                                   ckpt.get("k_small", RPF_K_SMALL),
                                   ckpt.get("k_large", RPF_K_LARGE),
                                   ckpt.get("cv_folds", RPF_CV_FOLDS))
        dim = ckpt.get("rpf_dim", extractor.dim)
        if dim != extractor.dim:
            raise ValueError(
                f"Saved model has RPF dim={dim} but current code expects "
                f"{extractor.dim}. Set FORCE_RETRAIN=True to retrain.")
        self.extractor = extractor
        self._set_drop(ckpt.get("drop_features", []))
        self.context = bool(ckpt.get("context", False))
        self.balance = bool(ckpt.get("balance", False))
        self.scorer = ckpt.get("scorer", "net")
        self.trees_params = {**TREES_DEFAULTS, **ckpt.get("trees_params", {})}
        self.trees = None
        if self.scorer == "trees":
            import joblib
            self.trees = joblib.load(path + ".trees")
        self.net = RPFNetPoisonNet(input_dim=self._input_dim()).to(self.device)
        self.net.load_state_dict(ckpt["net"])
        self._threshold = ckpt.get("threshold", 0.5)
        self._fitted = ckpt.get("fitted", True)
        ns = ckpt.get("null_scores")    # map_location may have put it on the GPU
        self.null_scores = None if ns is None else np.asarray(
            ns.cpu() if torch.is_tensor(ns) else ns, dtype=np.float32)
        if "rate_head" in ckpt:
            self.rate_head = RateEstimatorHead().to(self.device)
            self.rate_head.load_state_dict(ckpt["rate_head"])
            self.rate_head.eval()
        self.net.eval()
        v = ckpt.get("version", 1)
        print(f"\n[load] RPFNetPoison v{v} ← {path}  "
              f"threshold={self._threshold:.4f}  rpf_dim={dim}")
class HybridEnsembleDetector:

    def __init__(self, RPFNet: RPFNetPoisonDetector, fusion_w: float = 0.55):
        self.RPFNet = RPFNet
        self.fusion_w = fusion_w
        self.rate_head = None

    def _iso_scores(self, X):
        iso = IsolationForest(n_estimators=100, random_state=42, n_jobs=-1)
        return (-iso.fit(X).score_samples(X)).astype(np.float32)

    def _rank_fuse(self, RPFNet_s, iso_s, w=None):
        if w is None: w = self.fusion_w
        mr = rankdata(RPFNet_s).astype(np.float32) / len(RPFNet_s)
        ir = rankdata(iso_s).astype(np.float32) / len(iso_s)
        return w * mr + (1 - w) * ir

    def score(self, X, y, y_cont=None):
        RPFNet_s, rpf = self.RPFNet.score(X, y, y_cont=y_cont)
        iso_s = self._iso_scores(X)
        combined = self._rank_fuse(RPFNet_s, iso_s)
        return combined, RPFNet_s, iso_s

    def _estimate_rate(self, scores):
        """Estimate the contamination rate from RPFNet's raw probability scores."""
        return estimate_contamination_rate(np.asarray(scores, dtype=np.float64))

    def attach_rate_head(self, rate_head):
        self.rate_head = rate_head
        self.rate_head.to(self.RPFNet.device)
        self.rate_head.eval()

    def predict_adaptive(self, X, y, assumed_rate=ASSUMED_RATE, y_cont=None):
        # Rank fusion decides the ORDER of points; the rate comes from RPFNet's
        # calibrated scores (the fused ranks are uniform and carry no scale).
        combined, RPFNet_s, _ = self.score(X, y, y_cont=y_cont)
        n = len(combined)
        est_rate = self._estimate_rate(RPFNet_s)

        k = max(1, int(n * float(np.clip(est_rate, 0.001, 0.40))))
        pred = np.zeros(n, dtype=int)
        pred[np.argsort(combined)[-k:]] = 1
        return pred, combined

    def predict_topk(self, X, y, rate, y_cont=None):
        combined, _, _ = self.score(X, y, y_cont=y_cont)
        k = max(1, int(len(combined) * rate))
        pred = np.zeros(len(combined), dtype=int)
        pred[np.argsort(combined)[-k:]] = 1
        return pred, combined

    def calibrate_fusion_weight(self, datasets, attacks=None, rates=None,
                                 seeds=2, verbose=True):
        if attacks is None:
            attacks = ["label_flip", "backdoor", "feat_perturb", "repr_inversion",
                       "dist_shift", "clean_label", "boundary_flip", "feat_dropout"]
        if rates is None: rates = [0.05, 0.10, 0.20]

        if verbose:
            print(f"\n  [fusion-cal] Grid search over w ∈ [0.30, 0.75] ...")

        RPFNet_s_list, iso_s_list, y_list = [], [], []

        for ds_name, ds_val in datasets.items():
            if len(ds_val) == 3:
                X, y, y_cont = ds_val
            else:
                X, y = ds_val
                y_cont = None

            is_reg = y_cont is not None
            ds_attacks = (["feat_perturb", "backdoor", "repr_inversion",
                           "dist_shift", "feat_dropout"]
                          if is_reg else attacks)

            for atk in ds_attacks:
                for rate in rates:
                    for seed in range(seeds):
                        try:
                            Xp, yp, pidx, y_cont_p = apply_attack(
                                atk, X, y, fraction=rate,
                                seed=attack_seed(atk, seed),
                                y_cont=y_cont, return_y_cont=True)
                            if len(pidx) < 3: continue
                            ytrue = np.zeros(len(Xp), np.float32)
                            ytrue[pidx] = 1.0
                            cache_key = ("fusion", self.RPFNet.extractor.feature_version,
                                         ds_name, atk, rate, seed)
                            rpf = get_rpf_cached(self.RPFNet.extractor, Xp, yp,
                                                  cache_key, y_cont=y_cont_p)
                            ms = self.RPFNet._score_rpf(rpf)
                            iso_s = self._iso_scores(Xp)
                            RPFNet_s_list.append(ms)
                            iso_s_list.append(iso_s)
                            y_list.append(ytrue)
                        except Exception as e:
                            if verbose:
                                print(
                                    f"[fusion-cal] skip {ds_name}|{atk}|{rate}|{seed} "
                                    f"due to {type(e).__name__}: {e}"
                                )
                            continue

        if not RPFNet_s_list:
            if verbose: print("  [fusion-cal] No data — keeping default w=0.55")
            return

        best_w, best_f1 = self.fusion_w, 0.0
        grid = np.arange(0.30, 0.76, 0.05)

        for w in grid:
            f1s = []
            for ms, iso_s, yt in zip(RPFNet_s_list, iso_s_list, y_list):
                combined = self._rank_fuse(ms, iso_s, w)
                k = max(1, int(len(combined) * 0.10))
                pred = np.zeros(len(combined), dtype=int)
                pred[np.argsort(combined)[-k:]] = 1
                f1s.append(f1_score(yt, pred, zero_division=0))
            avg = float(np.mean(f1s))
            if avg > best_f1:
                best_f1, best_w = avg, float(w)

        self.fusion_w = best_w
        if verbose:
            print(f"  [fusion-cal] Best w={best_w:.2f}  val_F1={best_f1:.4f}")

    def save(self, path):
        self.RPFNet.save(path.replace(".pt", "_base.pt"))
        payload = {"fusion_w": self.fusion_w}
        if self.rate_head is not None:
            payload["rate_head"] = self.rate_head.state_dict()
        torch.save(payload, path)
        print(f"  [fusion] w={self.fusion_w:.2f} → {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.RPFNet.device)
        self.fusion_w = ckpt.get("fusion_w", 0.55)
        if "rate_head" in ckpt:
            rh = RateEstimatorHead()
            rh.load_state_dict(ckpt["rate_head"])
            self.attach_rate_head(rh)
