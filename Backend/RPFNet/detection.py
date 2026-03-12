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

# RATES        = [0.01, 0.05, 0.10, 0.15, 0.2, 0.25]
RATES        = [0.15]
ASSUMED_RATE = 0.10

RPFNet_MODEL_PATH = "./RPFNet_universal.pt"
FORCE_RETRAIN   = True

RPFNet_TRAIN_ATTACKS = [
    "label_flip", "tree_aware", "grad_flip", "boundary_flip",
    # "gauss_noise", "interpolation", "targeted_class",
    # "clean_label", "backdoor", "null_feature", "feat_perturb",
    # "backdoor_heavy", "repr_inversion", "dist_shift",
    # "outlier_inject", "feat_dropout",
]

# Regression attacks added to RPFNet-training
RPFNet_TRAIN_ATTACKS_REG = [
    # "target_shift", "leverage_attack", "target_flip_extreme",
    # "feat_perturb", "backdoor", "repr_inversion",
    # "dist_shift", "feat_dropout", "outlier_inject",
    # "gauss_noise",
]

RPFNet_TRAIN_RATES  = [ 0.15]
RPFNet_TRAIN_SEEDS  = 1
RPFNet_EPOCHS       = 30
RPFNet_BATCH_SIZE   = 512
RPFNet_LR           = 1e-3
EVAL_SEEDS = 2

RPF_K_SMALL  = 5
RPF_K_LARGE  = 15
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
from RPFNet.Attack import apply_attack
from RPFNet.RPFExtractor import RPFExtractor
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

#  ATTACK TAXONOMY — UPDATED with regression attacks

ATTACK_RPFNet = {
    "label_flip"     : ("A. Label flip",           True),
    "tree_aware"     : ("B. Tree-aware + flip",     True),
    "grad_flip"      : ("E. Gradient + flip",       True),
    "boundary_flip"  : ("F. Boundary flip",         True),
    "gauss_noise"    : ("G. Gauss noise + flip",    True),
    "interpolation"  : ("H. Interpolation",         True),
    "targeted_class" : ("J. Targeted class",        True),
    "clean_label"    : ("C. Clean-label",           False),
    "backdoor"       : ("D. Backdoor 4σ",           False),
    "null_feature"   : ("I. Null feature",          False),
    "feat_perturb"   : ("K. Feat perturb ±4σ",      False),
    "backdoor_heavy" : ("L. Backdoor heavy 10σ",    False),
    "repr_inversion" : ("M. Repr inversion −x",     False),
    "dist_shift"     : ("N. Distribution shift",    False),
    "outlier_inject" : ("O. Outlier injection",     False),
    "feat_dropout"   : ("P. Feat dropout 30%",      False),
    # Regression-specific attacks
    "target_shift"        : ("T. Target shift ±3σ",     False),
    "leverage_attack"     : ("U. Leverage attack",       False),
    "target_flip_extreme" : ("V. Target flip extreme",   True),
    # Composite attacks — training only
    "combo_flip_perturb" : ("Q. Combo flip+perturb",  True),
    "combo_flip_noise"   : ("R. Combo flip+noise",    True),
    "adaptive_blend"     : ("S. Adaptive blend",      True),
}
#Utils

def focal_loss(logits, targets, gamma: float = 2.0, pos_weight: float = 4.0):
    pw  = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pw, reduction="none")
    pt   = torch.exp(-F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"))
    return ((1 - pt) ** gamma * bce).mean()
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

#  RPFNetPoisonDetector
class RPFNetPoisonDetector:

    def __init__(self, device: str = "cpu", epochs: int = 400,
                 batch_size: int = 512, lr: float = 1e-3):
        self.device     = torch.device(device)
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.extractor  = RPFExtractor(k_small=RPF_K_SMALL,
                                        k_large=RPF_K_LARGE,
                                        cv_folds=RPF_CV_FOLDS)
        self.net        = RPFNetPoisonNet(input_dim=RPFExtractor.DIM).to(self.device)
        self.rate_head  = RateEstimatorHead().to(self.device)
        self._fitted    = False
        self._threshold = 0.5

    def RPFNet_fit(self, datasets: dict, attacks=None, rates=None,
                 seeds: int = 3, verbose: bool = True):
        """
        datasets: dict of name → (X, y) OR name → (X, y, y_cont)
        """
        if attacks is None: attacks = RPFNet_TRAIN_ATTACKS
        if rates   is None: rates   = RPFNet_TRAIN_RATES

        n_combos = len(datasets) * len(attacks) * len(rates) * seeds
        if verbose:
            print(f"\n[RPFNet-fit] {len(datasets)} datasets × {len(attacks)} attacks × "
                  f"{len(rates)} rates × {seeds} seeds = {n_combos} combos")

        rpf_pool, label_pool = [], []
        t0 = time.time()
        ok = 0

        for ds_name, ds_val in datasets.items():
            # Unpack: support both (X, y) and (X, y, y_cont) tuples
            if len(ds_val) == 3:
                X, y, y_cont = ds_val
            else:
                X, y = ds_val
                y_cont = None

            # Determine which attacks to use
            is_reg = y_cont is not None
            ds_attacks = (RPFNet_TRAIN_ATTACKS_REG if is_reg
                          else attacks)

            for atk in ds_attacks:
                for rate in rates:
                    for seed in range(seeds):
                        try:
                            Xp, yp, pidx = apply_attack(
                                atk, X, y, fraction=rate,
                                seed=seed * 1000 + hash(atk) % 997,
                                y_cont=y_cont)
                            if len(pidx) < 3: continue
                            ytrue       = np.zeros(len(Xp), np.float32)
                            ytrue[pidx] = 1.0
                            cache_key = ("RPFNet", ds_name, atk, rate, seed)
                            rpf = get_rpf_cached(self.extractor, Xp, yp,
                                                  cache_key, y_cont=y_cont)
                            rpf_pool.append(rpf)
                            label_pool.append(ytrue)
                            ok += 1
                        except Exception as e:
                            if verbose and "Unknown attack" in str(e):
                                pass  # skip silently
                            elif verbose:
                                print(f"[RPFNet-FIT FAIL] {ds_name}|{atk}|{rate}|{seed}: {e}")

        if not rpf_pool:
            raise RuntimeError("No RPF batches generated.")

        X_pool = np.concatenate(rpf_pool,   axis=0).astype(np.float32)
        y_pool = np.concatenate(label_pool, axis=0).astype(np.float32)
        elapsed = time.time() - t0
        if verbose:
            print(f"  → {ok} combos  |  "
                  f"pool: {len(X_pool):,} samples  |  "
                  f"poison frac: {y_pool.mean():.3f}  |  {elapsed:.1f}s")
            print(f"  Phase 2 — Training RPFNetPoisonNet ({self.epochs} epochs)...")

        opt   = torch.optim.AdamW(self.net.parameters(),
                                   lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=self.epochs, eta_min=self.lr * 0.01)
        rng   = np.random.default_rng(42)
        n     = len(X_pool)
        best  = float("inf")
        hist  = []

        for ep in range(self.epochs):
            perm    = rng.permutation(n)
            ep_loss = []
            for i in range(0, n, self.batch_size):
                idx = perm[i: i + self.batch_size]
                if len(idx) < 2: continue
                xb     = torch.tensor(X_pool[idx], device=self.device)
                yb     = torch.tensor(y_pool[idx], device=self.device)
                logits = self.net(xb)
                loss   = focal_loss(logits, yb)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                ep_loss.append(loss.item())
            sched.step()
            ml = float(np.mean(ep_loss)) if ep_loss else 0.
            hist.append(ml)
            if ml < best: best = ml
            if verbose and (ep + 1) % 80 == 0:
                print(f"  [ep {ep+1:>4}/{self.epochs}]  "
                      f"loss={ml:.4f}  best={best:.4f}")

        self._fitted = True

        if verbose:
            print("  Phase 3a — Calibrating fallback threshold...")
        self._calibrate(datasets)

        if verbose:
            print("  Phase 3b — Training RateEstimatorHead ...")

        rate_feats, rate_targets = [], []

        for ds_name, ds_val in datasets.items():
            if len(ds_val) == 3:
                X, y, y_cont = ds_val
            else:
                X, y = ds_val
                y_cont = None

            is_reg = y_cont is not None
            ds_attacks = (RPFNet_TRAIN_ATTACKS_REG if is_reg else attacks)

            for atk in ds_attacks:
                for rate_val in rates:
                    for seed in range(seeds):
                        try:
                            Xp, yp, pidx = apply_attack(
                                atk, X, y, fraction=rate_val,
                                seed=seed * 1000 + hash(atk) % 997,
                                y_cont=y_cont)
                            if len(pidx) < 3: continue
                            cache_key = ("RPFNet", ds_name, atk, rate_val, seed)
                            rpf   = get_rpf_cached(self.extractor, Xp, yp,
                                                    cache_key, y_cont=y_cont)
                            s     = self._score_rpf(rpf)
                            s_rk  = rankdata(s).astype(np.float32) / len(s)
                            feats = score_distribution_features(s_rk)
                            rate_feats.append(feats)
                            rate_targets.append(np.float32(rate_val))
                        except Exception:
                            pass

        if len(rate_feats) >= 16:
            Xr = np.stack(rate_feats).astype(np.float32)
            yr = np.array(rate_targets, dtype=np.float32)

            rh_opt = torch.optim.Adam(self.rate_head.parameters(), lr=1e-3)
            rng_rh = np.random.default_rng(0)

            for ep in range(200):
                perm = rng_rh.permutation(len(Xr))
                for i in range(0, len(Xr), 64):
                    idx = perm[i: i + 64]
                    xb  = torch.tensor(Xr[idx], device=self.device)
                    yb  = torch.tensor(yr[idx], device=self.device)
                    loss = F.mse_loss(self.rate_head(xb), yb)
                    rh_opt.zero_grad(); loss.backward(); rh_opt.step()

            self.rate_head.eval()
            preds = self.rate_head(
                torch.tensor(Xr, device=self.device)).detach().cpu().numpy()
            mae = float(np.abs(preds - yr).mean())
            if verbose:
                print(f"  ✓ RateEstimatorHead trained  n={len(Xr)}  MAE={mae:.4f}")

        if verbose:
            n_params = sum(p.numel() for p in self.net.parameters())
            print(f"\n  ✓ Done.  best_loss={best:.4f}  "
                  f"threshold={self._threshold:.4f}  params={n_params:,}")
        return hist

    def _calibrate(self, datasets: dict):
        all_s = []
        for _, ds_val in datasets.items():
            try:
                X, y = ds_val[0], ds_val[1]
                y_cont = ds_val[2] if len(ds_val) == 3 else None
                rpf = self.extractor.extract(X, y, y_cont=y_cont)
                all_s.append(self._score_rpf(rpf))
            except Exception:
                pass
        if all_s:
            self._threshold = float(np.percentile(np.concatenate(all_s), 95))

    def score(self, X: np.ndarray, y: np.ndarray, y_cont=None):
        rpf = self.extractor.extract(X, y, y_cont=y_cont)
        return self._score_rpf(rpf), rpf

    def _score_rpf(self, rpf: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            t = torch.tensor(rpf, dtype=torch.float32, device=self.device)
            s = torch.sigmoid(self.net(t)).cpu().numpy()
        return s.astype(np.float32)

    def predict(self, X, y, threshold=None, y_cont=None):
        scores, _ = self.score(X, y, y_cont=y_cont)
        if threshold is None:
            return (rankdata(scores) / len(scores) > 0.85).astype(int)
        return (scores > threshold).astype(int)

    def predict_adaptive(self, X, y, assumed_rate=ASSUMED_RATE, y_cont=None):
        scores, _ = self.score(X, y, y_cont=y_cont)
        est_rate  = estimate_contamination_rate(scores)
        n         = len(scores)
        k         = max(1, int(n * est_rate))
        pred      = np.zeros(n, dtype=int)
        pred[np.argsort(scores)[-k:]] = 1
        return pred, scores

    def predict_topk(self, X, y, rate, y_cont=None):
        scores, _ = self.score(X, y, y_cont=y_cont)
        k    = max(1, int(len(scores) * rate))
        pred = np.zeros(len(scores), dtype=int)
        pred[np.argsort(scores)[-k:]] = 1
        return pred, scores

    def feature_importance(self, X, y, y_cont=None):
        rpf = self.extractor.extract(X, y, y_cont=y_cont)
        t   = torch.tensor(rpf, dtype=torch.float32,
                            device=self.device, requires_grad=True)
        self.net.eval()
        self.net(t).sum().backward()
        return t.grad.abs().mean(0).cpu().numpy()

    def ablate_block(self, rpf, block):
        _, sl = RPFExtractor.BLOCK_NAMES[block]
        r = rpf.copy(); r[:, sl] = 0.0
        return r

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "net":       self.net.state_dict(),
            "threshold": self._threshold,
            "fitted":    self._fitted,
            "k_small":   self.extractor.k_small,
            "k_large":   self.extractor.k_large,
            "cv_folds":  self.extractor.cv_folds,
            "rpf_dim":   RPFExtractor.DIM,
            "version":   4,
            "rate_head": self.rate_head.state_dict(),
        }, path)
        n = sum(p.numel() for p in self.net.parameters())
        print(f"\n[save] RPFNetPoison v3 → {path}  params={n:,}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        dim  = ckpt.get("rpf_dim", RPFExtractor.DIM)
        if dim != RPFExtractor.DIM:
            raise ValueError(
                f"Saved model has RPF dim={dim} but current code expects "
                f"{RPFExtractor.DIM}. Set FORCE_RETRAIN=True to retrain.")
        self.net = RPFNetPoisonNet(input_dim=dim).to(self.device)
        self.net.load_state_dict(ckpt["net"])
        self._threshold          = ckpt.get("threshold", 0.5)
        self._fitted             = ckpt.get("fitted", True)
        self.extractor.k_small   = ckpt.get("k_small", RPF_K_SMALL)
        self.extractor.k_large   = ckpt.get("k_large", RPF_K_LARGE)
        self.extractor.cv_folds  = ckpt.get("cv_folds", RPF_CV_FOLDS)
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
        self.RPFNet      = RPFNet
        self.fusion_w  = fusion_w
        self.rate_head = None

    def _iso_scores(self, X):
        iso = IsolationForest(n_estimators=100, random_state=42, n_jobs=-1)
        return (-iso.fit(X).score_samples(X)).astype(np.float32)

    def _rank_fuse(self, RPFNet_s, iso_s, w=None):
        if w is None: w = self.fusion_w
        mr = rankdata(RPFNet_s).astype(np.float32) / len(RPFNet_s)
        ir = rankdata(iso_s).astype(np.float32)  / len(iso_s)
        return w * mr + (1 - w) * ir

    def score(self, X, y, y_cont=None):
        RPFNet_s, rpf = self.RPFNet.score(X, y, y_cont=y_cont)
        iso_s       = self._iso_scores(X)
        combined    = self._rank_fuse(RPFNet_s, iso_s)
        return combined, RPFNet_s, iso_s

    def _estimate_rate(self, scores):
        ranked = rankdata(scores).astype(np.float32) / len(scores)

        if self.rate_head is not None:
            try:
                feats = score_distribution_features(ranked)
                t     = torch.tensor(feats, dtype=torch.float32,
                                     device=self.RPFNet.device).unsqueeze(0)
                with torch.no_grad():
                    rate = float(self.rate_head(t).item())
                return float(np.clip(rate, 0.01, 0.40))
            except Exception:
                pass

        return estimate_contamination_rate(ranked)

    def attach_rate_head(self, rate_head):
        self.rate_head = rate_head
        self.rate_head.to(self.RPFNet.device)
        self.rate_head.eval()

    def predict_adaptive(self, X, y, assumed_rate=ASSUMED_RATE, y_cont=None):
        combined, _, _ = self.score(X, y, y_cont=y_cont)
        n              = len(combined)
        est_rate       = self._estimate_rate(combined)

        vote_rates = np.clip(
            [est_rate * 0.50, est_rate * 0.75, est_rate,
             est_rate * 1.25, est_rate * 1.50],
            0.01, 0.40
        )

        votes = np.zeros(n, dtype=np.float32)
        for vr in vote_rates:
            k    = max(1, int(n * vr))
            mask = np.zeros(n, dtype=int)
            mask[np.argsort(combined)[-k:]] = 1
            votes += mask

        pred = (votes >= 3).astype(int)
        return pred, combined

    def predict_topk(self, X, y, rate, y_cont=None):
        combined, _, _ = self.score(X, y, y_cont=y_cont)
        k    = max(1, int(len(combined) * rate))
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
                            Xp, yp, pidx = apply_attack(
                                atk, X, y, fraction=rate,
                                seed=seed * 500 + hash(atk) % 499,
                                y_cont=y_cont)
                            if len(pidx) < 3: continue
                            ytrue       = np.zeros(len(Xp), np.float32)
                            ytrue[pidx] = 1.0
                            cache_key = ("fusion", ds_name, atk, rate, seed)
                            rpf = get_rpf_cached(self.RPFNet.extractor, Xp, yp,
                                                  cache_key, y_cont=y_cont)
                            ms      = self.RPFNet._score_rpf(rpf)
                            iso_s   = self._iso_scores(Xp)
                            RPFNet_s_list.append(ms)
                            iso_s_list.append(iso_s)
                            y_list.append(ytrue)
                        except Exception:
                            pass

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

#  BASELINES
def find_kept_mask(original, cleaned):
    mask = np.zeros(len(original), dtype=bool)
    for i, row in enumerate(original):
        mask[i] = np.any(np.all(np.isclose(cleaned, row, atol=1e-8), axis=1))
    return mask

def run_trim_itrim(Xp, yp, contamination):
    if not HAS_ITRIM:
        return None, None
    try:
        result  = defense_both_trim_and_itrim(
            x=Xp, y=yp.astype(float),
            regressor=LinearRegression,
            eps_hat=contamination)
        mask_trim  = find_kept_mask(Xp, result["defense_trim"][0])
        mask_itrim = find_kept_mask(Xp, result["defense_itrim"][0])
        tm = np.where(mask_trim, 0, 1)
        im = np.where(mask_itrim, 0, 1)
        return tm, im
    except Exception:
        return None, None


def run_lof(Xp, contamination):
    try:
        if contamination == 'auto':
            cont = 'auto'
        else:
            cont = float(np.clip(contamination, 0.001, 0.499))
        lof  = LocalOutlierFactor(n_neighbors=20, contamination=cont, n_jobs=-1)
        pred = (lof.fit_predict(Xp) == -1).astype(int)
        return pred
    except Exception:
        return None


def run_ocsvm(Xp, contamination):
    try:
        nu   = float(np.clip(contamination, 0.001, 0.499))
        n    = len(Xp)
        if n > 2000:
            rng  = np.random.default_rng(42)
            idx  = rng.choice(n, 2000, replace=False)
            Xfit = Xp[idx]
        else:
            Xfit = Xp
        ocsvm = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
        ocsvm.fit(Xfit)
        pred  = (ocsvm.predict(Xp) == -1).astype(int)
        return pred
    except Exception:
        return None

def run_spectral_signature(Xp, yp, contamination):
    try:
        n, d = Xp.shape
        classes = np.unique(yp)
        scores = np.zeros(n)

        for c in classes:
            idx = np.where(yp == c)[0]
            if len(idx) < 5:
                continue

            Xc = Xp[idx]
            Xc_centered = Xc - Xc.mean(axis=0, keepdims=True)

            U, S, Vt = np.linalg.svd(Xc_centered, full_matrices=False)
            top_v = Vt[0]

            proj = (Xc_centered @ top_v) ** 2
            scores[idx] = proj

        k = max(1, int(n * contamination))
        pred = np.zeros(n, dtype=int)
        pred[np.argsort(scores)[-k:]] = 1
        return pred
    except Exception:
        return None
    
def run_influence_baseline(Xp, yp, contamination):
    try:
        n, d = Xp.shape
        if len(np.unique(yp)) < 2:
            return None

        # Fit logistic regression
        lr = LogisticRegression(max_iter=300, random_state=42)
        lr.fit(Xp, yp)

        proba = lr.predict_proba(Xp)
        classes = lr.classes_.tolist()

        # Compute gradient per sample
        grads = []
        for i in range(n):
            yi = yp[i]
            col = classes.index(yi)
            p = proba[i, col]
            grad = (p - 1) * Xp[i]
            grads.append(grad)

        grads = np.array(grads)
        influence_score = np.linalg.norm(grads, axis=1)

        k = max(1, int(n * contamination))
        pred = np.zeros(n, dtype=int)
        pred[np.argsort(influence_score)[-k:]] = 1
        return pred
    except Exception:
        return None
        
def run_sever(Xp, yp, contamination, n_iters=5):
    try:
        n = len(Xp)
        cont = float(np.clip(contamination, 0.001, 0.499))
        n_remove_total = max(1, int(n * cont))

        # Track which original indices are still in the "clean" set
        remaining = np.arange(n)
        removed = set()

        for iteration in range(n_iters):
            if len(remaining) < 10:
                break

            X_cur = Xp[remaining]
            y_cur = yp[remaining]

            # Need at least 2 classes to fit
            if len(np.unique(y_cur)) < 2:
                break

            # Fit logistic regression on current subset
            lr = LogisticRegression(C=1.0, max_iter=300, random_state=42)
            lr.fit(X_cur, y_cur)

            # Compute per-sample gradient approximation
            # For logistic regression: grad_i = (p_i - y_i) * x_i
            proba = lr.predict_proba(X_cur)
            classes = lr.classes_.tolist()

            # Build per-sample gradient vectors
            y_onehot = np.zeros((len(X_cur), len(classes)), dtype=np.float64)
            for i, yi in enumerate(y_cur):
                col = classes.index(yi) if yi in classes else 0
                y_onehot[i, col] = 1.0

            residuals = proba - y_onehot  # (n_cur, K)
            # Use first class residual as score signal (for binary, this is sufficient)
            resid_scalar = residuals[:, 0] if residuals.shape[1] >= 1 else residuals.sum(1)
            grads = resid_scalar[:, np.newaxis] * X_cur  # (n_cur, d)

            # Center the gradients
            grads_centered = grads - grads.mean(axis=0, keepdims=True)

            # Compute top singular vector via SVD
            try:
                # Use truncated SVD for efficiency
                U, S, Vt = np.linalg.svd(grads_centered, full_matrices=False)
                top_v = Vt[0]  # top right singular vector
            except np.linalg.LinAlgError:
                break

            # Score each point by squared projection onto top singular vector
            projections = (grads_centered @ top_v) ** 2

            # Remove a fraction of points per iteration
            n_remove_iter = max(1, n_remove_total // n_iters)
            n_remove_iter = min(n_remove_iter, len(remaining) - 5)

            if n_remove_iter <= 0:
                break

            # Indices within current subset to remove
            remove_local = np.argsort(projections)[-n_remove_iter:]
            remove_global = remaining[remove_local]

            removed.update(remove_global.tolist())
            remaining = np.array([i for i in remaining if i not in removed])

        # Build prediction array
        pred = np.zeros(n, dtype=int)
        for idx in removed:
            pred[idx] = 1

        if pred.sum() < n_remove_total and len(remaining) > 0:
            X_final = Xp[remaining]
            y_final = yp[remaining]
            if len(np.unique(y_final)) >= 2:
                try:
                    lr_final = LogisticRegression(C=1.0, max_iter=300,
                                                   random_state=42)
                    lr_final.fit(X_final, y_final)
                    proba_f = lr_final.predict_proba(X_final)
                    classes_f = lr_final.classes_.tolist()
                    y_oh_f = np.zeros((len(X_final), len(classes_f)), dtype=np.float64)
                    for i, yi in enumerate(y_final):
                        col = classes_f.index(yi) if yi in classes_f else 0
                        y_oh_f[i, col] = 1.0
                    resid_f = proba_f - y_oh_f
                    rs_f = resid_f[:, 0] if resid_f.shape[1] >= 1 else resid_f.sum(1)
                    grads_f = rs_f[:, np.newaxis] * X_final
                    grads_f_c = grads_f - grads_f.mean(axis=0, keepdims=True)
                    U_f, S_f, Vt_f = np.linalg.svd(grads_f_c, full_matrices=False)
                    proj_f = (grads_f_c @ Vt_f[0]) ** 2
                    n_extra = n_remove_total - pred.sum()
                    extra_local = np.argsort(proj_f)[-n_extra:]
                    pred[remaining[extra_local]] = 1
                except Exception:
                    pass

        return pred
    except Exception:
        return None

def run_deep_knn(Xp, contamination):
    try:
        pca = PCA(n_components=min(10, Xp.shape[1]))
        Z = pca.fit_transform(Xp)

        nbrs = NearestNeighbors(n_neighbors=10).fit(Z)
        D, _ = nbrs.kneighbors(Z)
        density_score = D.mean(axis=1)

        k = max(1, int(len(Xp) * contamination))
        pred = np.zeros(len(Xp), dtype=int)
        pred[np.argsort(density_score)[-k:]] = 1
        return pred
    except Exception:
        return None

def run_autoencoder_detector(Xp, contamination, seed=42):
    try:
        n, d = Xp.shape
        if n < 20 or d < 2:
            return None

        h1 = max(8, d)
        h2 = max(4, d // 2)
        bottleneck = max(2, d // 4)

        encoder = nn.Sequential(
            nn.Linear(d, h1), nn.LayerNorm(h1), nn.GELU(),
            nn.Linear(h1, h2), nn.LayerNorm(h2), nn.GELU(),
            nn.Linear(h2, bottleneck),
        )
        decoder = nn.Sequential(
            nn.Linear(bottleneck, h2), nn.LayerNorm(h2), nn.GELU(),
            nn.Linear(h2, h1), nn.LayerNorm(h1), nn.GELU(),
            nn.Linear(h1, d),
        )

        ae = nn.Sequential(encoder, decoder)

        # Xavier init
        for m in ae.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # Training
        device = torch.device("cpu")  # baselines run on CPU for fairness
        ae = ae.to(device)
        ae.train()

        X_tensor = torch.tensor(Xp, dtype=torch.float32, device=device)
        optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-5)

        batch_size = min(256, n)
        n_epochs = 50  # enough to converge on small datasets; fast

        rng = np.random.default_rng(seed)

        for epoch in range(n_epochs):
            perm = rng.permutation(n)
            for i in range(0, n, batch_size):
                idx = perm[i: i + batch_size]
                xb = X_tensor[idx]
                xb_hat = ae(xb)
                loss = F.mse_loss(xb_hat, xb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        #  Score: per-sample reconstruction error 
        ae.eval()
        with torch.no_grad():
            X_hat = ae(X_tensor)
            recon_error = ((X_tensor - X_hat) ** 2).mean(dim=1).cpu().numpy()

        #  Predict: top-k by reconstruction error 
        k = max(1, int(n * contamination))
        pred = np.zeros(n, dtype=int)
        pred[np.argsort(recon_error)[-k:]] = 1
        return pred

    except Exception:
        return None
    
#  EVALUATION
def _m(yt, yp, s=None):
    return {"F1":  f1_score(yt, yp, zero_division=0),
            "P":   precision_score(yt, yp, zero_division=0),
            "R":   recall_score(yt, yp, zero_division=0),
            "AUC": roc_auc_score(yt, s) if s is not None else 0.0}


def run_one(RPFNet, hybrid, X_tr, y_tr, atk, rate, ds_key="", y_cont=None):
    label, has_flip = ATTACK_RPFNet[atk]
    Xp, yp, pidx    = apply_attack(atk, X_tr, y_tr, fraction=rate, y_cont=y_cont)
    ytrue           = np.zeros(len(Xp)); ytrue[pidx] = 1

    if pidx.shape[0] < 3 or len(Xp) < 30:
        return None

    cache_key = ("eval", ds_key, atk, rate)
    rpf = get_rpf_cached(RPFNet.extractor, Xp, yp, cache_key, y_cont=y_cont)
    scores_m = RPFNet._score_rpf(rpf)

    est_rate = estimate_contamination_rate(scores_m)
    k_unk = max(1, int(len(scores_m) * est_rate))
    pred_unk_m = np.zeros(len(scores_m), dtype=int)
    pred_unk_m[np.argsort(scores_m)[-k_unk:]] = 1

    k_kno = max(1, int(len(scores_m) * rate))
    pred_kno_m = np.zeros(len(scores_m), dtype=int)
    pred_kno_m[np.argsort(scores_m)[-k_kno:]] = 1

    iso_model_auto = IsolationForest(contamination='auto', random_state=42)
    iso_model_rate = IsolationForest(contamination=rate, random_state=42)

    iso_unk = (iso_model_auto.fit_predict(Xp) == -1).astype(int)
    iso_kno = (iso_model_rate.fit_predict(Xp) == -1).astype(int)

    iso_scores = -iso_model_auto.score_samples(Xp)

    mr = rankdata(scores_m) / len(scores_m)
    ir = rankdata(iso_scores) / len(iso_scores)
    scores_h = hybrid.fusion_w * mr + (1 - hybrid.fusion_w) * ir

    pred_unk_h, _ = hybrid.predict_adaptive(Xp, yp, y_cont=y_cont)

    k_kno_h = max(1, int(len(scores_h) * rate))
    pred_kno_h = np.zeros(len(scores_h), dtype=int)
    pred_kno_h[np.argsort(scores_h)[-k_kno_h:]] = 1

    lof_unk = run_lof(Xp, 'auto')
    lof_kno = run_lof(Xp, rate)

    ocsvm_unk = run_ocsvm(Xp, ASSUMED_RATE)
    ocsvm_kno = run_ocsvm(Xp, rate)

    sever_unk = run_sever(Xp, yp, ASSUMED_RATE)
    sever_kno = run_sever(Xp, yp, rate)

    infl_unk = run_influence_baseline(Xp, yp, ASSUMED_RATE)
    infl_kno = run_influence_baseline(Xp, yp, rate)

    spec_unk = run_spectral_signature(Xp, yp, ASSUMED_RATE)
    spec_kno = run_spectral_signature(Xp, yp, rate)

    deepknn_unk = run_deep_knn(Xp, ASSUMED_RATE)
    deepknn_kno = run_deep_knn(Xp, rate)

    ae_unk = run_autoencoder_detector(Xp, ASSUMED_RATE)
    ae_kno = run_autoencoder_detector(Xp, rate)

    unk = {
        "RPFNet": _m(ytrue, pred_unk_m, scores_m),
        "Hybrid":       _m(ytrue, pred_unk_h, scores_h),
        "IsoForest":    _m(ytrue, iso_unk)
    }

    kno = {
        "RPFNet": _m(ytrue, pred_kno_m, scores_m),
        "Hybrid":       _m(ytrue, pred_kno_h, scores_h),
        "IsoForest":    _m(ytrue, iso_kno)
    }

    for lbl, d_unk, d_kno in [("LOF", lof_unk, lof_kno), ("OCSVM", ocsvm_unk, ocsvm_kno),
                               ("SEVER", sever_unk, sever_kno), ("Influence", infl_unk, infl_kno), ("SpecSig", spec_unk, spec_kno),
                               ("DeepkNN", deepknn_unk, deepknn_kno), ("AE-Recon", ae_unk, ae_kno)]:
        if d_unk is not None: unk[lbl] = _m(ytrue, d_unk)
        if d_kno is not None: kno[lbl] = _m(ytrue, d_kno)

    tm_u, im_u = run_trim_itrim(Xp, yp, ASSUMED_RATE)
    tm_k, im_k = run_trim_itrim(Xp, yp, rate)
    if tm_u is not None: unk["TRIM"] = _m(ytrue, tm_u); unk["iTRIM"] = _m(ytrue, im_u)
    if tm_k is not None: kno["TRIM"] = _m(ytrue, tm_k); kno["iTRIM"] = _m(ytrue, im_k)

    return {
        "unknown": unk,
        "known": kno,
        "label": label,
        "has_flip": has_flip,
        "n_poison": int(pidx.shape[0]),
        "n_total": len(Xp),
        "ytrue": ytrue,
        "hybrid_pred_unknown": pred_unk_h
    }

#  ABLATION
def ablation_study(RPFNet, X_tr, y_tr, attacks, rate=0.10, n_trials=3, y_cont=None):
    drops = {b: [] for b in RPFExtractor.BLOCK_NAMES}
    for trial in range(n_trials):
        for atk in attacks:
            try:
                Xp, yp, pidx = apply_attack(atk, X_tr, y_tr,
                                             fraction=rate, seed=trial * 31,
                                             y_cont=y_cont)
                if len(pidx) < 3: continue
                ytrue = np.zeros(len(Xp)); ytrue[pidx] = 1
                k     = max(1, int(len(Xp) * rate))

                rpf_full  = RPFNet.extractor.extract(Xp, yp, y_cont=y_cont)
                sc_full   = RPFNet._score_rpf(rpf_full)
                pred_full = np.zeros(len(Xp), dtype=int)
                pred_full[np.argsort(sc_full)[-k:]] = 1
                f1_full   = f1_score(ytrue, pred_full, zero_division=0)

                for block in RPFExtractor.BLOCK_NAMES:
                    rpf_m  = RPFNet.ablate_block(rpf_full, block)
                    sc_m   = RPFNet._score_rpf(rpf_m)
                    pred_m = np.zeros(len(Xp), dtype=int)
                    pred_m[np.argsort(sc_m)[-k:]] = 1
                    f1_m   = f1_score(ytrue, pred_m, zero_division=0)
                    drops[block].append(f1_full - f1_m)
            except Exception:
                pass
    return {b: float(np.mean(v)) if v else 0. for b, v in drops.items()}

def _bar(f, w=20):
    b = int(round(max(0., min(1., f)) * w))
    return "█" * b + "░" * (w - b)

def print_result(atk, rate, res):
    if res is None:
        print(f"  {atk} @ {rate:.0%}  SKIPPED"); return
    flip = "" if res["has_flip"] else "  ⚑ no-flip"
    print(f"\n  ┌─ {res['label']}  rate={rate:.0%}"
          f"  ({res['n_poison']}/{res['n_total']}){flip}")
    for mk, mlabel in [
        ("unknown", f"Mode 1 — unknown rate (assume {ASSUMED_RATE:.0%})"),
        ("known",   "Mode 2 — known rate   (oracle)"),
    ]:
        r    = res[mk]
        dets = list(r.keys())
        best = max(r[d]["F1_mean"] for d in dets)
        print(f"  │  {mlabel}")
        print(f"  │  {'':18}", end="")
        for d in dets: print(f"  {d:>14}", end="")
        print()
        for met in ["F1", "P", "R"]:
            print(f"  │  {met:<18}", end="")
            for d in dets:
                v   = r[d][f"{met}_mean"]
                std = r[d].get(f"{met}_std", 0.0)

                star = "★" if met == "F1" and abs(v - best) < 1e-4 else " "
                fail = "✗" if (
                    met == "F1"
                    and not res["has_flip"]
                    and d in ("TRIM", "iTRIM")
                    and v < 0.20
                ) else " "

                # SINGLE properly formatted column
                print(f"  {v:>6.4f}±{std:<6.4f}{star}{fail}", end="")
            print()
        f1h = r.get("Hybrid", r.get("RPFNet", {})).get("F1_mean", 0.)
        print(f"  │  Hybrid visual    {_bar(f1h)}  {f1h:.3f}")
    print(f"  └{'─'*68}")

def significance_test(ds_res, mode_key="unknown"):
    RPFNet_vals = []
    hybrid_vals = []

    for res in ds_res.values():
        if res is None:
            continue

        if "RPFNet" in res[mode_key] and "Hybrid" in res[mode_key]:
            RPFNet_vals.extend(res[mode_key]["RPFNet"]["F1_all"])
            hybrid_vals.extend(res[mode_key]["Hybrid"]["F1_all"])

    if len(RPFNet_vals) > 1:
        t_stat, p_t = ttest_rel(hybrid_vals, RPFNet_vals)
        w_stat, p_w = wilcoxon(hybrid_vals, RPFNet_vals)

        print(f"\n  Statistical Test (Hybrid vs RPFNetPoisonV3):")
        print(f"    Paired t-test p-value      : {p_t:.4e}")
        print(f"    Wilcoxon signed-rank p-val : {p_w:.4e}")
        
def print_summary(results, mode_key="unknown", heading="Summary"):
    valid = {k: v for k, v in results.items() if v}
    if not valid: return
    dets  = list(next(iter(valid.values()))[mode_key].keys())
    print(f"\n  ── {heading}")
    print(f"  {'Attack':<26} {'Rate':>5}", end="")
    for d in dets: print(f"  {d:>14}", end="")
    print(f"  {'Flip':>5}")

    bucket = {d: [] for d in dets}
    nf     = {d: [] for d in dets}

    for (atk, rate), res in sorted(results.items()):
        if res is None: continue
        r    = res[mode_key]
        best = max(r[d]["F1_mean"] for d in dets if d in r)
        print(f"  {res['label'][:26]:<26} {rate:>4.0%}", end="")
        for d in dets:
            if d not in r: continue
            f1   = r[d]["F1_mean"]
            star = "★" if abs(f1 - best) < 1e-4 else " "
            fail = "✗" if (not res["has_flip"] and d in ("TRIM","iTRIM") and f1 < 0.20) else " "
            mean = r[d]["F1_mean"]
            std  = r[d].get("F1_std", 0.0)
            print(f"  {mean:>6.4f}±{std:<6.4f}{star}{fail}", end="")
            bucket[d].append(f1)
            if not res["has_flip"]: nf[d].append(f1)
        print(f"  {'yes' if res['has_flip'] else 'NO':>5}")

    bg = max((np.mean(v) for v in bucket.values() if v), default=0.)
    print(f"\n  {'AVERAGE':<32}", end="")
    for d in dets:
        vals = bucket[d]
        if vals:
            mean = np.mean(vals)
            std  = np.std(vals)
        else:
            mean, std = 0.0, 0.0
        t = "★" if abs(mean - bg) < 1e-4 and bg > 0 else " "
        print(f"  {mean:>6.4f}±{std:<6.4f}{t} ", end="")
    print()
    if any(nf.values()):
        print(f"  {'no-flip attacks only':<32}", end="")
        for d in dets:
            a    = np.mean(nf[d]) if nf[d] else 0.
            fail = "✗" if d in ("TRIM","iTRIM") and a < 0.20 else " "
            print(f"  {a:>12.4f}{fail} ", end="")
        print()


def print_grand_summary(grand_results, eval_ds):
    W = 82
    print("\n" + "=" * W)
    print("  GRAND SUMMARY — Averaged Across ALL Datasets")
    print("=" * W)

    all_dets = []
    for dr in grand_results.values():
        for res in dr.values():
            if res:
                for d in res["unknown"]:
                    if d not in all_dets: all_dets.append(d)

    for mode_key, mode_label in [
        ("unknown", f"Mode 1 — rate UNKNOWN (assume {ASSUMED_RATE:.0%})"),
        ("known",   "Mode 2 — rate KNOWN   (oracle)"),
    ]:
        print(f"\n  ── {mode_label}")
        print(f"\n  {'Dataset':<28} {'Type':>7}", end="")
        for d in all_dets: print(f"  {d:>14}", end="")
        print()
        print(f"  {'─'*36}", end="")
        for _ in all_dets: print(f"  {'─'*14}", end="")
        print()

        overall = {d: [] for d in all_dets}
        nf_all  = {d: [] for d in all_dets}

        for ds_disp, dr in grand_results.items():
            avgs = {}
            for d in all_dets:
                vals = [r[mode_key][d]["F1_mean"]
                        for r in dr.values()
                        if r and d in r.get(mode_key, {})]
                avgs[d] = np.mean(vals) if vals else 0.
                overall[d].extend(vals)
            best = max(avgs.values()) if avgs else 0.
            zs   = next((v["zero_shot"] for k, v in eval_ds.items()
                          if v["display"] == ds_disp), False)
            is_reg = next((v.get("is_regression", False) for k, v in eval_ds.items()
                           if v["display"] == ds_disp), False)
            tag = "⭐ ZS" if zs else "seen"
            if is_reg: tag += " 📈"
            print(f"  {ds_disp[:28]:<28} {tag:>7}", end="")
            for d in all_dets:
                t = "★" if abs(avgs.get(d, 0) - best) < 1e-4 and best > 0 else " "
                print(f"  {avgs.get(d, 0):>12.4f}{t} ", end="")
            print()
            for res in dr.values():
                if res and not res["has_flip"]:
                    for d in all_dets:
                        if d in res.get(mode_key, {}):
                            nf_all[d].append(res[mode_key][d]["F1_mean"])

        bg = max((np.mean(v) for v in overall.values() if v), default=0.)
        print(f"\n  {'GRAND AVERAGE':<36}", end="")
        for d in all_dets:
            a = np.mean(overall[d]) if overall[d] else 0.
            s = np.std(overall[d])  if overall[d] else 0.
            t = "★" if abs(a - bg) < 1e-4 and bg > 0 else " "
            print(f"  {a:>6.4f}±{s:<6.4f}{t}", end="")
        print()

        zs_disps = {v["display"] for k, v in eval_ds.items() if v["zero_shot"]}
        zs_vals  = {d: [] for d in all_dets}
        for ds_disp, dr in grand_results.items():
            if ds_disp not in zs_disps: continue
            for res in dr.values():
                if res:
                    for d in all_dets:
                        if d in res.get(mode_key, {}):
                            zs_vals[d].append(res[mode_key][d]["F1_mean"])
        if any(zs_vals.values()):
            bz = max((np.mean(v) for v in zs_vals.values() if v), default=0.)
            print(f"  {'ZERO-SHOT ONLY ⭐':<36}", end="")
            for d in all_dets:
                a = np.mean(zs_vals[d]) if zs_vals[d] else 0.
                s = np.std(zs_vals[d])  if zs_vals[d] else 0.
                t = "★" if abs(a - bz) < 1e-4 and bz > 0 else " "
                print(f"  {a:>6.4f}±{s:<6.4f}{t}", end="")
            print()

        print(f"  {'no-flip attacks only':<36}", end="")
        for d in all_dets:
            a    = np.mean(nf_all[d]) if nf_all[d] else 0.
            s    = np.std(nf_all[d])  if nf_all[d] else 0.
            fail = "✗" if d in ("TRIM","iTRIM") and a < 0.20 else " "
            print(f"  {a:>6.4f}±{s:<6.4f}{fail}", end="")
        print()

    #  Attack Family Summary Table
    ATTACK_FAMILIES = {
        "Label Flip": [
            "label_flip", "boundary_flip", "targeted_class",
            "grad_flip", "tree_aware", "target_flip_extreme",
        ],
        "Noise / Perturbation": [
            "feat_perturb", "gauss_noise", "feat_dropout",
            "combo_flip_perturb", "combo_flip_noise",
        ],
        "Clean-Label / Stealth": [
            "clean_label", "interpolation", "adaptive_blend",
        ],
        "Backdoor": [
            "backdoor", "backdoor_heavy",
        ],
        "Structural / Distribution": [
            "repr_inversion", "dist_shift", "outlier_inject",
            "null_feature",
        ],
        "Regression-Specific": [
            "target_shift", "leverage_attack",
        ],
    }

    print("\n" + "=" * W)
    print("  ATTACK FAMILY SUMMARY — Averaged Across All Datasets & Rates")
    print("=" * W)

    # Collect all detectors from grand_results
    all_dets_fam = []
    for dr in grand_results.values():
        for res in dr.values():
            if res:
                for mk in ("unknown", "known"):
                    for d in res[mk]:
                        if d not in all_dets_fam:
                            all_dets_fam.append(d)

    for mode_key, mode_label in [
        ("unknown", f"Mode 1 — rate UNKNOWN (assume {ASSUMED_RATE:.0%})"),
        ("known",   "Mode 2 — rate KNOWN   (oracle)"),
    ]:
        print(f"\n  ── {mode_label}")
        print(f"\n  {'Attack Family':<28}", end="")
        for d in all_dets_fam:
            print(f"  {d:>14}", end="")
        print(f"  {'#runs':>6}")
        print(f"  {'─' * 28}", end="")
        for _ in all_dets_fam:
            print(f"  {'─' * 14}", end="")
        print(f"  {'─' * 6}")

        grand_fam = {d: [] for d in all_dets_fam}

        for fam_name, fam_attacks in ATTACK_FAMILIES.items():
            fam_vals = {d: [] for d in all_dets_fam}
            for dr in grand_results.values():
                for (atk, rate), res in dr.items():
                    if res is None:
                        continue
                    if atk not in fam_attacks:
                        continue
                    for d in all_dets_fam:
                        if d in res.get(mode_key, {}):
                            fam_vals[d].append(res[mode_key][d]["F1_mean"])

            n_runs = max(len(v) for v in fam_vals.values()) if fam_vals else 0
            if n_runs == 0:
                continue

            best_fam = max(
                (np.mean(v) for v in fam_vals.values() if v), default=0.
            )

            print(f"  {fam_name:<28}", end="")
            for d in all_dets_fam:
                if fam_vals[d]:
                    a = np.mean(fam_vals[d])
                    s = np.std(fam_vals[d])
                    grand_fam[d].extend(fam_vals[d])
                    t = "★" if abs(a - best_fam) < 1e-4 and best_fam > 0 else " "
                    print(f"  {a:>6.4f}±{s:<5.4f}{t}", end="")
                else:
                    print(f"  {'—':>14}", end="")
            print(f"  {n_runs:>6}")

        # Grand row across all families
        bg_fam = max(
            (np.mean(v) for v in grand_fam.values() if v), default=0.
        )
        print(f"  {'─' * 28}", end="")
        for _ in all_dets_fam:
            print(f"  {'─' * 14}", end="")
        print(f"  {'─' * 6}")
        print(f"  {'ALL FAMILIES':<28}", end="")
        for d in all_dets_fam:
            if grand_fam[d]:
                a = np.mean(grand_fam[d])
                s = np.std(grand_fam[d])
                t = "★" if abs(a - bg_fam) < 1e-4 and bg_fam > 0 else " "
                print(f"  {a:>6.4f}±{s:<5.4f}{t}", end="")
            else:
                print(f"  {'—':>14}", end="")
        total_runs = max(len(v) for v in grand_fam.values()) if grand_fam else 0
        print(f"  {total_runs:>6}")

def aggregate_seed_results(seed_results):
    """
    Aggregate results across multiple seeds.
    Returns mean and std for each metric.
    """

    def aggregate_mode(mode_key):
        out = {}
        detectors = seed_results[0][mode_key].keys()

        for d in detectors:
            f1s = [r[mode_key][d]["F1"] for r in seed_results if d in r[mode_key]]
            ps  = [r[mode_key][d]["P"]  for r in seed_results if d in r[mode_key]]
            rs  = [r[mode_key][d]["R"]  for r in seed_results if d in r[mode_key]]

            if len(f1s) == 0:
                continue

            out[d] = {
                "F1_mean": float(np.mean(f1s)),
                "F1_std":  float(np.std(f1s)),
                "P_mean":  float(np.mean(ps)),
                "P_std":   float(np.std(ps)),
                "R_mean":  float(np.mean(rs)),
                "R_std":   float(np.std(rs)),
                "F1_all":  f1s  # keep raw values for statistical tests
            }

        return out

    return {
        "unknown": aggregate_mode("unknown"),
        "known":   aggregate_mode("known"),
        "label":   seed_results[0]["label"],
        "has_flip": seed_results[0]["has_flip"],
        "n_poison": seed_results[0]["n_poison"],
        "n_total":  seed_results[0]["n_total"],
    }



#  MAIN
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(42)
    W = 82

    print("=" * W)
    print("  RPFNetPoison v3: Universal Poison Detection + Regression Support")
    print(f"  RPF dims : {RPFExtractor.DIM}  (v4=61, +Block G Structural Echo)")
    print(f"  Device   : {device}   Attacks: {len(ATTACKS)}   Rates: {RATES}")
    print(f"  Model    : {RPFNet_MODEL_PATH}")
    print(f"  RPFNet-train datasets: {len(RPFNet_TRAIN_DATASETS)} (expanded for OOD generalization)")
    print("=" * W)

    # Load RPFNet-training datasets
    print(f"\n  RPFNet-training datasets: {RPFNet_TRAIN_DATASETS}")
    RPFNet_data = {}
    for bname in RPFNet_TRAIN_DATASETS:
        try:
            result = load_builtin(bname)
            Xtr = result[0]
            ytr = result[2]
            is_reg = result[6]
            y_cont = result[7]
            if is_reg and y_cont is not None:
                RPFNet_data[bname] = (Xtr, ytr, y_cont)
            else:
                RPFNet_data[bname] = (Xtr, ytr)
        except Exception as e:
            print(f"  ⚠ {bname}: {e}")

    if not RPFNet_data:
        raise RuntimeError("No RPFNet-training datasets loaded.")

    print(f"\n  Successfully loaded {len(RPFNet_data)}/{len(RPFNet_TRAIN_DATASETS)} "
          f"RPFNet-training datasets")

    # Train or load
    RPFNet_det = RPFNetPoisonDetector(device=device, epochs=RPFNet_EPOCHS,
                                   batch_size=RPFNet_BATCH_SIZE, lr=RPFNet_LR)
    hybrid_det = HybridEnsembleDetector(RPFNet_det)
    fusion_path = RPFNet_MODEL_PATH.replace(".pt", "_fusion.pt")

    if not FORCE_RETRAIN and os.path.exists(RPFNet_MODEL_PATH):
        RPFNet_det.load(RPFNet_MODEL_PATH)
        if os.path.exists(fusion_path):
            hybrid_det.load(fusion_path)
        else:
            hybrid_det.calibrate_fusion_weight(RPFNet_data, verbose=True)
    else:
        print(f"\n  RPFNet-training on: {list(RPFNet_data.keys())}")
        RPFNet_det.RPFNet_fit(RPFNet_data, attacks=RPFNet_TRAIN_ATTACKS,
                          rates=RPFNet_TRAIN_RATES, seeds=RPFNet_TRAIN_SEEDS,
                          verbose=True)
        RPFNet_det.save(RPFNet_MODEL_PATH)
        hybrid_det.attach_rate_head(RPFNet_det.rate_head)
        hybrid_det.calibrate_fusion_weight(RPFNet_data, verbose=True)
        hybrid_det.save(fusion_path)

    # Build eval set
    eval_ds = {}
    for bname in EVAL_BUILTIN:
        try:
            result = load_builtin(bname)
            Xtr = result[0]
            ytr = result[2]
            is_reg = result[6]
            y_cont = result[7]
            eval_ds[bname] = {
                "display":       bname.replace("_", " ").title(),
                "X_tr": Xtr, "y_tr": ytr,
                "zero_shot":     bname not in RPFNet_TRAIN_DATASETS,
                "is_regression": is_reg,
                "y_cont":        y_cont,
            }
        except Exception as e:
            print(f"  ⚠ {bname}: {e}")

    for display, csv_path, tcol in CSV_DATASETS:
        if os.path.exists(csv_path):
            try:
                result = load_csv(csv_path, tcol)
                Xtr = result[0]
                ytr = result[2]
                key = display.lower().replace(" ", "_")
                eval_ds[key] = {"display": display, "X_tr": Xtr, "y_tr": ytr,
                                 "zero_shot": True, "is_regression": False,
                                 "y_cont": None}
            except Exception as e:
                print(f"  ⚠ {display}: {e}")
        else:
            print(f"\n  ⚠ CSV not found: {csv_path} — skipping")

    seen  = [v["display"] for v in eval_ds.values() if not v["zero_shot"]]
    zshot = [v["display"] for v in eval_ds.values() if v["zero_shot"]]
    regs  = [v["display"] for v in eval_ds.values() if v.get("is_regression")]
    print(f"\n  Eval seen:       {seen}")
    print(f"  Eval zero-shot:  {zshot} ⭐")
    print(f"  Eval regression: {regs} 📈")

    # Per-dataset loop
    grand_results = {}

    for ds_key, dsinfo in eval_ds.items():
        ds_disp = dsinfo["display"]
        Xtr, ytr = dsinfo["X_tr"], dsinfo["y_tr"]
        y_cont   = dsinfo.get("y_cont")
        is_reg   = dsinfo.get("is_regression", False)
        zs_tag   = " ⭐ ZERO-SHOT" if dsinfo["zero_shot"] else " (seen)"
        if is_reg: zs_tag += " 📈 REGRESSION"

        print("\n" + "─" * W)
        print(f"  DATASET: {ds_disp}{zs_tag}")
        print("─" * W)

        # Choose attacks based on dataset type
        ds_attacks = REGRESSION_ATTACKS if is_reg else ATTACKS

        ds_res = {}
        for atk in ds_attacks:
            if atk not in ATTACK_RPFNet:
                continue
            for rate in RATES:
                label, has_flip = ATTACK_RPFNet[atk]
                tag = "" if has_flip else " [no-flip]"
                print(f"\n  {label} @ {rate:.0%}{tag}... ", end="", flush=True)
                try:
                    seed_results = []

                    for s in range(EVAL_SEEDS):
                        try:
                            res_s = run_one(
                                RPFNet_det, hybrid_det,
                                Xtr, ytr, atk, rate,
                                ds_key=f"{ds_key}_seed{s}",
                                y_cont=y_cont
                            )
                            if res_s is not None:
                                seed_results.append(res_s)
                        except:
                            pass

                    if seed_results:
                        res = aggregate_seed_results(seed_results)
                    else:
                        res = None

                    ds_res[(atk, rate)] = res
                    if res:
                        f1h = res["unknown"]["Hybrid"]["F1_mean"]
                        f1m = res["unknown"]["RPFNet"]["F1_mean"]
                        f1i = res["unknown"]["IsoForest"]["F1_mean"]
                        print(f"Hybrid={f1h:.3f}  RPFNet={f1m:.3f}  Iso={f1i:.3f}")
                        print_result(atk, rate, res)
                    else:
                        print("SKIPPED")
                except Exception as e:
                    import traceback
                    print(f"ERROR: {e}")
                    traceback.print_exc()
                    ds_res[(atk, rate)] = None

        grand_results[ds_disp] = ds_res

        print_summary(ds_res, "unknown",
                      f"Mode 1 — rate UNKNOWN (assume {ASSUMED_RATE:.0%})")
        print_summary(ds_res, "known",
                      "Mode 2 — rate KNOWN (oracle)")

        significance_test(ds_res, "unknown")

        # Ablation
        print(f"\n  ── Ablation (block masking) — {ds_disp}")
        try:
            abl_attacks = (["feat_perturb", "repr_inversion", "dist_shift"]
                           if is_reg else
                           ["label_flip", "feat_perturb", "repr_inversion", "dist_shift"])
            abl = ablation_study(RPFNet_det, Xtr, ytr, abl_attacks,
                                  rate=0.10, y_cont=y_cont)
            print(f"  {'Block':>8}  {'Description':<32}  {'F1 drop':>8}  Importance")
            print(f"  {'─'*70}")
            for bk, (bdesc, _) in RPFExtractor.BLOCK_NAMES.items():
                drop = abl.get(bk, 0.)
                bar  = "█" * max(0, int(drop * 50)) if drop > 0 else "─"
                flag = ""
                print(f"  Block {bk:>2}  {bdesc[:32]:<32}  {drop:>+.4f}  {bar}{flag}")
        except Exception as e:
            print(f"  (ablation failed: {e})")

    # Grand summary
    print_grand_summary(grand_results, eval_ds)
    generate_all_figures(grand_results, eval_ds, RPFNet_det, hybrid_det)



