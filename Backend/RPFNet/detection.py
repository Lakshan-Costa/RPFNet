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

RPFNet_TRAIN_RATES = [ 0.15]
RPFNet_TRAIN_SEEDS = 1
RPFNet_EPOCHS = 30
RPFNet_BATCH_SIZE = 512
RPFNet_LR = 1e-3
EVAL_SEEDS = 2

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

#Utils
def focal_loss(logits, targets, gamma: float = 2.0, pos_weight: float = 4.0):
    pw = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pw, reduction="none")
    pt = torch.exp(-F.binary_cross_entropy_with_logits(
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
        self.device = torch.device(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.extractor = RPFExtractor(k_small=RPF_K_SMALL,
                                        k_large=RPF_K_LARGE,
                                        cv_folds=RPF_CV_FOLDS)
        self.net = RPFNetPoisonNet(input_dim=RPFExtractor.DIM).to(self.device)
        self.rate_head = RateEstimatorHead().to(self.device)
        self._fitted = False
        self._threshold = 0.5

    def RPFNet_fit(self, datasets: dict, attacks=None, rates=None,
                 seeds: int = 3, verbose: bool = True):
        """
        datasets: dict of name → (X, y) OR name → (X, y, y_cont)
        """
        if attacks is None: attacks = RPFNet_TRAIN_ATTACKS
        if rates is None: rates = RPFNet_TRAIN_RATES

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

        opt = torch.optim.AdamW(self.net.parameters(),
                                   lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=self.epochs, eta_min=self.lr * 0.01)
        rng = np.random.default_rng(42)
        n = len(X_pool)
        best = float("inf")
        hist = []

        for ep in range(self.epochs):
            perm = rng.permutation(n)
            ep_loss = []
            for i in range(0, n, self.batch_size):
                idx = perm[i: i + self.batch_size]
                if len(idx) < 2: continue
                xb = torch.tensor(X_pool[idx], device=self.device)
                yb = torch.tensor(y_pool[idx], device=self.device)
                logits = self.net(xb)
                loss = focal_loss(logits, yb)
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
                            rpf = get_rpf_cached(self.extractor, Xp, yp,
                                                    cache_key, y_cont=y_cont)
                            s = self._score_rpf(rpf)
                            s_rk = rankdata(s).astype(np.float32) / len(s)
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
                    xb = torch.tensor(Xr[idx], device=self.device)
                    yb = torch.tensor(yr[idx], device=self.device)
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
        rpf = self.extractor.extract(X, y, y_cont=y_cont)
        t = torch.tensor(rpf, dtype=torch.float32,
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
            "net": self.net.state_dict(),
            "threshold": self._threshold,
            "fitted": self._fitted,
            "k_small": self.extractor.k_small,
            "k_large": self.extractor.k_large,
            "cv_folds": self.extractor.cv_folds,
            "rpf_dim": RPFExtractor.DIM,
            "version": 4,
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
        self._threshold = ckpt.get("threshold", 0.5)
        self._fitted = ckpt.get("fitted", True)
        self.extractor.k_small = ckpt.get("k_small", RPF_K_SMALL)
        self.extractor.k_large = ckpt.get("k_large", RPF_K_LARGE)
        self.extractor.cv_folds = ckpt.get("cv_folds", RPF_CV_FOLDS)
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
        iso_s = self._iso_scores(X)
        combined = self._rank_fuse(RPFNet_s, iso_s)
        return combined, RPFNet_s, iso_s

    def _estimate_rate(self, scores):
        ranked = rankdata(scores).astype(np.float32) / len(scores)

        if self.rate_head is not None:
            try:
                feats = score_distribution_features(ranked)
                t = torch.tensor(feats, dtype=torch.float32,
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
        n = len(combined)
        est_rate = self._estimate_rate(combined)

        vote_rates = np.clip(
            [est_rate * 0.50, est_rate * 0.75, est_rate,
             est_rate * 1.25, est_rate * 1.50],
            0.01, 0.40
        )

        votes = np.zeros(n, dtype=np.float32)
        for vr in vote_rates:
            k = max(1, int(n * vr))
            mask = np.zeros(n, dtype=int)
            mask[np.argsort(combined)[-k:]] = 1
            votes += mask

        pred = (votes >= 3).astype(int)
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
                            ms = self.RPFNet._score_rpf(rpf)
                            iso_s = self._iso_scores(Xp)
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
