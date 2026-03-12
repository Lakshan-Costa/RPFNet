META_TRAIN_DATASETS = [
    "breast_cancer", "wine", "digits",
    "california_large", "adult", "credit_g",
    # Regression datasets for meta-training
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
RATES        = [0.05, 0.1, 0.15, 0.2, 0.25]
ASSUMED_RATE = 0.10

META_MODEL_PATH = "./metapoison_v3_universal.pt"
FORCE_RETRAIN   = True

META_TRAIN_ATTACKS = [
    "label_flip", "tree_aware", "grad_flip", "boundary_flip",
    "gauss_noise", "interpolation", "targeted_class",
    "clean_label", "backdoor", "null_feature", "feat_perturb",
    "backdoor_heavy", "repr_inversion", "dist_shift",
    "outlier_inject", "feat_dropout",
]

# Regression attacks added to meta-training
META_TRAIN_ATTACKS_REG = [
    "target_shift", "leverage_attack", "target_flip_extreme",
    "feat_perturb", "backdoor", "repr_inversion",
    "dist_shift", "feat_dropout", "outlier_inject",
    "gauss_noise",
]

META_TRAIN_RATES  = [0.05, 0.1, 0.15, 0.2, 0.25]
META_TRAIN_SEEDS  = 2
META_EPOCHS       = 30
META_BATCH_SIZE   = 512
META_LR           = 1e-3
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
from sklearn.ensemble import (RandomForestClassifier, IsolationForest,
                               GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.datasets import (load_breast_cancer, load_wine, load_digits,
                               load_iris, load_diabetes, fetch_covtype,
                               fetch_california_housing, fetch_openml,
                               make_classification, make_moons, make_circles,
                               make_friedman1, make_regression)

try:
    from iTrim.regressiondatapoisoning.src.defenses import defense_both_trim_and_itrim
    from sklearn.linear_model import LinearRegression
    HAS_ITRIM = True
except ImportError:
    HAS_ITRIM = False

def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

#  RPF CACHE
RPF_CACHE = {}

def get_rpf_cached(extractor, X, y, key, y_cont=None):
    """y_cont: optional continuous target for regression features."""
    if key not in RPF_CACHE:
        RPF_CACHE[key] = extractor.extract(X, y, y_cont=y_cont)
    return RPF_CACHE[key]

#  ATTACK TAXONOMY — UPDATED with regression attacks

ATTACK_META = {
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


#  DATASET LOADERS — UPDATED with expanded diverse datasets
def _split_scale(X, y, seed=42):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    sc  = StandardScaler()
    Xtr = sc.fit_transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    return Xtr, Xte, ytr, yte, sc


def _split_scale_reg(X, y, seed=42):
    """Split + scale for regression (no stratify)."""
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed)
    sc  = StandardScaler()
    Xtr = sc.fit_transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    # Also scale targets for regression
    y_mu, y_sd = ytr.mean(), ytr.std() + 1e-8
    ytr_s = ((ytr - y_mu) / y_sd).astype(np.float32)
    yte_s = ((yte - y_mu) / y_sd).astype(np.float32)
    return Xtr, Xte, ytr_s, yte_s, sc, y_mu, y_sd


def _load_openml_classification(data_id, name, max_n=5000, seed=42):            # ← NEW helper
    """
    Generic OpenML classification dataset loader.
    Handles mixed types, missing values, subsampling, and returns
    the standard 8-tuple.
    """
    bundle = fetch_openml(data_id=data_id, as_frame=True, parser='auto')
    df = bundle.frame.copy()
    tgt_name = bundle.target.name if hasattr(bundle.target, 'name') else 'class'
    tgt = df.pop(tgt_name) if tgt_name in df.columns else bundle.target

    # Encode target to int
    if tgt.dtype == object or str(tgt.dtype) == 'category':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(tgt.astype(str).values)
    else:
        y = tgt.values.astype(int)

    # Handle features
    for c in df.select_dtypes(include='number').columns:
        df[c] = df[c].fillna(df[c].median())
    for c in df.select_dtypes(exclude='number').columns:
        df[c] = df[c].astype(str).fillna('missing')
    df = pd.get_dummies(df, drop_first=True)
    X = df.values.astype(np.float32)

    # Subsample if too large
    if len(X) > max_n:
        rng_sub = np.random.default_rng(seed)
        idx = rng_sub.choice(len(X), max_n, replace=False)
        X, y = X[idx], y[idx]

    n_classes = len(np.unique(y))
    n_feat = X.shape[1]
    desc = f"{name} ({n_feat} feat, {n_classes} classes, n={len(X)})"

    Xtr, Xte, ytr, yte, sc = _split_scale(X, y, seed=seed)
    print(f"\n  [builtin:{name.lower().replace(' ','_')}] {desc}  "
          f"train={Xtr.shape}  balance={ytr.mean():.2f}")
    return Xtr, Xte, ytr, yte, [], sc, False, None


def load_builtin(name):
    is_regression = False
    y_cont = None

    # Regression datasets
    if name == "california_reg":
        d = fetch_california_housing()
        X = d.data.astype(np.float32)
        y_raw = d.target.astype(np.float32)
        # Subsample to 3000
        rng_sub = np.random.default_rng(42)
        idx = rng_sub.choice(len(X), min(3000, len(X)), replace=False)
        X, y_raw = X[idx], y_raw[idx]
        Xtr, Xte, ytr_cont, yte_cont, sc, y_mu, y_sd = _split_scale_reg(X, y_raw)
        # Binarize for RPF extraction
        ytr = (ytr_cont > 0).astype(int)  # above/below mean (scaled)
        yte = (yte_cont > 0).astype(int)
        desc = f"California Reg (8 feat, n={len(X)})"
        is_regression = True
        print(f"\n  [builtin:{name}] {desc}  train={Xtr.shape}  "
              f"balance={ytr.mean():.2f}  [REGRESSION→binarized]")
        return Xtr, Xte, ytr, yte, [], sc, is_regression, ytr_cont

    elif name == "diabetes_reg":
        d = load_diabetes()
        X = d.data.astype(np.float32)
        y_raw = d.target.astype(np.float32)
        Xtr, Xte, ytr_cont, yte_cont, sc, y_mu, y_sd = _split_scale_reg(X, y_raw)
        ytr = (ytr_cont > 0).astype(int)
        yte = (yte_cont > 0).astype(int)
        desc = f"Diabetes Reg (10 feat, n={len(X)})"
        is_regression = True
        print(f"\n  [builtin:{name}] {desc}  train={Xtr.shape}  "
              f"balance={ytr.mean():.2f}  [REGRESSION→binarized]")
        return Xtr, Xte, ytr, yte, [], sc, is_regression, ytr_cont

    elif name == "boston_reg":
        # Synthetic stand-in since real Boston is deprecated
        X, y_raw = make_regression(n_samples=1200, n_features=13,
                                    n_informative=8, noise=10.0,
                                    random_state=42)
        X = X.astype(np.float32)
        y_raw = y_raw.astype(np.float32)
        Xtr, Xte, ytr_cont, yte_cont, sc, y_mu, y_sd = _split_scale_reg(X, y_raw)
        ytr = (ytr_cont > 0).astype(int)
        yte = (yte_cont > 0).astype(int)
        desc = f"Boston-like Reg (13 feat, n={len(X)})"
        is_regression = True
        print(f"\n  [builtin:{name}] {desc}  train={Xtr.shape}  "
              f"balance={ytr.mean():.2f}  [REGRESSION→binarized]")
        return Xtr, Xte, ytr, yte, [], sc, is_regression, ytr_cont

    elif name == "friedman_reg":
        X, y_raw = make_friedman1(n_samples=2000, n_features=10,
                                    noise=1.0, random_state=42)
        X = X.astype(np.float32)
        y_raw = y_raw.astype(np.float32)
        Xtr, Xte, ytr_cont, yte_cont, sc, y_mu, y_sd = _split_scale_reg(X, y_raw)
        ytr = (ytr_cont > 0).astype(int)
        yte = (yte_cont > 0).astype(int)
        desc = f"Friedman1 Reg (10 feat, n={len(X)})"
        is_regression = True
        print(f"\n  [builtin:{name}] {desc}  train={Xtr.shape}  "
              f"balance={ytr.mean():.2f}  [REGRESSION→binarized]")
        return Xtr, Xte, ytr, yte, [], sc, is_regression, ytr_cont

    elif name == "abalone_reg":
        bundle = fetch_openml(data_id=183, as_frame=True, parser='auto')
        df = bundle.frame.copy()
        tgt_name = bundle.target.name if hasattr(bundle.target, 'name') else 'Rings'
        tgt = df.pop(tgt_name) if tgt_name in df.columns else bundle.target
        y_raw = tgt.astype(float).values.astype(np.float32)
        for c in df.select_dtypes(include='number').columns:
            df[c] = df[c].fillna(df[c].median())
        for c in df.select_dtypes(exclude='number').columns:
            df[c] = df[c].astype(str).fillna('missing')
        df = pd.get_dummies(df, drop_first=True)
        X = df.values.astype(np.float32)
        # Subsample
        rng_sub = np.random.default_rng(42)
        if len(X) > 4000:
            idx = rng_sub.choice(len(X), 4000, replace=False)
            X, y_raw = X[idx], y_raw[idx]
        Xtr, Xte, ytr_cont, yte_cont, sc, y_mu, y_sd = _split_scale_reg(X, y_raw)
        ytr = (ytr_cont > 0).astype(int)
        yte = (yte_cont > 0).astype(int)
        desc = f"Abalone Reg ({X.shape[1]} feat, n={len(X)})"
        is_regression = True
        print(f"\n  [builtin:{name}] {desc}  train={Xtr.shape}  "
              f"balance={ytr.mean():.2f}  [REGRESSION→binarized]")
        return Xtr, Xte, ytr, yte, [], sc, is_regression, ytr_cont

    elif name == "energy_reg":
        X, y_raw = make_regression(n_samples=1500, n_features=8,
                                    n_informative=6, noise=5.0,
                                    random_state=99)
        X = X.astype(np.float32)
        y_raw = y_raw.astype(np.float32)
        Xtr, Xte, ytr_cont, yte_cont, sc, y_mu, y_sd = _split_scale_reg(X, y_raw)
        ytr = (ytr_cont > 0).astype(int)
        yte = (yte_cont > 0).astype(int)
        desc = f"Energy-like Reg (8 feat, n={len(X)})"
        is_regression = True
        print(f"\n  [builtin:{name}] {desc}  train={Xtr.shape}  "
              f"balance={ytr.mean():.2f}  [REGRESSION→binarized]")
        return Xtr, Xte, ytr, yte, [], sc, is_regression, ytr_cont

    elif name == "ionosphere":
        return _load_openml_classification(data_id=59, name="Ionosphere")

    elif name == "spambase":
        return _load_openml_classification(data_id=44, name="Spambase")

    elif name == "vehicle":
        return _load_openml_classification(data_id=54, name="Vehicle")

    elif name == "segment":
        return _load_openml_classification(data_id=36, name="Segment")

    elif name == "pendigits":
        return _load_openml_classification(data_id=32, name="Pendigits", max_n=5000)

    elif name == "satimage":
        return _load_openml_classification(data_id=182, name="Satimage", max_n=5000)

    elif name == "waveform":
        return _load_openml_classification(data_id=60, name="Waveform", max_n=5000)

    elif name == "glass":
        return _load_openml_classification(data_id=41, name="Glass")

    elif name == "yeast":
        return _load_openml_classification(data_id=181, name="Yeast")

    elif name == "page_blocks":
        return _load_openml_classification(data_id=30, name="Page Blocks")

    elif name == "vowel":
        return _load_openml_classification(data_id=307, name="Vowel")

    elif name == "optdigits":
        return _load_openml_classification(data_id=28, name="Optdigits", max_n=5000)

    elif name == "steel_plates":
        return _load_openml_classification(data_id=1504, name="Steel Plates")

    elif name == "mfeat_factors":
        return _load_openml_classification(data_id=12, name="MFeat Factors")

    elif name == "cardiotocography":
        return _load_openml_classification(data_id=1466, name="Cardiotocography")

    # Classification datasets
    if name == "breast_cancer":
        d = load_breast_cancer()
        X, y = d.data.astype(np.float32), d.target
        desc = "Breast Cancer (30 feat)"
    elif name == "wine":
        d = load_wine()
        X = d.data.astype(np.float32)
        y = d.target
        desc = "Wine (13 feat, 3 classes)"
    elif name == "digits":
        d = load_digits()
        X = d.data.astype(np.float32)
        y = d.target
        desc = "Digits (64 feat, 10 classes)"
    elif name == "iris":
        d = load_iris()
        X = d.data.astype(np.float32)
        y = d.target
        desc = "Iris (4 feat, 3 classes)"
    elif name == "diabetes_bin":
        d = load_diabetes()
        X = d.data.astype(np.float32)
        y = (d.target > np.median(d.target)).astype(int)
        desc = "Diabetes binary (10 feat)"
    elif name == "moons":
        X, y = make_moons(n_samples=1200, noise=0.25, random_state=42)
        X = X.astype(np.float32)
        desc = "Moons (2 feat, nonlinear)"
    elif name == "syn_hd":
        X, y = make_classification(n_samples=1500, n_features=30,
                                    n_informative=12, n_redundant=5,
                                    n_clusters_per_class=2, random_state=42)
        X = X.astype(np.float32)
        desc = "Synthetic-30feat (30 feat)"
    elif name == "california_large":
        d = fetch_california_housing()
        X = d.data.astype(np.float32)
        y = (d.target > np.median(d.target)).astype(int)
        desc = f"California Housing full (8 feat, n={len(X)})"
    elif name == "adult":
        bundle = fetch_openml(data_id=1590, as_frame=True, parser='auto')
        df = bundle.frame.copy()
        tgt_name = bundle.target.name if hasattr(bundle.target, 'name') else 'class'
        tgt = df.pop(tgt_name) if tgt_name in df.columns else bundle.target
        y_str = tgt.astype(str).str.strip().str.lower()
        y_full = y_str.isin(['1', '>50k', '>50k.']).astype(int).values
        for c in df.select_dtypes(include='number').columns:
            df[c] = df[c].fillna(df[c].median())
        for c in df.select_dtypes(exclude='number').columns:
            df[c] = df[c].astype(str).fillna('missing')
        df = pd.get_dummies(df, drop_first=True)
        X_full = df.values.astype(np.float32)
        rng_sub = np.random.default_rng(42)
        c0 = np.where(y_full==0)[0]; c1 = np.where(y_full==1)[0]
        n0 = min(3750, len(c0)); n1 = min(1250, len(c1))
        idx = np.concatenate([rng_sub.choice(c0, n0, replace=False),
                               rng_sub.choice(c1, n1, replace=False)])
        rng_sub.shuffle(idx)
        X = X_full[idx]; y = y_full[idx]
        desc = f"Adult income (mixed, n={len(X)}, {X.shape[1]} feat)"
    elif name == "credit_g":
        bundle = fetch_openml(data_id=31, as_frame=True, parser='auto')
        df = bundle.frame.copy()
        tgt_name = bundle.target.name if hasattr(bundle.target, 'name') else 'class'
        tgt = df.pop(tgt_name) if tgt_name in df.columns else bundle.target
        y_str = tgt.astype(str).str.strip().str.lower()
        y = y_str.isin(['1', 'good']).astype(int).values
        for c in df.select_dtypes(include='number').columns:
            df[c] = df[c].fillna(df[c].median())
        for c in df.select_dtypes(exclude='number').columns:
            df[c] = df[c].astype(str).fillna('missing')
        df = pd.get_dummies(df, drop_first=True)
        X = df.values.astype(np.float32)
        desc = f"German Credit (mixed, n={len(X)}, {X.shape[1]} feat)"

    elif name == "nigerian_fraud":
        print("\n  [builtin:nigerian_fraud] Loading from HuggingFace...")
        ds = load_dataset(
            "electricsheepafrica/Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset",
            split="train"
        )

        df = ds.to_pandas()
        MAX_ROWS = 7000
        if len(df) > MAX_ROWS:
            df = df.sample(n=MAX_ROWS, random_state=42)

        target_col = None
        for cand in ["is_fraud", "fraud", "label", "target", "Class"]:
            if cand in df.columns:
                target_col = cand
                break
        if target_col is None:
            target_col = df.columns[-1]

        print(f"        Using target column: {target_col}")

        y = df[target_col].astype(int).values
        Xdf = df.drop(columns=[target_col])

        drop_cols = []
        for c in Xdf.columns:
            if Xdf[c].nunique() > 0.9 * len(Xdf):
                drop_cols.append(c)
        if drop_cols:
            Xdf = Xdf.drop(columns=drop_cols)

        for c in Xdf.select_dtypes(include="number").columns:
            Xdf[c] = Xdf[c].fillna(Xdf[c].median())

        for c in Xdf.select_dtypes(exclude="number").columns:
            Xdf[c] = Xdf[c].fillna("missing")

        Xdf = pd.get_dummies(Xdf, drop_first=True)

        X = Xdf.values.astype(np.float32)

        Xtr, Xte, ytr, yte, sc = _split_scale(X, y)

        desc = f"Nigerian Fraud (n={len(X)}, {X.shape[1]} feat)"

        print(f"        train={Xtr.shape}  balance={ytr.mean():.3f}")

        return Xtr, Xte, ytr, yte, [], sc, False, None
    else:
        raise ValueError(f"Unknown built-in: '{name}'")

    Xtr, Xte, ytr, yte, sc = _split_scale(X, y)
    print(f"\n  [builtin:{name}] {desc}  n={X.shape[0]}  "
          f"train={Xtr.shape}  balance={ytr.mean():.2f}")
    # Return 8-tuple for consistency; is_regression=False, y_cont=None
    return Xtr, Xte, ytr, yte, [], sc, False, None


def load_csv(path, target_col=None):
    """Same as v2 but returns 8-tuple."""
    print(f"\n  [csv] {path}")
    df = pd.read_csv(path)
    print(f"        {df.shape[0]} rows × {df.shape[1]} cols")
    if target_col is None:
        for cand in ["loan_status","Loan_Status","deposit","default","Default",
                     "label","target","y","class","outcome"]:
            if cand in df.columns:
                target_col = cand; break
        if target_col is None:
            target_col = df.columns[-1]
    print(f"        Target: '{target_col}'")
    ys = df[target_col].copy()
    if ys.dtype == object or str(ys.dtype) == "category":
        pos = {"yes","y","1","true","paid","fully paid","current","no default"}
        yb  = ys.astype(str).str.strip().str.lower().isin(pos)
        ys  = yb.astype(int).values
    else:
        u = np.unique(ys.dropna())
        ys = (ys == u[-1]).astype(int).values if len(u) == 2 else \
             (ys > float(np.median(ys.dropna()))).astype(int).values
    Xdf = df.drop(columns=[target_col])
    drop = [c for c in Xdf.columns
            if (Xdf[c].dtype not in (object, "category") and Xdf[c].nunique() == len(Xdf))
            or (Xdf[c].dtype == object and Xdf[c].nunique() > 0.9 * len(Xdf))]
    if drop: Xdf = Xdf.drop(columns=drop)
    Xdf = Xdf.dropna(axis=1, how="all")
    for c in Xdf.select_dtypes(include="number").columns:
        Xdf[c] = Xdf[c].fillna(Xdf[c].median())
    for c in Xdf.select_dtypes(exclude="number").columns:
        Xdf[c] = Xdf[c].fillna("missing")
    Xdf = pd.get_dummies(Xdf)
    X   = Xdf.values.astype(np.float32)
    Xtr, Xte, ytr, yte = train_test_split(X, ys, test_size=0.2,
                                           random_state=42, stratify=ys)
    sc  = StandardScaler()
    Xtr = sc.fit_transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    print(f"        Features: {X.shape[1]}  Train: {Xtr.shape}  Balance: {ytr.mean():.2f}")
    return Xtr, Xte, ytr, yte, list(Xdf.columns), sc, False, None

#  ATTACK LIBRARY
def _flip_label(yi, classes, rng):
    others = classes[classes != yi]
    if len(others) == 0: return yi
    return int(rng.choice(others))


def apply_attack(name, X, y, fraction, seed=42, y_cont=None):
    """
    y_cont: optional continuous target for regression attacks.
    For regression attacks, y is the binarized label, y_cont is the
    original continuous target. Attacks modify X and y_cont, then
    re-binarize to get new y.
    """
    rng     = np.random.default_rng(seed)
    n       = max(1, int(len(X) * fraction))
    Xp, yp  = X.copy(), y.copy()
    y_cont_p = y_cont.copy() if y_cont is not None else None
    pidx    = rng.choice(len(X), n, replace=False)
    classes = np.unique(y)
    is_bin  = len(classes) == 2

    #  Regression-specific attacks
    if name == "target_shift":
        # Shift continuous targets by ±3σ (then re-binarize)
        if y_cont is None:
            # Fallback: treat as feat_perturb if no continuous target
            name = "feat_perturb"
        else:
            y_std = y_cont.std() + 1e-8
            for i in pidx:
                y_cont_p[i] += 3.0 * y_std * rng.choice([-1., 1.])
            # Re-binarize
            yp = (y_cont_p > 0).astype(int)
            return Xp, yp, pidx

    if name == "leverage_attack":
        # Move points to high-leverage positions AND shift their targets
        if y_cont is None:
            name = "feat_perturb"
        else:
            stds = X.std(axis=0) + 1e-8
            y_std = y_cont.std() + 1e-8
            for i in pidx:
                # Push features to extreme positions (high leverage)
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

class RPFExtractor:
    """
    Relational Poison Fingerprint v3 — 55-dimensional.

    Block A  [0:8]   kNN Label Consistency
    Block B  [8:17]  Class-Cond. Geometry
    Block C [17:21] Scale Anomaly
    Block D  [21:34] Cross-Val Influence
    Block E  [34:47] Local Anomaly Extended
    Block F  [47:55] Regression/Influence Features [NEW]
    """

    DIM = 61

    BLOCK_NAMES = {
        "A"  : ("kNN Label Consistency",      slice(0,  8)),
        "B"  : ("Class-Cond. Geometry",       slice(8,  17)),
        "C" : ("Scale Anomaly",              slice(17, 21)),
        "D"  : ("Cross-Val Influence",        slice(21, 34)),
        "E"  : ("Local Anomaly Extended",     slice(34, 47)),
        "F"  : ("Regression/Influence [NEW]", slice(47, 55)),
    }

    def __init__(self, k_small: int = 5, k_large: int = 15,
                 cv_folds: int = 5):
        self.k_small  = k_small
        self.k_large  = k_large
        self.cv_folds = cv_folds

    def extract(self, X: np.ndarray, y: np.ndarray,
                y_cont: np.ndarray = None) -> np.ndarray:
        """
        y_cont: optional continuous targets for regression features.
        If None, uses y (cast to float) for Block E regression features.
        """
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

    # ── Block B (unchanged) ───────────────────────────────────────────

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

    # ── Block C (unchanged) ─────────────────────────────────────────

    def _block_c(self, X, y, rpf):
        n       = len(X)
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
        """
        Regression-inspired influence features. Works for both
        classification (treats y as numeric) and regression (uses y_cont).

        Features:
          47: leverage (hat matrix diagonal)
          48: studentized residual (rank-normalized)
          49: Cook's distance (rank-normalized)
          50: DFFITS (rank-normalized)
          51: local residual consistency (kNN residual agreement)
          52: prediction rank within class
          53: residual-leverage interaction
          54: leave-one-out influence estimate
        """
        n, d = X.shape

        # Use continuous target if available, else cast labels to float
        y_reg = y_cont.astype(np.float64) if y_cont is not None \
                else y.astype(np.float64)

        # Fit ridge regression: y ~ X
        try:
            ridge = Ridge(alpha=1.0).fit(X.astype(np.float64), y_reg)
            y_hat = ridge.predict(X.astype(np.float64))
            resid = y_reg - y_hat
        except Exception:
            rpf[:, 47:55] = 0.0
            return

        # ── Leverage (hat matrix diagonal) ────────────────────────────
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

        # ── Studentized residual ──────────────────────────────────────
        mse = (resid ** 2).mean() + 1e-8
        stud_resid = resid / np.sqrt(mse * (1 - np.clip(H_diag, 0, 0.999) + 1e-8))
        rpf[:, 48] = rankdata(np.abs(stud_resid)).astype(np.float32) / n

        # ── Cook's distance ───────────────────────────────────────────
        p = min(d, n - 1)
        cooks_d = (stud_resid ** 2 / max(p, 1)) * (H_diag / (1 - H_diag + 1e-8))
        rpf[:, 49] = rankdata(cooks_d).astype(np.float32) / n

        # ── DFFITS ────────────────────────────────────────────────────
        dffits = stud_resid * np.sqrt(H_diag / (1 - H_diag + 1e-8))
        rpf[:, 50] = rankdata(np.abs(dffits)).astype(np.float32) / n

        # ── Local residual consistency ────────────────────────────────
        # Do neighbors have similar residual signs?
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

        # ── Prediction rank within class ──────────────────────────────
        pred_rank = np.zeros(n, np.float32)
        classes = np.unique(y)
        for c in classes:
            idx = np.where(y == c)[0]
            if len(idx) < 2: continue
            pred_rank[idx] = rankdata(y_hat[idx]).astype(np.float32) / len(idx)
        rpf[:, 52] = pred_rank

        # ── Residual × leverage interaction ───────────────────────────
        rpf[:, 53] = rankdata(np.abs(resid) * H_diag).astype(np.float32) / n

        # ── Leave-one-out influence estimate (approximate) ────────────
        # Sherman-Morrison approximation: Δŷ_i ≈ resid_i * h_ii / (1 - h_ii)
        loo_influence = np.abs(resid * H_diag / (1 - H_diag + 1e-8))
        rpf[:, 54] = rankdata(loo_influence).astype(np.float32) / n


# =============================================================================
#  IMPROVED RATE ESTIMATION
# =============================================================================

def _otsu_rate(scores: np.ndarray) -> float:
    """
    Otsu's method: find threshold that minimizes within-class variance
    of scores, treating them as two populations (clean vs poison).
    Returns estimated fraction of poison (above threshold).
    """
    sorted_s = np.sort(scores.astype(np.float64))
    n = len(sorted_s)
    if n < 10:
        return 0.10

    # Efficient Otsu: compute running stats
    cum_sum = np.cumsum(sorted_s)
    cum_sq  = np.cumsum(sorted_s ** 2)
    total   = cum_sum[-1]

    best_var = np.inf
    best_k   = n // 2

    # Only search in plausible range [50%, 98%] of sorted scores
    lo_idx = max(1, int(n * 0.50))
    hi_idx = min(n - 1, int(n * 0.98))

    for k in range(lo_idx, hi_idx):
        w0 = k / n
        w1 = 1.0 - w0
        if w0 <= 0 or w1 <= 0:
            continue

        m0 = cum_sum[k - 1] / k
        m1 = (total - cum_sum[k - 1]) / (n - k)

        v0 = cum_sq[k - 1] / k - m0 ** 2
        v1 = (cum_sq[-1] - cum_sq[k - 1]) / (n - k) - m1 ** 2

        within_var = w0 * max(v0, 0) + w1 * max(v1, 0)

        if within_var < best_var:
            best_var = within_var
            best_k   = k

    return (n - best_k) / n


def _bimodality_coefficient(scores: np.ndarray) -> float:
    """
    Sarle's bimodality coefficient. Values > 0.555 suggest bimodal
    distribution (clear separation between clean and poison scores).
    """
    from scipy.stats import skew, kurtosis
    n = len(scores)
    if n < 4:
        return 0.0
    g = float(skew(scores))
    k = float(kurtosis(scores, fisher=True))
    bc = (g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    return float(np.clip(bc, 0, 1))


def score_distribution_features(scores: np.ndarray) -> np.ndarray:
    """16 summary statistics of a score distribution."""
    s = np.sort(scores.astype(np.float64))
    n = len(s)

    pcts    = np.percentile(s, [50, 60, 70, 75, 80, 85, 90, 92, 95, 97, 99])
    diffs   = np.diff(s[max(0, n // 2):])
    top_gap = float(diffs.max()) if len(diffs) > 0 else 0.0

    return np.array([
        *pcts,
        float(s.mean()),
        float(s.std()),
        float((s > s.mean() + 2 * s.std()).mean()),
        top_gap,
        float(s[-1] - s[n // 2]),
    ], dtype=np.float32)


class RateEstimatorHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32), nn.LayerNorm(32), nn.GELU(),
            nn.Linear(32, 16), nn.GELU(),
            nn.Linear(16, 1),  nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1) * 0.40


def estimate_contamination_rate(scores: np.ndarray,
                                 lo: float = 0.01,
                                 hi: float = 0.40) -> float:
    """
    IMPROVED ensemble — 4 estimators + bimodality-aware weighting.

    When bimodality is weak (scores don't clearly separate), we
    down-weight estimators that tend to overestimate, fixing the
    massive overestimation at 5% for subtle attacks.
    """
    n        = len(scores)
    sorted_s = np.sort(scores)
    estimates = []
    weights   = []

    bc = _bimodality_coefficient(scores)

    # ── 1. GMM ────────────────────────────────────────────────────────
    try:
        gmm = GaussianMixture(
            n_components=2, random_state=42, max_iter=200,
            covariance_type="full", reg_covar=1e-4
        ).fit(scores.reshape(-1, 1))
        poison_comp = int(np.argmax(gmm.means_.flatten()))
        est_gmm = float(gmm.weights_[poison_comp])
        estimates.append(est_gmm)
        # GMM is more reliable when scores are bimodal
        weights.append(2.0 if bc > 0.555 else 0.5)
    except Exception:
        pass

    # ── 2. Score-gap ──────────────────────────────────────────────────
    start = max(1, int(n * 0.40))
    diffs = np.diff(sorted_s[start:])
    if len(diffs) > 0:
        gap_pos = int(np.argmax(diffs))
        n_above = len(diffs) - gap_pos
        est_gap = n_above / n
        estimates.append(est_gap)
        weights.append(1.5 if bc > 0.555 else 0.5)

    # ── 3. Otsu's method ─────────────────────────────────────────────
    est_otsu = _otsu_rate(scores)
    estimates.append(est_otsu)
    weights.append(2.0)  # Otsu is generally well-calibrated

    # ── 4. Excess-mass (anchor — biased toward conservative) ──────────
    half  = sorted_s[:max(10, n // 2)]
    mu    = float(half.mean())
    sigma = float(half.std()) + 1e-8
    est_em = float((scores > mu + 2.0 * sigma).mean())
    estimates.append(est_em)
    weights.append(1.0)

    if not estimates:
        return 0.10

    # ── Weighted median ───────────────────────────────────────────────
    # When bimodality is weak, shrink toward conservative estimate
    estimates = np.array(estimates)
    weights   = np.array(weights)

    if bc < 0.40:
        # Weak separation: poison signal is subtle, shrink estimates down
        # This fixes the overestimation at 5% for boundary flip etc.
        shrink_factor = 0.6
        estimates = estimates * shrink_factor

    # Weighted median
    sorted_idx = np.argsort(estimates)
    cum_w = np.cumsum(weights[sorted_idx])
    mid   = cum_w[-1] / 2.0
    median_idx = np.searchsorted(cum_w, mid)
    rate = float(estimates[sorted_idx[min(median_idx, len(estimates) - 1)]])

    rate = float(np.clip(rate, lo, hi))
    return rate


# =============================================================================
#  MetaPoisonNet v3  (input_dim=55)
# =============================================================================

class MetaPoisonNet(nn.Module):
    def __init__(self, input_dim: int = 55,
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


def focal_loss(logits, targets, gamma: float = 2.0, pos_weight: float = 4.0):
    pw  = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pw, reduction="none")
    pt   = torch.exp(-F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"))
    return ((1 - pt) ** gamma * bce).mean()


# =============================================================================
#  MetaPoisonDetector v3 — UPDATED for regression + improved rate est.
# =============================================================================

class MetaPoisonDetector:

    def __init__(self, device: str = "cpu", epochs: int = 400,
                 batch_size: int = 512, lr: float = 1e-3):
        self.device     = torch.device(device)
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.extractor  = RPFExtractor(k_small=RPF_K_SMALL,
                                        k_large=RPF_K_LARGE,
                                        cv_folds=RPF_CV_FOLDS)
        self.net        = MetaPoisonNet(input_dim=RPFExtractor.DIM).to(self.device)
        self.rate_head  = RateEstimatorHead().to(self.device)
        self._fitted    = False
        self._threshold = 0.5

    def meta_fit(self, datasets: dict, attacks=None, rates=None,
                 seeds: int = 3, verbose: bool = True):
        """
        datasets: dict of name → (X, y) OR name → (X, y, y_cont)
        """
        if attacks is None: attacks = META_TRAIN_ATTACKS
        if rates   is None: rates   = META_TRAIN_RATES

        n_combos = len(datasets) * len(attacks) * len(rates) * seeds
        if verbose:
            print(f"\n[meta-fit] {len(datasets)} datasets × {len(attacks)} attacks × "
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
            ds_attacks = (META_TRAIN_ATTACKS_REG if is_reg
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
                            cache_key = ("meta", ds_name, atk, rate, seed)
                            rpf = get_rpf_cached(self.extractor, Xp, yp,
                                                  cache_key, y_cont=y_cont)
                            rpf_pool.append(rpf)
                            label_pool.append(ytrue)
                            ok += 1
                        except Exception as e:
                            if verbose and "Unknown attack" in str(e):
                                pass  # skip silently
                            elif verbose:
                                print(f"[META-FIT FAIL] {ds_name}|{atk}|{rate}|{seed}: {e}")

        if not rpf_pool:
            raise RuntimeError("No RPF batches generated.")

        X_pool = np.concatenate(rpf_pool,   axis=0).astype(np.float32)
        y_pool = np.concatenate(label_pool, axis=0).astype(np.float32)
        elapsed = time.time() - t0
        if verbose:
            print(f"  → {ok} combos  |  "
                  f"pool: {len(X_pool):,} samples  |  "
                  f"poison frac: {y_pool.mean():.3f}  |  {elapsed:.1f}s")
            print(f"  Phase 2 — Training MetaPoisonNet ({self.epochs} epochs)...")

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
            ds_attacks = (META_TRAIN_ATTACKS_REG if is_reg else attacks)

            for atk in ds_attacks:
                for rate_val in rates:
                    for seed in range(seeds):
                        try:
                            Xp, yp, pidx = apply_attack(
                                atk, X, y, fraction=rate_val,
                                seed=seed * 1000 + hash(atk) % 997,
                                y_cont=y_cont)
                            if len(pidx) < 3: continue
                            cache_key = ("meta", ds_name, atk, rate_val, seed)
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
            "version":   3,
            "rate_head": self.rate_head.state_dict(),
        }, path)
        n = sum(p.numel() for p in self.net.parameters())
        print(f"\n[save] MetaPoison v3 → {path}  params={n:,}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        dim  = ckpt.get("rpf_dim", RPFExtractor.DIM)
        if dim != RPFExtractor.DIM:
            raise ValueError(
                f"Saved model has RPF dim={dim} but current code expects "
                f"{RPFExtractor.DIM}. Set FORCE_RETRAIN=True to retrain.")
        self.net = MetaPoisonNet(input_dim=dim).to(self.device)
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
        print(f"\n[load] MetaPoison v{v} ← {path}  "
              f"threshold={self._threshold:.4f}  rpf_dim={dim}")


# =============================================================================
#  HybridEnsembleDetector — UPDATED with improved rate estimation
# =============================================================================

class HybridEnsembleDetector:

    def __init__(self, meta: MetaPoisonDetector, fusion_w: float = 0.55):
        self.meta      = meta
        self.fusion_w  = fusion_w
        self.rate_head = None

    def _iso_scores(self, X):
        iso = IsolationForest(n_estimators=100, random_state=42, n_jobs=-1)
        return (-iso.fit(X).score_samples(X)).astype(np.float32)

    def _rank_fuse(self, meta_s, iso_s, w=None):
        if w is None: w = self.fusion_w
        mr = rankdata(meta_s).astype(np.float32) / len(meta_s)
        ir = rankdata(iso_s).astype(np.float32)  / len(iso_s)
        return w * mr + (1 - w) * ir

    def score(self, X, y, y_cont=None):
        meta_s, rpf = self.meta.score(X, y, y_cont=y_cont)
        iso_s       = self._iso_scores(X)
        combined    = self._rank_fuse(meta_s, iso_s)
        return combined, meta_s, iso_s

    def _estimate_rate(self, scores):
        ranked = rankdata(scores).astype(np.float32) / len(scores)

        if self.rate_head is not None:
            try:
                feats = score_distribution_features(ranked)
                t     = torch.tensor(feats, dtype=torch.float32,
                                     device=self.meta.device).unsqueeze(0)
                with torch.no_grad():
                    rate = float(self.rate_head(t).item())
                return float(np.clip(rate, 0.01, 0.40))
            except Exception:
                pass

        return estimate_contamination_rate(ranked)

    def attach_rate_head(self, rate_head):
        self.rate_head = rate_head
        self.rate_head.to(self.meta.device)
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

        meta_s_list, iso_s_list, y_list = [], [], []

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
                            rpf = get_rpf_cached(self.meta.extractor, Xp, yp,
                                                  cache_key, y_cont=y_cont)
                            ms      = self.meta._score_rpf(rpf)
                            iso_s   = self._iso_scores(Xp)
                            meta_s_list.append(ms)
                            iso_s_list.append(iso_s)
                            y_list.append(ytrue)
                        except Exception:
                            pass

        if not meta_s_list:
            if verbose: print("  [fusion-cal] No data — keeping default w=0.55")
            return

        best_w, best_f1 = self.fusion_w, 0.0
        grid = np.arange(0.30, 0.76, 0.05)

        for w in grid:
            f1s = []
            for ms, iso_s, yt in zip(meta_s_list, iso_s_list, y_list):
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
        self.meta.save(path.replace(".pt", "_base.pt"))
        payload = {"fusion_w": self.fusion_w}
        if self.rate_head is not None:
            payload["rate_head"] = self.rate_head.state_dict()
        torch.save(payload, path)
        print(f"  [fusion] w={self.fusion_w:.2f} → {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.meta.device)
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
    
#  EVALUATION — UPDATED for regression support
def _m(yt, yp, s=None):
    return {"F1":  f1_score(yt, yp, zero_division=0),
            "P":   precision_score(yt, yp, zero_division=0),
            "R":   recall_score(yt, yp, zero_division=0),
            "AUC": roc_auc_score(yt, s) if s is not None else 0.0}


def run_one(meta, hybrid, X_tr, y_tr, atk, rate, ds_key="", y_cont=None):
    label, has_flip = ATTACK_META[atk]
    Xp, yp, pidx    = apply_attack(atk, X_tr, y_tr, fraction=rate, y_cont=y_cont)
    ytrue           = np.zeros(len(Xp)); ytrue[pidx] = 1

    if pidx.shape[0] < 3 or len(Xp) < 30:
        return None

    cache_key = ("eval", ds_key, atk, rate)
    rpf = get_rpf_cached(meta.extractor, Xp, yp, cache_key, y_cont=y_cont)
    scores_m = meta._score_rpf(rpf)

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
        "MetaPoisonV3": _m(ytrue, pred_unk_m, scores_m),
        "Hybrid":       _m(ytrue, pred_unk_h, scores_h),
        "IsoForest":    _m(ytrue, iso_unk)
    }

    kno = {
        "MetaPoisonV3": _m(ytrue, pred_kno_m, scores_m),
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


# =============================================================================
#  ABLATION (unchanged logic, updated for v3 blocks)

def ablation_study(meta, X_tr, y_tr, attacks, rate=0.10, n_trials=3, y_cont=None):
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

                rpf_full  = meta.extractor.extract(Xp, yp, y_cont=y_cont)
                sc_full   = meta._score_rpf(rpf_full)
                pred_full = np.zeros(len(Xp), dtype=int)
                pred_full[np.argsort(sc_full)[-k:]] = 1
                f1_full   = f1_score(ytrue, pred_full, zero_division=0)

                for block in RPFExtractor.BLOCK_NAMES:
                    rpf_m  = meta.ablate_block(rpf_full, block)
                    sc_m   = meta._score_rpf(rpf_m)
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
        f1h = r.get("Hybrid", r.get("MetaPoisonV3", {})).get("F1_mean", 0.)
        print(f"  │  Hybrid visual    {_bar(f1h)}  {f1h:.3f}")
    print(f"  └{'─'*68}")

def significance_test(ds_res, mode_key="unknown"):
    meta_vals = []
    hybrid_vals = []

    for res in ds_res.values():
        if res is None:
            continue

        if "MetaPoisonV3" in res[mode_key] and "Hybrid" in res[mode_key]:
            meta_vals.extend(res[mode_key]["MetaPoisonV3"]["F1_all"])
            hybrid_vals.extend(res[mode_key]["Hybrid"]["F1_all"])

    if len(meta_vals) > 1:
        t_stat, p_t = ttest_rel(hybrid_vals, meta_vals)
        w_stat, p_w = wilcoxon(hybrid_vals, meta_vals)

        print(f"\n  Statistical Test (Hybrid vs MetaPoisonV3):")
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

#  RPFNet
#
#  Append this block AFTER the main evaluation loop in metapoison_v3.py,
#  right before or after print_grand_summary(grand_results, eval_ds).
#
#  It reuses: grand_results, eval_ds, meta_det, hybrid_det, RATES,
#             ATTACKS, REGRESSION_ATTACKS, ATTACK_META, RPFExtractor,
#             ASSUMED_RATE, META_TRAIN_DATASETS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_curve, auc, precision_recall_curve

PLOT_DIR = "./figures"
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Consistent style ─────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.1,
})

COLORS = {
    "Hybrid":       "#2563EB",
    "MetaPoisonV3": "#7C3AED",
    "IsoForest":    "#6B7280",
    "LOF":          "#9CA3AF",
    "OCSVM":        "#D1D5DB",
    "SEVER":        "#F59E0B",
    "Influence":    "#EF4444",
    "SpecSig":      "#10B981",
    "PCA-KNN":      "#F97316",
    "TRIM":         "#EC4899",
    "iTRIM":        "#A855F7",
    "AE-Recon":     "#06B6D4",
}

def _get_color(d):
    return COLORS.get(d, "#888888")


# =====================================================================
#  FIGURE 1 — F1 Heatmap: Attacks × Detectors (per dataset or grand)
# =====================================================================

def plot_f1_heatmap(grand_results, eval_ds, mode_key="unknown"):
    """
    One heatmap per dataset: rows = attacks, cols = detectors, cell = F1.
    Also produces a grand-average heatmap across all datasets.
    """
    for ds_disp, ds_res in grand_results.items():
        valid = {k: v for k, v in ds_res.items() if v}
        if not valid:
            continue

        dets = []
        for res in valid.values():
            for d in res[mode_key]:
                if d not in dets:
                    dets.append(d)

        attacks_sorted = sorted(valid.keys(), key=lambda x: x[1])
        row_labels = [f"{ATTACK_META[a][0][:22]} @{r:.0%}" for a, r in attacks_sorted]
        matrix = np.full((len(attacks_sorted), len(dets)), np.nan)

        for ri, key in enumerate(attacks_sorted):
            res = valid[key]
            for ci, d in enumerate(dets):
                if d in res[mode_key]:
                    matrix[ri, ci] = res[mode_key][d]["F1_mean"]

        fig, ax = plt.subplots(figsize=(max(6, len(dets) * 1.1),
                                         max(4, len(row_labels) * 0.38)))
        cmap = LinearSegmentedColormap.from_list("rg", ["#fee2e2","#fef9c3","#dcfce7"], N=256)
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap,
                    xticklabels=dets, yticklabels=row_labels,
                    vmin=0, vmax=1, linewidths=0.5, linecolor="white",
                    ax=ax, cbar_kws={"label": "F1 Score"})
        ax.set_title(f"F1 Scores — {ds_disp} ({mode_key} rate)", fontweight="bold")
        ax.set_xlabel("Detector")
        ax.set_ylabel("")
        plt.tight_layout()
        safe = ds_disp.replace(" ", "_").lower()
        path = f"{PLOT_DIR}/fig1_heatmap_{safe}_{mode_key}.pdf"
        fig.savefig(path)
        plt.close(fig)
        print(f"  [fig1] Saved {path}")

    # ── Grand-average heatmap ─────────────────────────────────────
    all_dets = []
    for dr in grand_results.values():
        for res in dr.values():
            if res:
                for d in res[mode_key]:
                    if d not in all_dets:
                        all_dets.append(d)

    ds_names = list(grand_results.keys())
    matrix_g = np.full((len(ds_names), len(all_dets)), np.nan)
    for ri, ds_disp in enumerate(ds_names):
        for ci, d in enumerate(all_dets):
            vals = [r[mode_key][d]["F1_mean"]
                    for r in grand_results[ds_disp].values()
                    if r and d in r.get(mode_key, {})]
            if vals:
                matrix_g[ri, ci] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(max(6, len(all_dets) * 1.1),
                                     max(3, len(ds_names) * 0.5)))
    cmap = LinearSegmentedColormap.from_list("rg", ["#fee2e2","#fef9c3","#dcfce7"], N=256)
    sns.heatmap(matrix_g, annot=True, fmt=".2f", cmap=cmap,
                xticklabels=all_dets, yticklabels=ds_names,
                vmin=0, vmax=1, linewidths=0.5, linecolor="white",
                ax=ax, cbar_kws={"label": "Mean F1"})
    ax.set_title(f"Mean F1 by Dataset × Detector ({mode_key} rate)", fontweight="bold")
    plt.tight_layout()
    path = f"{PLOT_DIR}/fig1_heatmap_grand_{mode_key}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig1] Saved {path}")


# =====================================================================
#  FIGURE 2 — Grouped Bar: Detector comparison across poison rates
# =====================================================================

def plot_rate_comparison(grand_results, mode_key="unknown"):
    """
    For each rate, show average F1 of each detector across all
    datasets/attacks at that rate. Grouped bar chart.
    """
    all_dets = []
    for dr in grand_results.values():
        for res in dr.values():
            if res:
                for d in res[mode_key]:
                    if d not in all_dets:
                        all_dets.append(d)

    rates_seen = sorted(set(r for dr in grand_results.values()
                             for (_, r) in dr.keys()))
    rate_det_f1 = {r: {d: [] for d in all_dets} for r in rates_seen}

    for dr in grand_results.values():
        for (atk, rate), res in dr.items():
            if res is None:
                continue
            for d in all_dets:
                if d in res[mode_key]:
                    rate_det_f1[rate][d].append(res[mode_key][d]["F1_mean"])

    x = np.arange(len(rates_seen))
    n_det = len(all_dets)
    width = 0.8 / n_det

    fig, ax = plt.subplots(figsize=(max(7, len(rates_seen) * 1.8), 4.5))
    for i, d in enumerate(all_dets):
        means = [np.mean(rate_det_f1[r][d]) if rate_det_f1[r][d] else 0
                 for r in rates_seen]
        stds  = [np.std(rate_det_f1[r][d])  if rate_det_f1[r][d] else 0
                 for r in rates_seen]
        ax.bar(x + i * width - 0.4 + width / 2, means, width,
               yerr=stds, label=d, color=_get_color(d),
               edgecolor="white", linewidth=0.5, capsize=2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.0%}" for r in rates_seen])
    ax.set_xlabel("Poison Rate")
    ax.set_ylabel("Mean F1 Score")
    ax.set_title(f"Detector Performance by Poison Rate ({mode_key} rate)", fontweight="bold")
    ax.legend(loc="upper left", ncol=min(4, n_det), framealpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{PLOT_DIR}/fig2_rate_comparison_{mode_key}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig2] Saved {path}")


# =====================================================================
#  FIGURE 3 — Ablation Study Bar Chart (Block Importance)
# =====================================================================

def plot_ablation(meta_det, eval_ds):
    """
    Bar chart showing F1 drop when each RPF block is zeroed out,
    averaged across datasets.
    """
    block_drops_all = {b: [] for b in RPFExtractor.BLOCK_NAMES}

    for ds_key, dsinfo in eval_ds.items():
        Xtr, ytr = dsinfo["X_tr"], dsinfo["y_tr"]
        y_cont   = dsinfo.get("y_cont")
        is_reg   = dsinfo.get("is_regression", False)
        abl_attacks = (["feat_perturb", "repr_inversion", "dist_shift"]
                       if is_reg else
                       ["label_flip", "feat_perturb", "repr_inversion", "dist_shift"])
        try:
            abl = ablation_study(meta_det, Xtr, ytr, abl_attacks,
                                  rate=0.10, y_cont=y_cont)
            for b in RPFExtractor.BLOCK_NAMES:
                if b in abl:
                    block_drops_all[b].append(abl[b])
        except Exception:
            pass

    blocks  = list(RPFExtractor.BLOCK_NAMES.keys())
    descs   = [RPFExtractor.BLOCK_NAMES[b][0] for b in blocks]
    means   = [np.mean(block_drops_all[b]) if block_drops_all[b] else 0
               for b in blocks]
    stds    = [np.std(block_drops_all[b])  if block_drops_all[b] else 0
               for b in blocks]

    colors = ["#3B82F6", "#8B5CF6", "#EC4899", "#F59E0B", "#10B981", "#EF4444"]
    while len(colors) < len(blocks):
        colors.append("#6B7280")

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(range(len(blocks)), means, xerr=stds,
                   color=colors[:len(blocks)], edgecolor="white",
                   linewidth=0.5, capsize=3)
    ax.set_yticks(range(len(blocks)))
    ax.set_yticklabels([f"Block {b}: {d}" for b, d in zip(blocks, descs)],
                       fontsize=9)
    ax.set_xlabel("Mean F1 Drop When Block Zeroed")
    ax.set_title("RPF Block Ablation Study", fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    for bar, m in zip(bars, means):
        if m > 0.005:
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"+{m:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    path = f"{PLOT_DIR}/fig3_ablation.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig3] Saved {path}")


# =====================================================================
#  FIGURE 4 — ROC Curves (Hybrid vs baselines, representative attack)
# =====================================================================

def plot_roc_curves(meta_det, hybrid_det, eval_ds):
    """
    ROC curves for Hybrid, MetaPoison, IsoForest on a representative
    dataset + attack. Picks the first classification dataset available.
    """
    # Pick a representative dataset
    ds_key = None
    for k, v in eval_ds.items():
        if not v.get("is_regression", False):
            ds_key = k
            break
    if ds_key is None:
        ds_key = list(eval_ds.keys())[0]

    dsinfo  = eval_ds[ds_key]
    Xtr     = dsinfo["X_tr"]
    ytr     = dsinfo["y_tr"]
    y_cont  = dsinfo.get("y_cont")
    ds_disp = dsinfo["display"]

    # Pick representative attacks at 10%
    rep_attacks = ["label_flip", "feat_perturb", "clean_label", "backdoor"]
    if dsinfo.get("is_regression"):
        rep_attacks = ["target_shift", "feat_perturb", "leverage_attack"]

    fig, axes = plt.subplots(1, min(4, len(rep_attacks)),
                              figsize=(min(4, len(rep_attacks)) * 3.5, 3.5),
                              sharey=True)
    if len(rep_attacks) == 1:
        axes = [axes]

    for ax_idx, atk in enumerate(rep_attacks[:4]):
        if atk not in ATTACK_META:
            continue
        ax = axes[ax_idx]
        try:
            Xp, yp, pidx = apply_attack(atk, Xtr, ytr, fraction=0.10,
                                          y_cont=y_cont)
            ytrue = np.zeros(len(Xp))
            ytrue[pidx] = 1

            # Hybrid scores
            combined, meta_s, iso_s = hybrid_det.score(Xp, yp, y_cont=y_cont)

            for label, scores, color, ls in [
                ("Hybrid",       combined, COLORS["Hybrid"],       "-"),
                ("MetaPoisonV3", meta_s,   COLORS["MetaPoisonV3"], "--"),
                ("IsoForest",    iso_s,    COLORS["IsoForest"],    ":"),
            ]:
                fpr, tpr, _ = roc_curve(ytrue, scores)
                roc_auc_val = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=1.5,
                        label=f"{label} (AUC={roc_auc_val:.2f})")

            ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.5)
            ax.set_xlabel("FPR")
            if ax_idx == 0:
                ax.set_ylabel("TPR")
            ax.set_title(ATTACK_META[atk][0][:20], fontsize=10)
            ax.legend(fontsize=7, loc="lower right")
            ax.grid(alpha=0.2)
        except Exception as e:
            ax.set_title(f"{atk} (failed)")

    fig.suptitle(f"ROC Curves — {ds_disp} @ 10% poison", fontweight="bold", y=1.02)
    plt.tight_layout()
    path = f"{PLOT_DIR}/fig4_roc_{ds_key}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig4] Saved {path}")


# =====================================================================
#  FIGURE 5 — Box Plot: F1 Distribution per Detector
# =====================================================================

def plot_f1_boxplot(grand_results, mode_key="unknown"):
    """
    Box plot showing F1 distribution across all attacks/rates for
    each detector. Highlights spread and robustness.
    """
    all_dets = []
    for dr in grand_results.values():
        for res in dr.values():
            if res:
                for d in res[mode_key]:
                    if d not in all_dets:
                        all_dets.append(d)

    det_vals = {d: [] for d in all_dets}
    for dr in grand_results.values():
        for res in dr.values():
            if res is None:
                continue
            for d in all_dets:
                if d in res[mode_key]:
                    det_vals[d].append(res[mode_key][d]["F1_mean"])

    # Sort detectors by median F1 (descending)
    sorted_dets = sorted(all_dets,
                          key=lambda d: np.median(det_vals[d]) if det_vals[d] else 0,
                          reverse=True)

    fig, ax = plt.subplots(figsize=(max(6, len(sorted_dets) * 0.8), 4.5))
    bp_data = [det_vals[d] for d in sorted_dets]
    bp = ax.boxplot(bp_data, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                    markeredgecolor="black", markersize=4),
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(linewidth=0.8),
                    flierprops=dict(markersize=3, alpha=0.5))

    for patch, d in zip(bp["boxes"], sorted_dets):
        patch.set_facecolor(_get_color(d))
        patch.set_alpha(0.8)

    ax.set_xticklabels(sorted_dets, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score")
    ax.set_title(f"F1 Distribution Across All Attacks ({mode_key} rate)",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(-0.02, 1.05)
    plt.tight_layout()
    path = f"{PLOT_DIR}/fig5_boxplot_{mode_key}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig5] Saved {path}")


# =====================================================================
#  FIGURE 6 — Flip vs No-Flip Attack Comparison
# =====================================================================

def plot_flip_vs_noflip(grand_results, mode_key="unknown"):
    """
    Grouped bars: each detector's mean F1 on flip attacks vs no-flip
    attacks. This highlights TRIM/iTRIM's known weakness.
    """
    all_dets = []
    for dr in grand_results.values():
        for res in dr.values():
            if res:
                for d in res[mode_key]:
                    if d not in all_dets:
                        all_dets.append(d)

    flip_vals   = {d: [] for d in all_dets}
    noflip_vals = {d: [] for d in all_dets}

    for dr in grand_results.values():
        for res in dr.values():
            if res is None:
                continue
            bucket = flip_vals if res["has_flip"] else noflip_vals
            for d in all_dets:
                if d in res[mode_key]:
                    bucket[d].append(res[mode_key][d]["F1_mean"])

    x = np.arange(len(all_dets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, len(all_dets) * 0.9), 4.5))
    means_f  = [np.mean(flip_vals[d])   if flip_vals[d]   else 0 for d in all_dets]
    means_nf = [np.mean(noflip_vals[d]) if noflip_vals[d] else 0 for d in all_dets]

    ax.bar(x - width / 2, means_f,  width, label="Label-flip attacks",
           color="#3B82F6", edgecolor="white")
    ax.bar(x + width / 2, means_nf, width, label="No-flip attacks (harder)",
           color="#EF4444", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(all_dets, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean F1 Score")
    ax.set_title("Flip vs No-Flip Attack Performance", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{PLOT_DIR}/fig6_flip_vs_noflip_{mode_key}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig6] Saved {path}")


# =====================================================================
#  FIGURE 7 — Zero-Shot vs Seen Dataset Performance
# =====================================================================

def plot_zeroshot_vs_seen(grand_results, eval_ds, mode_key="unknown"):
    """
    Compares mean F1 for seen (meta-trained) vs zero-shot datasets.
    """
    all_dets = []
    for dr in grand_results.values():
        for res in dr.values():
            if res:
                for d in res[mode_key]:
                    if d not in all_dets:
                        all_dets.append(d)

    seen_vals = {d: [] for d in all_dets}
    zs_vals   = {d: [] for d in all_dets}

    zs_disps = {v["display"] for v in eval_ds.values() if v["zero_shot"]}

    for ds_disp, dr in grand_results.items():
        bucket = zs_vals if ds_disp in zs_disps else seen_vals
        for res in dr.values():
            if res is None:
                continue
            for d in all_dets:
                if d in res[mode_key]:
                    bucket[d].append(res[mode_key][d]["F1_mean"])

    x = np.arange(len(all_dets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, len(all_dets) * 0.9), 4.5))
    means_s = [np.mean(seen_vals[d]) if seen_vals[d] else 0 for d in all_dets]
    means_z = [np.mean(zs_vals[d])   if zs_vals[d]   else 0 for d in all_dets]

    ax.bar(x - width / 2, means_s, width, label="Seen (meta-trained)",
           color="#8B5CF6", edgecolor="white")
    ax.bar(x + width / 2, means_z, width, label="Zero-shot ⭐",
           color="#F59E0B", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(all_dets, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean F1 Score")
    ax.set_title("Generalization: Seen vs Zero-Shot Datasets", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{PLOT_DIR}/fig7_zeroshot_{mode_key}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig7] Saved {path}")


# =====================================================================
#  FIGURE 8 — Feature Importance (RPF dimensions)
# =====================================================================

def plot_feature_importance(meta_det, eval_ds):
    """
    Gradient-based feature importance across RPF dimensions,
    color-coded by block.
    """
    all_imp = []
    for ds_key, dsinfo in eval_ds.items():
        try:
            imp = meta_det.feature_importance(dsinfo["X_tr"], dsinfo["y_tr"],
                                               y_cont=dsinfo.get("y_cont"))
            all_imp.append(imp)
        except Exception:
            pass

    if not all_imp:
        print("  [fig8] No feature importance data — skipping")
        return

    mean_imp = np.mean(all_imp, axis=0)
    dim = len(mean_imp)

    block_colors = {
        "A": "#3B82F6", "B": "#8B5CF6", "C": "#EC4899",
        "D": "#F59E0B", "E": "#10B981", "F": "#EF4444",
    }
    colors = []
    block_labels = []
    for b, (desc, sl) in RPFExtractor.BLOCK_NAMES.items():
        for i in range(sl.start, min(sl.stop, dim)):
            colors.append(block_colors.get(b, "#6B7280"))
            block_labels.append(b)

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(range(dim), mean_imp[:dim], color=colors[:dim],
                  edgecolor="white", linewidth=0.3)
    ax.set_xlabel("RPF Feature Index")
    ax.set_ylabel("Mean |Gradient| Importance")
    ax.set_title("RPF Feature Importance (gradient-based)", fontweight="bold")

    # Add block boundary annotations
    for b, (desc, sl) in RPFExtractor.BLOCK_NAMES.items():
        mid = (sl.start + min(sl.stop, dim)) / 2
        ax.axvline(sl.start - 0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.text(mid, ax.get_ylim()[1] * 0.95, f"Block {b}",
                ha="center", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=block_colors.get(b, "#ccc"),
                          alpha=0.3))

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{PLOT_DIR}/fig8_feature_importance.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig8] Saved {path}")


# =====================================================================
#  FIGURE 9 — Rate Estimation Calibration (estimated vs true rate)
# =====================================================================

def plot_rate_calibration(meta_det, eval_ds):
    """
    Scatter: true poison rate vs estimated rate. Perfect calibration
    = diagonal line.
    """
    true_rates = []
    est_rates  = []

    rates_to_test = [0.01, 0.05, 0.10, 0.15, 0.20]
    test_attacks = ["label_flip", "feat_perturb", "backdoor", "repr_inversion"]

    for ds_key, dsinfo in eval_ds.items():
        if dsinfo.get("is_regression"):
            continue  # stick to classification for cleaner calibration plot
        Xtr, ytr = dsinfo["X_tr"], dsinfo["y_tr"]
        for atk in test_attacks:
            if atk not in ATTACK_META:
                continue
            for rate in rates_to_test:
                try:
                    Xp, yp, pidx = apply_attack(atk, Xtr, ytr, fraction=rate)
                    scores, _ = meta_det.score(Xp, yp)
                    est = estimate_contamination_rate(scores)
                    true_rates.append(rate)
                    est_rates.append(est)
                except Exception:
                    pass

    if not true_rates:
        print("  [fig9] No calibration data — skipping")
        return

    true_rates = np.array(true_rates)
    est_rates  = np.array(est_rates)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true_rates + np.random.normal(0, 0.002, len(true_rates)),
               est_rates, alpha=0.4, s=20, c="#2563EB", edgecolors="white",
               linewidth=0.3)
    ax.plot([0, 0.25], [0, 0.25], "k--", linewidth=1, label="Perfect calibration")
    ax.set_xlabel("True Poison Rate")
    ax.set_ylabel("Estimated Poison Rate")
    ax.set_title("Rate Estimation Calibration", fontweight="bold")
    ax.set_xlim(-0.01, 0.25)
    ax.set_ylim(-0.01, 0.45)
    ax.legend()
    ax.grid(alpha=0.3)

    mae = float(np.abs(true_rates - est_rates).mean())
    ax.text(0.02, 0.40, f"MAE = {mae:.3f}", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    path = f"{PLOT_DIR}/fig9_rate_calibration.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig9] Saved {path}")


#  FIGURE 10 — Radar Chart: Detector profiles across attack families

def plot_radar(grand_results, mode_key="unknown"):
    """
    Radar/spider chart comparing top detectors across attack categories.
    """
    # Group attacks into families
    families = {
        "Label Flip":    ["label_flip", "boundary_flip", "targeted_class",
                          "target_flip_extreme"],
        "Feature Pert.": ["feat_perturb", "gauss_noise", "feat_dropout"],
        "Clean-Label":   ["clean_label", "interpolation"],
        "Backdoor":      ["backdoor", "backdoor_heavy"],
        "Structural":    ["repr_inversion", "dist_shift", "outlier_inject"],
        "Regression":    ["target_shift", "leverage_attack"],
    }

    # Collect per-family F1 for key detectors
    key_dets = ["Hybrid", "MetaPoisonV3", "IsoForest", "SEVER", "LOF"]
    # Only keep detectors that actually appear
    available_dets = set()
    for dr in grand_results.values():
        for res in dr.values():
            if res:
                available_dets.update(res[mode_key].keys())
    key_dets = [d for d in key_dets if d in available_dets]

    fam_det_f1 = {fam: {d: [] for d in key_dets} for fam in families}

    for dr in grand_results.values():
        for (atk, rate), res in dr.items():
            if res is None:
                continue
            for fam, fam_attacks in families.items():
                if atk in fam_attacks:
                    for d in key_dets:
                        if d in res[mode_key]:
                            fam_det_f1[fam][d].append(res[mode_key][d]["F1_mean"])

    # Filter out families with no data
    active_fams = [f for f in families if any(fam_det_f1[f][d] for d in key_dets)]
    if len(active_fams) < 3:
        print("  [fig10] Not enough attack families — skipping radar")
        return

    N = len(active_fams)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for d in key_dets:
        values = [np.mean(fam_det_f1[f][d]) if fam_det_f1[f][d] else 0
                  for f in active_fams]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, label=d, color=_get_color(d))
        ax.fill(angles, values, alpha=0.08, color=_get_color(d))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(active_fams, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Detector Profiles Across Attack Families", fontweight="bold",
                 pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    path = f"{PLOT_DIR}/fig10_radar_{mode_key}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig10] Saved {path}")


#  FIGURE 11 — Classification vs Regression dataset performance

def plot_cls_vs_reg(grand_results, eval_ds, mode_key="unknown"):
    """
    Compares detector F1 on classification vs regression datasets.
    Only relevant if regression datasets exist.
    """
    cls_disps = {v["display"] for v in eval_ds.values() if not v.get("is_regression")}
    reg_disps = {v["display"] for v in eval_ds.values() if v.get("is_regression")}

    if not reg_disps:
        print("  [fig11] No regression datasets — skipping")
        return

    all_dets = []
    for dr in grand_results.values():
        for res in dr.values():
            if res:
                for d in res[mode_key]:
                    if d not in all_dets:
                        all_dets.append(d)

    cls_vals = {d: [] for d in all_dets}
    reg_vals = {d: [] for d in all_dets}

    for ds_disp, dr in grand_results.items():
        bucket = reg_vals if ds_disp in reg_disps else cls_vals
        for res in dr.values():
            if res is None:
                continue
            for d in all_dets:
                if d in res[mode_key]:
                    bucket[d].append(res[mode_key][d]["F1_mean"])

    x = np.arange(len(all_dets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, len(all_dets) * 0.9), 4.5))
    means_c = [np.mean(cls_vals[d]) if cls_vals[d] else 0 for d in all_dets]
    means_r = [np.mean(reg_vals[d]) if reg_vals[d] else 0 for d in all_dets]

    ax.bar(x - width / 2, means_c, width, label="Classification",
           color="#3B82F6", edgecolor="white")
    ax.bar(x + width / 2, means_r, width, label="Regression 📈",
           color="#10B981", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(all_dets, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean F1 Score")
    ax.set_title("Classification vs Regression Dataset Performance", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{PLOT_DIR}/fig11_cls_vs_reg_{mode_key}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig11] Saved {path}")


#  FIGURE 12 — Score Distribution (clean vs poison)

def plot_score_distributions(meta_det, hybrid_det, eval_ds):
    """
    Overlapping histograms of Hybrid scores for clean vs poisoned points.
    Shows separation quality on a representative attack.
    """
    ds_key = None
    for k, v in eval_ds.items():
        if not v.get("is_regression", False):
            ds_key = k
            break
    if ds_key is None:
        ds_key = list(eval_ds.keys())[0]

    dsinfo = eval_ds[ds_key]
    Xtr, ytr = dsinfo["X_tr"], dsinfo["y_tr"]
    y_cont = dsinfo.get("y_cont")

    attacks_to_show = ["label_flip", "clean_label", "feat_perturb", "backdoor"]
    if dsinfo.get("is_regression"):
        attacks_to_show = ["target_shift", "feat_perturb", "leverage_attack",
                           "backdoor"]

    attacks_to_show = [a for a in attacks_to_show if a in ATTACK_META]

    fig, axes = plt.subplots(1, min(4, len(attacks_to_show)),
                              figsize=(min(4, len(attacks_to_show)) * 3.5, 3),
                              sharey=True)
    if not hasattr(axes, '__len__'):
        axes = [axes]

    for ax_idx, atk in enumerate(attacks_to_show[:4]):
        ax = axes[ax_idx]
        try:
            Xp, yp, pidx = apply_attack(atk, Xtr, ytr, fraction=0.10,
                                          y_cont=y_cont)
            ytrue = np.zeros(len(Xp))
            ytrue[pidx] = 1

            combined, _, _ = hybrid_det.score(Xp, yp, y_cont=y_cont)

            clean_s  = combined[ytrue == 0]
            poison_s = combined[ytrue == 1]

            ax.hist(clean_s, bins=40, alpha=0.6, color="#3B82F6",
                    label="Clean", density=True)
            ax.hist(poison_s, bins=40, alpha=0.6, color="#EF4444",
                    label="Poison", density=True)
            ax.set_xlabel("Hybrid Score")
            if ax_idx == 0:
                ax.set_ylabel("Density")
            ax.set_title(ATTACK_META[atk][0][:20], fontsize=10)
            ax.legend(fontsize=7)
        except Exception:
            ax.set_title(f"{atk} (failed)")

    fig.suptitle(f"Score Distributions — {dsinfo['display']} @ 10%",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    path = f"{PLOT_DIR}/fig12_score_dist_{ds_key}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig12] Saved {path}")

# Generate all figures

def generate_all_figures(grand_results, eval_ds, meta_det, hybrid_det):
    """Call this at the end of __main__ to produce all paper figures."""
    print("\n" + "=" * 82)
    print("  GENERATING PAPER FIGURES")
    print("=" * 82)

    plot_f1_heatmap(grand_results, eval_ds, "unknown")
    plot_f1_heatmap(grand_results, eval_ds, "known")
    plot_rate_comparison(grand_results, "unknown")
    plot_rate_comparison(grand_results, "known")
    plot_ablation(meta_det, eval_ds)
    plot_roc_curves(meta_det, hybrid_det, eval_ds)
    plot_f1_boxplot(grand_results, "unknown")
    plot_f1_boxplot(grand_results, "known")
    plot_flip_vs_noflip(grand_results, "unknown")
    plot_zeroshot_vs_seen(grand_results, eval_ds, "unknown")
    plot_feature_importance(meta_det, eval_ds)
    plot_rate_calibration(meta_det, eval_ds)
    plot_radar(grand_results, "unknown")
    plot_cls_vs_reg(grand_results, eval_ds, "unknown")
    plot_score_distributions(meta_det, hybrid_det, eval_ds)

    print(f"\n  ✓ All figures saved to {PLOT_DIR}/")
    print("=" * 82)

#  MAIN
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(42)
    W = 82

    print("=" * W)
    print("  MetaPoison v3: Universal Poison Detection + Regression Support")
    print(f"  RPF dims : {RPFExtractor.DIM}  (v3=55, +Block E Influence)")
    print(f"  Device   : {device}   Attacks: {len(ATTACKS)}   Rates: {RATES}")
    print(f"  Model    : {META_MODEL_PATH}")
    print(f"  Meta-train datasets: {len(META_TRAIN_DATASETS)} (expanded for OOD generalization)")
    print("=" * W)

    # Load meta-training datasets
    print(f"\n  Meta-training datasets: {META_TRAIN_DATASETS}")
    meta_data = {}
    for bname in META_TRAIN_DATASETS:
        try:
            result = load_builtin(bname)
            Xtr = result[0]
            ytr = result[2]
            is_reg = result[6]
            y_cont = result[7]
            if is_reg and y_cont is not None:
                meta_data[bname] = (Xtr, ytr, y_cont)
            else:
                meta_data[bname] = (Xtr, ytr)
        except Exception as e:
            print(f"  ⚠ {bname}: {e}")

    if not meta_data:
        raise RuntimeError("No meta-training datasets loaded.")

    print(f"\n  Successfully loaded {len(meta_data)}/{len(META_TRAIN_DATASETS)} "
          f"meta-training datasets")

    # Train or load
    meta_det = MetaPoisonDetector(device=device, epochs=META_EPOCHS,
                                   batch_size=META_BATCH_SIZE, lr=META_LR)
    hybrid_det = HybridEnsembleDetector(meta_det)
    fusion_path = META_MODEL_PATH.replace(".pt", "_fusion.pt")

    if not FORCE_RETRAIN and os.path.exists(META_MODEL_PATH):
        meta_det.load(META_MODEL_PATH)
        if os.path.exists(fusion_path):
            hybrid_det.load(fusion_path)
        else:
            hybrid_det.calibrate_fusion_weight(meta_data, verbose=True)
    else:
        print(f"\n  Meta-training on: {list(meta_data.keys())}")
        meta_det.meta_fit(meta_data, attacks=META_TRAIN_ATTACKS,
                          rates=META_TRAIN_RATES, seeds=META_TRAIN_SEEDS,
                          verbose=True)
        meta_det.save(META_MODEL_PATH)
        hybrid_det.attach_rate_head(meta_det.rate_head)
        hybrid_det.calibrate_fusion_weight(meta_data, verbose=True)
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
                "zero_shot":     bname not in META_TRAIN_DATASETS,
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
            if atk not in ATTACK_META:
                continue
            for rate in RATES:
                label, has_flip = ATTACK_META[atk]
                tag = "" if has_flip else " [no-flip]"
                print(f"\n  {label} @ {rate:.0%}{tag}... ", end="", flush=True)
                try:
                    seed_results = []

                    for s in range(EVAL_SEEDS):
                        try:
                            res_s = run_one(
                                meta_det, hybrid_det,
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
                        f1m = res["unknown"]["MetaPoisonV3"]["F1_mean"]
                        f1i = res["unknown"]["IsoForest"]["F1_mean"]
                        print(f"Hybrid={f1h:.3f}  Meta={f1m:.3f}  Iso={f1i:.3f}")
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
            abl = ablation_study(meta_det, Xtr, ytr, abl_attacks,
                                  rate=0.10, y_cont=y_cont)
            print(f"  {'Block':>8}  {'Description':<32}  {'F1 drop':>8}  Importance")
            print(f"  {'─'*70}")
            for bk, (bdesc, _) in RPFExtractor.BLOCK_NAMES.items():
                drop = abl.get(bk, 0.)
                bar  = "█" * max(0, int(drop * 50)) if drop > 0 else "─"
                flag = " ← NEW" if bk == "E" else ""
                print(f"  Block {bk:>2}  {bdesc[:32]:<32}  {drop:>+.4f}  {bar}{flag}")
        except Exception as e:
            print(f"  (ablation failed: {e})")

    # Grand summary
    print_grand_summary(grand_results, eval_ds)
    generate_all_figures(grand_results, eval_ds, meta_det, hybrid_det)