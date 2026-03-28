import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch.nn.functional as F
from datasets import load_dataset


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

#  DATASET LOADERS — UPDATED with expanded diverse datasets
def _split_scale(X, y, seed=42):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    return Xtr, Xte, ytr, yte, sc


def _split_scale_reg(X, y, seed=42):
    """Split + scale for regression (no stratify)."""
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    # Also scale targets for regression
    y_mu, y_sd = ytr.mean(), ytr.std() + 1e-8
    ytr_s = ((ytr - y_mu) / y_sd).astype(np.float32)
    yte_s = ((yte - y_mu) / y_sd).astype(np.float32)
    return Xtr, Xte, ytr_s, yte_s, sc, y_mu, y_sd


def _load_openml_classification(data_id, name, max_n=5000, seed=42):
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
        c0 = np.where(y_full==0)[0]
        c1 = np.where(y_full==1)[0]
        n0 = min(3750, len(c0))
        n1 = min(1250, len(c1))
        idx = np.concatenate([rng_sub.choice(c0, n0, replace=False),
                               rng_sub.choice(c1, n1, replace=False)])
        rng_sub.shuffle(idx)
        X = X_full[idx]
        y = y_full[idx]
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
                target_col = cand
                break
        if target_col is None:
            target_col = df.columns[-1]
    print(f"        Target: '{target_col}'")
    ys = df[target_col].copy()
    if ys.dtype == object or str(ys.dtype) == "category":
        pos = {"yes","y","1","true","paid","fully paid","current","no default"}
        yb = ys.astype(str).str.strip().str.lower().isin(pos)
        ys = yb.astype(int).values
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
    X = Xdf.values.astype(np.float32)
    Xtr, Xte, ytr, yte = train_test_split(X, ys, test_size=0.2,
                                           random_state=42, stratify=ys)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    print(f" Features: {X.shape[1]}  Train: {Xtr.shape}  Balance: {ytr.mean():.2f}")
    return Xtr, Xte, ytr, yte, list(Xdf.columns), sc, False, None