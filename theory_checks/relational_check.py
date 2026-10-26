"""Measures the quantities of the relational proposition on real datasets."""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Backend"), ROOT]
warnings.filterwarnings("ignore")

from RPFNet.Attack import apply_attack, attack_seed  # noqa: E402
from RPFNet import evaluation as ev                 # noqa: E402

DATASETS = ["breast_cancer", "wine", "digits", "california_large", "adult", "credit_g", "ionosphere",
            "spambase", "vehicle", "segment", "pendigits", "satimage", "waveform", "glass", "yeast",
            "page_blocks", "vowel", "optdigits", "steel_plates", "mfeat_factors", "cardiotocography",
            "moons", "syn_hd", "nigerian_fraud", "bank_marketing", "loan_data"]
# Oblivious: rows chosen uniformly at random and each poison row depends only on the row it replaces.
OBLIVIOUS = ["label_flip", "gauss_noise", "repr_inversion", "dist_shift", "feat_dropout", "combo_flip_noise"]
# Poison values use statistics of the whole training split (weak dependence on the remaining clean rows).
GLOBAL = ["feat_perturb", "clean_label", "interpolation", "adaptive_blend"]
# Which rows are poisoned depends on the data (selection), or poison copies specific clean rows.
ADAPTIVE = ["boundary_flip", "targeted_class", "badnets", "subpopulation"]
ATTACKS = OBLIVIOUS + GLOBAL + ADAPTIVE
RATES = [0.01, 0.05, 0.1, 0.2]
SEEDS = [0, 1, 2]
KS = [5, 15]
MAX_ROWS = 10000
OUT = os.path.join(ROOT, "results", "relational_check")


def load(name):
    """Same loading and training-row cap as run_experiments.py."""
    d = ev.load_named(name, ROOT)
    if d["is_reg"]:
        raise ValueError("regression dataset")
    X, y = d["X"], d["y"]
    if len(X) > MAX_ROWS:
        idx = np.sort(np.random.default_rng(0).choice(len(X), MAX_ROWS, replace=False))
        X, y = X[idx], y[idx]
    return np.asarray(X, float), np.asarray(d["X_te"], float), np.asarray(y), np.asarray(d["y_te"])


def vote(Xtr, ytr, Xq, k, loo):
    """Plurality vote of the k nearest rows; ties go to the smallest label."""
    nn = NearestNeighbors(n_neighbors=k + int(loo)).fit(Xtr)
    idx = nn.kneighbors(Xq, return_distance=False)
    if loo:
        own = idx == np.arange(len(Xq))[:, None]
        has = own.any(1)
        idx = np.where(has[:, None], np.array([r[~o][:k] for r, o in zip(idx, own)]), idx[:, :k])
    labels = np.unique(ytr)
    counts = np.stack([(ytr[idx] == c).sum(1) for c in labels], 1)
    return labels[counts.argmax(1)]


def run(name):
    Xtr, Xte, ytr, yte = load(name)
    rows = []
    for attack in ATTACKS:
        for rate in RATES:
            for seed in SEEDS:
                try:
                    Xp, yp, pidx = apply_attack(attack, Xtr, ytr, rate, seed=attack_seed(attack, seed))[:3]
                except Exception as e:  # attack not applicable to this dataset
                    print(f"  skip {attack} {rate} {seed}: {type(e).__name__}")
                    continue
                pidx = np.asarray(pidx, int)
                if len(pidx) == 0:
                    continue
                is_q = np.zeros(len(yp), bool); is_q[pidx] = True
                keep = ~is_q  # clean rows that remain (unmodified)
                for k in KS:
                    if keep.sum() <= k + 1:
                        continue
                    c = vote(Xp, yp, Xp, k, loo=True) != yp
                    c0 = vote(Xtr[keep], ytr[keep], Xtr[keep], k, loo=True) != ytr[keep]
                    err_p = np.mean(vote(Xp, yp, Xte, k, loo=False) != yte)
                    err_0 = np.mean(vote(Xtr[keep], ytr[keep], Xte, k, loo=False) != yte)
                    rows.append(dict(dataset=name, attack=attack, rate=rate, seed=seed, k=k,
                                     n=len(yp), n_P=int(keep.sum()), n_test=len(yte),
                                     C_P=int(c[keep].sum()), C_Q=int(c[is_q].sum()), C_0=int(c0.sum()),
                                     err_p=err_p, err_0=err_0))
    return pd.DataFrame(rows)


def main():
    names = sys.argv[1:] or DATASETS
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "relational.csv")
    done = set(pd.read_csv(path)["dataset"]) if os.path.exists(path) else set()
    for name in names:
        if name in done:
            print(f"{name}: done"); continue
        try:
            df = run(name)
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {e}"); continue
        df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        print(f"{name}: {len(df)} rows")


if __name__ == "__main__":
    main()
