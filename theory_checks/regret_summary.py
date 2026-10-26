"""Regret against the best baseline of each (dataset, attack) cell (Table 9)."""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

R = sys.argv[1] if len(sys.argv) > 1 else "results/v2"
BASE = ["IsoForest", "LOF", "OCSVM", "SpecSig", "GradNorm", "SEVER", "GradSpectral", "kNNLabel", "kNNDist",
        "ConfidentLearning", "ECOD", "IsoForest+CL", "AE-Recon", "TRIM", "iTRIM", "ECOD+CL"]
FIXED = [b for b in BASE if b not in ("TRIM", "iTRIM")]
HELD = ["influence_lr", "subpopulation", "adaptive_dup", "catback"]


def boot(x, B=5000, seed=0):
    r = np.random.default_rng(seed)
    x = np.asarray(x)
    return np.percentile([x[r.integers(0, len(x), len(x))].mean() for _ in range(B)], [2.5, 97.5])


def regret(d):
    cell = d.groupby(["dataset", "attack", "method"]).AUPRC.mean().unstack("method")
    return cell.sub(cell[[b for b in BASE if b in cell]].max(axis=1), axis=0).groupby(level="dataset").mean()


def main():
    m = pd.read_csv(os.path.join(R, "dataset_kfold", "results.csv"), low_memory=False)
    m = m[~m.method.str.startswith("_")]
    for tag, d in [("seen attack types", m[~m.attack.isin(HELD)]), ("four held-out attacks", m[m.attack.isin(HELD)])]:
        reg = regret(d)
        best = max(FIXED, key=lambda b: reg[b].mean())
        print(f"{tag}: {len(reg)} datasets; RPFNet {reg.RPFNet.mean():.3f} {boot(reg.RPFNet).round(3)}; best fixed "
              f"{best} {reg[best].mean():.3f} {boot(reg[best]).round(3)}; RPFNet smaller on "
              f"{(reg.RPFNet > reg[best]).sum()}/{len(reg)}, p={wilcoxon(reg.RPFNet, reg[best]).pvalue:.2g}")


if __name__ == "__main__":
    main()
