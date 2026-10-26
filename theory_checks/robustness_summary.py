"""Robustness of the main comparison with ECOD (Appendix H, Table tab:robust). Run from the repository root."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

K = ["dataset", "attack", "rate", "seed"]
EASY = ["catback", "feat_perturb", "leverage_attack"]


def boot(x, B=5000, seed=0):
    r = np.random.default_rng(seed)
    x = np.asarray(x)
    return np.percentile([x[r.integers(0, len(x), len(x))].mean() for _ in range(B)], [2.5, 97.5])


m = pd.read_csv("results/v2/dataset_kfold/results.csv", low_memory=False)
m = m[m.method.isin(["RPFNet", "ECOD"])]


def row(name, d, metric="AUPRC", weight_family=False):
    p = d.pivot_table(index=K + ["family"], columns="method", values=metric).dropna()
    if weight_family:
        pooled = p.groupby(level="family").mean().mean()
        pdm = p.groupby(level=["dataset", "family"]).mean().groupby(level="dataset").mean()
    else:
        pooled = p.mean()
        pdm = p.groupby(level="dataset").mean()
    diff = pdm.RPFNet - pdm.ECOD
    lo, hi = boot(diff)
    print(f"{name:34s} pooled {pooled.RPFNet:.3f} {pooled.ECOD:.3f} | per dataset {pdm.RPFNet.mean():.3f} "
          f"{pdm.ECOD.mean():.3f} diff {diff.mean():.3f} [{lo:.3f}, {hi:.3f}] higher {(diff > 0).sum()}/{len(diff)} "
          f"p={wilcoxon(pdm.RPFNet, pdm.ECOD).pvalue:.1g}")


row("AUPRC, all configurations", m)
row("AUPRC, without easy attacks", m[~m.attack.isin(EASY)])
row("AUPRC, families weighted equally", m, weight_family=True)
row("AUPRC at 0.5%", m[m.rate == 0.005])
row("AUPRC at 1%", m[m.rate == 0.01])
row("Recall at FPR 1%", m, "R@FPR1")
row("Recall at FPR 5%", m, "R@FPR5")
