"""Held-out attack families on all five folds (Table 3 right, Table 9 last row, Table 15)."""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

R = sys.argv[1] if len(sys.argv) > 1 else "results/v2"
BASE = ["IsoForest", "LOF", "OCSVM", "SpecSig", "GradNorm", "SEVER", "GradSpectral", "kNNLabel", "kNNDist",
        "ConfidentLearning", "ECOD", "IsoForest+CL", "AE-Recon", "TRIM", "iTRIM", "ECOD+CL"]
FIXED = [b for b in BASE if b not in ("TRIM", "iTRIM")]
KEY = ["dataset", "attack", "rate", "seed"]


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    out, run = np.empty(len(p)), 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        out[i] = run
    return out


def boot(x, B=5000, seed=0):
    r = np.random.default_rng(seed)
    x = np.asarray(x)
    return np.percentile([x[r.integers(0, len(x), len(x))].mean() for _ in range(B)], [2.5, 97.5])


def main():
    fr = []
    for f in ["family_out"] + [f"family_out_f{i}" for i in range(1, 5)]:
        x = pd.read_csv(os.path.join(R, f, "results.csv"), low_memory=False)
        x["fold"] = f
        fr.append(x)
    d = pd.concat(fr)
    d = d[~d.method.astype(str).str.startswith("_")]
    d = d[d.family == d.split.str.replace("no_", "")]
    main_ = pd.read_csv(os.path.join(R, "dataset_kfold", "results.csv"), low_memory=False)
    seen = main_[main_.method == "RPFNet"].set_index(KEY)["AUPRC"].dropna()

    rows = []
    for fam, g in d.groupby("family"):
        piv = g.pivot_table(index=KEY, columns="method", values="AUPRC")
        bl = [b for b in BASE if b in piv and piv[b].notna().all()]
        best = max(bl, key=lambda b: piv[b].mean())
        pdm = piv.groupby(level="dataset").mean()
        idx = piv.index.intersection(seen.index)
        rows.append(dict(family=fam, N=len(piv), datasets=len(pdm), seen=seen.loc[idx].mean(),
                         unseen_same=piv.loc[idx, "RPFNet"].mean(), unseen_all=piv.RPFNet.mean(), best=best,
                         best_val=piv[best].mean(), better=int((pdm.RPFNet > pdm[best]).sum()),
                         p=wilcoxon(pdm.RPFNet, pdm[best]).pvalue if len(pdm) > 1 else np.nan))
    T = pd.DataFrame(rows)
    T["p_holm"] = holm(T.p.fillna(1))
    pd.set_option("display.width", 250)
    print(T.round(4).to_string())

    piv = d.pivot_table(index=KEY + ["family"], columns="method", values="AUPRC")
    pdm = piv.groupby(level="dataset").mean()
    print(f"\npooled over {len(piv)} left-out configurations: RPFNet {piv.RPFNet.mean():.3f}",
          {b: round(piv[b].mean(), 3) for b in ["ConfidentLearning", "ECOD", "IsoForest"]})
    for b in ["ConfidentLearning", "ECOD"]:
        print(f"  vs {b}: higher on {(pdm.RPFNet > pdm[b]).sum()}/{len(pdm)}, p={wilcoxon(pdm.RPFNet, pdm[b]).pvalue:.2g}")
    r = piv.reset_index().set_index(KEY)["RPFNet"]
    idx = r.index.intersection(seen.index)
    a = pd.DataFrame({"unseen": r.loc[idx], "seen": seen.loc[idx]}).reset_index()
    pa = a.groupby("dataset")[["unseen", "seen"]].mean()
    print(f"cost of leaving the family out, {len(idx)} configurations: seen {a.seen.mean():.3f} -> unseen "
          f"{a.unseen.mean():.3f}, lower on {(pa.unseen < pa.seen).sum()}/{len(pa)} datasets")

    cell = d.groupby(["dataset", "attack", "method"]).AUPRC.mean().unstack("method")
    reg = cell.sub(cell[[b for b in BASE if b in cell]].max(axis=1), axis=0).groupby(level="dataset").mean()
    best = max(FIXED, key=lambda b: reg[b].mean())
    print(f"regret: RPFNet {reg.RPFNet.mean():.3f} {boot(reg.RPFNet).round(3)}; best fixed {best} "
          f"{reg[best].mean():.3f} {boot(reg[best]).round(3)}; smaller on {(reg.RPFNet > reg[best]).sum()}/{len(reg)}, "
          f"p={wilcoxon(reg.RPFNet, reg[best]).pvalue:.2g}")


if __name__ == "__main__":
    main()
