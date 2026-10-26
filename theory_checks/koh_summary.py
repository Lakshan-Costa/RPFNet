"""Summary of the sanitiser-aware attacks (configs/koh_attacks.json."""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

R = sys.argv[1] if len(sys.argv) > 1 else "results/v2"
BASE = ["IsoForest", "LOF", "OCSVM", "SpecSig", "GradNorm", "SEVER", "GradSpectral", "kNNLabel", "kNNDist",
        "ConfidentLearning", "ECOD", "IsoForest+CL", "AE-Recon", "ECOD+CL"]
KEY = ["dataset", "attack", "rate", "seed"]


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    out, run = np.empty(len(p)), 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        out[i] = run
    return out


def load(path):
    d = pd.read_csv(path, low_memory=False)
    if "error" in d:
        bad = d["error"].notna()
        if bad.any():
            print(f"  {bad.sum()} rows with errors, e.g. {d.loc[bad, 'error'].iloc[0]}")
        d = d[~bad]
    if "skipped" in d:
        d = d[d["skipped"] != True]   # noqa: E712
    return d


def damage(d):
    ds = d[d.method.isin(["_none", "_clean"])].pivot_table(index=KEY, columns="method", values="downstream")
    return (ds["_clean"] - ds["_none"]).rename("dmg_lr")


def main():
    k = load(os.path.join(R, "koh_attacks", "results.csv"))
    k = k[~k.method.str.startswith("_") | k.method.isin(["_none", "_clean"])]
    det = k[~k.method.str.startswith("_")]
    piv = det.pivot_table(index=KEY, columns="method", values="AUPRC")
    pivr = det.pivot_table(index=KEY, columns="method", values="AUROC")
    print(f"configurations: {len(piv)}, datasets: {piv.index.get_level_values(0).nunique()}")
    methods = [m for m in ["RPFNet", "Stack", "MinP", "Net"] if m in piv] + [b for b in BASE if b in piv]
    tab = pd.DataFrame({"AUROC": pivr[methods].mean(), "AUPRC": piv[methods].mean()})
    for atk in sorted(piv.index.get_level_values("attack").unique()):
        tab[f"AUPRC {atk}"] = piv.xs(atk, level="attack")[methods].mean()
    pdm = piv.groupby(level="dataset").mean()
    rows = []
    for m in methods[1:]:
        ok = pdm[["RPFNet", m]].dropna()
        p = wilcoxon(ok.RPFNet, ok[m]).pvalue if len(ok) > 5 and (ok.RPFNet != ok[m]).any() else np.nan
        rows.append((m, int((ok.RPFNet > ok[m]).sum()), len(ok), p))
    t = pd.DataFrame(rows, columns=["method", "RPFNet_better", "datasets", "p"]).set_index("method")
    t["p_holm"] = holm(t.p.fillna(1))
    tab = tab.join(t)
    pd.set_option("display.width", 250)
    print(tab.sort_values("AUPRC", ascending=False).round(4).to_string())
    print("\nAUPRC by rate:")
    print(piv[["RPFNet", "ECOD", "ConfidentLearning", "IsoForest", "kNNLabel"]].groupby(level="rate").mean().round(3))

    dm = damage(k)
    main_ = load(os.path.join(R, "dataset_kfold", "results.csv"))
    dmain = damage(main_[~main_.dataset.str.endswith("_reg")])
    print("\nLogistic-regression accuracy lost (points), mean by rate:")
    print(pd.DataFrame({atk: dm.xs(atk, level="attack").groupby(level="rate").mean() * 100
                        for atk in dm.index.get_level_values("attack").unique()}
                       | {"main benchmark (cls)": dmain.groupby(level="rate").mean() * 100}).round(2))
    x = piv.join(dm)
    for lo, hi in [(-1, 0.02), (0.02, 1)]:
        g = x[(x.dmg_lr > lo) & (x.dmg_lr <= hi)]
        if len(g):
            print(f"LR damage in ({lo},{hi}]: N={len(g)}  RPFNet {g.RPFNet.mean():.3f}  ECOD {g.ECOD.mean():.3f}  "
                  f"CL {g.ConfidentLearning.mean():.3f}  IForest {g.IsoForest.mean():.3f}")

    dpath = os.path.join(R, "koh_attacks", "diagnostics.csv")
    if os.path.exists(dpath):
        dg = pd.read_csv(dpath)
        print("\nDiagnostics (seed 0):")
        print(dg.groupby("attack")[["harm_ratio", "spike_ratio", "damage_proxy", "damage_rf"]].median().round(3))
        print("share rho > 1:", dg.groupby("attack").harm_ratio.apply(lambda v: (v > 1).mean()).round(3).to_dict(),
              " rho < 1/4:", dg.groupby("attack").harm_ratio.apply(lambda v: (v < 0.25).mean()).round(3).to_dict())
        print("RF damage > 2 points:", dg.groupby("attack").damage_rf.apply(lambda v: (v > 0.02).mean()).round(3).to_dict())
        y = dg.merge(piv.reset_index(), on=KEY)
        for lo, hi in [(-1, 0.25), (0.25, 1), (1, np.inf)]:
            g = y[(y.harm_ratio > lo) & (y.harm_ratio <= hi)]
            if len(g):
                print(f"rho in ({lo},{hi}]: N={len(g)} RPFNet {g.RPFNet.mean():.3f} GradNorm {g.GradNorm.mean():.3f} "
                      f"ECOD {g.ECOD.mean():.3f} CL {g.ConfidentLearning.mean():.3f}")


if __name__ == "__main__":
    main()
