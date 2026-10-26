"""Dataset-level test (Proposition 5)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Backend")]
from RPFNet.evaluation import FAMILY  # noqa: E402

NAME = sys.argv[1] if len(sys.argv) > 1 else "dataset_test"
ALPHA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
R = os.path.join("results", "v2")
refs = pd.read_csv(os.path.join(R, NAME, "dataset_test_refs.csv"))
items = pd.read_csv(os.path.join(R, NAME, "dataset_test_items.csv"))
items["ref"] = items["ref"].fillna("")
KEY = ["split", "dataset", "attack", "rate", "seed"]

rows = []
for (split, m), it in items.groupby(["split", "method"]):
    rf = refs[(refs.split == split) & (refs.method == m)].set_index("ref")["T"]
    K = len(rf)
    if m == "RPFNet":     # jackknife+: compare per reference scorer
        it = it.assign(Tref=it.ref.map(rf))
        g = it.groupby(KEY)
        p = (1 + g.apply(lambda d: int((d.Tref >= d["T"]).sum()), include_groups=False)) / (K + 1)
    else:                 # exact conformal
        g = it.set_index(KEY)["T"]
        p = (1 + g.apply(lambda t: int((rf.values >= t).sum()))) / (K + 1)
    df = p.rename("p").reset_index()
    df["method"], df["K"] = m, K
    rows.append(df)
P = pd.concat(rows, ignore_index=True)
P["clean"] = P.attack == "clean"
P["family"] = P.attack.map(lambda a: "clean" if a == "clean" else FAMILY.get(a, "other"))
P["reject"] = P.p <= ALPHA
pd.set_option("display.width", 200)
print(f"alpha = {ALPHA}; references per fold (K): {sorted(P.K.unique())}; min p = {P.p.min():.3f}")
print("\nFalse alarms on clean datasets (guarantee: RPFNet <= 2 alpha, baselines <= alpha):")
c = P[P.clean].groupby("method").agg(datasets=("reject", "size"), false_alarms=("reject", "sum"), rate=("reject", "mean"))
print(c.round(3).to_string())
pw = P[~P.clean]
print("\nPower (share of poisoned configurations rejected), by rate:")
print(pw.pivot_table(index="rate", columns="method", values="reject").round(3).to_string())
print("\nPower by family:")
print(pw.pivot_table(index="family", columns="method", values="reject").round(3).to_string())
print("\nOverall power:", pw.groupby("method").reject.mean().round(3).to_dict())
# per-dataset power comparison RPFNet vs each baseline
from scipy.stats import wilcoxon
d = pw.groupby(["dataset", "method"]).reject.mean().unstack()
for b in [m for m in d.columns if m != "RPFNet"]:
    x = (d.RPFNet - d[b]).dropna()
    pv = wilcoxon(x).pvalue if (x != 0).any() else 1.0
    print(f"  RPFNet vs {b}: higher power on {(x > 0).sum()}/{len(x)} datasets, lower on {(x < 0).sum()}, p={pv:.2g}")

# gated filter: RPFNet's own flags only where the RPFNet test rejects
res = os.path.join(R, "dataset_kfold", "results.csv")
if os.path.exists(res):
    m = pd.read_csv(res, low_memory=False)
    m = m[m.method == "RPFNet"][["dataset", "attack", "rate", "seed", "F1_own", "R_own", "P_own"]]
    g = pw[pw.method == "RPFNet"].merge(m, on=["dataset", "attack", "rate", "seed"])
    g["F1_gated"] = np.where(g.reject, g.F1_own, 0.0)
    print(f"\nGated filter on {len(g)} poisoned configurations: F1 ungated {g.F1_own.mean():.3f}, gated "
          f"{g.F1_gated.mean():.3f}; by rate:")
    print(g.groupby("rate")[["F1_own", "F1_gated", "reject"]].mean().round(3).to_string())
    cl = P[P.clean & (P.method == "RPFNet")]
    print(f"Clean datasets on which the gated filter flags anything: {int(cl.reject.sum())} of {len(cl)} "
          f"(ungated filter: all {len(cl)})")
P.to_csv(os.path.join(R, NAME, "dataset_test_pvalues.csv"), index=False)
