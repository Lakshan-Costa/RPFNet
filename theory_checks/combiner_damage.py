"""Supervised combiners against RPFNet by downstream damage and on the held-out attacks (Appendix H)."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

R = "results/v2/"
K = ["dataset", "attack", "rate", "seed"]
HELD = ["influence_lr", "subpopulation", "adaptive_dup", "catback"]
M = ["RPFNet", "Base-GB", "FP-GB", "FP+Base-GB"]


def fp(p):
    if p >= 0.01:
        return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\cdot10^{{{int(e)}}}$"


c = pd.read_csv(R + "combiner_check/scores.csv").pivot_table(index=K, columns="method", values="AUPRC")
m = pd.read_csv(R + "dataset_kfold/results.csv", low_memory=False)
p = c.join(m[m.method == "RPFNet"].set_index(K)["AUPRC"].rename("RPFNet"), how="inner")
dg = pd.read_csv(R + "dataset_kfold_ext/diagnostics.csv")
x = dg[K + ["damage_rf", "damage_proxy"]].merge(p.reset_index(), on=K)
groups = [("All seed-0 classification", x),
          ("Random forest loses $>2$ points", x[x.damage_rf > 0.02]),
          ("Logistic proxy loses $>2$ points", x[x.damage_proxy > 0.02]),
          ("Four held-out attacks (all seeds)", p.reset_index()[p.reset_index().attack.isin(HELD)])]
rows = []
for name, g in groups:
    d = g.groupby("dataset")[M].mean()
    t1 = wilcoxon(d.RPFNet, d["Base-GB"]).pvalue
    t2 = wilcoxon(d["FP-GB"], d["Base-GB"]).pvalue
    rows.append(f"{name} & {len(g)} & {len(d)} & " + " & ".join(f"{g[k].mean():.3f}" for k in M)
                + f" & {int((d.RPFNet > d['Base-GB']).sum())}/{len(d)} ({fp(t1)}) & "
                  f"{int((d['FP-GB'] > d['Base-GB']).sum())}/{len(d)} ({fp(t2)})\\\\")
    print(rows[-1])
open("tab_combdamage.tex", "w").write("\n".join(rows) + "\n")
