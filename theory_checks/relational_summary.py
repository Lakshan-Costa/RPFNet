"""Summarises results/relational_check/relational.csv (see relational_check.py)."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relational_check import OBLIVIOUS, GLOBAL  # noqa: E402

d = pd.read_csv(os.path.join(ROOT, "results", "relational_check", "relational.csv"))
d["group"] = np.select([d.attack.isin(OBLIVIOUS), d.attack.isin(GLOBAL)], ["oblivious", "global"], "adaptive")
d["dC"] = (d.C_P - d.C_0) / d.n_P                 # change of the clean rows' contradiction rate
d["harm"] = d.err_p - d.err_0                     # harm to k-NN on the clean test split
d["dd"] = d.dC - d.harm                           # identity: mean 0 for oblivious attacks
d["excess"] = (d.C_P + d.C_Q - d.C_0) / d.n_P     # all rows: predicted >= harm
d["slack"] = d.excess - d.harm


def summ(g):
    m = g.groupby("dataset")[["dC", "harm", "dd", "excess", "slack"]].mean()
    se = m.std(ddof=1) / np.sqrt(len(m))
    pos = g[g.harm > 0]
    return pd.Series({"datasets": len(m), "configs": len(g), "harm": m.harm.mean(), "dC": m.dC.mean(),
                      "dd": m.dd.mean(), "dd_se": se.dd, "excess": m.excess.mean(),
                      "slack": m.slack.mean(), "slack_se": se.slack,
                      "clean_share": (pos.C_P - pos.C_0).sum() / max(1, (pos.C_P + pos.C_Q - pos.C_0).sum()),
                      "rho_excess_harm": g[["excess", "harm"]].corr("spearman").iloc[0, 1]})


pd.set_option("display.width", 250, "display.max_columns", 20, "display.precision", 4)
out = []
for k, gk in d.groupby("k"):
    a = gk.groupby("group").apply(summ); a.insert(0, "k", k); out.append(a)
    b = gk.groupby("attack").apply(summ); b.insert(0, "k", k); out.append(b)
res = pd.concat(out)
print(res)
res.to_csv(os.path.join(ROOT, "results", "relational_check", "summary.csv"))
