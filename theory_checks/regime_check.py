"""Both regimes of Proposition 4 on real attacks (Section 7.3, Appendix B.4, Table tab:regimes)."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OBLIVIOUS = ["label_flip", "gauss_noise", "repr_inversion", "dist_shift", "feat_dropout", "combo_flip_noise"]
K = ["dataset", "attack", "rate"]

r = pd.read_csv("results/relational_check/relational.csv")
r = r[r.k == 5].copy()
r["d"] = r.err_p - r.err_0
r["ex"] = (r.C_P + r.C_Q - r.C_0) / r.n_P
g = r.groupby(K).agg(delta=("d", "mean"), sd=("d", "std"), excess=("ex", "mean"), n_test=("n_test", "first"),
                     draws=("d", "size")).reset_index()
g["se"] = np.maximum(g.sd.fillna(0) / np.sqrt(g.draws), 1 / g.n_test)
dg = pd.read_csv("results/v2/dataset_kfold_ext/diagnostics.csv")
dg = dg[dg.seed == 0][K + ["harm_ratio", "spike_ratio"]]
x = g.merge(dg, on=K)
print(f"joined configurations: {len(x)}, datasets: {x.dataset.nunique()}, attacks: {x.attack.nunique()}")


def boot_auc(y, s, ds, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    u = ds.unique()
    out = []
    for _ in range(B):
        pick = rng.choice(u, len(u), replace=True)
        idx = np.concatenate([np.where(ds.values == d)[0] for d in pick])
        if len(set(y[idx])) == 2:
            out.append(roc_auc_score(y[idx], s[idx]))
    return np.percentile(out, [2.5, 97.5])


for z in (2, 3):
    harm = x.delta > z * x.se
    none = x.delta.abs() <= z * x.se
    cells = [("rho<1/4, kNN harm", (x.harm_ratio < 0.25) & harm), ("rho>1, no kNN harm", (x.harm_ratio > 1) & none),
             ("rho<1/4, no kNN harm", (x.harm_ratio < 0.25) & none), ("rho>1, kNN harm", (x.harm_ratio > 1) & harm)]
    for name, c in cells[: 4 if z == 2 else 2]:
        s = x[c]
        print(f"z={z} {name:22s} N={len(s):4d} ({s.attack.isin(OBLIVIOUS).sum()}) datasets={s.dataset.nunique():2d} "
              f"attacks={s.attack.nunique()} spectral={s.spike_ratio.median():.2f} excess={s.excess.median():.3f} "
              f"Delta={s.delta.mean():.3f}")
# AUROCs on the configurations whose k-NN harm is decided at z = 2 (detectable or absent)
dec = x[(x.delta > 2 * x.se) | (x.delta.abs() <= 2 * x.se)].reset_index(drop=True)
harm = (dec.delta > 2 * dec.se).values
big = (dec.harm_ratio > 1).values
print(f"decided configurations: {len(dec)}")
for name, col in [("Contradiction excess", "excess"), ("Spectral ratio", "spike_ratio")]:
    s = dec[col].values
    a1, a2 = roc_auc_score(harm, s), roc_auc_score(big, s)
    b1, b2 = boot_auc(harm, s, dec.dataset), boot_auc(big, s, dec.dataset)
    print(f"AUROC {name:22s} kNN harm {a1:.3f} [{b1[0]:.3f}, {b1[1]:.3f}]   rho>1 {a2:.3f} [{b2[0]:.3f}, {b2[1]:.3f}]")
