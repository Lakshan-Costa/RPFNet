"""LaTeX rows of the adaptive-attacker table (Appendix): run from the repository root."""
import pandas as pd

f = pd.read_csv("results/v2/frontier/frontier.csv")
f = f[f.get("error", pd.Series(index=f.index, dtype=object)).isna()]
f["sol"] = [f"Uniform $a={a:.2f}$" if s == "uniform" else f"Adaptive $\\lambda={l:g}$"
            for s, a, l in zip(f.setting, f.mean_a.round(2), f.lam)]
order = [f"Uniform $a={a:.2f}$" for a in (0, .25, .5, .75, 1)] + [f"Adaptive $\\lambda={l:g}$" for l in (0, .5, 2, 8)]
au = f.pivot_table(index="sol", columns="method", values="AUROC").loc[order]
r = f[f.method == "RPFNet"].groupby("sol").agg(a=("mean_a", "mean"), lr=("dmg_lr", "mean"), rf=("dmg_rf", "mean"),
                                               rho=("harm_ratio", "median"), spike=("spike_ratio", "median")).loc[order]
rows = []
for s in order:
    rows.append(f"{s} & {r.a[s]:.2f} & {au.RPFNet[s]:.3f} & {au.ConfidentLearning[s]:.3f} & {au.kNNLabel[s]:.3f} & "
                f"{au.GradNorm[s]:.3f} & {au.SEVER[s]:.3f} & {au.ECOD[s]:.3f} & {100 * r.lr[s]:.1f} & {100 * r.rf[s]:.1f} & "
                f"{r.rho[s]:.3f} & {r.spike[s]:.2f}\\\\")
open("tab_frontier.tex", "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
