"""Figure 1: mean AUPRC by harm ratio rho, seed-0 classification configurations. Writes fig_comp.pdf."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = "results/v2/"
K = ["dataset", "attack", "rate", "seed"]
dg = pd.read_csv(R + "dataset_kfold_ext/diagnostics.csv")
m = pd.read_csv(R + "dataset_kfold/results.csv", low_memory=False)
gn = m[m.method == "GradNorm"].set_index(K)["AUPRC"].rename("GradNorm")
rp = m[m.method == "RPFNet"].set_index(K)["AUPRC"].rename("RPFNet")
x = dg.set_index(K)[["harm_ratio", "auprc_blockH"]].join(rp).join(gn).reset_index()
x["bin"] = pd.cut(x.harm_ratio, [-np.inf, 0.25, 1, np.inf], labels=["<1/4", "1/4–1", ">1"])
fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.75), sharey=True)
style = [("RPFNet", "RPFNet", "#1f77b4", "o", "-"), ("GradNorm", "Gradient norm", "#2ca02c", "s", "--"),
         ("auprc_blockH", "Spectral score (H)", "#ff7f0e", "^", ":")]
for ax, (title, rates) in zip(axes, [("0.5–1% contamination", [0.005, 0.01]), ("5–20% contamination", [0.05, 0.1, 0.2])]):
    g = x[x.rate.isin(rates)].groupby("bin", observed=False)[[s[0] for s in style]].mean()
    for col, lab, c, mk, ls in style:
        ax.plot(range(3), g[col].values, color=c, marker=mk, ls=ls, lw=1.2, ms=3.5, label=lab)
    ax.set_xticks(range(3)); ax.set_xticklabels(g.index, fontsize=6.5)
    ax.set_title(title, fontsize=7); ax.set_xlabel("harm ratio ρ", fontsize=6.5)
    ax.set_ylim(0, 1); ax.tick_params(labelsize=6.5); ax.grid(alpha=0.3, lw=0.5)
    print(title); print(g.round(3))
axes[0].set_ylabel("AUPRC", fontsize=6.5)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig("fig_comp.pdf", bbox_inches="tight")
