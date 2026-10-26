"""LaTeX rows of the sanitiser-aware attack table (Appendix): run from the repository root."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

NAMES = {"RPFNet": r"\method{}", "Stack": r"\method{}-Stack", "MinP": r"\method{}-MinP", "Net": r"\method{}-Net",
         "IsoForest+CL": "Isolation Forest + CL", "ECOD+CL": "ECOD + CL", "ConfidentLearning": "Confident learning",
         "ECOD": "ECOD", "GradNorm": "Gradient norm", "IsoForest": "Isolation Forest", "GradSpectral": "Gradient spectral",
         "SEVER": "SEVER", "LOF": "LOF", "AE-Recon": "Autoencoder", "OCSVM": "One-class SVM", "kNNLabel": "$k$-NN label",
         "SpecSig": "Spectral signatures", "kNNDist": "$k$-NN distance"}


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); out, run = np.empty(len(p)), 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i])); out[i] = run
    return out


def fp(p):
    if p >= 0.01:
        return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\cdot10^{{{int(e)}}}$"


d = pd.read_csv("results/v2/koh_attacks/results.csv", low_memory=False)
d = d[~d.method.str.startswith("_")]
key = ["dataset", "attack", "rate", "seed"]
pr = d.pivot_table(index=key, columns="method", values="AUPRC")
ro = d.pivot_table(index=key, columns="method", values="AUROC")
pdm = pr.groupby(level="dataset").mean()
others = [m for m in NAMES if m != "RPFNet"]
p = [wilcoxon(pdm.RPFNet, pdm[m]).pvalue for m in others]
ph = dict(zip(others, holm(p)))
rows = []
for m in sorted(NAMES, key=lambda m: -pr[m].mean()):
    b = "--" if m == "RPFNet" else f"{int((pdm.RPFNet > pdm[m]).sum())}/{len(pdm)}"
    q = "--" if m == "RPFNet" else fp(ph[m])
    rows.append(f"{NAMES[m]} & {ro[m].mean():.3f} & {pr[m].mean():.3f} & {pr.xs('koh_minmax', level='attack')[m].mean():.3f}"
                f" & {pr.xs('koh_influence', level='attack')[m].mean():.3f} & {b} & {q}\\\\")
open("tab_koh.tex", "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
