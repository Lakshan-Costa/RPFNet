"""Summary of the white-box adaptive attacker (configs/frontier.json, --mode frontier."""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R = sys.argv[1] if len(sys.argv) > 1 else "results/v2"
EVADE_AUROC, HARM_PTS = 0.7, 2.0


def main():
    f = pd.read_csv(os.path.join(R, "frontier", "frontier.csv"))
    if "error" in f:
        bad = f["error"].notna()
        print(f"tasks with errors: {f.loc[bad, ['dataset', 'seed']].drop_duplicates().shape[0]}"
              + (f", e.g. {f.loc[bad, 'error'].iloc[0]}" if bad.any() else ""))
        f = f[~bad]
    f["sol"] = np.where(f.setting == "uniform", "a=" + f.mean_a.round(2).astype(str), "adaptive lam=" + f.lam.astype(str))
    sol_cols = ["dataset", "seed", "sol"]
    f["dmg_lr"], f["dmg_rf"] = f.dmg_lr * 100, f.dmg_rf * 100     # accuracy points
    rp = f[f.method == "RPFNet"].set_index(sol_cols)
    print(f"datasets: {f.dataset.nunique()}, tasks: {f[['dataset', 'seed']].drop_duplicates().shape[0]}")
    pd.set_option("display.width", 250)
    au = f.pivot_table(index="sol", columns="method", values="AUROC")
    meas = rp.groupby(level="sol")[["mean_a", "dmg_lr", "dmg_rf", "harm_ratio", "spike_ratio", "queries"]].mean()
    print("\nMean over tasks (AUROC of each detector; damage in accuracy points lost):")
    print(au.join(meas).round(3).to_string())

    ad = rp[rp.setting == "adaptive"]
    print(f"\nAdaptive solutions: {len(ad)}")
    ev = ad[(ad.AUROC <= EVADE_AUROC) & (ad.dmg_lr >= HARM_PTS)]
    print(f"evade RPFNet (AUROC <= {EVADE_AUROC}) while costing the logistic regression >= {HARM_PTS:.0f} points: "
          f"{len(ev)} of {len(ad)}" + (f" on {ev.index.get_level_values(0).nunique()} datasets" if len(ev) else ""))
    ev_any = ad[ad.AUROC <= EVADE_AUROC]
    print(f"evade RPFNet at any damage: {len(ev_any)}; their median rho {ev_any.harm_ratio.median():.3f}, "
          f"share rho < 1/4: {(ev_any.harm_ratio < 0.25).mean():.2f}, share spectral ratio > 1.5: "
          f"{(ev_any.spike_ratio > 1.5).mean():.2f}" if len(ev_any) else "")
    allsol = rp.reset_index()
    rs = [spearmanr(g.AUROC, g.dmg_lr).statistic for _, g in allsol.groupby(["dataset", "seed"]) if g.AUROC.nunique() > 2]
    rr = [spearmanr(g.AUROC, g.harm_ratio).statistic for _, g in allsol.groupby(["dataset", "seed"]) if g.AUROC.nunique() > 2]
    print(f"within-task Spearman, RPFNet AUROC vs LR damage: median {np.nanmedian(rs):.2f}; "
          f"vs harm ratio: median {np.nanmedian(rr):.2f}")
    # best detector per solution
    best = f.loc[f.groupby(sol_cols).AUROC.idxmax()]
    print("\nDetector with the highest AUROC, by solution type:")
    print(best.groupby(np.where(best.setting == "uniform", "uniform", "adaptive")).method.value_counts().unstack(fill_value=0))
    # the trade-off curve for the strongest evasion weight
    lam_max = ad.lam.max() if len(ad) else None
    if lam_max is not None:
        a = f[(f.setting == "adaptive") & (f.lam == lam_max)].pivot_table(index=["dataset", "seed"], columns="method",
                                                                            values="AUROC")
        print(f"\nAt lam={lam_max}: datasets where RPFNet has the highest AUROC: "
              f"{(a.idxmax(axis=1) == 'RPFNet').sum()} of {len(a)}")


if __name__ == "__main__":
    main()
