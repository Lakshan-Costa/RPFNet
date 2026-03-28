import os, time, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.stats import rankdata
from scipy.stats import ttest_rel, wilcoxon
from sklearn.ensemble import (RandomForestClassifier, IsolationForest,
                               GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.datasets import (load_breast_cancer, load_wine, load_digits,
                               load_iris, load_diabetes, fetch_covtype,
                               fetch_california_housing, fetch_openml,
                               make_classification, make_moons, make_circles,
                               make_friedman1, make_regression)
from RPFNet.Attack import apply_attack
from RPFNet.RPFExtractor import RPFExtractor
from RPFNet.Dataset import load_builtin, load_csv
from RPFNet.RateEstimator import estimate_contamination_rate, score_distribution_features, RateEstimatorHead

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_curve, auc, precision_recall_curve

PLOT_DIR = "./figures"
os.makedirs(PLOT_DIR, exist_ok=True)

ATTACK_META = {
    "label_flip" : ("A. Label flip", True),
    "tree_aware" : ("B. Tree-aware + flip", True),
    "grad_flip" : ("E. Gradient + flip", True),
    "boundary_flip" : ("F. Boundary flip", True),
    "gauss_noise" : ("G. Gauss noise + flip", True),
    "interpolation" : ("H. Interpolation", True),
    "targeted_class" : ("J. Targeted class", True),
    "clean_label" : ("C. Clean-label", False),
    "backdoor" : ("D. Backdoor 4σ",  False),
    "null_feature" : ("I. Null feature", False),
    "feat_perturb" : ("K. Feat perturb ±4σ", False),
    "backdoor_heavy" : ("L. Backdoor heavy 10σ", False),
    "repr_inversion" : ("M. Repr inversion −x", False),
    "dist_shift" : ("N. Distribution shift", False),
    "outlier_inject" : ("O. Outlier injection", False),
    "feat_dropout" : ("P. Feat dropout 30%", False),
    # Regression-specific attacks
    "target_shift" : ("T. Target shift ±3σ", False),
    "leverage_attack" : ("U. Leverage attack", False),
    "target_flip_extreme" : ("V. Target flip extreme", True),
    # Composite attacks — training only
    "combo_flip_perturb" : ("Q. Combo flip+perturb", True),
    "combo_flip_noise" : ("R. Combo flip+noise", True),
    "adaptive_blend" : ("S. Adaptive blend", True),
}

def ablation_study(meta, X_tr, y_tr, attacks, rate=0.10, n_trials=3, y_cont=None):
    drops = {b: [] for b in RPFExtractor.BLOCK_NAMES}
    for trial in range(n_trials):
        for atk in attacks:
            try:
                Xp, yp, pidx = apply_attack(atk, X_tr, y_tr,
                                             fraction=rate, seed=trial * 31,
                                             y_cont=y_cont)
                if len(pidx) < 3: continue
                ytrue = np.zeros(len(Xp))
                ytrue[pidx] = 1
                k = max(1, int(len(Xp) * rate))

                rpf_full = meta.extractor.extract(Xp, yp, y_cont=y_cont)
                sc_full = meta._score_rpf(rpf_full)
                pred_full = np.zeros(len(Xp), dtype=int)
                pred_full[np.argsort(sc_full)[-k:]] = 1
                f1_full = f1_score(ytrue, pred_full, zero_division=0)

                for block in RPFExtractor.BLOCK_NAMES:
                    rpf_m = meta.ablate_block(rpf_full, block)
                    sc_m = meta._score_rpf(rpf_m)
                    pred_m = np.zeros(len(Xp), dtype=int)
                    pred_m[np.argsort(sc_m)[-k:]] = 1
                    f1_m = f1_score(ytrue, pred_m, zero_division=0)
                    drops[block].append(f1_full - f1_m)
            except Exception as exc:
                print(f"[ablation_study] Warning: attack {atk} trial {trial} failed: {exc}")
                continue
    return {b: float(np.mean(v)) if v else 0. for b, v in drops.items()}

# ── Consistent style ─────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

COLORS = {
    "Hybrid": "#2563EB",
    "MetaPoisonV3": "#7C3AED",
    "IsoForest": "#6B7280",
    "LOF": "#9CA3AF",
    "OCSVM": "#D1D5DB",
    "SEVER": "#F59E0B",
    "Influence": "#EF4444",
    "SpecSig": "#10B981",
    "PCA-KNN": "#F97316",
    "TRIM": "#EC4899",
    "iTRIM": "#A855F7",
    "AE-Recon": "#06B6D4",
}

def _get_color(d):
    return COLORS.get(d, "#888888")


#  FIGURE 1 — F1 Heatmap: Attacks × Detectors (per dataset or grand)
def plot_f1_heatmap(grand_results, eval_ds, mode_key="unknown"):
    """
    One heatmap per dataset: rows = attacks, cols = detectors, cell = F1.
    Also produces a grand-average heatmap across all datasets.
    """
    for ds_disp, ds_res in grand_results.items():
        valid = {k: v for k, v in ds_res.items() if v}
        if not valid:
            continue

        dets = []
        for res in valid.values():
            for d in res[mode_key]:
                if d not in dets:
                    dets.append(d)

        attacks_sorted = sorted(valid.keys(), key=lambda x: x[1])
        row_labels = [f"{ATTACK_META[a][0][:22]} @{r:.0%}" for a, r in attacks_sorted]
        matrix = np.full((len(attacks_sorted), len(dets)), np.nan)

        for ri, key in enumerate(attacks_sorted):
            res = valid[key]
            for ci, d in enumerate(dets):
                if d in res[mode_key]:
                    matrix[ri, ci] = res[mode_key][d]["F1_mean"]

        fig, ax = plt.subplots(figsize=(max(6, len(dets) * 1.1),
                                         max(4, len(row_labels) * 0.38)))
        cmap = LinearSegmentedColormap.from_list("rg", ["#fee2e2","#fef9c3","#dcfce7"], N=256)
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap,
                    xticklabels=dets, yticklabels=row_labels,
                    vmin=0, vmax=1, linewidths=0.5, linecolor="white",
                    ax=ax, cbar_kws={"label": "F1 Score"})
        ax.set_title(f"F1 Scores — {ds_disp} ({mode_key} rate)", fontweight="bold")
        ax.set_xlabel("Detector")
        ax.set_ylabel("")
        plt.tight_layout()
        safe = ds_disp.replace(" ", "_").lower()
        path = f"{PLOT_DIR}/fig1_heatmap_{safe}_{mode_key}.pdf"
        fig.savefig(path)
        plt.close(fig)
        print(f"  [fig1] Saved {path}")

    # ── Grand-average heatmap ─────────────────────────────────────
    all_dets = []
    for dr in grand_results.values():
        for res in dr.values():
            if res:
                for d in res[mode_key]:
                    if d not in all_dets:
                        all_dets.append(d)

    ds_names = list(grand_results.keys())
    matrix_g = np.full((len(ds_names), len(all_dets)), np.nan)
    for ri, ds_disp in enumerate(ds_names):
        for ci, d in enumerate(all_dets):
            vals = [r[mode_key][d]["F1_mean"]
                    for r in grand_results[ds_disp].values()
                    if r and d in r.get(mode_key, {})]
            if vals:
                matrix_g[ri, ci] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(max(6, len(all_dets) * 1.1),
                                     max(3, len(ds_names) * 0.5)))
    cmap = LinearSegmentedColormap.from_list("rg", ["#fee2e2","#fef9c3","#dcfce7"], N=256)
    sns.heatmap(matrix_g, annot=True, fmt=".2f", cmap=cmap,
                xticklabels=all_dets, yticklabels=ds_names,
                vmin=0, vmax=1, linewidths=0.5, linecolor="white",
                ax=ax, cbar_kws={"label": "Mean F1"})
    ax.set_title(f"Mean F1 by Dataset × Detector ({mode_key} rate)", fontweight="bold")
    plt.tight_layout()
    path = f"{PLOT_DIR}/fig1_heatmap_grand_{mode_key}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig1] Saved {path}")


# =====================================================================
#  FIGURE 2 — Grouped Bar: Detector comparison across poison rates
# =====================================================================

def plot_rate_comparison(grand_results, mode_key="unknown"):
    """
    For each rate, show average F1 of each detector across all
    datasets/attacks at that rate. Grouped bar chart.
    """
    all_dets = []
    for dr in grand_results.values():
        for res in dr.values():
            if res:
                for d in res[mode_key]:
                    if d not in all_dets:
                        all_dets.append(d)

    rates_seen = sorted(set(r for dr in grand_results.values()
                             for (_, r) in dr.keys()))
    rate_det_f1 = {r: {d: [] for d in all_dets} for r in rates_seen}

    for dr in grand_results.values():
        for (atk, rate), res in dr.items():
            if res is None:
                continue
            for d in all_dets:
                if d in res[mode_key]:
                    rate_det_f1[rate][d].append(res[mode_key][d]["F1_mean"])

    x = np.arange(len(rates_seen))
    n_det = len(all_dets)
    width = 0.8 / n_det

    fig, ax = plt.subplots(figsize=(max(7, len(rates_seen) * 1.8), 4.5))
    for i, d in enumerate(all_dets):
        means = [np.mean(rate_det_f1[r][d]) if rate_det_f1[r][d] else 0
                 for r in rates_seen]
        stds = [np.std(rate_det_f1[r][d]) if rate_det_f1[r][d] else 0
                 for r in rates_seen]
        ax.bar(x + i * width - 0.4 + width / 2, means, width,
               yerr=stds, label=d, color=_get_color(d),
               edgecolor="white", linewidth=0.5, capsize=2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.0%}" for r in rates_seen])
    ax.set_xlabel("Poison Rate")
    ax.set_ylabel("Mean F1 Score")
    ax.set_title(f"Detector Performance by Poison Rate ({mode_key} rate)", fontweight="bold")
    ax.legend(loc="upper left", ncol=min(4, n_det), framealpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{PLOT_DIR}/fig2_rate_comparison_{mode_key}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig2] Saved {path}")


# =====================================================================
#  FIGURE 3 — Ablation Study Bar Chart (Block Importance)
# =====================================================================

def plot_ablation(meta_det, eval_ds):
    """
    Bar chart showing F1 drop when each RPF block is zeroed out,
    averaged across datasets.
    """
    block_drops_all = {b: [] for b in RPFExtractor.BLOCK_NAMES}

    for ds_key, dsinfo in eval_ds.items():
        Xtr, ytr = dsinfo["X_tr"], dsinfo["y_tr"]
        y_cont = dsinfo.get("y_cont")
        is_reg = dsinfo.get("is_regression", False)
        abl_attacks = (["feat_perturb", "repr_inversion", "dist_shift"]
                       if is_reg else
                       ["label_flip", "feat_perturb", "repr_inversion", "dist_shift"])
        try:
            abl = ablation_study(meta_det, Xtr, ytr, abl_attacks,
                                  rate=0.10, y_cont=y_cont)
            for b in RPFExtractor.BLOCK_NAMES:
                if b in abl:
                    block_drops_all[b].append(abl[b])
        except Exception as exc:
            print(f"[plot_ablation] Warning: dataset {ds_key} failed: {exc}")
            continue

    blocks = list(RPFExtractor.BLOCK_NAMES.keys())
    descs = [RPFExtractor.BLOCK_NAMES[b][0] for b in blocks]
    means = [np.mean(block_drops_all[b]) if block_drops_all[b] else 0
               for b in blocks]
    stds = [np.std(block_drops_all[b]) if block_drops_all[b] else 0
               for b in blocks]

    colors = ["#3B82F6", "#8B5CF6", "#EC4899", "#F59E0B", "#10B981", "#EF4444"]
    while len(colors) < len(blocks):
        colors.append("#6B7280")

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(range(len(blocks)), means, xerr=stds,
                   color=colors[:len(blocks)], edgecolor="white",
                   linewidth=0.5, capsize=3)
    ax.set_yticks(range(len(blocks)))
    ax.set_yticklabels([f"Block {b}: {d}" for b, d in zip(blocks, descs)],
                       fontsize=9)
    ax.set_xlabel("Mean F1 Drop When Block Zeroed")
    ax.set_title("RPF Block Ablation Study", fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    for bar, m in zip(bars, means):
        if m > 0.005:
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"+{m:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    path = f"{PLOT_DIR}/fig3_ablation.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig3] Saved {path}")


# =====================================================================
#  FIGURE 4 — ROC Curves (Hybrid vs baselines, representative attack)
# =====================================================================

def plot_roc_curves(meta_det, hybrid_det, eval_ds):
    """
    ROC curves for Hybrid, MetaPoison, IsoForest on a representative
    dataset + attack. Picks the first classification dataset available.
    """
    # Pick a representative dataset
    ds_key = None
    for k, v in eval_ds.items():
        if not v.get("is_regression", False):
            ds_key = k
            break
    if ds_key is None:
        ds_key = list(eval_ds.keys())[0]

    dsinfo = eval_ds[ds_key]
    Xtr = dsinfo["X_tr"]
    ytr = dsinfo["y_tr"]
    y_cont = dsinfo.get("y_cont")
    ds_disp = dsinfo["display"]

    # Pick representative attacks at 10%
    rep_attacks = ["label_flip", "feat_perturb", "clean_label", "backdoor"]
    if dsinfo.get("is_regression"):
        rep_attacks = ["target_shift", "feat_perturb", "leverage_attack"]

    fig, axes = plt.subplots(1, min(4, len(rep_attacks)),
                              figsize=(min(4, len(rep_attacks)) * 3.5, 3.5),
                              sharey=True)
    if len(rep_attacks) == 1:
        axes = [axes]

    for ax_idx, atk in enumerate(rep_attacks[:4]):
        if atk not in ATTACK_META:
            continue
        ax = axes[ax_idx]
        try:
            Xp, yp, pidx = apply_attack(atk, Xtr, ytr, fraction=0.10,
                                          y_cont=y_cont)
            ytrue = np.zeros(len(Xp))
            ytrue[pidx] = 1

            # Hybrid scores
            combined, meta_s, iso_s = hybrid_det.score(Xp, yp, y_cont=y_cont)

            for label, scores, color, ls in [
                ("Hybrid", combined, COLORS["Hybrid"], "-"),
                ("MetaPoisonV3", meta_s, COLORS["MetaPoisonV3"], "--"),
                ("IsoForest", iso_s, COLORS["IsoForest"], ":"),
            ]:
                fpr, tpr, _ = roc_curve(ytrue, scores)
                roc_auc_val = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=1.5,
                        label=f"{label} (AUC={roc_auc_val:.2f})")

            ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.5)
            ax.set_xlabel("FPR")
            if ax_idx == 0:
                ax.set_ylabel("TPR")
            ax.set_title(ATTACK_META[atk][0][:20], fontsize=10)
            ax.legend(fontsize=7, loc="lower right")
            ax.grid(alpha=0.2)
        except Exception as e:
            ax.set_title(f"{atk} (failed)")

    fig.suptitle(f"ROC Curves — {ds_disp} @ 10% poison", fontweight="bold", y=1.02)
    plt.tight_layout()
    path = f"{PLOT_DIR}/fig4_roc_{ds_key}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig4] Saved {path}")


# =====================================================================
#  FIGURE 5 — Box Plot: F1 Distribution per Detector
# =====================================================================

def plot_f1_boxplot(grand_results, mode_key="unknown"):
    """
    Box plot showing F1 distribution across all attacks/rates for
    each detector. Highlights spread and robustness.
    """
    all_dets = []
    for dr in grand_results.values():
        for res in dr.values():
            if res:
                for d in res[mode_key]:
                    if d not in all_dets:
                        all_dets.append(d)

    det_vals = {d: [] for d in all_dets}
    for dr in grand_results.values():
        for res in dr.values():
            if res is None:
                continue
            for d in all_dets:
                if d in res[mode_key]:
                    det_vals[d].append(res[mode_key][d]["F1_mean"])

    # Sort detectors by median F1 (descending)
    sorted_dets = sorted(all_dets,
                          key=lambda d: np.median(det_vals[d]) if det_vals[d] else 0,
                          reverse=True)

    fig, ax = plt.subplots(figsize=(max(6, len(sorted_dets) * 0.8), 4.5))
    bp_data = [det_vals[d] for d in sorted_dets]
    bp = ax.boxplot(bp_data, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                    markeredgecolor="black", markersize=4),
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(linewidth=0.8),
                    flierprops=dict(markersize=3, alpha=0.5))

    for patch, d in zip(bp["boxes"], sorted_dets):
        patch.set_facecolor(_get_color(d))
        patch.set_alpha(0.8)

    ax.set_xticklabels(sorted_dets, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score")
    ax.set_title(f"F1 Distribution Across All Attacks ({mode_key} rate)",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(-0.02, 1.05)
    plt.tight_layout()
    path = f"{PLOT_DIR}/fig5_boxplot_{mode_key}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig5] Saved {path}")


# =====================================================================
#  FIGURE 6 — Flip vs No-Flip Attack Comparison
# =====================================================================

def plot_flip_vs_noflip(grand_results, mode_key="unknown"):
    """
    Grouped bars: each detector's mean F1 on flip attacks vs no-flip
    attacks. This highlights TRIM/iTRIM's known weakness.
    """
    all_dets = []
    for dr in grand_results.values():
        for res in dr.values():
            if res:
                for d in res[mode_key]:
                    if d not in all_dets:
                        all_dets.append(d)

    flip_vals = {d: [] for d in all_dets}
    noflip_vals = {d: [] for d in all_dets}

    for dr in grand_results.values():
        for res in dr.values():
            if res is None:
                continue
            bucket = flip_vals if res["has_flip"] else noflip_vals
            for d in all_dets:
                if d in res[mode_key]:
                    bucket[d].append(res[mode_key][d]["F1_mean"])

    x = np.arange(len(all_dets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, len(all_dets) * 0.9), 4.5))
    means_f = [np.mean(flip_vals[d]) if flip_vals[d] else 0 for d in all_dets]
    means_nf = [np.mean(noflip_vals[d]) if noflip_vals[d] else 0 for d in all_dets]

    ax.bar(x - width / 2, means_f, width, label="Label-flip attacks",
           color="#3B82F6", edgecolor="white")
    ax.bar(x + width / 2, means_nf, width, label="No-flip attacks (harder)",
           color="#EF4444", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(all_dets, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean F1 Score")
    ax.set_title("Flip vs No-Flip Attack Performance", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{PLOT_DIR}/fig6_flip_vs_noflip_{mode_key}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig6] Saved {path}")


# =====================================================================
#  FIGURE 7 — Zero-Shot vs Seen Dataset Performance
# =====================================================================

def plot_zeroshot_vs_seen(grand_results, eval_ds, mode_key="unknown"):
    """
    Compares mean F1 for seen (meta-trained) vs zero-shot datasets.
    """
    all_dets = []
    for dr in grand_results.values():
        for res in dr.values():
            if res:
                for d in res[mode_key]:
                    if d not in all_dets:
                        all_dets.append(d)

    seen_vals = {d: [] for d in all_dets}
    zs_vals = {d: [] for d in all_dets}

    zs_disps = {v["display"] for v in eval_ds.values() if v["zero_shot"]}

    for ds_disp, dr in grand_results.items():
        bucket = zs_vals if ds_disp in zs_disps else seen_vals
        for res in dr.values():
            if res is None:
                continue
            for d in all_dets:
                if d in res[mode_key]:
                    bucket[d].append(res[mode_key][d]["F1_mean"])

    x = np.arange(len(all_dets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, len(all_dets) * 0.9), 4.5))
    means_s = [np.mean(seen_vals[d]) if seen_vals[d] else 0 for d in all_dets]
    means_z = [np.mean(zs_vals[d]) if zs_vals[d] else 0 for d in all_dets]

    ax.bar(x - width / 2, means_s, width, label="Seen (meta-trained)",
           color="#8B5CF6", edgecolor="white")
    ax.bar(x + width / 2, means_z, width, label="Zero-shot ⭐",
           color="#F59E0B", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(all_dets, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean F1 Score")
    ax.set_title("Generalization: Seen vs Zero-Shot Datasets", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{PLOT_DIR}/fig7_zeroshot_{mode_key}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig7] Saved {path}")


# =====================================================================
#  FIGURE 8 — Feature Importance (RPF dimensions)
# =====================================================================

def plot_feature_importance(meta_det, eval_ds):
    """
    Gradient-based feature importance across RPF dimensions,
    color-coded by block.
    """
    all_imp = []
    for ds_key, dsinfo in eval_ds.items():
        try:
            imp = meta_det.feature_importance(dsinfo["X_tr"], dsinfo["y_tr"],
                                               y_cont=dsinfo.get("y_cont"))
            all_imp.append(imp)
        except Exception as exc:
            print(f"[plot_feature_importance] Warning: dataset {ds_key} failed: {exc}")
            continue

    if not all_imp:
        print("  [fig8] No feature importance data — skipping")
        return

    mean_imp = np.mean(all_imp, axis=0)
    dim = len(mean_imp)

    block_colors = {
        "A": "#3B82F6", "B": "#8B5CF6", "C": "#EC4899",
        "D": "#F59E0B", "E": "#10B981", "F": "#EF4444",
    }
    colors = []
    block_labels = []
    for b, (desc, sl) in RPFExtractor.BLOCK_NAMES.items():
        for i in range(sl.start, min(sl.stop, dim)):
            colors.append(block_colors.get(b, "#6B7280"))
            block_labels.append(b)

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(range(dim), mean_imp[:dim], color=colors[:dim],
                  edgecolor="white", linewidth=0.3)
    ax.set_xlabel("RPF Feature Index")
    ax.set_ylabel("Mean |Gradient| Importance")
    ax.set_title("RPF Feature Importance (gradient-based)", fontweight="bold")

    # Add block boundary annotations
    for b, (desc, sl) in RPFExtractor.BLOCK_NAMES.items():
        mid = (sl.start + min(sl.stop, dim)) / 2
        ax.axvline(sl.start - 0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.text(mid, ax.get_ylim()[1] * 0.95, f"Block {b}",
                ha="center", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=block_colors.get(b, "#ccc"),
                          alpha=0.3))

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{PLOT_DIR}/fig8_feature_importance.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig8] Saved {path}")


# =====================================================================
#  FIGURE 9 — Rate Estimation Calibration (estimated vs true rate)
# =====================================================================

def plot_rate_calibration(meta_det, eval_ds):
    """
    Scatter: true poison rate vs estimated rate. Perfect calibration
    = diagonal line.
    """
    true_rates = []
    est_rates = []

    rates_to_test = [0.01, 0.05, 0.10, 0.15, 0.20]
    test_attacks = ["label_flip", "feat_perturb", "backdoor", "repr_inversion"]

    for ds_key, dsinfo in eval_ds.items():
        if dsinfo.get("is_regression"):
            continue  # stick to classification for cleaner calibration plot
        Xtr, ytr = dsinfo["X_tr"], dsinfo["y_tr"]
        for atk in test_attacks:
            if atk not in ATTACK_META:
                continue
            for rate in rates_to_test:
                try:
                    Xp, yp, pidx = apply_attack(atk, Xtr, ytr, fraction=rate)
                    scores, _ = meta_det.score(Xp, yp)
                    est = estimate_contamination_rate(scores)
                    true_rates.append(rate)
                    est_rates.append(est)
                except Exception as exc:
                    print(f"[plot_rate_calibration] Warning: attack {atk} on ds {ds_key} rate {rate} failed: {exc}")
                    continue

    if not true_rates:
        print("  [fig9] No calibration data — skipping")
        return

    true_rates = np.array(true_rates)
    est_rates = np.array(est_rates)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true_rates + np.random.normal(0, 0.002, len(true_rates)),
               est_rates, alpha=0.4, s=20, c="#2563EB", edgecolors="white",
               linewidth=0.3)
    ax.plot([0, 0.25], [0, 0.25], "k--", linewidth=1, label="Perfect calibration")
    ax.set_xlabel("True Poison Rate")
    ax.set_ylabel("Estimated Poison Rate")
    ax.set_title("Rate Estimation Calibration", fontweight="bold")
    ax.set_xlim(-0.01, 0.25)
    ax.set_ylim(-0.01, 0.45)
    ax.legend()
    ax.grid(alpha=0.3)

    mae = float(np.abs(true_rates - est_rates).mean())
    ax.text(0.02, 0.40, f"MAE = {mae:.3f}", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    path = f"{PLOT_DIR}/fig9_rate_calibration.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig9] Saved {path}")


#  FIGURE 10 — Radar Chart: Detector profiles across attack families

def plot_radar(grand_results, mode_key="unknown"):
    """
    Radar/spider chart comparing top detectors across attack categories.
    """
    # Group attacks into families
    families = {
        "Label Flip": ["label_flip", "boundary_flip", "targeted_class",
                          "target_flip_extreme"],
        "Feature Pert.": ["feat_perturb", "gauss_noise", "feat_dropout"],
        "Clean-Label": ["clean_label", "interpolation"],
        "Backdoor": ["backdoor", "backdoor_heavy"],
        "Structural": ["repr_inversion", "dist_shift", "outlier_inject"],
        "Regression": ["target_shift", "leverage_attack"],
    }

    # Collect per-family F1 for key detectors
    key_dets = ["Hybrid", "MetaPoisonV3", "IsoForest", "SEVER", "LOF"]
    # Only keep detectors that actually appear
    available_dets = set()
    for dr in grand_results.values():
        for res in dr.values():
            if res:
                available_dets.update(res[mode_key].keys())
    key_dets = [d for d in key_dets if d in available_dets]

    fam_det_f1 = {fam: {d: [] for d in key_dets} for fam in families}

    for dr in grand_results.values():
        for (atk, rate), res in dr.items():
            if res is None:
                continue
            for fam, fam_attacks in families.items():
                if atk in fam_attacks:
                    for d in key_dets:
                        if d in res[mode_key]:
                            fam_det_f1[fam][d].append(res[mode_key][d]["F1_mean"])

    # Filter out families with no data
    active_fams = [f for f in families if any(fam_det_f1[f][d] for d in key_dets)]
    if len(active_fams) < 3:
        print("  [fig10] Not enough attack families — skipping radar")
        return

    N = len(active_fams)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for d in key_dets:
        values = [np.mean(fam_det_f1[f][d]) if fam_det_f1[f][d] else 0
                  for f in active_fams]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, label=d, color=_get_color(d))
        ax.fill(angles, values, alpha=0.08, color=_get_color(d))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(active_fams, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Detector Profiles Across Attack Families", fontweight="bold",
                 pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    path = f"{PLOT_DIR}/fig10_radar_{mode_key}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig10] Saved {path}")


#  FIGURE 11 — Classification vs Regression dataset performance

def plot_cls_vs_reg(grand_results, eval_ds, mode_key="unknown"):
    """
    Compares detector F1 on classification vs regression datasets.
    Only relevant if regression datasets exist.
    """
    cls_disps = {v["display"] for v in eval_ds.values() if not v.get("is_regression")}
    reg_disps = {v["display"] for v in eval_ds.values() if v.get("is_regression")}

    if not reg_disps:
        print("  [fig11] No regression datasets — skipping")
        return

    all_dets = []
    for dr in grand_results.values():
        for res in dr.values():
            if res:
                for d in res[mode_key]:
                    if d not in all_dets:
                        all_dets.append(d)

    cls_vals = {d: [] for d in all_dets}
    reg_vals = {d: [] for d in all_dets}

    for ds_disp, dr in grand_results.items():
        bucket = reg_vals if ds_disp in reg_disps else cls_vals
        for res in dr.values():
            if res is None:
                continue
            for d in all_dets:
                if d in res[mode_key]:
                    bucket[d].append(res[mode_key][d]["F1_mean"])

    x = np.arange(len(all_dets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, len(all_dets) * 0.9), 4.5))
    means_c = [np.mean(cls_vals[d]) if cls_vals[d] else 0 for d in all_dets]
    means_r = [np.mean(reg_vals[d]) if reg_vals[d] else 0 for d in all_dets]

    ax.bar(x - width / 2, means_c, width, label="Classification",
           color="#3B82F6", edgecolor="white")
    ax.bar(x + width / 2, means_r, width, label="Regression 📈",
           color="#10B981", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(all_dets, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean F1 Score")
    ax.set_title("Classification vs Regression Dataset Performance", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{PLOT_DIR}/fig11_cls_vs_reg_{mode_key}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig11] Saved {path}")


#  FIGURE 12 — Score Distribution (clean vs poison)

def plot_score_distributions(meta_det, hybrid_det, eval_ds):
    """
    Overlapping histograms of Hybrid scores for clean vs poisoned points.
    Shows separation quality on a representative attack.
    """
    ds_key = None
    for k, v in eval_ds.items():
        if not v.get("is_regression", False):
            ds_key = k
            break
    if ds_key is None:
        ds_key = list(eval_ds.keys())[0]

    dsinfo = eval_ds[ds_key]
    Xtr, ytr = dsinfo["X_tr"], dsinfo["y_tr"]
    y_cont = dsinfo.get("y_cont")

    attacks_to_show = ["label_flip", "clean_label", "feat_perturb", "backdoor"]
    if dsinfo.get("is_regression"):
        attacks_to_show = ["target_shift", "feat_perturb", "leverage_attack",
                           "backdoor"]

    attacks_to_show = [a for a in attacks_to_show if a in ATTACK_META]

    fig, axes = plt.subplots(1, min(4, len(attacks_to_show)),
                              figsize=(min(4, len(attacks_to_show)) * 3.5, 3),
                              sharey=True)
    if not hasattr(axes, '__len__'):
        axes = [axes]

    for ax_idx, atk in enumerate(attacks_to_show[:4]):
        ax = axes[ax_idx]
        try:
            Xp, yp, pidx = apply_attack(atk, Xtr, ytr, fraction=0.10,
                                          y_cont=y_cont)
            ytrue = np.zeros(len(Xp))
            ytrue[pidx] = 1

            combined, _, _ = hybrid_det.score(Xp, yp, y_cont=y_cont)

            clean_s = combined[ytrue == 0]
            poison_s = combined[ytrue == 1]

            ax.hist(clean_s, bins=40, alpha=0.6, color="#3B82F6",
                    label="Clean", density=True)
            ax.hist(poison_s, bins=40, alpha=0.6, color="#EF4444",
                    label="Poison", density=True)
            ax.set_xlabel("Hybrid Score")
            if ax_idx == 0:
                ax.set_ylabel("Density")
            ax.set_title(ATTACK_META[atk][0][:20], fontsize=10)
            ax.legend(fontsize=7)
        except Exception:
            ax.set_title(f"{atk} (failed)")

    fig.suptitle(f"Score Distributions — {dsinfo['display']} @ 10%",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    path = f"{PLOT_DIR}/fig12_score_dist_{ds_key}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig12] Saved {path}")

# Generate all figures

def generate_all_figures(grand_results, eval_ds, meta_det, hybrid_det):
    """Call this at the end of __main__ to produce all paper figures."""
    print("\n" + "=" * 82)
    print("  GENERATING PAPER FIGURES")
    print("=" * 82)

    plot_f1_heatmap(grand_results, eval_ds, "unknown")
    plot_f1_heatmap(grand_results, eval_ds, "known")
    plot_rate_comparison(grand_results, "unknown")
    plot_rate_comparison(grand_results, "known")
    plot_ablation(meta_det, eval_ds)
    plot_roc_curves(meta_det, hybrid_det, eval_ds)
    plot_f1_boxplot(grand_results, "unknown")
    plot_f1_boxplot(grand_results, "known")
    plot_flip_vs_noflip(grand_results, "unknown")
    plot_zeroshot_vs_seen(grand_results, eval_ds, "unknown")
    plot_feature_importance(meta_det, eval_ds)
    plot_rate_calibration(meta_det, eval_ds)
    plot_radar(grand_results, "unknown")
    plot_cls_vs_reg(grand_results, eval_ds, "unknown")
    plot_score_distributions(meta_det, hybrid_det, eval_ds)

    print(f"\n  ✓ All figures saved to {PLOT_DIR}/")
    print("=" * 82)