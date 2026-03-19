import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


def set_style():
    """Paper-quality plot style."""
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": False,
    })


def fig_violation_heatmap(coverage_results, save_path="figures/fig_violation_heatmap.pdf"):
    set_style()

    matrix = coverage_results["attack_invariant_matrix"]
    attack_cov = coverage_results["attack_coverage"]

    inv_names = list(list(matrix.values())[0].keys())
    inv_short = ["I₁: Neigh.", "I₂: Geom.", "I₃: Infl.", "I₄: Struct."]
    attacks = sorted(matrix.keys())

    # Build data matrix
    n_atk = len(attacks)
    n_inv = len(inv_names) + 1  # +1 for "Any" column
    data = np.zeros((n_atk, n_inv))

    for i, atk in enumerate(attacks):
        for j, inv_name in enumerate(inv_names):
            data[i, j] = matrix[atk].get(inv_name, {}).get("hit_rate", 0)
        data[i, -1] = attack_cov.get(atk, {}).get("coverage", 0)

    fig, ax = plt.subplots(figsize=(7, max(4, n_atk * 0.45)))

    # Custom colormap: white → orange → dark red
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["#FFFFFF", "#FAEEDA", "#EF9F27", "#D85A30", "#993C1D"]
    cmap = LinearSegmentedColormap.from_list("violation", colors)

    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # Labels
    attack_labels = [a.replace("_", " ").title() for a in attacks]
    col_labels = inv_short + ["Any ≥1"]

    ax.set_xticks(range(n_inv))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(n_atk))
    ax.set_yticklabels(attack_labels)

    # Add text annotations
    for i in range(n_atk):
        for j in range(n_inv):
            val = data[i, j]
            color = "white" if val > 0.6 else "black"
            weight = "bold" if j == n_inv - 1 else "normal"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    color=color, fontsize=8, fontweight=weight)

    # Separator line before "Any" column
    ax.axvline(x=n_inv - 1.5, color="black", linewidth=1.5)

    plt.colorbar(im, ax=ax, label="Violation rate", shrink=0.8)
    ax.set_title("Attack × Invariant violation matrix", fontweight="bold")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def fig_coverage_bar(coverage_results, save_path="figures/fig_coverage_bar.pdf"):
    set_style()

    attack_cov = coverage_results["attack_coverage"]
    attacks = sorted(attack_cov.keys(), key=lambda x: -attack_cov[x]["coverage"])

    labels = [a.replace("_", " ").title() for a in attacks]
    values = [attack_cov[a]["coverage"] for a in attacks]

    fig, ax = plt.subplots(figsize=(8, max(3, len(attacks) * 0.35)))

    colors = ["#1D9E75" if v >= 0.8 else "#EF9F27" if v >= 0.5
              else "#E24B4A" for v in values]

    bars = ax.barh(range(len(attacks)), values, color=colors, height=0.7)

    ax.set_yticks(range(len(attacks)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Coverage rate (fraction of trials with ≥1 violated invariant)")
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.8, color="gray", linestyle="--", alpha=0.5, label="80% threshold")

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.02, i, f"{val:.0%}", va="center", fontsize=8)

    ax.invert_yaxis()
    ax.set_title("Invariant violation coverage by attack type", fontweight="bold")
    ax.legend(loc="lower right")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def fig_transferability(trans_results, save_path="figures/fig_transferability.pdf"):
    set_style()

    summary = trans_results["summary"]
    inv_names = list(summary.keys())
    inv_short = ["I₁: Neigh.", "I₂: Geom.", "I₃: Infl.", "I₄: Struct."]

    means = [summary[n]["mean_ks"] for n in inv_names]
    stds = [summary[n]["std_ks"] for n in inv_names]
    maxs = [summary[n]["max_ks"] for n in inv_names]

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(inv_names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, means, width, yerr=stds,
                   label="Mean KS", color="#378ADD", capsize=3)
    bars2 = ax.bar(x + width / 2, maxs, width,
                   label="Max KS", color="#D85A30", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(inv_short)
    ax.set_ylabel("KS statistic (lower = more transferable)")
    ax.set_title("Cross-dataset distribution similarity\n"
                 "(rank-normalized clean invariant statistics)",
                 fontweight="bold")
    ax.legend()

    # Reference line
    ax.axhline(y=0.15, color="green", linestyle="--", alpha=0.5,
               label="Strong transferability threshold")

    ax.set_ylim(0, max(maxs) * 1.3 + 0.05)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def fig_complementarity(coverage_results, save_path="figures/fig_complementarity.pdf"):
    set_style()

    comp = coverage_results["complementarity"]
    inv_names = list(comp.keys())
    inv_short = ["I₁: Neigh.", "I₂: Geom.", "I₃: Infl.", "I₄: Struct."]

    total = [comp[n]["total_catches"] for n in inv_names]
    exclusive = [comp[n]["exclusive_catches"] for n in inv_names]

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(inv_names))
    width = 0.35

    ax.bar(x - width / 2, total, width, label="Total catches",
           color="#378ADD")
    ax.bar(x + width / 2, exclusive, width, label="Exclusive catches",
           color="#D85A30")

    ax.set_xticks(x)
    ax.set_xticklabels(inv_short)
    ax.set_ylabel("Number of attack instances")
    ax.set_title("Invariant complementarity\n"
                 "(exclusive = only this invariant detected the attack)",
                 fontweight="bold")
    ax.legend()

    # Mark necessary invariants
    for i, name in enumerate(inv_names):
        if comp[name]["is_necessary"]:
            ax.text(i + width / 2, exclusive[i] + 1, "★",
                    ha="center", fontsize=14, color="#1D9E75")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def fig_synthetic_snr(snr_results, save_path="figures/fig_synthetic_snr.pdf"):
    set_style()

    deltas = sorted(snr_results.keys())
    attacks = list(snr_results[deltas[0]].keys())

    fig, axes = plt.subplots(1, len(deltas), figsize=(4 * len(deltas), 4),
                             sharey=True)

    inv_colors = {
        "I₁: Neighborhood consistency": "#378ADD",
        "I₂: Geometric coherence": "#1D9E75",
        "I₃: Influence boundedness": "#D85A30",
        "I₄: Structural stability": "#534AB7",
    }

    for ax_idx, delta in enumerate(deltas):
        ax = axes[ax_idx] if len(deltas) > 1 else axes
        results = snr_results[delta]

        atk_labels = [a.replace("_", " ").title() for a in attacks]
        x = np.arange(len(attacks))

        for inv_idx, (inv_name, color) in enumerate(inv_colors.items()):
            values = []
            for atk in attacks:
                r = results[atk]
                inv_data = r["invariants"].get(inv_name, {})
                values.append(inv_data.get("cohens_d", 0))

            offset = (inv_idx - 1.5) * 0.18
            ax.bar(x + offset, values, 0.16, label=inv_name[:12],
                   color=color, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(atk_labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"δ = {delta} (SNR = {delta / np.sqrt(20):.1f})",
                     fontweight="bold")
        ax.axhline(y=0.2, color="red", linestyle="--", alpha=0.3,
                   label="d=0.2 threshold" if ax_idx == 0 else None)

        if ax_idx == 0:
            ax.set_ylabel("Cohen's d (effect size)")

    handles, labels = axes[0].get_legend_handles_labels() if len(deltas) > 1 \
                      else axes.get_legend_handles_labels()
    fig.legend(handles[:5], labels[:5], loc="upper center",
               ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.08))

    fig.suptitle("Invariant violations across class separation levels",
                 fontweight="bold", y=1.12)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def generate_all_invariant_figures(coverage_results, trans_results,
                                    snr_results, output_dir="figures"):
    """Generate all figures for the paper."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  Generating invariant analysis figures...")

    fig_violation_heatmap(
        coverage_results,
        os.path.join(output_dir, "fig_violation_heatmap.pdf")
    )
    fig_coverage_bar(
        coverage_results,
        os.path.join(output_dir, "fig_coverage_bar.pdf")
    )
    fig_complementarity(
        coverage_results,
        os.path.join(output_dir, "fig_complementarity.pdf")
    )

    if trans_results is not None:
        fig_transferability(
            trans_results,
            os.path.join(output_dir, "fig_transferability.pdf")
        )

    if snr_results is not None:
        fig_synthetic_snr(
            snr_results,
            os.path.join(output_dir, "fig_synthetic_snr.pdf")
        )

    print(f"All invariant figures saved to {output_dir}/")