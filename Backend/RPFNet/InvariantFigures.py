"""
InvariantFigures.py - Invariant-level ablation via PERMUTATION.

WHY PERMUTATION NOT ZEROING:
  RPFExtractor.extract() z-normalises each column to mean~0, std~1.
  Zeroing a column = setting it to its mean = the network sees "average"
  features = predictions barely change = F1 drop ~ 0.

  Permutation shuffles each column independently across samples.
  This preserves the marginal distribution but DESTROYS the per-sample
  correlation with the poison label. Standard feature importance method.

Usage:
    from InvariantFigures import (
        discover_block_layout,
        run_invariant_ablation,
        generate_invariant_ablation_figures,
    )

    discover_block_layout()   # run ONCE to see block keys
    ablation = run_invariant_ablation(detector, X, y, attacks, apply_attack)
    generate_invariant_ablation_figures(ablation, output_dir="figures")
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

from sklearn.metrics import f1_score


# ============================================================================
#  BLOCK DISCOVERY
# ============================================================================

def discover_block_layout():
    from Backend.RPFNet.RPFExtractor import RPFExtractor
    print("\n" + "=" * 70)
    print("  RPFExtractor.BLOCK_NAMES")
    print("=" * 70)
    print(f"  DIM={RPFExtractor.DIM}  blocks={len(RPFExtractor.BLOCK_NAMES)}  "
          f"key_type={type(list(RPFExtractor.BLOCK_NAMES.keys())[0]).__name__}")
    for key, (desc, sl) in RPFExtractor.BLOCK_NAMES.items():
        w = sl.stop - sl.start if isinstance(sl, slice) else "?"
        print(f"    '{key}' : {desc:<35} {sl}  (width={w})")
    print("=" * 70)
    return RPFExtractor.BLOCK_NAMES


# ============================================================================
#  INVARIANT -> BLOCK MAPPING  (string keys matching RPFExtractor)
# ============================================================================

INVARIANT_TO_BLOCKS = {
    "I1: Neighbourhood consistency": ["A", "E"],
    "I2: Geometric coherence": ["B"],
    "I3: Influence boundedness": ["D", "F"],
    "I4: Structural stability": ["G"],
    "I5: Scale consistency": ["C"],
}


def _build_validated_mapping(block_names_dict, invariant_map=None):
    if invariant_map is None:
        invariant_map = INVARIANT_TO_BLOCKS

    actual_keys = set(block_names_dict.keys())
    all_referenced = set()
    for blk_ids in invariant_map.values():
        all_referenced.update(blk_ids)

    matches = all_referenced & actual_keys
    if matches == all_referenced:
        return invariant_map

    missing = all_referenced - actual_keys
    if missing:
        print(f"  WARNING: keys {missing} not in BLOCK_NAMES {sorted(actual_keys)}")

    if not matches:
        sorted_actual = sorted(actual_keys)
        converted = {}
        for inv_name, blk_ids in invariant_map.items():
            new_ids = []
            for idx in blk_ids:
                if isinstance(idx, int) and idx < len(sorted_actual):
                    new_ids.append(sorted_actual[idx])
                else:
                    new_ids.append(idx)
            converted[inv_name] = new_ids
        conv_refs = set()
        for ids in converted.values():
            conv_refs.update(ids)
        if conv_refs & actual_keys:
            print(f"  Auto-converted int -> string keys")
            return converted

    return invariant_map


# ============================================================================
#  ATTACK FAMILIES
# ============================================================================

ATTACK_FAMILIES = {
    "Label Flip": [
        "label_flip", "boundary_flip", "targeted_class",
        "grad_flip", "tree_aware", "target_flip_extreme",
    ],
    "Noise / Perturb": [
        "feat_perturb", "gauss_noise", "feat_dropout",
        "combo_flip_perturb", "combo_flip_noise",
    ],
    "Clean-Label": [
        "clean_label", "interpolation", "adaptive_blend",
    ],
    "Backdoor": [
        "backdoor", "backdoor_heavy",
    ],
    "Structural / Dist": [
        "repr_inversion", "dist_shift", "outlier_inject",
        "null_feature",
    ],
    "Regression": [
        "target_shift", "leverage_attack",
    ],
}


def _attack_to_family(attack_name):
    for fam, members in ATTACK_FAMILIES.items():
        if attack_name in members:
            return fam
    return "Other"


# ============================================================================
#  STYLE
# ============================================================================

def _set_style():
    plt.rcParams.update({
        "font.size": 10, "font.family": "serif",
        "axes.labelsize": 11, "axes.titlesize": 12,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "figure.dpi": 300,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.grid": False,
    })


# ============================================================================
#  CORE ABLATION - PERMUTATION-BASED
# ============================================================================

def _ablate_invariant_permute(rpf, block_keys, block_names_dict, rng):
    """
    Permutation-ablate: shuffle columns belonging to the given blocks.
    Each column is independently permuted across samples.
    Returns (ablated_rpf, number_of_permuted_columns).
    """
    rpf_abl = rpf.copy()
    n_cols = 0
    for blk_key in block_keys:
        if blk_key not in block_names_dict:
            continue
        _, sl = block_names_dict[blk_key]
        for col in range(sl.start, sl.stop):
            rpf_abl[:, col] = rng.permutation(rpf_abl[:, col])
            n_cols += 1
    return rpf_abl, n_cols


def run_invariant_ablation(
    detector,
    X, y,
    attacks,
    apply_attack_fn,
    rate=0.10,
    n_trials=3,
    n_permutations=5,
    y_cont=None,
    invariant_map=None,
    verbose=True,
):
    """
    Invariant-level ablation via PERMUTATION.

    For each (attack, trial):
      1. Extract full RPF, compute full-model F1
      2. For each invariant, permute its columns n_permutations times,
         compute mean ablated F1
      3. drop = full_F1 - mean(ablated_F1)

    Parameters
    ----------
    n_permutations : int
        Number of independent permutations per invariant per trial.
        Averaging over multiple shuffles reduces noise.
    """
    from Backend.RPFNet.RPFExtractor import RPFExtractor

    block_names = RPFExtractor.BLOCK_NAMES
    validated_map = _build_validated_mapping(block_names, invariant_map)
    inv_names = list(validated_map.keys())

    if verbose:
        print(f"\n  === Invariant ablation (PERMUTATION method) ===")
        print(f"  {len(inv_names)} invariants, {len(attacks)} attacks, "
              f"{n_trials} trials, {n_permutations} permutations each")
        print(f"  Block keys: {sorted(block_names.keys())} "
              f"(type={type(list(block_names.keys())[0]).__name__})")
        for inv, blks in validated_map.items():
            descs = [block_names[b][0] for b in blks if b in block_names]
            cols = sum(
                (block_names[b][1].stop - block_names[b][1].start)
                for b in blks if b in block_names
            )
            print(f"    {inv} -> {blks} = {cols} cols ({', '.join(descs)})")

    # Sanity check
    total_cols = 0
    for inv_name in inv_names:
        for blk_key in validated_map[inv_name]:
            if blk_key in block_names:
                sl = block_names[blk_key][1]
                total_cols += sl.stop - sl.start

    if total_cols == 0:
        print("\n  FATAL: No columns mapped! Keys don't match BLOCK_NAMES.")
        print(f"    Expected keys like: {sorted(block_names.keys())}")
        print(f"    Got: {set(k for ids in validated_map.values() for k in ids)}")
        return {
            "per_invariant": {inv: {"mean_drop": 0, "std_drop": 0, "drops": []}
                              for inv in inv_names},
            "per_attack": {}, "per_family": {},
            "full_f1s": {}, "invariant_names": inv_names,
            "ERROR": "Block key mismatch",
        }

    if verbose:
        print(f"  Total ablatable columns: {total_cols} / {RPFExtractor.DIM}")

    # Run ablation
    rng = np.random.default_rng(42)
    global_drops = {inv: [] for inv in inv_names}
    per_attack = {}
    full_f1s = {}

    for atk in attacks:
        per_attack[atk] = {inv: [] for inv in inv_names}
        full_f1s[atk] = []

        for trial in range(n_trials):
            try:
                seed = trial * 31 + hash(atk) % 997
                Xp, yp, pidx = apply_attack_fn(
                    atk, X, y, fraction=rate, seed=seed, y_cont=y_cont
                )
                if len(pidx) < 3 or len(Xp) < 30:
                    continue

                ytrue = np.zeros(len(Xp), dtype=np.float32)
                ytrue[pidx] = 1.0
                k = max(1, int(len(Xp) * rate))

                # Full model
                rpf_full = detector.extractor.extract(Xp, yp, y_cont=y_cont)
                sc_full = detector._score_rpf(rpf_full)
                pred_full = np.zeros(len(Xp), dtype=int)
                pred_full[np.argsort(sc_full)[-k:]] = 1
                f1_full = f1_score(ytrue, pred_full, zero_division=0)
                full_f1s[atk].append(f1_full)

                # Ablate each invariant (permutation, averaged)
                for inv_name in inv_names:
                    blk_keys = validated_map[inv_name]
                    f1_ablated_runs = []

                    for perm_seed in range(n_permutations):
                        perm_rng = np.random.default_rng(
                            seed * 1000 + trial * 100 + perm_seed)
                        rpf_abl, _ = _ablate_invariant_permute(
                            rpf_full, blk_keys, block_names, perm_rng)
                        sc_abl = detector._score_rpf(rpf_abl)
                        pred_abl = np.zeros(len(Xp), dtype=int)
                        pred_abl[np.argsort(sc_abl)[-k:]] = 1
                        f1_abl = f1_score(ytrue, pred_abl, zero_division=0)
                        f1_ablated_runs.append(f1_abl)

                    mean_f1_abl = float(np.mean(f1_ablated_runs))
                    drop = f1_full - mean_f1_abl
                    global_drops[inv_name].append(drop)
                    per_attack[atk][inv_name].append(drop)

            except Exception as e:
                if verbose:
                    print(f"    [skip] {atk} trial {trial}: {e}")

        if verbose and full_f1s[atk]:
            drops_str = "  ".join(
                f"{inv.split(':')[0]}={np.mean(per_attack[atk][inv]):+.4f}"
                for inv in inv_names if per_attack[atk][inv]
            )
            print(f"    {atk:>22s}  F1={np.mean(full_f1s[atk]):.3f}  {drops_str}")

    # Aggregate
    per_invariant = {}
    for inv in inv_names:
        drops = global_drops[inv]
        per_invariant[inv] = {
            "mean_drop": float(np.mean(drops)) if drops else 0.0,
            "std_drop": float(np.std(drops)) if drops else 0.0,
            "drops": drops,
        }

    per_attack_agg = {}
    for atk in attacks:
        per_attack_agg[atk] = {}
        for inv in inv_names:
            drops = per_attack[atk][inv]
            per_attack_agg[atk][inv] = {
                "mean_drop": float(np.mean(drops)) if drops else 0.0,
                "std_drop": float(np.std(drops)) if drops else 0.0,
                "drops": drops,
            }

    per_family = {}
    for fam_name in ATTACK_FAMILIES:
        per_family[fam_name] = {inv: [] for inv in inv_names}
    for atk in attacks:
        fam = _attack_to_family(atk)
        if fam not in per_family:
            per_family[fam] = {inv: [] for inv in inv_names}
        for inv in inv_names:
            per_family[fam][inv].extend(per_attack[atk][inv])

    per_family_agg = {}
    for fam in per_family:
        per_family_agg[fam] = {}
        for inv in inv_names:
            drops = per_family[fam][inv]
            per_family_agg[fam][inv] = {
                "mean_drop": float(np.mean(drops)) if drops else 0.0,
                "std_drop": float(np.std(drops)) if drops else 0.0,
                "n": len(drops),
            }

    return {
        "per_invariant": per_invariant,
        "per_attack": per_attack_agg,
        "per_family": per_family_agg,
        "full_f1s": {a: float(np.mean(v)) if v else 0.0
                     for a, v in full_f1s.items()},
        "invariant_names": inv_names,
        "validated_map": validated_map,
    }


# ============================================================================
#  FIGURES
# ============================================================================

def fig_invariant_ablation_bar(ablation_results,
                               save_path="figures/fig_invariant_ablation_bar.pdf"):
    _set_style()
    inv_names = ablation_results["invariant_names"]
    per_inv = ablation_results["per_invariant"]

    means = [per_inv[inv]["mean_drop"] for inv in inv_names]
    stds = [per_inv[inv]["std_drop"] for inv in inv_names]
    short = [n.split(":")[0] + ":" + n.split(":")[1][:8] for n in inv_names]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    palette = ["#378ADD", "#1D9E75", "#D85A30", "#534AB7", "#C44E9B"]
    x = np.arange(len(inv_names))

    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=palette[:len(inv_names)], edgecolor="white",
                  linewidth=0.8, width=0.6, zorder=3)

    for i, (bar, m, s) in enumerate(zip(bars, means, stds)):
        y_pos = max(m + s, 0) + 0.003
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{m:+.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=20, ha="right")
    ax.set_ylabel("Mean F1 drop (full - ablated)")
    ax.set_title("Invariant ablation: F1 impact of removing each invariant",
                 fontweight="bold", pad=12)
    ax.axhline(y=0, color="gray", linewidth=0.8)
    low = min(min(means) - 0.02, -0.01)
    high = max(max(m + s for m, s in zip(means, stds)) + 0.02, 0.05)
    ax.set_ylim(low, high)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.text(0.98, 0.02,
            "",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
            color="gray", style="italic")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def fig_invariant_family_heatmap(ablation_results,
                                 save_path="figures/fig_invariant_family_heatmap.pdf"):
    _set_style()
    inv_names = ablation_results["invariant_names"]
    per_family = ablation_results["per_family"]

    families = [f for f in ATTACK_FAMILIES if f in per_family
                and any(per_family[f][inv]["n"] > 0 for inv in inv_names)]
    if not families:
        print("  [skip] No family data for heatmap.")
        return

    n_fam = len(families)
    n_inv = len(inv_names)
    data = np.zeros((n_fam, n_inv))
    counts = np.zeros((n_fam, n_inv), dtype=int)

    for i, fam in enumerate(families):
        for j, inv in enumerate(inv_names):
            data[i, j] = per_family[fam][inv]["mean_drop"]
            counts[i, j] = per_family[fam][inv]["n"]

    fig, ax = plt.subplots(figsize=(8, max(3.5, n_fam * 0.7 + 1)))
    vmax = max(abs(data.min()), abs(data.max()), 0.02)
    cmap = LinearSegmentedColormap.from_list(
        "drop", ["#2166AC", "#67A9CF", "#F7F7F7", "#EF8A62", "#B2182B"])
    im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    inv_short = [n.split(":")[0] + ":" + n.split(":")[1][:8] for n in inv_names]
    ax.set_xticks(range(n_inv))
    ax.set_xticklabels(inv_short, rotation=30, ha="right")
    ax.set_yticks(range(n_fam))
    ax.set_yticklabels(families)

    for i in range(n_fam):
        for j in range(n_inv):
            val = data[i, j]
            n = counts[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            txt = f"{val:+.3f}" if n > 0 else "---"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("F1 drop (positive = invariant helps)", fontsize=9)
    ax.set_title("Invariant x Attack Family ablation heatmap",
                 fontweight="bold", pad=12)

    for i in range(n_fam):
        row = data[i]
        if row.max() > 0:
            j_best = int(np.argmax(row))
            rect = plt.Rectangle((j_best - 0.48, i - 0.48), 0.96, 0.96,
                                 linewidth=2, edgecolor="#1D9E75",
                                 facecolor="none", zorder=5)
            ax.add_patch(rect)

    ax.text(0.98, -0.08,
            "",
            transform=ax.transAxes, fontsize=7.5, ha="right",
            color="#1D9E75", style="italic")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def fig_invariant_per_attack_detail(ablation_results,
                                    save_path="figures/fig_invariant_per_attack_detail.pdf"):
    _set_style()
    inv_names = ablation_results["invariant_names"]
    per_attack = ablation_results["per_attack"]

    attacks = [a for a in per_attack
               if any(per_attack[a][inv]["mean_drop"] != 0 for inv in inv_names)]
    if not attacks:
        print("  [skip] No attack data for detail chart.")
        return

    n_atk = len(attacks)
    n_inv = len(inv_names)
    palette = ["#378ADD", "#1D9E75", "#D85A30", "#534AB7", "#C44E9B"]

    fig, ax = plt.subplots(figsize=(max(8, n_atk * 0.9), 5))
    x = np.arange(n_atk)
    total_w = 0.75
    bar_w = total_w / n_inv

    for j, inv in enumerate(inv_names):
        vals = [per_attack[a][inv]["mean_drop"] for a in attacks]
        errs = [per_attack[a][inv]["std_drop"] for a in attacks]
        offset = (j - (n_inv - 1) / 2) * bar_w
        ax.bar(x + offset, vals, bar_w * 0.9, yerr=errs,
               capsize=2, label=inv.split(":")[0] + ":" + inv.split(":")[1][:8],
               color=palette[j % len(palette)], alpha=0.85, zorder=3)

    atk_labels = [a.replace("_", " ").title() for a in attacks]
    ax.set_xticks(x)
    ax.set_xticklabels(atk_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("F1 drop")
    ax.set_title("Invariant ablation per attack", fontweight="bold", pad=12)
    ax.axhline(y=0, color="gray", linewidth=0.8)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=7, ncol=2)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
#  CONSOLE SUMMARY
# ============================================================================

def print_invariant_ablation_summary(ablation_results):
    inv_names = ablation_results["invariant_names"]
    per_inv = ablation_results["per_invariant"]
    per_family = ablation_results["per_family"]

    W = 82
    print("\n" + "=" * W)
    print("  INVARIANT-LEVEL ABLATION STUDY (permutation method)")
    print("=" * W)

    print(f"\n  {'Invariant':<38} {'Mean dF1':>9} {'Std':>8} {'#trials':>8}")
    print(f"  {'_' * 65}")
    for inv in inv_names:
        d = per_inv[inv]
        n = len(d["drops"])
        bar = "#" * max(0, int(d["mean_drop"] * 80)) if d["mean_drop"] > 0 else "-"
        print(f"  {inv:<38} {d['mean_drop']:>+8.4f} {d['std_drop']:>8.4f} {n:>8}  {bar}")

    families = [f for f in ATTACK_FAMILIES if f in per_family]
    if families:
        print(f"\n  {'Family':<22}", end="")
        for inv in inv_names:
            short = inv.split(":")[0]
            print(f"  {short:>10}", end="")
        print()
        print(f"  {'_' * 22}", end="")
        for _ in inv_names:
            print(f"  {'_' * 10}", end="")
        print()

        for fam in families:
            has_data = any(per_family[fam][inv]["n"] > 0 for inv in inv_names)
            if not has_data:
                continue
            print(f"  {fam:<22}", end="")
            row_vals = []
            for inv in inv_names:
                v = per_family[fam][inv]["mean_drop"]
                row_vals.append(v)
                print(f"  {v:>+9.4f}", end="")
            best_idx = int(np.argmax(row_vals)) if max(row_vals) > 0 else -1
            best_inv = inv_names[best_idx].split(":")[0] if best_idx >= 0 else "---"
            print(f"   <- {best_inv}")

    print("=" * W)


# ============================================================================
#  PUBLIC ENTRY POINT
# ============================================================================

def generate_invariant_ablation_figures(ablation_results, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)

    if "ERROR" in ablation_results:
        print(f"\n  Skipping figures: {ablation_results['ERROR']}")
        return

    print(f"\n  Generating invariant ablation figures...")

    fig_invariant_ablation_bar(
        ablation_results,
        os.path.join(output_dir, "fig_invariant_ablation_bar.pdf"),
    )
    fig_invariant_family_heatmap(
        ablation_results,
        os.path.join(output_dir, "fig_invariant_family_heatmap.pdf"),
    )
    fig_invariant_per_attack_detail(
        ablation_results,
        os.path.join(output_dir, "fig_invariant_per_attack_detail.pdf"),
    )

    print_invariant_ablation_summary(ablation_results)
    print(f"\n  All invariant ablation figures saved to {output_dir}/")