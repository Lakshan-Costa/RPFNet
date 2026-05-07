import numpy as np
import os
import json
from Backend.RPFNet.InvariantAnalyzer import (
    InvariantAnalyzer,
    GaussianMixtureBounds,
    synthetic_validation,
)

def run_theoretical_analysis(RPFNet_data, eval_ds, grand_results,
                              apply_attack_fn, verbose=True):
    W = 82
    print("\n" + "=" * W)
    print("  THEORETICAL ANALYSIS - Relational Invariant Violation")
    print("=" * W)

    # PART 1: Synthetic Validation
    print("\n" + "-" * W)
    print("  PART 1: Synthetic Gaussian Mixture Validation")
    print("-" * W)

    # Run at multiple SNR levels to show the theory holds across regimes
    snr_results = {}
    for delta in [1.5, 3.0, 5.0]:
        print(f"\n  - δ = {delta} (SNR = {delta/np.sqrt(20):.2f})")
        results, analyzer = synthetic_validation(
            n=2000, d=20, delta=delta, sigma=1.0, k=15,
            poison_rate=0.10, verbose=verbose
        )
        snr_results[delta] = results

    # PART 2: Coverage Analysis on Real Data
    print("\n" + "-" * W)
    print("  PART 2: Coverage Analysis - Real Datasets")
    print("-" * W)

    # Use a subset of training datasets for coverage analysis
    coverage_datasets = {}
    ds_count = 0
    for name, val in RPFNet_data.items():
        if ds_count >= 8:
            break
        coverage_datasets[name] = val
        ds_count += 1

    # Also add eval datasets
    for ds_key, dsinfo in eval_ds.items():
        if ds_count >= 12:
            break
        X = dsinfo["X_tr"]
        y = dsinfo["y_tr"]
        y_cont = dsinfo.get("y_cont")
        if y_cont is not None:
            coverage_datasets[ds_key] = (X, y, y_cont)
        else:
            coverage_datasets[ds_key] = (X, y)
        ds_count += 1

    # Define attacks to test
    classification_attacks = [
        "label_flip", "clean_label", "backdoor", "feat_perturb",
        "repr_inversion", "dist_shift", "boundary_flip", "feat_dropout",
    ]

    analyzer = InvariantAnalyzer(alpha=0.05)

    coverage_results = analyzer.coverage_analysis(
        datasets=coverage_datasets,
        attacks=classification_attacks,
        apply_attack_fn=apply_attack_fn,
        rates=(0.05, 0.10, 0.20),
        seeds=2,
        verbose=verbose,
    )

    analyzer.print_coverage_report(coverage_results)

    # PART 3: Transferability Analysis
    print("\n" + "-" * W)
    print("  PART 3: Transferability Analysis")
    print("-" * W)

    trans_results = analyzer.transferability_analysis(
        coverage_datasets, verbose=verbose
    )
    analyzer.print_transferability_report(trans_results)

    # PART 4: Theoretical Bounds
    print("\n" + "-" * W)
    print("  PART 4: Theoretical Bounds (for paper Section IV)")
    print("-" * W)

    # Show bounds at different regimes
    for setting_name, params in [
        ("Low-dim binary (Breast Cancer-like)",
         {"delta": 3.0, "sigma": 1.0, "k": 15, "d": 30, "n": 500, "K": 2}),
        ("High-dim multiclass (Digits-like)",
         {"delta": 2.0, "sigma": 1.0, "k": 15, "d": 64, "n": 1500, "K": 10}),
        ("Small-n (Glass-like)",
         {"delta": 2.5, "sigma": 1.0, "k": 10, "d": 9, "n": 200, "K": 6}),
    ]:
        print(f"\n  Setting: {setting_name}")
        GaussianMixtureBounds.print_bounds(**params)


    # SAVE RESULTS
    save_data = {
        "coverage_rate": coverage_results["coverage_rate"],
        "n_total": coverage_results["n_total"],
        "n_covered": coverage_results["n_covered"],
        "attack_coverage": coverage_results["attack_coverage"],
        "attack_invariant_matrix": {},
    }

    for atk, inv_dict in coverage_results["attack_invariant_matrix"].items():
        save_data["attack_invariant_matrix"][atk] = {
            inv_name: {"hit_rate": v["hit_rate"], "n_trials": v["n_trials"]}
            for inv_name, v in inv_dict.items()
        }

    save_data["complementarity"] = coverage_results["complementarity"]
    save_data["transferability"] = trans_results["summary"]

    os.makedirs("results", exist_ok=True)
    with open("results/invariant_analysis.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\n  Results saved to results/invariant_analysis.json")

    return coverage_results, trans_results, snr_results

def generate_paper_table_coverage(coverage_results):
    inv_short = {
        "I1: Neighborhood consistency": "I1",
        "I2: Geometric coherence": "I2",
        "I3: Influence boundedness": "I3",
        "I4: Structural stability": "I4",
    }

    inv_names = list(inv_short.keys())

    print("\n% LaTeX table - Attack x Invariant Violation Matrix")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Attack-Invariant violation matrix. Each cell shows the")
    print("fraction of trials where the invariant was violated ($p<0.05$,")
    print("Cohen's $d>0.2$). Every attack violates at least one invariant.}")
    print("\\label{tab:invariant-coverage}")
    print("\\begin{tabular}{l" + "c" * len(inv_names) + "c}")
    print("\\hline")
    header = "Attack"
    for name in inv_names:
        header += f" & {inv_short[name]}"
    header += " & Any \\\\"
    print(header)
    print("\\hline")

    matrix = coverage_results["attack_invariant_matrix"]
    attack_cov = coverage_results["attack_coverage"]

    for atk in sorted(matrix.keys()):
        row = atk.replace("_", "\\_")
        for inv_name in inv_names:
            hit = matrix[atk].get(inv_name, {}).get("hit_rate", 0)
            if hit > 0.7:
                row += f" & \\textbf{{{hit:.0%}}}"
            elif hit > 0.3:
                row += f" & {hit:.0%}"
            else:
                row += f" & {hit:.0%}"
        any_cov = attack_cov.get(atk, {}).get("coverage", 0)
        row += f" & {any_cov:.0%} \\\\"
        print(row)

    print("\\hline")

    # Average row
    avg_row = "\\textbf{Average}"
    for inv_name in inv_names:
        vals = [matrix[atk].get(inv_name, {}).get("hit_rate", 0)
                for atk in matrix]
        avg_row += f" & {np.mean(vals):.0%}"
    all_cov = [attack_cov[atk]["coverage"] for atk in attack_cov]
    avg_row += f" & \\textbf{{{np.mean(all_cov):.0%}}} \\\\"
    print(avg_row)

    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


def generate_paper_table_transferability(trans_results):
    inv_short = {
        "I₁: Neighborhood consistency": "I₁",
        "I₂: Geometric coherence": "I₂",
        "I₃: Influence boundedness": "I₃",
        "I₄: Structural stability": "I₄",
    }

    print("\n% LaTeX table - Transferability (KS statistics)")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Cross-dataset KS statistics for rank-normalized")
    print("invariant distributions (lower = more transferable).}")
    print("\\label{tab:transferability}")
    print("\\begin{tabular}{lccc}")
    print("\\hline")
    print("Invariant & Mean KS & Max KS & Std KS \\\\")
    print("\\hline")

    for inv_name, info in trans_results["summary"].items():
        short = inv_short.get(inv_name, inv_name[:10])
        print(f"{short} & {info['mean_ks']:.4f} & "
              f"{info['max_ks']:.4f} & {info['std_ks']:.4f} \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

if __name__ == "__main__":
    print("\nRunning standalone synthetic validation...")
    synthetic_validation(verbose=True)