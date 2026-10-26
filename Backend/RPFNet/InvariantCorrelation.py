"""Correlation analysis between the relational-invariant families."""

import json
import os

import numpy as np
from scipy.stats import pearsonr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from Backend.RPFNet.RPFExtractor import RPFExtractor

try:
    from Backend.RPFNet.InvariantFigures import INVARIANT_TO_BLOCKS, _set_style
except Exception:  # keep this module usable stand-alone
    INVARIANT_TO_BLOCKS = {
        "I1: Neighbourhood consistency": ["A", "E"],
        "I2: Geometric coherence": ["B"],
        "I3: Influence boundedness": ["D", "F"],
        "I4: Structural stability": ["G"],
        "I5: Scale consistency": ["C"],
    }

    def _set_style():
        plt.rcParams.update({
            "font.size": 10, "font.family": "serif",
            "axes.labelsize": 11, "axes.titlesize": 12,
            "xtick.labelsize": 9, "ytick.labelsize": 9,
            "legend.fontsize": 9, "figure.dpi": 300,
            "savefig.dpi": 300, "savefig.bbox": "tight",
            "axes.grid": False,
        })


SHORT_LABELS = {
    "I1: Neighbourhood consistency": "I1:Neigh",
    "I2: Geometric coherence": "I2:Geom",
    "I3: Influence boundedness": "I3:Infl",
    "I4: Structural stability": "I4:Struct",
    "I5: Scale consistency": "I5:Scale",
}


def _short(name):
    if name in SHORT_LABELS:
        return SHORT_LABELS[name]
    parts = name.split(":")
    return parts[0] + ":" + parts[1].strip()[:8] if len(parts) > 1 else name[:10]


#  FAMILY -> COLUMN RESOLUTION

def invariant_family_columns(invariant_map=None, block_names=None):
    """Map each invariant family to the RPF column indices it owns."""
    invariant_map = invariant_map or INVARIANT_TO_BLOCKS
    block_names = block_names or RPFExtractor.BLOCK_NAMES

    cols = {}
    for inv_name, blk_keys in invariant_map.items():
        idx = []
        for bk in blk_keys:
            if bk not in block_names:
                continue
            sl = block_names[bk][1]
            idx.extend(range(sl.start, sl.stop))
        cols[inv_name] = np.asarray(sorted(set(idx)), dtype=int)
    return cols


def _zscore(A):
    """Column-wise z-score; constant columns are left at zero."""
    A = np.asarray(A, dtype=np.float64)
    mu = A.mean(axis=0, keepdims=True)
    sd = A.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = (A - mu) / sd
    return np.where(np.isfinite(Z), Z, 0.0)


def invariant_family_scores(rpf, invariant_map=None, block_names=None):
    """Reduce an (n, DIM) RPF matrix to (n, 5) family scores."""
    rpf = np.asarray(rpf, dtype=np.float64)
    if rpf.ndim != 2:
        raise ValueError(f"RPF must be 2-D, got shape {rpf.shape}")

    cols = invariant_family_columns(invariant_map, block_names)
    names = list(cols.keys())

    Z = _zscore(rpf)
    scores = np.zeros((rpf.shape[0], len(names)), dtype=np.float64)
    for j, nm in enumerate(names):
        idx = cols[nm]
        idx = idx[idx < rpf.shape[1]]
        if len(idx) == 0:
            continue
        scores[:, j] = Z[:, idx].mean(axis=1)
    return scores, names


#  CORRELATION

def invariant_correlation_matrix(scores):
    """Pairwise Pearson r and two-sided p-values for an (n, m) score matrix."""
    scores = np.asarray(scores, dtype=np.float64)
    m = scores.shape[1]
    C = np.eye(m, dtype=np.float64)
    P = np.zeros((m, m), dtype=np.float64)

    for i in range(m):
        for j in range(i + 1, m):
            a, b = scores[:, i], scores[:, j]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 3 or a[ok].std() < 1e-12 or b[ok].std() < 1e-12:
                r, p = 0.0, 1.0
            else:
                try:
                    r, p = pearsonr(a[ok], b[ok])
                except Exception:
                    r, p = 0.0, 1.0
            if not np.isfinite(r):
                r, p = 0.0, 1.0
            C[i, j] = C[j, i] = float(r)
            P[i, j] = P[j, i] = float(p)
    return C, P


def _fisher_mean(matrices, weights=None):
    """Average correlation matrices in Fisher-z space (diagonal forced to 1)."""
    if not matrices:
        return None
    M = np.stack([np.clip(np.asarray(m, dtype=np.float64), -0.999999, 0.999999)
                  for m in matrices])
    Z = np.arctanh(M)
    if weights is None:
        Zm = Z.mean(axis=0)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1, 1, 1)
        Zm = (Z * w).sum(axis=0) / max(w.sum(), 1e-12)
    out = np.tanh(Zm)
    np.fill_diagonal(out, 1.0)
    return out


#  OPTIONAL: REUSE OF RPFs PRODUCED DURING THE PIPELINE RUN

_COLLECTED = []          # list of (tag, (n, 5) family-score array)
_COLLECT_STATE = {"active": False, "orig": None, "max_sources": 64,
                  "max_rows": 20000, "seen": 0,
                  "rng": np.random.default_rng(20240)}


def enable_rpf_collection(max_sources=64, max_rows=20000):
    """Start caching the family scores of every RPF the pipeline extracts."""
    if _COLLECT_STATE["active"]:
        return
    orig = RPFExtractor.extract
    _COLLECT_STATE.update({"active": True, "orig": orig,
                           "max_sources": max_sources, "max_rows": max_rows,
                           "seen": 0,
                           "rng": np.random.default_rng(20240)})

    def _wrapped(self, X, y, y_cont=None):
        rpf = orig(self, X, y, y_cont=y_cont)
        try:
            arr = np.asarray(rpf)
            if arr.ndim == 2 and arr.shape[0] >= 8:
                if arr.shape[0] > _COLLECT_STATE["max_rows"]:
                    step = int(np.ceil(arr.shape[0]
                                       / _COLLECT_STATE["max_rows"]))
                    arr = arr[::step]
                sc, _ = invariant_family_scores(arr)
                _COLLECT_STATE["seen"] += 1
                seen = _COLLECT_STATE["seen"]
                tag = f"call_{seen:04d}_n{arr.shape[0]}"
                cap = _COLLECT_STATE["max_sources"]
                if len(_COLLECTED) < cap:
                    _COLLECTED.append((tag, sc))
                else:
                    j = int(_COLLECT_STATE["rng"].integers(0, seen))
                    if j < cap:
                        _COLLECTED[j] = (tag, sc)
        except Exception:
            pass  # collection must never affect the pipeline
        return rpf

    RPFExtractor.extract = _wrapped


def disable_rpf_collection():
    """Restore the original RPFExtractor.extract."""
    if _COLLECT_STATE["active"] and _COLLECT_STATE["orig"] is not None:
        RPFExtractor.extract = _COLLECT_STATE["orig"]
    _COLLECT_STATE.update({"active": False, "orig": None})


def collected_family_scores():
    return list(_COLLECTED)


def clear_collected_rpf():
    _COLLECTED.clear()
    _COLLECT_STATE["seen"] = 0


#  DATASET NORMALISATION (classification / regression, exactly as elsewhere)

def _normalize_datasets(datasets):
    """Accepts the shapes already used across the project and returns [(name, X, y, y_cont), ...]."""
    out = []
    if datasets is None:
        return out

    items = datasets.items() if isinstance(datasets, dict) \
        else [(None, d) for d in datasets]

    for key, val in items:
        try:
            if isinstance(val, dict):
                X = val.get("X_tr")
                y = val.get("y_tr")
                y_cont = val.get("y_cont")
                name = val.get("display", key)
            elif isinstance(val, (tuple, list)):
                if key is None:
                    name, X, y = val[0], val[1], val[2]
                    y_cont = val[3] if len(val) > 3 else None
                else:
                    name = key
                    X, y = val[0], val[1]
                    y_cont = val[2] if len(val) > 2 else None
            else:
                continue

            if X is None or y is None:
                continue
            out.append((str(name), np.asarray(X), np.asarray(y),
                        None if y_cont is None else np.asarray(y_cont)))
        except Exception:
            continue
    return out


def _sources_from_datasets(datasets, extractor=None, max_rows=8000,
                           verbose=True):
    """Reduce each dataset's RPF to family scores using the existing extractor."""
    extractor = extractor or RPFExtractor()
    sources = []
    for name, X, y, y_cont in _normalize_datasets(datasets):
        try:
            if len(X) < 8:
                continue
            note = ""
            if len(X) > max_rows:
                # Deterministic stride subsample keeps this analysis cheap; it
                # never touches the pipeline's own RPFs.
                step = int(np.ceil(len(X) / max_rows))
                X, y = X[::step], y[::step]
                y_cont = None if y_cont is None else y_cont[::step]
                note = f"  (stride-{step} subsample of {len(X) * step})"
            rpf = extractor.extract(X, y, y_cont=y_cont)
            sc, _ = invariant_family_scores(np.asarray(rpf))
            sources.append((name, sc))
            if verbose:
                kind = "regression" if y_cont is not None else "classification"
                print(f"    {name:<28} n={sc.shape[0]:>6}  ({kind}){note}")
        except Exception as e:
            if verbose:
                print(f"    [skip] {name}: {type(e).__name__}: {e}")
    return sources


#  REPORTING

def print_invariant_correlations(result):
    """Readable console table of the 5x5 Pearson matrix."""
    W = 82
    names = result["invariant_names"]
    shorts = [_short(n) for n in names]
    C = np.asarray(result["correlation_matrix"])
    P = np.asarray(result["p_values"])

    print("\n" + "=" * W)
    print("  INVARIANT-FAMILY CORRELATION ANALYSIS (Pearson, RPF-based)")
    print("=" * W)
    print(f"\n  Source        : {result['source']}")
    print(f"  Sources pooled: {result['n_sources']}")
    print(f"  Samples       : {result['n_samples']:,}")
    print("  Family score  : mean of z-scored RPF features in the family")

    print(f"\n  Family -> RPF columns")
    for nm in names:
        cols = result["family_columns"][nm]
        blocks = "+".join(result["family_blocks"][nm])
        print(f"    {_short(nm):<10} blocks {blocks:<4} "
              f"{len(cols):>2} features  {result['family_column_spans'][nm]}")

    print(f"\n  Pearson correlation matrix (r)")
    print(f"  {'':<11}", end="")
    for s in shorts:
        print(f"{s:>11}", end="")
    print()
    for i, s_i in enumerate(shorts):
        print(f"  {s_i:<11}", end="")
        for j in range(len(shorts)):
            print(f"{C[i, j]:>11.4f}", end="")
        print()

    print(f"\n  Two-sided p-values")
    print(f"  {'':<11}", end="")
    for s in shorts:
        print(f"{s:>11}", end="")
    print()
    for i, s_i in enumerate(shorts):
        print(f"  {s_i:<11}", end="")
        for j in range(len(shorts)):
            if i == j:
                print(f"{'-':>11}", end="")
            else:
                print(f"{P[i, j]:>11.2e}", end="")
        print()

    m = len(names)
    iu = np.triu_indices(m, k=1)
    off = C[iu]
    order = np.argsort(-np.abs(off))
    print(f"\n  Off-diagonal summary")
    print(f"    mean |r| = {np.abs(off).mean():.4f}   "
          f"max |r| = {np.abs(off).max():.4f}   "
          f"pairs |r|>0.5 : {(np.abs(off) > 0.5).sum()}/{len(off)}")
    print(f"\n  Ranked pairs")
    for k in order:
        i, j = iu[0][k], iu[1][k]
        r = C[i, j]
        strength = ("strong" if abs(r) >= 0.7 else
                    "moderate" if abs(r) >= 0.4 else
                    "weak" if abs(r) >= 0.2 else "negligible")
        print(f"    {shorts[i]:<10} - {shorts[j]:<10} r = {r:>+7.4f}   "
              f"r^2 = {r ** 2:>6.4f}   [{strength}]")

    if result.get("mean_within_source") is not None:
        Cw = np.asarray(result["mean_within_source"])
        print(f"\n  Fisher-z mean of the per-source matrices "
              f"(controls for between-dataset offsets)")
        print(f"  {'':<11}", end="")
        for s in shorts:
            print(f"{s:>11}", end="")
        print()
        for i, s_i in enumerate(shorts):
            print(f"  {s_i:<11}", end="")
            for j in range(len(shorts)):
                print(f"{Cw[i, j]:>11.4f}", end="")
            print()

    print("\n  Note: the sign of a family score follows the orientation of its"
          "\n  underlying RPF features, not a violation direction; interpret the"
          "\n  magnitude |r| as the redundancy between two families.")
    print("=" * W)


def print_invariant_correlation_latex(result):
    """LaTeX table of the correlation matrix (lower triangle + diagonal)."""
    names = result["invariant_names"]
    shorts = [_short(n).split(":")[0] for n in names]
    C = np.asarray(result["correlation_matrix"])
    m = len(names)

    print("\n% LaTeX table - Invariant-family Pearson correlations")
    print("\\begin{table}[t]")
    print("\\centering")
    summ = result["off_diagonal_summary"]
    print("\\caption{Pairwise Pearson correlation between the five relational "
          "invariant families, computed on the RPF representation "
          f"({result['n_samples']} samples, {result['n_sources']} sources). "
          "Each family is summarised per sample by the mean of its z-scored "
          f"RPF features. Mean off-diagonal $|r|$ = {summ['mean_abs_r']:.3f}, "
          f"max $|r|$ = {summ['max_abs_r']:.3f}.}}")
    print("\\label{tab:invariant-correlations}")
    print("\\begin{tabular}{l" + "c" * m + "}")
    print("\\hline")
    print("Invariant & " + " & ".join(shorts) + " \\\\")
    print("\\hline")
    for i in range(m):
        row = [shorts[i]]
        for j in range(m):
            row.append("---" if j > i else f"{C[i, j]:.3f}")
        print(" & ".join(row) + " \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


#  FIGURE

def fig_invariant_correlations(result,
                               save_path="figures/fig_invariant_correlations.pdf"):
    """Diverging heatmap of the 5x5 Pearson matrix (new filename, no overwrite)."""
    _set_style()
    names = result["invariant_names"]
    shorts = [_short(n) for n in names]
    C = np.asarray(result["correlation_matrix"])
    m = len(names)

    # Diverging: two hues + neutral midpoint, symmetric domain about r = 0.
    cmap = LinearSegmentedColormap.from_list(
        "corr", ["#2166AC", "#67A9CF", "#F7F7F7", "#EF8A62", "#B2182B"])
    cmap.set_bad("#E8E8E8")  # diagonal is uninformative; keep it recessive

    masked = np.ma.array(C, mask=np.eye(m, dtype=bool))

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(masked, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="equal")

    ax.set_xticks(range(m))
    ax.set_xticklabels(shorts, rotation=30, ha="right")
    ax.set_yticks(range(m))
    ax.set_yticklabels(shorts)

    for i in range(m):
        for j in range(m):
            if i == j:
                ax.text(j, i, "1.000", ha="center", va="center",
                        color="#8A8A8A", fontsize=9)
                continue
            val = C[i, j]
            color = "white" if abs(val) > 0.55 else "#1A1A1A"
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold")

    # 2px surface gap between cells
    ax.set_xticks(np.arange(-0.5, m, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, m, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.03,
                        ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.set_label("Pearson r", fontsize=9)
    cbar.outline.set_visible(False)

    ax.set_title("Correlation between invariant families (RPF)",
                 fontweight="bold", pad=12)
    ax.text(0.5, -0.20,
            f"{result['n_samples']:,} samples from {result['n_sources']} "
            f"sources; family score = mean of z-scored RPF features",
            transform=ax.transAxes, fontsize=7.5, ha="center", va="top",
            color="gray", style="italic")

    d = os.path.dirname(save_path)
    if d:
        os.makedirs(d, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


#  SAVING

def save_invariant_correlations(result, results_dir="results",
                                csv_name="invariant_correlations.csv",
                                json_name="invariant_correlations.json"):
    os.makedirs(results_dir, exist_ok=True)
    names = result["invariant_names"]
    shorts = [_short(n) for n in names]
    C = np.asarray(result["correlation_matrix"])

    csv_path = os.path.join(results_dir, csv_name)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("invariant," + ",".join(shorts) + "\n")
        for i, s in enumerate(shorts):
            f.write(s + "," + ",".join(f"{C[i, j]:.6f}"
                                       for j in range(len(shorts))) + "\n")

    json_path = os.path.join(results_dir, json_name)
    payload = {
        "method": "Pearson correlation between invariant-family scores; "
                  "family score = mean of z-scored RPF features of that family",
        "source": result["source"],
        "n_sources": result["n_sources"],
        "n_samples": result["n_samples"],
        "invariant_names": names,
        "invariant_short": shorts,
        "family_blocks": result["family_blocks"],
        "family_columns": {k: [int(c) for c in v]
                           for k, v in result["family_columns"].items()},
        "correlation_matrix": C.tolist(),
        "p_values": np.asarray(result["p_values"]).tolist(),
        "off_diagonal_summary": result["off_diagonal_summary"],
        "per_source": {k: np.asarray(v).tolist()
                       for k, v in result["per_source"].items()},
        "per_source_n": result["per_source_n"],
        "mean_within_source": (None if result.get("mean_within_source") is None
                               else np.asarray(
                                   result["mean_within_source"]).tolist()),
    }
    if result.get("pipeline_collected") is not None:
        pc = result["pipeline_collected"]
        payload["pipeline_collected"] = {
            "n_sources": pc["n_sources"],
            "n_samples": pc["n_samples"],
            "correlation_matrix": np.asarray(pc["correlation_matrix"]).tolist(),
        }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"  Saved: {csv_path}")
    print(f"  Saved: {json_path}")
    return csv_path, json_path


#  ENTRY POINT

def compute_invariant_correlations(sources, invariant_map=None,
                                   block_names=None, source_label=""):
    """Pool a list of (name, (n,5) scores) into the full correlation result."""
    cols = invariant_family_columns(invariant_map, block_names)
    names = list(cols.keys())
    invariant_map = invariant_map or INVARIANT_TO_BLOCKS

    if not sources:
        raise ValueError("No RPF sources available for correlation analysis.")

    stacked = np.vstack([np.asarray(s) for _, s in sources])
    C, P = invariant_correlation_matrix(stacked)

    per_source, per_source_n, per_mats, per_w = {}, {}, [], []
    for nm, sc in sources:
        sc = np.asarray(sc)
        if sc.shape[0] < 8:
            continue
        Ci, _ = invariant_correlation_matrix(sc)
        per_source[nm] = Ci
        per_source_n[nm] = int(sc.shape[0])
        per_mats.append(Ci)
        per_w.append(sc.shape[0])

    m = len(names)
    iu = np.triu_indices(m, k=1)
    off = C[iu]

    spans = {}
    for nm in names:
        idx = cols[nm]
        if len(idx) == 0:
            spans[nm] = "(none)"
            continue
        runs, start, prev = [], int(idx[0]), int(idx[0])
        for c in idx[1:]:
            c = int(c)
            if c == prev + 1:
                prev = c
                continue
            runs.append((start, prev))
            start = prev = c
        runs.append((start, prev))
        spans[nm] = ", ".join(f"{a}-{b}" if a != b else f"{a}"
                              for a, b in runs)

    return {
        "invariant_names": names,
        "family_blocks": {nm: list(invariant_map.get(nm, [])) for nm in names},
        "family_columns": {nm: cols[nm] for nm in names},
        "family_column_spans": spans,
        "correlation_matrix": C,
        "p_values": P,
        "per_source": per_source,
        "per_source_n": per_source_n,
        "mean_within_source": _fisher_mean(per_mats, per_w) if per_mats else None,
        "n_sources": len(sources),
        "n_samples": int(stacked.shape[0]),
        "source": source_label or f"{len(sources)} RPF sources",
        "off_diagonal_summary": {
            "mean_abs_r": float(np.abs(off).mean()),
            "max_abs_r": float(np.abs(off).max()),
            "n_pairs": int(len(off)),
            "n_pairs_abs_r_gt_0.5": int((np.abs(off) > 0.5).sum()),
        },
    }


def run_invariant_correlation_analysis(detector=None, datasets=None,
                                       rpf_matrices=None,
                                       prefer_collected=False,
                                       include_collected=True,
                                       results_dir="results",
                                       figures_dir="figures",
                                       figure_name="fig_invariant_correlations.pdf",
                                       max_rows=8000,
                                       invariant_map=None,
                                       emit_latex=True,
                                       verbose=True):
    """Compute, print, save and plot the 5x5 invariant-family correlation matrix."""
    if verbose:
        print("\n" + "-" * 82)
        print("  INVARIANT-FAMILY CORRELATION ANALYSIS (additive; no retraining)")
        print("-" * 82)

    collected = collected_family_scores()
    sources, label = [], ""

    if rpf_matrices is not None:
        items = (rpf_matrices.items() if isinstance(rpf_matrices, dict)
                 else [(f"rpf_{i + 1}", r) for i, r in enumerate(rpf_matrices)])
        for nm, rpf in items:
            arr = np.asarray(rpf)
            if arr.ndim != 2 or arr.shape[0] < 8:
                continue
            if arr.shape[1] == len(invariant_family_columns(invariant_map)):
                sc = arr  # already family scores
            else:
                sc, _ = invariant_family_scores(arr, invariant_map)
            sources.append((str(nm), sc))
        label = "supplied RPF matrices"

    elif prefer_collected and collected:
        sources = collected
        label = "RPFs generated during the pipeline run (collected, not recomputed)"

    else:
        extractor = getattr(detector, "extractor", None) or RPFExtractor()
        if verbose:
            print("\n  Reducing the RPF of the loaded datasets "
                  "(existing extractor, no attacks, no training):")
        sources = _sources_from_datasets(datasets, extractor,
                                         max_rows=max_rows, verbose=verbose)
        label = "RPF of the evaluation/training datasets (clean)"
        if not sources and collected:
            sources = collected
            label = "RPFs generated during the pipeline run (collected)"

    result = compute_invariant_correlations(sources, invariant_map=invariant_map,
                                            source_label=label)

    if include_collected and collected and sources is not collected:
        try:
            stacked = np.vstack([np.asarray(s) for _, s in collected])
            Cc, _ = invariant_correlation_matrix(stacked)
            result["pipeline_collected"] = {
                "n_sources": len(collected),
                "n_samples": int(stacked.shape[0]),
                "correlation_matrix": Cc,
            }
        except Exception:
            result["pipeline_collected"] = None

    print_invariant_correlations(result)

    if result.get("pipeline_collected"):
        pc = result["pipeline_collected"]
        shorts = [_short(n) for n in result["invariant_names"]]
        Cc = np.asarray(pc["correlation_matrix"])
        print(f"\n  Cross-check on the RPFs collected during evaluation "
              f"({pc['n_samples']:,} samples, {pc['n_sources']} extractions)")
        print(f"  {'':<11}", end="")
        for s in shorts:
            print(f"{s:>11}", end="")
        print()
        for i, s_i in enumerate(shorts):
            print(f"  {s_i:<11}", end="")
            for j in range(len(shorts)):
                print(f"{Cc[i, j]:>11.4f}", end="")
            print()

    save_invariant_correlations(result, results_dir=results_dir)
    fig_invariant_correlations(result,
                               os.path.join(figures_dir, figure_name))
    if emit_latex:
        print_invariant_correlation_latex(result)

    return result


if __name__ == "__main__":
    # Stand-alone smoke test on a synthetic Gaussian mixture.
    rng = np.random.default_rng(42)
    n, d = 600, 12
    X = np.vstack([rng.normal(0.0, 1.0, (n // 2, d)),
                   rng.normal(2.5, 1.0, (n // 2, d))]).astype(np.float32)
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    run_invariant_correlation_analysis(
        datasets={"synthetic_gaussian": (X, y)},
        results_dir="results", figures_dir="figures", verbose=True)
