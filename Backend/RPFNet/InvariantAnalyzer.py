import numpy as np
from scipy import stats
from scipy.stats import rankdata, ks_2samp, mannwhitneyu
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score
from collections import defaultdict


#  INVARIANT DEFINITIONS
class RelationalInvariant:
    def __init__(self, name, short_name, description, rpf_slices):
        self.name = name
        self.short_name = short_name
        self.description = description
        self.rpf_slices = rpf_slices

    def compute_statistic(self, X, y, rpf=None, y_cont=None):
        raise NotImplementedError

    def estimate_clean_bound(self, stats_clean, alpha=0.05):
        return float(np.percentile(stats_clean, 100 * (1 - alpha)))


class NeighborhoodConsistency(RelationalInvariant):
    def __init__(self):
        super().__init__(
            "I1: Neighborhood consistency", "I1:Neigh",
            "k-NN label agreement (excess over chance) bounded below",
            [slice(0, 8)])

    def compute_statistic(self, X, y, rpf=None, y_cont=None):
        n = len(X)
        K = len(np.unique(y))
        k = min(15, n - 2)
        chance = 1.0 / max(K, 2)
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(X)
        _, I_neighbors = nbrs.kneighbors(X)
        I_neighbors = I_neighbors[:, 1:]
        agreement = (y[I_neighbors] == y[:, np.newaxis]).mean(1)
        denom = max(1.0 - chance, 1e-8)
        excess = np.clip((agreement - chance) / denom, 0.0, 1.0)
        return (1.0 - excess).astype(np.float32)


class GeometricCoherence(RelationalInvariant):
    def __init__(self):
        super().__init__(
            "I2: Geometric coherence", "I2:Geom",
            "Distance to own centroid < distance to nearest other",
            [slice(8, 17)])

    def compute_statistic(self, X, y, rpf=None, y_cont=None):
        n = len(X)
        classes = np.unique(y)
        if len(classes) < 2:
            return np.zeros(n, np.float32)
        mus = {}
        for c in classes:
            idx = np.where(y == c)[0]
            mus[c] = X[idx].mean(0) if len(idx) >= 2 else X.mean(0)
        violation = np.zeros(n, np.float32)
        for i in range(n):
            d_own = np.linalg.norm(X[i] - mus.get(y[i], X.mean(0)))
            d_other = min(
                (np.linalg.norm(X[i] - mus[c]) for c in classes if c != y[i]),
                default=d_own * 2)
            violation[i] = np.log1p(d_own + 1e-8) - np.log1p(d_other + 1e-8)
        return violation


class InfluenceBoundedness(RelationalInvariant):
    def __init__(self):
        super().__init__(
            "I3: Influence boundedness", "I3:Infl",
            "LOO influence bounded for clean sub-Gaussian data",
            [slice(21, 34), slice(47, 55)])

    def compute_statistic(self, X, y, rpf=None, y_cont=None):
        n, d = X.shape
        y_reg = y_cont.astype(np.float64) if y_cont is not None \
                else y.astype(np.float64)
        try:
            ridge = Ridge(alpha=1.0).fit(X.astype(np.float64), y_reg)
            y_hat = ridge.predict(X.astype(np.float64))
            resid = y_reg - y_hat
            XtX_inv = np.linalg.inv(
                X.astype(np.float64).T @ X.astype(np.float64)
                + np.eye(d, dtype=np.float64))
            H_diag = (X.astype(np.float64) @ XtX_inv
                      * X.astype(np.float64)).sum(1)
            mse = (resid ** 2).mean() + 1e-8
            stud = resid / np.sqrt(mse * (1 - np.clip(H_diag, 0, 0.999) + 1e-8))
            cooks = (stud ** 2 / max(min(d, n-1), 1)) * (H_diag / (1 - H_diag + 1e-8))
            return cooks.astype(np.float32)
        except Exception:
            return np.zeros(n, np.float32)


class StructuralStability(RelationalInvariant):
    def __init__(self):
        super().__init__(
            "I4: Structural stability", "I4:Struct",
            "Graph centrality x label disagreement bounded above",
            [slice(55, 61)])

    def compute_statistic(self, X, y, rpf=None, y_cont=None):
        n = len(X)
        K = len(np.unique(y))
        k = min(15, n - 2)
        chance = 1.0 / max(K, 2)
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(X)
        _, I_neighbors = nbrs.kneighbors(X)
        I_neighbors = I_neighbors[:, 1:]
        rev_counts = np.zeros(n, np.float32)
        np.add.at(rev_counts, I_neighbors.ravel(), 1.0)
        centrality = rev_counts / (rev_counts.mean() + 1e-8)
        raw_agree = (y[I_neighbors] == y[:, np.newaxis]).mean(1).astype(np.float32)
        denom = max(1.0 - chance, 1e-8)
        excess = np.clip((raw_agree - chance) / denom, 0.0, 1.0)
        return (centrality * (1.0 - excess + 0.01)).astype(np.float32)


class ScaleConsistency(RelationalInvariant):
    def __init__(self):
        super().__init__(
            "I5: Scale consistency", "I5:Scale",
            "Feature magnitude consistent within class",
            [slice(17, 21)])

    def compute_statistic(self, X, y, rpf=None, y_cont=None):
        n = len(X)
        classes = np.unique(y)
        l2 = np.linalg.norm(X, axis=1)

        l2_z = np.zeros(n, np.float32)
        rel_scale = np.zeros(n, np.float32)
        for c in classes:
            idx = np.where(y == c)[0]
            if len(idx) < 2:
                continue
            mu = l2[idx].mean()
            sd = l2[idx].std() + 1e-8
            l2_z[idx] = np.abs((l2[idx] - mu) / sd)
            rel_scale[idx] = np.abs(np.log(l2[idx] / (mu + 1e-8) + 1e-8))

        return ((l2_z + rel_scale) / 2.0).astype(np.float32)

def compute_row_violations(self, X, y, y_cont=None):
    assert self._fitted, "Call fit_clean_bounds() first"

    results = {}

    for inv in self.invariants:
        stat = inv.compute_statistic(X, y, y_cont=y_cont)
        bound = self.clean_bounds[inv.name]["bound"]

        # Simple threshold-based violation
        results[inv.short_name.split(":")[0]] = (stat > bound)

    return results
#  EFFECTIVENESS FILTER
def measure_attack_effectiveness(X_clean, y_clean, X_poisoned, y_poisoned,
                                  min_accuracy_drop=0.01):
    try:
        n = len(X_clean)
        if len(np.unique(y_clean)) < 2:
            return True, 0.0, 0.0, 0.0

        rng = np.random.default_rng(42)
        test_size = max(20, int(n * 0.2))
        test_idx = rng.choice(n, test_size, replace=False)
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False

        X_test, y_test = X_clean[test_idx], y_clean[test_idx]

        clf_c = LogisticRegression(C=1.0, max_iter=300, random_state=42)
        clf_c.fit(X_clean[train_mask], y_clean[train_mask])
        acc_c = accuracy_score(y_test, clf_c.predict(X_test))

        clf_p = LogisticRegression(C=1.0, max_iter=300, random_state=42)
        clf_p.fit(X_poisoned, y_poisoned)
        acc_p = accuracy_score(y_test, clf_p.predict(X_test))

        drop = acc_c - acc_p
        return drop >= min_accuracy_drop, float(drop), float(acc_c), float(acc_p)
    except Exception:
        return True, 0.0, 0.0, 0.0

#  INVARIANT ANALYZER
class InvariantAnalyzer:
    """v2: 5 invariants, effectiveness filter, primary-detector complementarity."""

    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.invariants = [
            NeighborhoodConsistency(),
            GeometricCoherence(),
            InfluenceBoundedness(),
            StructuralStability(),
            ScaleConsistency(),
        ]
        self.clean_bounds = {}
        self.clean_stats = {}
        self._fitted = False

    def fit_clean_bounds(self, X, y, y_cont=None, n_bootstrap=10, seed=42):
        rng = np.random.default_rng(seed)
        n = len(X)
        for inv in self.invariants:
            full_stats = inv.compute_statistic(X, y, y_cont=y_cont)
            self.clean_stats[inv.name] = full_stats
            bounds = []
            for _ in range(n_bootstrap):
                idx = rng.choice(n, n, replace=True)
                yc = y_cont[idx] if y_cont is not None else None
                s = inv.compute_statistic(X[idx], y[idx], y_cont=yc)
                bounds.append(inv.estimate_clean_bound(s, self.alpha))
            self.clean_bounds[inv.name] = {
                "bound": float(np.percentile(bounds, 95)),
                "mean": float(np.mean(bounds)),
                "std": float(np.std(bounds)),
                "clean_violation_rate": float(
                    (full_stats > np.percentile(bounds, 95)).mean()),
            }
        self._fitted = True
        return self.clean_bounds

    def analyze_attack(self, X, y, poison_mask, y_cont=None,
                       attack_name="unknown"):
        assert self._fitted, "Call fit_clean_bounds() first"
        results = {
            "attack": attack_name,
            "n_poison": int(poison_mask.sum()),
            "n_clean": int((~poison_mask).sum()),
            "invariants": {},
            "any_violated": False,
            "n_violated": 0,
            "violation_signature": [],
            "primary_invariant": None,
            "primary_cohens_d": 0.0,
        }
        clean_idx = np.where(~poison_mask)[0]
        poison_idx = np.where(poison_mask)[0]
        if len(poison_idx) < 3 or len(clean_idx) < 3:
            return results

        best_d, best_inv = -np.inf, None

        for inv in self.invariants:
            stat = inv.compute_statistic(X, y, y_cont=y_cont)
            bound = self.clean_bounds[inv.name]["bound"]
            sc, sp = stat[clean_idx], stat[poison_idx]
            vr_p = float((sp > bound).mean())
            vr_c = float((sc > bound).mean())

            try:
                _, p_val = mannwhitneyu(sp, sc, alternative="greater")
                p_val = float(p_val)
            except Exception:
                p_val = 1.0

            pooled = np.sqrt(
                (sc.var() * len(sc) + sp.var() * len(sp))
                / (len(sc) + len(sp))) + 1e-8
            d = float((sp.mean() - sc.mean()) / pooled)

            try:
                ks_s, ks_p = ks_2samp(sp, sc)
            except Exception:
                ks_s, ks_p = 0.0, 1.0

            violated = p_val < 0.05 and d > 0.2 and vr_p > vr_c * 1.5

            results["invariants"][inv.name] = {
                "violation_rate_poison": vr_p,
                "violation_rate_clean": vr_c,
                "violation_rate_ratio": vr_p / (vr_c + 1e-8),
                "mann_whitney_p": p_val,
                "cohens_d": d,
                "ks_statistic": float(ks_s),
                "ks_p": float(ks_p),
                "is_violated": violated,
                "mean_poison": float(sp.mean()),
                "mean_clean": float(sc.mean()),
                "bound": bound,
            }
            if violated:
                results["any_violated"] = True
                results["n_violated"] += 1
                results["violation_signature"].append(inv.name)
            if d > best_d:
                best_d, best_inv = d, inv.name

        results["primary_invariant"] = best_inv
        results["primary_cohens_d"] = float(best_d)
        return results

    def coverage_analysis(self, datasets, attacks, apply_attack_fn,
                          rates=(0.05, 0.10, 0.20), seeds=3,
                          min_accuracy_drop=0.01, verbose=True):
        """Effectiveness filter + primary-detector complementarity."""
        all_results = []
        attack_inv_hits = defaultdict(lambda: defaultdict(list))
        per_attack_cov = defaultdict(list)
        inv_names = [inv.name for inv in self.invariants]
        n_effective, n_ineffective = 0, 0
        ineffective_log = []

        for ds_name, ds_val in datasets.items():
            X, y = ds_val[0], ds_val[1]
            y_cont = ds_val[2] if len(ds_val) == 3 else None
            self.fit_clean_bounds(X, y, y_cont=y_cont)

            for atk in attacks:
                for rate in rates:
                    for seed in range(seeds):
                        try:
                            Xp, yp, pidx = apply_attack_fn(
                                atk, X, y, fraction=rate,
                                seed=seed * 1000 + hash(atk) % 997,
                                y_cont=y_cont)
                            if len(pidx) < 3:
                                continue

                            # Effectiveness filter
                            eff, drop, ca, pa = measure_attack_effectiveness(
                                X, y, Xp, yp,
                                min_accuracy_drop=min_accuracy_drop)
                            if not eff:
                                n_ineffective += 1
                                ineffective_log.append(
                                    (ds_name, atk, rate, drop))
                                continue
                            n_effective += 1

                            mask = np.zeros(len(Xp), dtype=bool)
                            mask[pidx] = True
                            r = self.analyze_attack(
                                Xp, yp, mask, y_cont=y_cont,
                                attack_name=atk)
                            r["dataset"] = ds_name
                            r["rate"] = rate
                            r["seed"] = seed
                            r["accuracy_drop"] = drop
                            all_results.append(r)
                            per_attack_cov[atk].append(r["any_violated"])
                            for iname, ires in r["invariants"].items():
                                attack_inv_hits[atk][iname].append(
                                    ires["is_violated"])
                        except Exception as e:
                            if verbose:
                                print(f"  [cov] {ds_name}|{atk}|{rate}: {e}")

        total = len(all_results)
        covered = sum(1 for r in all_results if r["any_violated"])

        attack_coverage = {
            atk: {"coverage": float(np.mean(f)), "n_trials": len(f)}
            for atk, f in per_attack_cov.items()}

        attack_inv_matrix = {}
        for atk, inv_dict in attack_inv_hits.items():
            attack_inv_matrix[atk] = {
                iname: {
                    "hit_rate": float(np.mean(inv_dict.get(iname, [0]))),
                    "n_trials": len(inv_dict.get(iname, [])),
                } for iname in inv_names}

        # Primary-detector complementarity
        primary_counts = defaultdict(lambda: defaultdict(int))
        for r in all_results:
            if r["primary_invariant"]:
                primary_counts[r["attack"]][r["primary_invariant"]] += 1

        complementarity = {}
        for inv in self.invariants:
            exclusive = sum(
                1 for r in all_results
                if inv.name in r["violation_signature"]
                and len(r["violation_signature"]) == 1)
            total_c = sum(
                1 for r in all_results
                if inv.name in r["violation_signature"])
            primary_for = []
            for atk, counts in primary_counts.items():
                t = sum(counts.values())
                frac = counts.get(inv.name, 0) / t if t > 0 else 0
                if frac > 0.3:
                    primary_for.append((atk, frac))

            complementarity[inv.name] = {
                "total_catches": total_c,
                "exclusive_catches": exclusive,
                "primary_for": primary_for,
                "is_necessary": exclusive > 0 or len(primary_for) > 0,
                "necessity_reason": (
                    "exclusive" if exclusive > 0
                    else "primary" if primary_for
                    else "redundant"),
            }

        return {
            "coverage_rate": covered / total if total > 0 else 0.0,
            "n_total": total, "n_covered": covered,
            "n_effective": n_effective, "n_ineffective": n_ineffective,
            "attack_coverage": attack_coverage,
            "attack_invariant_matrix": attack_inv_matrix,
            "complementarity": complementarity,
            "hardest_attacks": sorted(
                attack_coverage.items(),
                key=lambda x: x[1]["coverage"])[:5],
            "all_results": all_results,
            "ineffective_log": ineffective_log,
        }

    def transferability_analysis(self, datasets, verbose=True):
        inv_names = [inv.name for inv in self.invariants]
        ds_stats = {}
        jitter_rng = np.random.default_rng(123)

        for ds_name, ds_val in datasets.items():
            X, y = ds_val[0], ds_val[1]
            y_cont = ds_val[2] if len(ds_val) == 3 else None
            ds_stats[ds_name] = {}
            for inv in self.invariants:
                raw = inv.compute_statistic(X, y, y_cont=y_cont)
                # Add tiny jitter to break ties (magnitude << data scale)
                jitter = jitter_rng.uniform(-1e-7, 1e-7, len(raw))
                ranked = rankdata(raw + jitter).astype(np.float32) / len(raw)
                ds_stats[ds_name][inv.name] = ranked

        ds_names = list(ds_stats.keys())
        n_ds = len(ds_names)
        ks_matrices = {nm: np.zeros((n_ds, n_ds)) for nm in inv_names}
        for i in range(n_ds):
            for j in range(i + 1, n_ds):
                for nm in inv_names:
                    ks, _ = ks_2samp(
                        ds_stats[ds_names[i]][nm],
                        ds_stats[ds_names[j]][nm])
                    ks_matrices[nm][i, j] = ks
                    ks_matrices[nm][j, i] = ks

        summary = {}
        for nm in inv_names:
            u = ks_matrices[nm][np.triu_indices(n_ds, k=1)]
            summary[nm] = {
                "mean_ks": float(u.mean()) if len(u) > 0 else 0.0,
                "max_ks": float(u.max()) if len(u) > 0 else 0.0,
                "std_ks": float(u.std()) if len(u) > 0 else 0.0,
            }
        return {"summary": summary, "ks_matrices": ks_matrices,
                "dataset_names": ds_names}

    # Reporting

    def print_coverage_report(self, cr):
        W = 84
        print("\n" + "=" * W)
        print("  RELATIONAL INVARIANT VIOLATION -- COVERAGE (v2)")
        print("=" * W)
        print(f"\n  Effective attacks: {cr['n_effective']}  "
              f"(filtered {cr['n_ineffective']} ineffective)")
        print(f"  Coverage: {cr['coverage_rate']:.1%} "
              f"({cr['n_covered']}/{cr['n_total']})")

        inv_names = [inv.name for inv in self.invariants]
        shorts = [inv.short_name for inv in self.invariants]
        matrix = cr["attack_invariant_matrix"]
        ac = cr["attack_coverage"]

        print(f"\n  {'Attack':<22}", end="")
        for s in shorts:
            print(f" {s:>9}", end="")
        print(f" {'Covg':>6}")
        print(f"  {'─' * (22 + 10 * len(shorts) + 7)}")

        for atk in sorted(matrix.keys()):
            print(f"  {atk:<22}", end="")
            for nm in inv_names:
                h = matrix[atk].get(nm, {}).get("hit_rate", 0)
                m = "+" if h > 0.5 else ("~" if h > 0.2 else "-")
                print(f" {h:>6.0%} {m}", end="")
            print(f" {ac.get(atk, {}).get('coverage', 0):>5.0%}")

        print(f"\n  COMPLEMENTARITY (primary-detector analysis):")
        print(f"  {'Invariant':<36} {'Catch':>5} {'Excl':>5} "
              f"{'Primary for':>28} {'Nec?':>12}")
        print(f"  {'─' * 90}")
        for inv in self.invariants:
            c = cr["complementarity"][inv.name]
            pstr = ", ".join(f"{a}({d:.0%})" for a, d in c["primary_for"]
                             ) if c["primary_for"] else "---"
            if len(pstr) > 26:
                pstr = pstr[:26] + ".."
            nec = f"YES({c['necessity_reason']})" if c["is_necessary"] else "no"
            print(f"  {inv.name:<36} {c['total_catches']:>5} "
                  f"{c['exclusive_catches']:>5} {pstr:>28} {nec:>12}")

    def print_transferability_report(self, tr):
        W = 84
        print("\n" + "=" * W)
        print("  TRANSFERABILITY (v2: class-count-invariant)")
        print("=" * W)
        for nm, info in tr["summary"].items():
            q = ("strong" if info["mean_ks"] < 0.15
                 else "moderate" if info["mean_ks"] < 0.30 else "WEAK")
            bar = "█" * max(0, int((1 - info["mean_ks"]) * 25))
            print(f"  {nm:<36}  KS={info['mean_ks']:.4f}"
                  f"+-{info['std_ks']:.4f}  max={info['max_ks']:.4f}"
                  f"  {bar}  [{q}]")
        print(f"\n  Overall: {np.mean([v['mean_ks'] for v in tr['summary'].values()]):.4f}")


#  GAUSSIAN MIXTURE BOUNDS
class GaussianMixtureBounds:
    @staticmethod
    def print_bounds(delta, sigma, k, d, n, K=2):
        from scipy.stats import norm
        snr = delta / (sigma * np.sqrt(2))
        p_same = max(1.0/K, min(1.0 - (K-1)*norm.cdf(-snr/2), 1.0))
        eps = np.sqrt(np.log(20.0) / (2 * k))
        d_own = sigma * np.sqrt(d)
        d_other = np.sqrt(delta**2 + d * sigma**2)
        cooks_bound = (2*d)/n + np.sqrt(np.log(20)/n)
        z_bound = np.sqrt(2 * np.log(40.0))

        print(f"\n  Theoretical Bounds (Gaussian Mixture, 5 invariants)")
        print(f"  delta={delta}, sigma={sigma}, k={k}, d={d}, n={n}, K={K}")
        print(f"  SNR = {delta/(sigma*np.sqrt(d)):.2f}")
        print(f"  I1: E[agreement] >= {p_same:.3f}, "
              f"violation if < {max(0, p_same - eps):.3f}")
        print(f"  I2: E[d_own] ~ {d_own:.2f}, E[d_other] ~ {d_other:.2f}")
        print(f"  I3: Cook's bound = {cooks_bound:.4f}")
        print(f"  I4: Structural bound = {0.5/K + 0.1:.3f}")
        print(f"  I5: Scale z-bound = {z_bound:.2f}")


#  SYNTHETIC VALIDATION (7 attacks including dist_shift, repr_inversion)
def synthetic_validation(n=2000, d=20, delta=3.0, sigma=1.0, k=15,
                         poison_rate=0.10, verbose=True):
    rng = np.random.default_rng(42)
    n2 = n // 2
    mu0 = np.zeros(d)
    mu1 = np.zeros(d)
    mu1[0] = delta
    X = np.vstack([
        rng.normal(mu0, sigma, (n2, d)),
        rng.normal(mu1, sigma, (n2, d))
    ]).astype(np.float32)
    y = np.array([0]*n2 + [1]*n2)

    if verbose:
        print(f"\n  Synthetic: n={n}, d={d}, delta={delta}, sigma={sigma}")
        GaussianMixtureBounds.print_bounds(delta, sigma, k, d, n)

    ana = InvariantAnalyzer(alpha=0.05)
    ana.fit_clean_bounds(X, y)

    if verbose:
        print(f"\n  Empirical clean bounds:")
        for inv in ana.invariants:
            b = ana.clean_bounds[inv.name]
            print(f"    {inv.name}: bound={b['bound']:.4f} "
                  f"(clean viol={b['clean_violation_rate']:.1%})")

    np_ = int(n * poison_rate)
    attacks = {}

    # 1. Label flip
    Xp, yp = X.copy(), y.copy()
    pidx = rng.choice(n, np_, replace=False)
    yp[pidx] = 1 - yp[pidx]
    attacks["label_flip"] = (Xp, yp, pidx)

    # 2. Feature perturbation
    Xp, yp = X.copy(), y.copy()
    pidx = rng.choice(n, np_, replace=False)
    Xp[pidx] += 4.0 * sigma * rng.choice([-1., 1.], (np_, d))
    attacks["feat_perturb"] = (Xp, yp, pidx)

    # 3. Clean-label
    Xp, yp = X.copy(), y.copy()
    pidx = rng.choice(n, np_, replace=False)
    for i in pidx:
        Xp[i] += 0.7 * ((mu1 if y[i] == 0 else mu0) - X[i])
    attacks["clean_label"] = (Xp, yp, pidx)

    # 4. Backdoor
    Xp, yp = X.copy(), y.copy()
    pidx = rng.choice(n, np_, replace=False)
    Xp[pidx, :3] = 5.0
    yp[pidx] = 1 - yp[pidx]
    attacks["backdoor"] = (Xp, yp, pidx)

    # 5. Boundary flip
    Xp, yp = X.copy(), y.copy()
    bd = np.abs(X[:, 0] - delta / 2)
    cands = np.argsort(bd)[:np_ * 3]
    pidx = rng.choice(cands, np_, replace=False)
    yp[pidx] = 1 - yp[pidx]
    attacks["boundary_flip"] = (Xp, yp, pidx)

    # 6. Distribution shift (needs I5)
    Xp, yp = X.copy(), y.copy()
    pidx = rng.choice(n, np_, replace=False)
    Xp[pidx] = 2.0 * Xp[pidx]
    attacks["dist_shift"] = (Xp, yp, pidx)

    # 7. Representation inversion
    Xp, yp = X.copy(), y.copy()
    pidx = rng.choice(n, np_, replace=False)
    Xp[pidx] = -Xp[pidx]
    attacks["repr_inversion"] = (Xp, yp, pidx)

    ni = len(ana.invariants)
    results = {}
    for aname, (Xp, yp, pidx) in attacks.items():
        mask = np.zeros(len(Xp), dtype=bool)
        mask[pidx] = True
        r = ana.analyze_attack(Xp, yp, mask, attack_name=aname)
        results[aname] = r
        if verbose:
            print(f"\n  {aname}: violated={r['n_violated']}/{ni}  "
                  f"primary={r['primary_invariant']}")
            for iname, ir in r["invariants"].items():
                tag = "VIOLATED" if ir["is_violated"] else "ok"
                pri = " <-PRIMARY" if iname == r["primary_invariant"] \
                      and ir["is_violated"] else ""
                print(f"    {iname[:30]:<30}  d={ir['cohens_d']:>5.2f}  "
                      f"p={ir['mann_whitney_p']:.1e}  "
                      f"vr={ir['violation_rate_poison']:.0%}  {tag}{pri}")

    nc = sum(1 for r in results.values() if r["any_violated"])
    ok = nc == len(results)
    if verbose:
        print(f"\n  COVERAGE: {'ALL DETECTED' if ok else 'GAP'}  "
              f"({nc}/{len(results)})")
    return results, ana