"""Evaluation primitives for the RPFNet paper."""
import hashlib
import json
import os
import sys
import time
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, spearmanr, studentized_range, wilcoxon
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from RPFNet.Attack import apply_attack, attack_seed
from RPFNet.Dataset import load_builtin, load_csv
from RPFNet.RPFExtractor import RPFExtractor
from RPFNet.effect_blocks import make_extractor
from RPFNet.RateEstimator import (estimate_contamination_rate, _shape_based_rate,
                                  empirical_pvalues, bh_select, storey_rate)
from RPFNet.baselines import BASELINES, REGRESSION_BASELINES, ASSUMED_RATE

CSV_DATASETS = {
    "bank_marketing": ("Datasets/csv/bank.csv", "deposit"),
    "loan_data": ("Datasets/csv/loan_data.csv", None),
}

# Attack families (the notebook's ATTACK_FAMILIES, plus the new attacks).
FAMILY = {
    **dict.fromkeys(["label_flip", "boundary_flip", "targeted_class", "grad_flip",
                     "tree_aware", "target_flip_extreme"], "label_flip"),
    **dict.fromkeys(["feat_perturb", "gauss_noise", "feat_dropout",
                     "combo_flip_perturb", "combo_flip_noise"], "noise"),
    **dict.fromkeys(["clean_label", "interpolation", "adaptive_blend",
                     "gradient_matching", "feature_collision"], "clean_label"),
    **dict.fromkeys(["badnets", "badnets_heavy", "backdoor", "backdoor_heavy", "catback"], "backdoor"),
    **dict.fromkeys(["repr_inversion", "dist_shift", "outlier_inject",
                     "null_feature"], "structural"),
    **dict.fromkeys(["target_shift", "leverage_attack"], "regression"),
    **dict.fromkeys(["influence_lr", "subpopulation", "adaptive_dup", "evasive"], "optimised"),
    **dict.fromkeys(["koh_minmax", "koh_influence", "camouflage"], "sanitiser_aware"),
}

IFOREST_FEATURES = ["E_iforest", "E_iforest_x_impurity", "E_iforest_rank"]


# ---------------------------------------------------------------- data
def load_named(name, root="."):
    """Load a dataset by name into a dict with train and test parts."""
    if name in CSV_DATASETS:
        path, target = CSV_DATASETS[name]
        r = load_csv(os.path.join(root, path), target)
        extra = None
    else:
        r = load_builtin(name)
        extra = r[4] if r[6] else None
    return {"X": r[0], "y": np.asarray(r[2]), "y_cont": r[7], "X_te": r[1],
            "y_te": np.asarray(r[3]), "y_te_cont": extra, "is_reg": bool(r[6])}


def dataset_folds(names, is_reg, n_folds, seed=0):
    """Split dataset names into folds, spreading regression sets evenly."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_folds)]
    i = 0
    for group in ([n for n in names if not is_reg[n]], [n for n in names if is_reg[n]]):
        for name in rng.permutation(group):
            folds[i % n_folds].append(str(name))
            i += 1
    return folds


def make_splits(cfg, is_reg):
    """Train/test splits for the configured protocol."""
    p = cfg["protocol"]
    names = cfg["datasets"]
    if p["scheme"] == "fixed":
        return [{"split": "fixed", "train": p["train"], "test": p["test"], "exclude_family": None}]
    folds = dataset_folds(names, is_reg, p["n_folds"], p.get("seed", 0))
    if p["scheme"] == "dataset_kfold":
        return [{"split": f"fold{f}", "test": folds[f],
                 "train": [n for g, fold in enumerate(folds) if g != f for n in fold],
                 "exclude_family": None} for f in range(p["n_folds"])]
    if p["scheme"] == "family_out":
        e = p.get("eval_fold", 0)
        train = [n for g, fold in enumerate(folds) if g != e for n in fold]
        t = cfg["train"]
        fams = sorted({FAMILY[a] for a in t["attacks"] + t.get("attacks_reg", [])})
        return [{"split": f"no_{fam}", "test": folds[e], "train": train,
                 "exclude_family": fam} for fam in fams]
    raise ValueError(f"Unknown protocol scheme {p['scheme']!r}")


def split_attacks(cfg, split, is_reg_test):
    """(train_cls, train_reg, test_attack_list) for a split."""
    t, e = cfg["train"], cfg["eval"]
    fam = split["exclude_family"]
    tr_c = [a for a in t["attacks"] if FAMILY.get(a) != fam]
    tr_r = [a for a in t.get("attacks_reg", []) if FAMILY.get(a) != fam]
    if fam is None:
        te = (e["attacks_reg"] if is_reg_test else e["attacks"]) + (
            [] if is_reg_test else e.get("unseen_attacks", []))
    else:   # every configured attack of the held-out family
        pool = (e["attacks_reg"] + t.get("attacks_reg", [])) if is_reg_test else (
            e["attacks"] + t["attacks"])
        te = sorted({a for a in pool if FAMILY.get(a) == fam})
    return tr_c, tr_r, te


# ---------------------------------------------------------------- cache
def source_hash(*paths):
    """Short hash of source files, ignoring line endings."""
    h = hashlib.sha1()
    for p in paths:
        with open(p, "rb") as f:
            h.update(f.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:8]


class RPFCache:
    """Disk cache of fingerprints, usable as `rpf_fn` in build_pool/_calibrate."""

    def __init__(self, root, extractor: RPFExtractor):
        self.ex = extractor
        # A composite extractor (version 3) caches each part separately, so its
        # version-2 part reuses the version-2 cache.
        self.parts = [RPFCache(root, p) for p in extractor.parts] if hasattr(extractor, "parts") else None
        if self.parts is not None:
            self.dir = None
            return
        files = getattr(extractor, "source_files", None) or [sys.modules[type(extractor).__module__].__file__]
        tag = getattr(extractor, "cache_tag", None) or (
            f"v{extractor.feature_version}_k{extractor.k_small}-{extractor.k_large}"
            f"_cv{extractor.cv_folds}_rs{extractor.random_state}")
        self.dir = os.path.join(root, f"{tag}_src{source_hash(*files)}")
        os.makedirs(self.dir, exist_ok=True)

    def _path(self, key, *arrays):
        h = hashlib.sha1(repr(tuple(key)).encode())
        for a in arrays:
            if a is not None:
                a = np.ascontiguousarray(a)
                h.update(f"{a.dtype.str}{a.shape}".encode())
                h.update(a.tobytes())
        return os.path.join(self.dir, h.hexdigest()[:24])

    def __call__(self, key, X, y, y_cont=None):
        return self.get(key, X, y, y_cont)[0]

    def get(self, key, X, y, y_cont=None):
        """Returns (rpf, timings); timings are those of the original extraction."""
        if self.parts is not None:
            fv = self.ex.feature_version
            outs = [c.get(tuple(getattr(c.ex, "feature_version", fv) if i == 1 and k == fv else k
                                for i, k in enumerate(key)), X, y, y_cont) for c in self.parts]
            return (np.hstack([o[0] for o in outs]).astype(np.float32),
                    {k: v for o in outs for k, v in o[1].items()})
        p = self._path(key, X, y, y_cont)
        if os.path.exists(p + ".npy") and os.path.exists(p + ".json"):
            with open(p + ".json") as f:
                return np.load(p + ".npy"), json.load(f)
        rpf = self.ex.extract(X, y, y_cont=y_cont)
        _atomic_save(p, rpf, self.ex.last_timings)
        return rpf, dict(self.ex.last_timings)


def _atomic_save(p, array, meta):
    """Write a cache entry atomically."""
    tmp = f"{p}.{os.getpid()}.tmp"
    with open(tmp + ".json", "w") as f:
        json.dump(meta, f)
    os.replace(tmp + ".json", p + ".json")
    np.save(tmp + ".npy", array)
    os.replace(tmp + ".npy", p + ".npy")


class BaselineCache:
    """Disk cache of baseline outputs."""

    def __init__(self, root):
        import RPFNet.baselines as B
        self.dir = os.path.join(root, f"baselines_src{source_hash(B.__file__)}")
        os.makedirs(self.dir, exist_ok=True)

    def get(self, method, X, t, seed):
        """t: the labels, or the continuous target for regression baselines."""
        h = hashlib.sha1(repr((method, int(seed))).encode())
        for a in (X, t):
            a = np.ascontiguousarray(a)
            h.update(f"{a.dtype.str}{a.shape}".encode())
            h.update(a.tobytes())
        p = os.path.join(self.dir, h.hexdigest()[:24])
        if os.path.exists(p + ".npy") and os.path.exists(p + ".json"):
            with open(p + ".json") as f:
                meta = json.load(f)
            v = np.load(p + ".npy")
            return v[0], (v[1] > 0.5 if meta["has_flags"] else None), meta["time_s"]
        fn = REGRESSION_BASELINES[method] if method in REGRESSION_BASELINES else BASELINES[method]
        t0 = time.perf_counter()
        sc, fl = fn(X, t, seed)
        dt = time.perf_counter() - t0
        sc = np.asarray(sc, dtype=np.float64)
        fl = None if fl is None else np.asarray(fl, dtype=bool)
        _atomic_save(p, np.stack([sc, np.zeros(len(sc)) if fl is None else fl.astype(np.float64)]),
                     {"has_flags": fl is not None, "time_s": dt})
        return sc, fl, dt


# ---------------------------------------------------------------- parallel precompute
_WORKER = {}


def _worker_state(cache_root, ex_args):
    """Per-process caches, and single-threaded libraries: the workers already
    run in parallel. None of these settings changes any output."""
    if _WORKER.get("args") != (cache_root, ex_args):
        import torch
        import RPFNet.Attack as A
        import RPFNet.RPFExtractor as RX
        RX.N_JOBS = A.N_JOBS = 1
        if RX.FAISS_AVAILABLE:
            RX.faiss.omp_set_num_threads(1)
        torch.set_num_threads(1)
        _WORKER.update(args=(cache_root, ex_args), cache=RPFCache(cache_root, make_extractor(*ex_args)),
                       bcache=BaselineCache(cache_root))
    return _WORKER["cache"], _WORKER["bcache"]


def _precompute_one(task, cache_root, ex_args):
    """One configuration: the same attack call and cache key as build_pool,
    _calibrate/minp_scorers (clean) and evaluate_config use, so they find it."""
    from threadpoolctl import threadpool_limits
    cache, bcache = _worker_state(cache_root, ex_args)
    fv, name, X, y, yc = ex_args[0], task["name"], task["X"], task["y"], task["y_cont"]
    try:
        with threadpool_limits(1):
            if task["kind"] == "clean":
                cache(("clean", fv, name), X, y, yc)
                return None
            atk, rate, seed = task["attack"], task["rate"], task["seed"]
            if task["kind"] == "pool":
                Xp, yp, pidx, ycp = apply_attack(atk, X, y, fraction=rate, seed=attack_seed(atk, seed),
                                                 y_cont=yc, return_y_cont=True)
                key = ("RPFNet", fv, name, atk, rate, seed)
            else:
                Xp, yp, pidx, ycp = poison({"X": X, "y": y, "y_cont": yc}, atk, rate, seed)
                key = ("eval", fv, name, atk, rate, seed)
            if len(pidx) < 3:
                return None
            cache(key, Xp, yp, ycp)
            for m in task["baselines"]:
                if m in REGRESSION_BASELINES and not task["is_reg"]:
                    continue
                bcache.get(m, Xp, ycp if m in REGRESSION_BASELINES else yp, seed)
    except Exception as exc:
        return f"{task['kind']}|{name}|{task.get('attack')}: {type(exc).__name__}: {exc}"
    return None


def precompute(tasks, cache_root, ex_args, n_jobs=None):
    """Fill the fingerprint and baseline caches for every task in parallel processes, largest datasets first."""
    from joblib import Parallel, delayed, cpu_count
    n_jobs = n_jobs or max(1, cpu_count(only_physical_cores=True))
    cost = {}
    for t in tasks:   # rough extraction cost: rows x features x classes
        if t["name"] not in cost:
            X, y = t["X"], t["y"]
            cost[t["name"]] = X.shape[0] * X.shape[1] * (1 if t["is_reg"] else max(1, len(np.unique(y)) - 1))
    tasks = sorted(tasks, key=lambda t: -cost[t["name"]])
    t0 = time.perf_counter()
    errors = [e for e in Parallel(n_jobs=n_jobs, verbose=2)(
        delayed(_precompute_one)(t, cache_root, ex_args) for t in tasks) if e]
    print(f"  precomputed {len(tasks)} configurations on {n_jobs} processes in {time.perf_counter() - t0:.0f}s"
          + (f"; {len(errors)} failed and will be redone sequentially, e.g. {errors[0]}" if errors else ""))


# ---------------------------------------------------------------- metrics
def dataset_stat(scores, frac=0.01, kmin=5, robust=False):
    """Dataset-level statistic of the dataset test (Proposition 5)."""
    s = np.asarray(scores, dtype=np.float64)
    if robust:
        q1, q2, q3 = np.percentile(s, [25, 50, 75])
        s = (s - q2) / max(q3 - q1, 1e-12)
    k = min(len(s), max(kmin, int(np.ceil(frac * len(s)))))
    return float(np.mean(np.sort(s)[-k:]))


def topk_flags(scores, rate):
    k = max(1, int(len(scores) * rate))
    flags = np.zeros(len(scores), dtype=bool)
    flags[np.argsort(scores)[-k:]] = True
    return flags


def prf(ytrue, flags):
    tp = float((flags & ytrue).sum())
    p = tp / max(flags.sum(), 1)
    r = tp / max(ytrue.sum(), 1)
    return p, r, (2 * p * r / (p + r) if p + r > 0 else 0.0)


def threshold_free(ytrue, scores):
    """AUROC, AUPRC, precision at k = #poison, recall at 1% and 5% FPR."""
    fpr, tpr, _ = roc_curve(ytrue, scores)
    k = int(ytrue.sum())
    return {"AUROC": roc_auc_score(ytrue, scores),
            "AUPRC": average_precision_score(ytrue, scores),
            "P@k": float(ytrue[np.argsort(scores)[-k:]].mean()),
            "R@FPR1": float(np.interp(0.01, fpr, tpr)),
            "R@FPR5": float(np.interp(0.05, fpr, tpr))}


P_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


def _nulls(det, reg):
    """The clean-score nulls for p-values."""
    nulls = {"": det.null_scores} if det.null_scores is not None else {}
    for tag, by_task in getattr(det, "extra_nulls", {}).items():
        if by_task.get(reg) is not None:
            nulls[tag] = by_task[reg]
    return nulls


def null_columns(s, ytrue, nulls, q, by=("_oof",)):
    """BH flags against each null (tag -> clean scores) and the calibration of its p-values on the clean rows."""
    out, n, clean = {}, len(s), ~ytrue
    for tag, null in nulls.items():
        p = empirical_pvalues(s, null)
        sets = {"bh": bh_select(p, q)}
        if tag in by:
            sets["by"] = bh_select(p, q / np.sum(1.0 / np.arange(1, n + 1)))
        for kind, f in sets.items():
            out[f"P_{kind}{tag}"], out[f"R_{kind}{tag}"], out[f"F1_{kind}{tag}"] = prf(ytrue, f)
            out[f"FDR_{kind}{tag}"] = float((f & clean).sum() / max(f.sum(), 1))
            out[f"n_flag_{kind}{tag}"] = int(f.sum())
        out[f"storey_rate{tag}"] = storey_rate(p)
        out[f"t_bh{tag}"] = q * max(int(sets["bh"].sum()), 1) / n
        for t in P_GRID:
            ok = clean.any() and t >= 1.0 / (len(null) + 1)
            out[f"c{t:.0e}{tag}"] = float((p[clean] <= t).mean() / t) if ok else float("nan")
    return out


def clean_alarms(det, rpf, reg, q=0.10):
    """What RPFNet reports on an unpoisoned dataset (every row clean)."""
    s = det._score_rpf(rpf)
    rate_fn = getattr(det, "rate_fn", None)
    rate = rate_fn(rpf, reg) if rate_fn else estimate_contamination_rate(s, probabilities=True)
    row = {"n": len(s), "est_rate_own": float(rate), "n_flag_own": int(topk_flags(s, rate).sum())}
    row.update(null_columns(s, np.zeros(len(s), bool), _nulls(det, reg), q))
    return row


def downstream_score(X, y, X_te, y_te, regression):
    """Clean-test accuracy (classification) or R^2 (regression) of a simple
    model trained on (X, y)."""
    if regression:
        return float(Ridge(alpha=1.0).fit(X, y).score(X_te, y_te))
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(LogisticRegression(max_iter=2000).fit(X, y).score(X_te, y_te))


SLOW_ATTACKS = {"koh_minmax", "koh_influence"}


def poison(data, atk, rate, seed):
    """Deterministic poisoning for evaluation; the seed space is disjoint from
    the one build_pool uses for training."""
    root = os.environ.get("RPF_POISON_CACHE")
    path = None
    if root and atk in SLOW_ATTACKS:
        import RPFNet.Attack as A
        import RPFNet.koh_attacks as KA
        h = hashlib.sha1(repr((atk, float(rate), int(seed))).encode())
        for a in (data["X"], data["y"]):
            a = np.ascontiguousarray(a)
            h.update(f"{a.dtype.str}{a.shape}".encode())
            h.update(a.tobytes())
        d = os.path.join(root, f"poison_src{source_hash(A.__file__, KA.__file__)}")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, h.hexdigest()[:24] + ".npz")
        if os.path.exists(path):
            z = np.load(path, allow_pickle=False)
            return z["Xp"], z["yp"], z["pidx"].astype(int), None
    Xp, yp, pidx, ycp = apply_attack(atk, data["X"], data["y"], fraction=rate,
                                     seed=attack_seed(atk, 10_000 + seed),
                                     y_cont=data["y_cont"], return_y_cont=True)
    if path is not None:
        tmp = path + f".{os.getpid()}.tmp.npz"
        np.savez(tmp, Xp=Xp, yp=yp, pidx=np.asarray(pidx, dtype=int))
        os.replace(tmp, path)
    return Xp, yp, np.asarray(pidx, dtype=int), ycp


def score_dataset(det, Xp, yp, ycp, pidx, data, info, baselines=(), cache=None, cache_key=None,
                  extra_scorers=None, q=0.10, downstream=True, seed=0, bcache=None):
    """Score one poisoned dataset with RPFNet, its variants and the baselines."""
    n = len(Xp)
    ytrue = np.zeros(n, dtype=bool)
    ytrue[pidx] = True
    rate = len(pidx) / n
    reg = data["is_reg"]
    base = dict(info, n=n, n_poison=int(len(pidx)))

    t0 = time.perf_counter()
    if cache is not None and cache_key is not None:
        rpf, timings = cache.get(cache_key, Xp, yp, ycp)
        t_ext = sum(timings.values())
    else:
        rpf = det.extractor.extract(Xp, yp, y_cont=ycp)
        t_ext = time.perf_counter() - t0
    t1 = time.perf_counter()
    s = det._score_rpf(rpf)
    rate_fn = getattr(det, "rate_fn", None)
    own_rate = rate_fn(rpf, reg) if rate_fn else estimate_contamination_rate(s, probabilities=True)
    results = {"RPFNet": (s, topk_flags(s, own_rate), t_ext + time.perf_counter() - t1)}
    alt = {}
    for name, fn in (extra_scorers or {}).items():
        t = time.perf_counter()
        out = fn(rpf, reg) if getattr(fn, "needs_task", False) else fn(rpf)   # reg: regression?
        sc, fl, *more = out if isinstance(out, tuple) else (out, None)
        results[name] = (sc, fl, t_ext + time.perf_counter() - t)
        alt[name] = more[0] if more else {}
    for m in baselines:
        if m in REGRESSION_BASELINES and not reg:
            continue
        tgt = ycp if m in REGRESSION_BASELINES else yp
        if bcache is not None:
            results[m] = bcache.get(m, Xp, tgt, seed)
            continue
        t = time.perf_counter()
        sc, fl = (REGRESSION_BASELINES if m in REGRESSION_BASELINES else BASELINES)[m](Xp, tgt, seed)
        results[m] = (np.asarray(sc, dtype=np.float64), fl, time.perf_counter() - t)

    rows = []
    for m, (sc, own, t) in results.items():
        own = own if own is not None else topk_flags(sc, ASSUMED_RATE)
        est_shared = _shape_based_rate(np.asarray(sc, dtype=np.float64))
        row = dict(base, method=m, time_s=t, **threshold_free(ytrue, sc))
        row["F1_known"] = prf(ytrue, topk_flags(sc, rate))[2]
        row["P_own"], row["R_own"], row["F1_own"] = prf(ytrue, own)
        row["F1_shared"] = prf(ytrue, topk_flags(sc, est_shared))[2]
        row["est_rate_shared"] = est_shared
        for k, f in alt.get(m, {}).items():
            row[f"F1_{k}"] = prf(ytrue, f)[2]
        if alt.get(m):
            row["est_rate_own"] = float(own.mean())
        if m == "RPFNet":
            row["est_rate_own"] = own_rate
            row.update(null_columns(s, ytrue, _nulls(det, reg), q))
        if downstream:
            keep = ~own
            yk = ycp if reg else yp
            row["downstream"] = downstream_score(Xp[keep], yk[keep], data["X_te"],
                                                 data["y_te_cont"] if reg else data["y_te"], reg)
        rows.append(row)
    if downstream:
        yk = ycp if reg else yp
        yte = data["y_te_cont"] if reg else data["y_te"]
        for m, Xk, yk2 in (("_none", Xp, yk), ("_oracle", Xp[~ytrue], yk[~ytrue]),
                           ("_clean", data["X"], data["y_cont"] if reg else data["y"])):
            rows.append(dict(base, method=m, downstream=downstream_score(Xk, yk2, data["X_te"], yte, reg)))
    return rows


def evaluate_config(det, data, ds_name, atk, rate, seed, **kw):
    Xp, yp, pidx, ycp = poison(data, atk, rate, seed)
    info = dict(dataset=ds_name, attack=atk, family=FAMILY.get(atk, "other"), rate=rate, seed=seed)
    if len(pidx) < 3:
        return [dict(info, n=len(Xp), n_poison=int(len(pidx)), method="RPFNet", skipped=True)]
    fv = det.extractor.feature_version
    return score_dataset(det, Xp, yp, ycp, pidx, data, info,
                         cache_key=("eval", fv, ds_name, atk, rate, seed), seed=seed, **kw)


# ---------------------------------------------------------------- statistics
def _methods(df):
    return sorted(m for m in df["method"].dropna().unique() if not str(m).startswith("_"))


def config_table(df, metric):
    """Configs x methods, averaged over seeds (seeds are not independent units)."""
    d = df[df["method"].isin(_methods(df))]
    if "skipped" in d:
        d = d[d["skipped"] != True]  # noqa: E712
    return d.groupby(["dataset", "attack", "rate", "method"])[metric].mean().unstack("method")


def _holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, (len(p) - i) * p[j])
        adj[j] = min(1.0, running)
    return adj


def pooled_tests(df, metric, ref="RPFNet"):
    """Paired Wilcoxon tests of `ref` against every other method, at the
    configuration level and at the dataset level, with Holm correction."""
    T = config_table(df, metric)
    D = T.groupby(level="dataset").mean()
    out = []
    for b in T.columns:
        if b == ref:
            continue
        dc = (T[ref] - T[b]).dropna()
        dd = (D[ref] - D[b]).dropna()
        pc = wilcoxon(dc).pvalue if (dc != 0).any() else 1.0
        pd_ = wilcoxon(dd).pvalue if len(dd) > 1 and (dd != 0).any() else 1.0
        out.append(dict(baseline=b, metric=metric, n_configs=len(dc), n_datasets=len(dd),
                        mean_diff=dc.mean(), win_rate_configs=float((dc > 0).mean()),
                        p_configs=pc, mean_diff_datasets=dd.mean(),
                        win_rate_datasets=float((dd > 0).mean()), p_datasets=pd_))
    res = pd.DataFrame(out)
    if len(res):
        res["p_configs_holm"] = _holm(res["p_configs"])
        res["p_datasets_holm"] = _holm(res["p_datasets"])
    return res


def friedman_nemenyi(df, metric, alpha=0.05):
    """Average ranks over datasets (1 = best), Friedman test and the Nemenyi
    critical difference (Demsar 2006). Methods missing on any dataset drop out."""
    D = config_table(df, metric).groupby(level="dataset").mean().dropna(axis=1)
    k, N = D.shape[1], D.shape[0]
    ranks = D.rank(axis=1, ascending=False).mean().sort_values()
    p = friedmanchisquare(*[D[c] for c in D.columns]).pvalue if k >= 3 and N >= 2 else float("nan")
    cd = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * N))
    return {"metric": metric, "n_datasets": N, "avg_rank": ranks.to_dict(),
            "friedman_p": p, "nemenyi_cd": float(cd)}


def bootstrap_ci(df, metric, method, ref=None, B=2000, seed=0):
    """Hierarchical bootstrap (datasets, then configs within them) of the mean
    of `method`, or of the difference ref - method. Returns (mean, lo, hi)."""
    T = config_table(df, metric)
    v = (T[ref] - T[method]) if ref else T[method]
    groups = [g.dropna().to_numpy() for _, g in v.groupby(level="dataset")]
    groups = [g for g in groups if len(g)]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(B):
        picks = rng.integers(len(groups), size=len(groups))
        stats.append(np.mean(np.concatenate([rng.choice(groups[i], len(groups[i])) for i in picks])))
    return float(v.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def per_rate_table(df, method="RPFNet"):
    """RPFNet by true contamination rate, including skipped configurations."""
    d = df[df["method"] == method]
    skipped = d.get("skipped", pd.Series(False, index=d.index)).fillna(False).astype(bool)
    cols = [c for c in ["est_rate_own", "P_own", "R_own", "F1_own", "F1_known", "AUROC",
                        "AUPRC", "F1_bh", "FDR_bh", "storey_rate"] if c in d]
    t = d[~skipped].groupby("rate")[cols].mean()
    t["n_valid"] = d[~skipped].groupby("rate").size()
    t["n_skipped"] = d[skipped].groupby("rate").size().reindex(t.index).fillna(0).astype(int)
    return t


# ---------------------------------------------------------------- ablation, stability, scaling
def ablation_variants(extractor, blocks=True, iforest=True):
    """Named drop_features lists: one per block, plus the Isolation Forest columns."""
    v = {"full": []}
    if blocks:
        for key, (_, sl) in extractor.block_names.items():
            v[f"no_{key}"] = extractor.feature_names[sl]
    if iforest:
        v["no_iforest"] = IFOREST_FEATURES
    return v


def stability(det, Xp, yp, ycp, k, n_rep=5, frac=0.8, seed=0):
    """Score repeatability under (a) the extractor's internal randomness and (b) 80% row subsamples."""
    ex = det.extractor
    orig = ex.random_state
    S, R = [], []
    try:
        for r in range(n_rep):
            ex.random_state = r
            rpf = ex.extract(Xp, yp, y_cont=ycp)
            R.append(rpf)
            S.append(det._score_rpf(rpf))
    finally:
        ex.random_state = orig

    def topk(s, kk):
        return set(np.argsort(s)[-kk:])

    def summarise(pairs):
        rho = [spearmanr(a, b)[0] for a, b, _ in pairs]   # works on old and new scipy
        jac = [len(topk(a, kk) & topk(b, kk)) / len(topk(a, kk) | topk(b, kk)) for a, b, kk in pairs]
        return float(np.mean(rho)), float(np.mean(jac))

    seed_rho, seed_jac = summarise([(a, b, k) for a, b in combinations(S, 2)])
    feat_sd = np.stack(R).std(0).mean(0)     # per feature, in fingerprint SD units

    rng = np.random.default_rng(seed)
    n = len(Xp)
    subs = []
    for _ in range(n_rep):
        idx = np.sort(rng.choice(n, int(frac * n), replace=False))
        s = np.full(n, np.nan)
        s[idx] = det.score(Xp[idx], yp[idx], y_cont=None if ycp is None else ycp[idx])[0]
        subs.append(s)
    pairs = []
    for a, b in combinations(subs, 2):
        shared = ~np.isnan(a) & ~np.isnan(b)
        kk = max(1, int(round(k * shared.mean())))
        pairs.append((a[shared], b[shared], kk))
    sub_rho, sub_jac = summarise(pairs)
    varying = feat_sd > 1e-6   # most columns are deterministic given the data
    return {"seed_spearman": seed_rho, "seed_topk_jaccard": seed_jac,
            "n_features_varying": int(varying.sum()),
            "feature_sd_mean_varying": float(feat_sd[varying].mean()) if varying.any() else 0.0,
            "feature_sd_max": float(feat_sd.max()),
            "subsample_spearman": sub_rho, "subsample_topk_jaccard": sub_jac}


def scaling(ns, d=20, feature_version=2, seed=0):
    """Extraction time per block on synthetic data of growing size."""
    rows = []
    for n in ns:
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, d)).astype(np.float32)
        y = (X[:, 0] + 0.5 * rng.standard_normal(n) > 0).astype(int)
        Xp, yp, _ = apply_attack("label_flip", X, y, fraction=0.1, seed=seed)
        ex = make_extractor(feature_version, 5, 15, 3)
        t = time.perf_counter()
        ex.extract(Xp, yp)
        rows.append(dict(n=n, d=d, total_s=time.perf_counter() - t,
                         **{f"{b}_s": v for b, v in ex.last_timings.items()}))
    return pd.DataFrame(rows)
