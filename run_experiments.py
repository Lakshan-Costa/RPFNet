"""Run the RPFNet evaluation protocol from a JSON config."""
import argparse
import glob
import json
import os
import platform
import re
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(ROOT, "Backend"), ROOT]

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from RPFNet import evaluation as ev
from RPFNet.Attack import evasive_attack
from RPFNet.detection import RPFNetPoisonDetector
from RPFNet.effect_blocks import EFFECT_NAMES
from RPFNet.combine import LearnedCombiner, MinPCombiner, RateModel, robust_scale
from RPFNet.diagnostics import gradient_diagnostics


def new_detector(cfg, drop=None):
    t = cfg["train"]
    device = "cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu"
    return RPFNetPoisonDetector(device=device, epochs=t["epochs"], batch_size=t["batch_size"],
                                lr=t["lr"], feature_version=t["feature_version"],
                                drop_features=drop, seed=t.get("seed", 0),
                                context=t.get("context", False),
                                balance=t.get("balance_configs", False),
                                scorer=t.get("scorer", "net"), trees=t.get("trees"))


def model_dir(cfg, out):
    """Where a split's scorer is stored."""
    src = cfg.get("models_from")
    return os.path.join(os.path.dirname(out), src, "models") if src else os.path.join(out, "models")


def precompute(cfg, splits, data, cache_root, baselines=()):
    """Fill the fingerprint and baseline caches in parallel."""
    ex = new_detector(cfg).extractor
    t, e = cfg["train"], cfg["eval"]
    tasks, seen = [], set()

    def add(kind, name, atk=None, rate=None, seed=None, bl=()):
        if (kind, name, atk, rate, seed) in seen:
            return
        seen.add((kind, name, atk, rate, seed))
        d = data[name]
        tasks.append(dict(kind=kind, name=name, attack=atk, rate=rate, seed=seed, X=d["X"], y=d["y"],
                          y_cont=d["y_cont"] if d["is_reg"] else None, is_reg=d["is_reg"], baselines=tuple(bl)))

    for split in splits:
        tr_c, tr_r, _ = ev.split_attacks(cfg, split, False)
        for n in split["train"]:
            add("clean", n)
            for atk in (tr_r if data[n]["is_reg"] else tr_c):
                for rate in t["rates"]:
                    for s in range(t["seeds"]):
                        add("pool", n, atk, rate, s)
        for n in split["test"]:
            for atk in ev.split_attacks(cfg, split, data[n]["is_reg"])[2]:
                for rate in e["rates"]:
                    for s in range(e["seeds"]):
                        add("eval", n, atk, rate, s, baselines)
    ev.precompute(tasks, cache_root, (ex.feature_version, ex.k_small, ex.k_large, ex.cv_folds, ex.random_state),
                  cfg.get("n_jobs"))


def train_split(cfg, split, data, cache_root, out, drop=None, tag="", linear=True, keep_pool=False):
    """Train (or load) the detector for one split. Returns (det, cache, extra scorers).
    keep_pool keeps the training pool as det._pool = (X, y) for the rate model and --mode dump."""
    det = new_detector(cfg, drop)
    cache = ev.RPFCache(cache_root, det.extractor)
    path = os.path.join(model_dir(cfg, out), f"{split['split']}{tag}.pt")
    train = {n: (data[n]["X"], data[n]["y"], data[n]["y_cont"]) if data[n]["is_reg"]
             else (data[n]["X"], data[n]["y"]) for n in split["train"]}
    is_reg_test = False
    tr_c, tr_r, _ = ev.split_attacks(cfg, split, is_reg_test)
    t = cfg["train"]
    X_pool, y_pool = det.build_pool(train, tr_c, t["rates"], t["seeds"], verbose=False,
                                    attacks_reg=tr_r, rpf_fn=cache)
    if os.path.exists(path):
        det.load(path)
    elif cfg.get("models_from"):
        sys.exit(f"models_from={cfg['models_from']!r}: {path} does not exist; refusing to retrain.")
    else:
        t0 = time.time()
        det.train_net(X_pool, y_pool, verbose=False)
        adv = t.get("adversarial")
        for rnd in range(adv.get("rounds", 1) if adv else 0):
            # Add datasets poisoned by the evasive attack aimed at the current
            # detector, then retrain from scratch on the larger pool.
            more = det.adversarial_pool(train, adv.get("rates", [0.05, 0.1]), iters=adv.get("iters", 10),
                                        lam=adv.get("lam", 1.0), base=adv.get("base", "label_flip"),
                                        seed=rnd, max_n=adv.get("max_n", 3000), rpf_fn=cache)
            if more is None:
                break
            groups = det._pool_groups + more[2]
            X_pool, y_pool = np.concatenate([X_pool, more[0]]), np.concatenate([y_pool, more[1]])
            det._pool_groups = groups
            det.reset_net()
            det.train_net(X_pool, y_pool, verbose=False)
        det._calibrate(train, rpf_fn=cache)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        det.save(path)
        print(f"  trained {split['split']}{tag} on {len(X_pool):,} rows in {time.time() - t0:.0f}s")
    if keep_pool:
        det._pool = (X_pool, y_pool)
    if not linear:
        return det, cache, {}
    # Linear model on the same fingerprints: does the MLP matter? 59 weights
    # need nowhere near the full pool; fitting all 7M+ rows took over an hour.
    sub = np.random.default_rng(0).permutation(len(X_pool))[:200_000]
    lin = LogisticRegression(max_iter=2000, class_weight="balanced").fit(det._mask(X_pool[sub]), y_pool[sub])
    return det, cache, {"RPF-LR": lambda rpf: lin.decision_function(det._mask(rpf))}


def train_variants(cfg, split, data, cache_root, out, main_det):
    """Train the detector variants in cfg["variants"]."""
    scorers = {}
    for name, overrides in cfg.get("variants", {}).items():
        vcfg = json.loads(json.dumps(cfg))
        vcfg["train"].update(overrides)
        tag = "_" + re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
        vdet = train_split(vcfg, split, data, cache_root, out, tag=tag, linear=False)[0]
        d = vdet.extractor.dim
        if list(main_det.extractor.feature_names[:d]) != list(vdet.extractor.feature_names):
            raise ValueError(f"variant {name!r}: its fingerprint is not a prefix of the main one")

        def score(rpf, det=vdet, d=d):
            s = det._score_rpf(rpf[:, :d])
            return s, ev.topk_flags(s, ev.estimate_contamination_rate(s, probabilities=True))
        scorers[name] = score
    return scorers


def signal_scorers(det):
    """Every block-I signal on its own, as a stand-alone detector (plan B3)."""
    ex = det.extractor
    if not hasattr(ex, "effect"):
        return {}
    return {f"Signal:{n[2:]}": (lambda rpf, j=ex.base.dim + i: rpf[:, j])
            for i, n in enumerate(EFFECT_NAMES)}


def minp_scorers(det, split, data, cache, pool=None):
    """Plan item D1a."""
    ex = det.extractor
    if not hasattr(ex, "effect"):
        return {}, {}
    cols = slice(ex.base.dim, ex.base.dim + ex.effect.dim)

    def scores(rpf):
        return np.column_stack([robust_scale(det._score_rpf(rpf)), rpf[:, cols]])

    null = {}
    for n in split["train"]:   # the fingerprints _calibrate cached
        d = data[n]
        rpf = cache(("clean", ex.feature_version, n), d["X"], d["y"], d["y_cont"] if d["is_reg"] else None)
        null.setdefault(bool(d["is_reg"]), []).append(scores(rpf))
    null[None] = [a for v in list(null.values()) for a in v]   # both tasks, if a split lacks one
    comb = {reg: (MinPCombiner(np.vstack(v)), MinPCombiner(np.vstack(v)[:, 1:])) for reg, v in null.items()}

    rate_model = {}
    X_pool, y_pool = pool if pool is not None else (None, None)
    bounds = np.cumsum([0] + list(det._pool_groups))
    for reg in (False, True) if X_pool is not None else ():
        groups = [(X_pool[a:b, cols].astype(np.float64), float(y_pool[a:b].mean()))
                  for a, b, m in zip(bounds[:-1], bounds[1:], det._pool_meta) if m[4] == reg]
        if len(groups) >= 10 and reg in null:
            rate_model[reg] = RateModel(comb[reg][1], np.vstack(null[reg])[:, 1:]).fit(groups)

    if det.scorer == "trees":
        # The tree scorer's probabilities are not calibrated like the network's,
        # so its own flags use the learned rate as well (score_dataset reads rate_fn).
        det.rate_fn = lambda rpf, reg: (rate_model[bool(reg)].predict(rpf[:, cols].astype(np.float64))
                                        if bool(reg) in rate_model else
                                        ev.estimate_contamination_rate(det._score_rpf(rpf), probabilities=True))

    def own_rate(rpf, reg):
        rf = getattr(det, "rate_fn", None)
        return rf(rpf, reg) if rf else ev.estimate_contamination_rate(det._score_rpf(rpf), probabilities=True)

    def make(i):
        def score(rpf, reg):
            S = scores(rpf)
            sc = comb.get(bool(reg), comb[None])[i].score(S if i == 0 else S[:, 1:])
            old = own_rate(rpf, reg)
            rm = rate_model.get(bool(reg))
            rate = rm.predict(S[:, 1:]) if rm is not None else old
            return sc, ev.topk_flags(sc, rate), {"rpfrate": ev.topk_flags(sc, old)}
        score.needs_task = True
        return score
    return {"MinP": make(0), "MinP-signals": make(1)}, rate_model


def oof_network_scores(cfg, split, det, pool, data, cache, out, n_inner=3, parts=None, tag=""):
    """Out-of-fold network scores for the training pool and for the clean training datasets."""
    names = sorted(split["train"])
    path = os.path.join(model_dir(cfg, out), f"{split['split']}_oof{tag}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["pool"], {n: z[f"clean{i}"] for i, n in enumerate(names)}
    X_pool, y_pool = pool
    meta, sizes = det._pool_meta, list(det._pool_groups)
    bounds = np.cumsum([0] + sizes)
    part = parts or {n: i % n_inner for i, n in enumerate(names)}
    fv = det.extractor.feature_version
    oof, clean = np.zeros(len(X_pool), np.float32), {}
    for k in range(n_inner):
        tr = [g for g, m in enumerate(meta) if part[m[0]] != k]
        idx = np.concatenate([np.arange(bounds[g], bounds[g + 1]) for g in tr])
        inner = new_detector(cfg)
        inner.train_net(X_pool[idx], y_pool[idx], verbose=False, groups=[sizes[g] for g in tr])
        for g, m in enumerate(meta):
            if part[m[0]] == k:
                oof[bounds[g]:bounds[g + 1]] = inner._score_rpf(X_pool[bounds[g]:bounds[g + 1]])
        for n in names:
            if part[n] == k:
                d = data[n]
                clean[n] = inner._score_rpf(cache(("clean", fv, n), d["X"], d["y"],
                                                  d["y_cont"] if d["is_reg"] else None))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, pool=oof, **{f"clean{i}": clean[n] for i, n in enumerate(names)})
    return oof, clean


def null_parts(cfg, names, data, n_inner=3):
    """Inner partition for the out-of-dataset nulls."""
    grp, units = cfg.get("groups", {}), {}
    for n in sorted(names):
        units.setdefault(grp.get(n, n), []).append(n)
    parts, i = {}, 0
    for reg in (True, False):
        for k in sorted(k for k, m in units.items() if any(data[x]["is_reg"] for x in m) == reg):
            parts.update(dict.fromkeys(units[k], i % n_inner))
            i += 1
    if len(set(parts.values())) < 2:
        raise ValueError("the out-of-dataset nulls need training datasets from at least two sources")
    return parts


def oof_nulls(cfg, split, det, pool, data, cache, out, n_q=10001):
    """Nulls for the BH flags, one per task type, each training dataset weighted equally (n_q quantiles each)."""
    names = sorted(split["train"])
    oof, clean = oof_network_scores(cfg, split, det, pool, data, cache, out,
                                    parts=null_parts(cfg, names, data), tag="_null")
    fv = det.extractor.feature_version
    ins = {n: det._score_rpf(cache(("clean", fv, n), data[n]["X"], data[n]["y"],
                                   data[n]["y_cont"] if data[n]["is_reg"] else None)) for n in names}
    bounds, y_pool = np.cumsum([0] + list(det._pool_groups)), pool[1]
    in_pool = {}
    for g, m in enumerate(det._pool_meta):
        a, b = bounds[g], bounds[g + 1]
        in_pool.setdefault(m[0], []).append(oof[a:b][y_pool[a:b] <= 0.5])
    grid, nulls = np.linspace(0, 1, n_q), {}
    for tag, per in (("_oof", clean), ("_pool", {n: np.concatenate(v) for n, v in in_pool.items()}),
                     ("_ins", ins)):
        nulls[tag] = {}
        for reg in (False, True):
            qs = [np.quantile(per[n], grid) for n in names
                  if data[n]["is_reg"] == reg and len(per.get(n, ()))]
            if qs:
                nulls[tag][reg] = np.sort(np.concatenate(qs)).astype(np.float32)
    return nulls


def clean_alarm_rows(cfg, split, det, cache, data):
    """RPFNet on the unpoisoned version of every test dataset (--mode main with
    eval.clean_alarms): false alarms and p-value calibration with no poison."""
    fv, q = det.extractor.feature_version, cfg["eval"].get("fdr_q", 0.10)
    rows = []
    for name in split["test"]:
        d = data[name]
        rpf = cache(("clean", fv, name), d["X"], d["y"], d["y_cont"] if d["is_reg"] else None)
        rows.append(dict(dataset=name, split=split["split"], **ev.clean_alarms(det, rpf, d["is_reg"], q)))
    return rows


def stack_scorers(cfg, split, det, pool, data, cache, out, rate_model):
    """Plan item D1b."""
    ex = det.extractor
    if not hasattr(ex, "effect") or pool is None:
        return {}
    cols = slice(ex.base.dim, ex.base.dim + ex.effect.dim)
    oof_pool, oof_clean = oof_network_scores(cfg, split, det, pool, data, cache, out)
    ref = {}
    for n in split["train"]:
        d = data[n]
        rpf = cache(("clean", ex.feature_version, n), d["X"], d["y"], d["y_cont"] if d["is_reg"] else None)
        ref.setdefault(bool(d["is_reg"]), []).append(np.column_stack([robust_scale(oof_clean[n]), rpf[:, cols]]))
    X_pool, y_pool = pool
    bounds = np.cumsum([0] + list(det._pool_groups))
    models = {}
    for reg, v in ref.items():
        sets = [(np.column_stack([robust_scale(oof_pool[a:b]), X_pool[a:b, cols]]), y_pool[a:b] > 0.5)
                for a, b, m in zip(bounds[:-1], bounds[1:], det._pool_meta) if m[4] == reg]
        if len(sets) >= 10:
            models[reg] = LearnedCombiner(MinPCombiner(np.vstack(v))).fit(sets)
    if not models:
        return {}

    def score(rpf, reg):
        S = np.column_stack([robust_scale(det._score_rpf(rpf)), rpf[:, cols]])
        m = models.get(bool(reg)) or next(iter(models.values()))   # a split without one task uses the other
        sc = m.score(S)
        rf = getattr(det, "rate_fn", None)
        old = rf(rpf, reg) if rf else ev.estimate_contamination_rate(det._score_rpf(rpf), probabilities=True)
        rm = rate_model.get(bool(reg))
        rate = rm.predict(S[:, 1:]) if rm is not None else old
        return sc, ev.topk_flags(sc, rate), {"rpfrate": ev.topk_flags(sc, old)}
    score.needs_task = True
    return {"Stack": score}


def dump_split(cfg, split, det, data, cache, path, pool_rows=200, seed=0):
    """Row-level data for offline research (--mode dump), one file per split."""
    e, fv, rng = cfg["eval"], det.extractor.feature_version, np.random.default_rng(seed)
    F, S, Y, L, C, configs = [], [], [], [], [], []
    for name in split["test"]:
        d = data[name]
        for atk in ev.split_attacks(cfg, split, d["is_reg"])[2]:
            for rate in e["rates"]:
                for s in range(e["seeds"]):
                    Xp, yp, pidx, ycp = ev.poison(d, atk, rate, s)
                    if len(pidx) < 3:
                        continue
                    rpf = cache(("eval", fv, name, atk, rate, s), Xp, yp, ycp)
                    t = np.zeros(len(Xp), bool)
                    t[pidx] = True
                    F.append(rpf); S.append(det._score_rpf(rpf)); Y.append(t); L.append(np.asarray(yp))
                    C.append(np.full(len(Xp), len(configs), np.int32))
                    configs.append(dict(dataset=name, attack=atk, family=ev.FAMILY.get(atk, "other"),
                                        rate=rate, seed=s, regression=bool(d["is_reg"])))
    clean = [cache(("clean", fv, n), data[n]["X"], data[n]["y"], data[n]["y_cont"] if data[n]["is_reg"] else None)
             for n in split["train"]]
    X_pool, y_pool = det.__dict__.pop("_pool")
    bounds = np.cumsum([0] + list(det._pool_groups))
    PF, PS, PY, PG = [], [], [], []
    for g, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        yy = y_pool[a:b] > 0.5
        keep = np.concatenate([np.where(yy)[0], rng.choice(np.where(~yy)[0], min(pool_rows, (~yy).sum()), replace=False)])
        PF.append(X_pool[a:b][keep]); PS.append(det._score_rpf(X_pool[a:b])[keep]); PY.append(yy[keep])
        PG.append(np.full(len(keep), g, np.int32))
    meta = dict(configs=configs, clean_datasets=list(split["train"]), feature_names=det.extractor.feature_names,
                pool_groups=[dict(dataset=m[0], attack=m[1], rate=m[2], seed=m[3], regression=m[4], n=int(n))
                             for m, n in zip(det._pool_meta, det._pool_groups)])
    np.savez_compressed(path, F=np.vstack(F).astype(np.float32), score=np.concatenate(S), y=np.concatenate(Y),
                        label=np.concatenate(L), config=np.concatenate(C),
                        clean=np.vstack(clean).astype(np.float32),
                        clean_dataset=np.concatenate([np.full(len(c), i, np.int32) for i, c in enumerate(clean)]),
                        pool_F=np.vstack(PF).astype(np.float32), pool_score=np.concatenate(PS),
                        pool_y=np.concatenate(PY), pool_group=np.concatenate(PG), meta=np.array(json.dumps(meta)))
    print(f"  wrote {path} ({len(configs)} configurations, {sum(map(len, F)):,} rows)")


def eval_split(cfg, split, det, cache, extra, data, rows_path, baselines, bcache=None):
    e = cfg["eval"]
    part_path = rows_path + ".partial"   # finished datasets, so an interrupted split resumes
    rows = pd.read_csv(part_path).to_dict("records") if os.path.exists(part_path) else []
    done = {r["dataset"] for r in rows}
    for name in split["test"]:
        if name in done:
            continue
        _, _, attacks = ev.split_attacks(cfg, split, data[name]["is_reg"])
        for atk in attacks:
            for rate in e["rates"]:
                for seed in range(e["seeds"]):
                    try:
                        r = ev.evaluate_config(det, data[name], name, atk, rate, seed,
                                               baselines=baselines, cache=cache, extra_scorers=extra,
                                               q=e.get("fdr_q", 0.10), downstream=e.get("downstream", True),
                                               bcache=bcache)
                    except Exception as exc:   # record, never hide
                        r = [dict(dataset=name, attack=atk, rate=rate, seed=seed, method="RPFNet",
                                  error=f"{type(exc).__name__}: {exc}")]
                    for row in r:
                        row["split"] = split["split"]
                    rows.extend(r)
        pd.DataFrame(rows).to_csv(part_path, index=False)
        print(f"    {split['split']}: {name} done ({len(rows)} rows)")
    pd.DataFrame(rows).to_csv(rows_path, index=False)
    if os.path.exists(part_path):
        os.remove(part_path)
    return rows


def summarise(df, out, ref="RPFNet"):
    ok = df[df.get("error", pd.Series(np.nan, index=df.index)).isna()]
    for metric in ["AUPRC", "AUROC", "F1_known", "F1_own", "F1_shared", "downstream"]:
        if metric in ok and ok[metric].notna().any():
            ev.pooled_tests(ok, metric, ref).to_csv(os.path.join(out, f"tests_{metric}.csv"), index=False)
    fr = {m: ev.friedman_nemenyi(ok, m) for m in ["AUPRC", "AUROC", "F1_own"] if m in ok}
    ci = {m: ev.bootstrap_ci(ok, "AUPRC", m) for m in ev._methods(ok)}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump({"friedman_nemenyi": fr, "auprc_bootstrap_ci": ci}, f, indent=2, default=float)
    ev.per_rate_table(ok).to_csv(os.path.join(out, "per_rate_RPFNet.csv"))
    fam = ok[ok["method"].isin(ev._methods(ok))].groupby(["family", "method"])[["AUPRC", "AUROC", "F1_own"]].mean()
    fam.to_csv(os.path.join(out, "per_family.csv"))
    g = ok[ok["method"].isin(ev._methods(ok))].groupby("method")
    means = g[["AUROC", "AUPRC", "P@k", "F1_known", "F1_own", "F1_shared", "time_s"]
              + [c for c in ["F1_rpfrate"] if c in ok]].mean()
    means["n_rows"] = g.size()   # methods such as TRIM run on a subset of datasets
    means.sort_values("AUPRC", ascending=False).to_csv(os.path.join(out, "method_means.csv"))
    print(means.sort_values("AUPRC", ascending=False).round(3).to_string())
    errors = df.get("error", pd.Series(dtype=object)).dropna()
    if len(errors):
        print(f"  {len(errors)} configurations raised errors; see the 'error' column.")


_FRONTIER = {}


def _frontier_task(task, cfg, out):
    """One (fold, dataset, seed) of --mode frontier."""
    from threadpoolctl import threadpool_limits
    import RPFNet.Attack as A
    import RPFNet.RPFExtractor as RX
    from RPFNet.adaptive_attack import CamouflageAttack, GRID
    from RPFNet.baselines import BASELINES
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    RX.N_JOBS = A.N_JOBS = 1
    if RX.FAISS_AVAILABLE:
        RX.faiss.omp_set_num_threads(1)
    torch.set_num_threads(1)
    f = cfg["frontier"]
    path = task["model"]
    if _FRONTIER.get("path") != path:
        det = new_detector(cfg)
        det.load(path)
        _FRONTIER.update(path=path, det=det)
    det = _FRONTIER["det"]
    X, y, X_te, y_te = task["X"], task["y"], task["X_te"], task["y_te"]
    seed = task["seed"]
    rows = []
    with threadpool_limits(1):
        atk = CamouflageAttack(X, y, f["rate"], seed=seed)
        score_fn = lambda Xc, yc: det.score(Xc, yc)[0]
        truth = np.zeros(len(X), bool)
        truth[atk.pidx] = True
        lr0 = LogisticRegression(C=1.0, max_iter=2000).fit(X, y).score(X_te, y_te)
        rf0 = RandomForestClassifier(200, random_state=seed, n_jobs=1).fit(X, y).score(X_te, y_te)

        def measure(a, setting, lam, queries, attacker):
            Xp, yp = atk.dataset(a)
            base = dict(dataset=task["name"], split=task["split"], seed=seed, rate=f["rate"], setting=setting,
                        lam=lam, mean_a=float(a.mean()), queries=queries, n=len(X), n_poison=int(truth.sum()),
                        attacker_damage=attacker.get("damage"), attacker_auroc=attacker.get("auroc"),
                        dmg_lr=lr0 - LogisticRegression(C=1.0, max_iter=2000).fit(Xp, yp).score(X_te, y_te),
                        dmg_rf=rf0 - RandomForestClassifier(200, random_state=seed, n_jobs=1).fit(Xp, yp).score(X_te, y_te))
            try:
                g = gradient_diagnostics(Xp, yp, atk.pidx, seed=seed)
                base.update(harm_ratio=g["harm_ratio"], spike_ratio=g["spike_ratio"], auroc_blockH=g.get("auroc_blockH"))
            except Exception as exc:
                base.update(diag_error=f"{type(exc).__name__}: {exc}")
            scores = {"RPFNet": det.score(Xp, yp)[0]}
            for m in f["baselines"]:
                scores[m] = BASELINES[m](Xp, yp, seed)[0]
            for m, sc in scores.items():
                sc = np.asarray(sc, dtype=np.float64)
                rows.append(dict(base, method=m, AUROC=roc_auc_score(truth, sc),
                                 AUPRC=average_precision_score(truth, sc)))

        # non-adaptive sweep (shared by every lam), then the adaptive search per lam
        sweep = []
        for g_ in GRID:
            a = np.full(len(atk.pidx), g_)
            Xp, yp = atk.dataset(a)
            d_, auc = atk.damage(Xp), roc_auc_score(truth, np.asarray(score_fn(Xp, yp), dtype=np.float64))
            sweep.append((a, d_, auc))
            measure(a, "uniform", np.nan, 1, dict(damage=d_, auroc=auc))
        for lam in f["lams"]:
            init = max(sweep, key=lambda t: t[1] - lam * t[2])[0]
            a, hist = atk.run(score_fn, lam, iters=f.get("iters", 20), init=init)
            last = max((h for h in hist if h.get("accepted") or h["kind"] == "init"), key=lambda h: h["J"])
            measure(a, "adaptive", lam, len(GRID) + len(hist), last)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default="main",
                    choices=["main", "ablation", "stability", "scaling", "diagnostics", "adaptive", "dump",
                             "frontier", "dataset_test"])
    ap.add_argument("--fresh", action="store_true",
                    help="delete this config's results first (the fingerprint cache is kept)")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    out = os.path.join(ROOT, cfg.get("output_dir", "results/v2"), cfg["name"])
    cache_root = os.path.join(ROOT, cfg.get("output_dir", "results/v2"), "rpf_cache")
    os.environ["RPF_POISON_CACHE"] = cache_root   # slow attacks (ev.SLOW_ATTACKS) are cached there
    if args.fresh and os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)
    code = ev.source_hash(*sorted(glob.glob(os.path.join(ROOT, "Backend", "RPFNet", "*.py"))),
                          os.path.abspath(__file__))
    env = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
           "numpy": np.__version__, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": args.mode}
    cfg_path = os.path.join(out, f"config_{args.mode}.json")
    if os.path.exists(cfg_path):   # finished splits are reused, so config and code must match
        with open(cfg_path) as f:
            saved = json.load(f)
        if ({**saved["config"], "only_splits": None} != {**cfg, "only_splits": None}
                or saved.get("code") != code):
            sys.exit(f"{out} holds results from a different config or code version, and its "
                     f"finished splits would be mixed with new ones. Add --fresh to start over.")
    with open(cfg_path, "w") as f:
        json.dump({"config": cfg, "code": code, "environment": env}, f, indent=2)

    if args.mode == "scaling":
        s = cfg["scaling"]
        df = ev.scaling(s["n"], s.get("d", 20), cfg["train"]["feature_version"])
        df.to_csv(os.path.join(out, "scaling.csv"), index=False)
        print(df.round(2).to_string())
        return

    data = {n: ev.load_named(n, ROOT) for n in cfg["datasets"]}
    max_rows = cfg.get("max_rows")
    for n, d in data.items():   # optional subsample of large training splits
        if max_rows and len(d["X"]) > max_rows:
            idx = np.sort(np.random.default_rng(0).choice(len(d["X"]), max_rows, replace=False))
            d["X"], d["y"] = d["X"][idx], d["y"][idx]
            d["y_cont"] = None if d["y_cont"] is None else d["y_cont"][idx]
    is_reg = {n: d["is_reg"] for n, d in data.items()}
    splits = ev.make_splits(cfg, is_reg)
    if cfg.get("only_splits"):
        splits = [s for s in splits if s["split"] in cfg["only_splits"]]
    baselines = cfg["eval"].get("baselines", [])

    if args.mode in ("main", "dump", "ablation"):
        todo = [s for s in splits if not os.path.exists(os.path.join(out, f"rows_{s['split']}.csv"))] \
            if args.mode == "main" else splits
        if todo:
            precompute(cfg, todo, data, cache_root, baselines if args.mode == "main" else ())
    bcache = ev.BaselineCache(cache_root)

    if args.mode == "main":
        all_rows, alarms = [], []
        for split in splits:
            rows_path = os.path.join(out, f"rows_{split['split']}.csv")
            alarm_path = os.path.join(out, f"alarms_{split['split']}.csv")
            if os.path.exists(rows_path):
                all_rows.extend(pd.read_csv(rows_path).to_dict("records"))
                if os.path.exists(alarm_path):
                    alarms.extend(pd.read_csv(alarm_path).to_dict("records"))
                continue
            print(f"[split {split['split']}] train on {len(split['train'])} datasets, "
                  f"test on {split['test']}")
            e = cfg["eval"]
            det, cache, extra = train_split(cfg, split, data, cache_root, out,
                                            keep_pool=bool(e.get("minp") or e.get("stack") or e.get("null")
                                                           or cfg["train"].get("scorer") == "trees"))
            pool = det.__dict__.pop("_pool", None)
            extra.update(train_variants(cfg, split, data, cache_root, out, det))
            if e.get("signal_baselines"):
                extra.update(signal_scorers(det))
            rate_model = {}
            if e.get("minp") or e.get("stack") or det.scorer == "trees":
                scorers, rate_model = minp_scorers(det, split, data, cache, pool)
                if e.get("minp"):
                    extra.update(scorers)
            if e.get("stack"):
                extra.update(stack_scorers(cfg, split, det, pool, data, cache, out, rate_model))
            if e.get("null") == "oof":
                det.extra_nulls = oof_nulls(cfg, split, det, pool, data, cache, out)
            del pool
            if e.get("clean_alarms"):
                a = clean_alarm_rows(cfg, split, det, cache, data)
                pd.DataFrame(a).to_csv(alarm_path, index=False)
                alarms.extend(a)
            all_rows.extend(eval_split(cfg, split, det, cache, extra, data, rows_path, baselines, bcache))
        if alarms:
            pd.DataFrame(alarms).to_csv(os.path.join(out, "clean_alarms.csv"), index=False)
        df = pd.DataFrame(all_rows)
        df.to_csv(os.path.join(out, "results.csv"), index=False)
        summarise(df, out)

    elif args.mode == "dump":
        os.makedirs(os.path.join(out, "dump"), exist_ok=True)
        for split in splits:
            det, cache, _ = train_split(cfg, split, data, cache_root, out, linear=False, keep_pool=True)
            dump_split(cfg, split, det, data, cache, os.path.join(out, "dump", f"{split['split']}.npz"))

    elif args.mode == "ablation":
        # Retrain without each block (and without the Isolation Forest columns).
        a = cfg.get("ablation", {})
        rows = []
        for split in splits[: a.get("n_splits", 1)]:
            for tseed in range(a.get("train_seeds", 3)):
                scfg = json.loads(json.dumps(cfg))
                scfg["train"]["seed"] = tseed
                for vname, drop in ev.ablation_variants(new_detector(scfg).extractor).items():
                    tag = f"_{vname}_s{tseed}"
                    rows_path = os.path.join(out, f"ablation_{split['split']}{tag}.csv")
                    if not os.path.exists(rows_path):
                        det, cache, _ = train_split(scfg, split, data, cache_root, out, drop=drop, tag=tag)
                        r = eval_split(scfg, split, det, cache, {}, data, rows_path, [])
                        pd.DataFrame([dict(x, variant=vname, train_seed=tseed) for x in r]).to_csv(
                            rows_path, index=False)
                    rows.extend(pd.read_csv(rows_path).to_dict("records"))
        df = pd.DataFrame(rows)
        df = df[(df["method"] == "RPFNet") & df["AUPRC"].notna()]
        per = df.groupby(["variant", "train_seed"])["AUPRC"].mean().unstack("variant")
        drop = per.sub(per["full"], axis=0).mul(-1)          # full minus variant, per seed
        res = pd.DataFrame({"mean_AUPRC_drop": drop.mean(), "sd_over_seeds": drop.std(),
                            "full_AUPRC_sd_over_seeds": per["full"].std()})
        res.sort_values("mean_AUPRC_drop", ascending=False).to_csv(os.path.join(out, "ablation_drops.csv"))
        print(res.sort_values("mean_AUPRC_drop", ascending=False).round(4).to_string())

    elif args.mode == "stability":
        s = cfg.get("stability", {})
        split = splits[0]
        det, _, _ = train_split(cfg, split, data, cache_root, out)
        rows = []
        for name in split["test"]:
            for atk in s.get("attacks", ["label_flip"]):
                if data[name]["is_reg"]:
                    continue
                Xp, yp, pidx, ycp = ev.poison(data[name], atk, s.get("rate", 0.1), 0)
                r = ev.stability(det, Xp, yp, ycp, len(pidx), s.get("n_rep", 5))
                rows.append(dict(dataset=name, attack=atk, **r))
                print(rows[-1])
        pd.DataFrame(rows).to_csv(os.path.join(out, "stability.csv"), index=False)

    elif args.mode == "diagnostics":
        d = cfg.get("diagnostics", {})
        part_path = os.path.join(out, "diagnostics.csv.partial")   # finished datasets, for resuming
        rows = pd.read_csv(part_path).to_dict("records") if os.path.exists(part_path) else []
        done = {r["dataset"] for r in rows}
        for name in cfg["datasets"]:
            if is_reg[name] or name in done:
                continue
            dn = data[name]
            for atk in d.get("attacks", cfg["eval"]["attacks"] + cfg["eval"].get("unseen_attacks", [])):
                for rate in d.get("rates", cfg["eval"]["rates"]):
                    for seed in range(d.get("seeds", 1)):
                        Xp, yp, pidx, _ = ev.poison(dn, atk, rate, seed)
                        if len(pidx) < 3:
                            continue
                        r = gradient_diagnostics(Xp, yp, pidx, test=(dn["X_te"], dn["y_te"]),
                                                 original=(dn["X"], dn["y"]), seed=seed)
                        rows.append(dict(dataset=name, attack=atk, rate=rate, seed=seed, **r))
            pd.DataFrame(rows).to_csv(part_path, index=False)
            print(f"  diagnostics: {name} done ({len(rows)} rows)")
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out, "diagnostics.csv"), index=False)
        main_rows = os.path.join(out, "results.csv")
        if os.path.exists(main_rows):   # relate detectability to RPFNet's AUROC
            m = pd.read_csv(main_rows)
            m = m[m["method"] == "RPFNet"][["dataset", "attack", "rate", "seed", "AUROC", "AUPRC"]]
            df = df.merge(m, on=["dataset", "attack", "rate", "seed"], how="left")
            df.to_csv(os.path.join(out, "diagnostics.csv"), index=False)
        if os.path.exists(part_path):
            os.remove(part_path)
        cols = ["thm1_holds", "gap_over_lower", "thm2_direction_holds", "thm2_auroc_holds",
                "detectability_D", "auroc_blockH", "auroc_bound", "cert_holds", "harm_ratio"]
        print(df.groupby("attack")[[c for c in cols if c in df]].mean().round(3).to_string())

    elif args.mode == "dataset_test":
        from RPFNet.baselines import BASELINES
        dt = cfg.get("dataset_test", {})
        bl = dt.get("baselines", ["ECOD", "IsoForest", "GradNorm"])
        refs_rows, item_rows = [], []
        fv = None
        for split in splits:
            part = os.path.join(out, f"dataset_test_{split['split']}.npz")
            if os.path.exists(part):
                z = np.load(part, allow_pickle=True)
                refs_rows += list(z["refs"]); item_rows += list(z["items"])
                continue
            det, cache, _ = train_split(cfg, split, data, cache_root, out, linear=False, keep_pool=True)
            X_pool, y_pool = det.__dict__.pop("_pool")
            fv = det.extractor.feature_version
            meta, sizes = det._pool_meta, list(det._pool_groups)
            bounds = np.cumsum([0] + sizes)
            refs = [n for n in split["train"] if not data[n]["is_reg"]]
            tests = [n for n in split["test"] if not data[n]["is_reg"]]
            # test items: (dataset, attack, rate, seed, fingerprint, X, y)
            items = []
            for name in tests:
                d = data[name]
                items.append((name, "clean", 0.0, -1, cache(("clean", fv, name), d["X"], d["y"], None), d["X"], d["y"]))
                for atk in ev.split_attacks(cfg, split, False)[2]:
                    for rate in cfg["eval"]["rates"]:
                        for seed in range(cfg["eval"]["seeds"]):
                            Xp, yp, pidx, _ = ev.poison(d, atk, rate, seed)
                            if len(pidx) < 3 or len(np.unique(yp)) < 2:   # same exclusions as the main protocol
                                continue
                            items.append((name, atk, rate, seed, cache(("eval", fv, name, atk, rate, seed), Xp, yp, None), Xp, yp))
            print(f"[{split['split']}] {len(refs)} references, {len(items)} test items")
            # baselines: exact conformal references on the clean training datasets
            for m in bl:
                for r in refs:
                    sc = BASELINES[m](data[r]["X"], data[r]["y"], 0)[0]
                    refs_rows.append((split["split"], m, r, ev.dataset_stat(sc, robust=True)))
                for (name, atk, rate, seed, _, Xp, yp) in items:
                    sc = bcache.get(m, Xp, yp, max(seed, 0))[0] if seed >= 0 else BASELINES[m](Xp, yp, 0)[0]
                    item_rows.append((split["split"], m, "", name, atk, rate, seed, ev.dataset_stat(sc, robust=True)))
            # RPFNet: leave-one-reference-dataset-out scorers
            t0 = time.time()
            for j, r in enumerate(refs):
                keep = [g for g, mm in enumerate(meta) if mm[0] != r]
                idx = np.concatenate([np.arange(bounds[g], bounds[g + 1]) for g in keep])
                inner = new_detector(cfg)
                inner.train_net(X_pool[idx], y_pool[idx], verbose=False, groups=[sizes[g] for g in keep])
                d = data[r]
                refs_rows.append((split["split"], "RPFNet", r,
                                  ev.dataset_stat(inner._score_rpf(cache(("clean", fv, r), d["X"], d["y"], None)))))
                for (name, atk, rate, seed, rpf, _, _) in items:
                    item_rows.append((split["split"], "RPFNet", r, name, atk, rate, seed,
                                      ev.dataset_stat(inner._score_rpf(rpf))))
                print(f"  [{split['split']}] scorer without {r}: {j + 1}/{len(refs)} ({time.time() - t0:.0f}s)")
            np.savez(part, refs=np.array(refs_rows[-(len(refs) * (len(bl) + 1)):], dtype=object),
                     items=np.array(item_rows[-(len(items) * (len(bl) + len(refs))):], dtype=object))
            del X_pool, y_pool
        pd.DataFrame(refs_rows, columns=["split", "method", "ref", "T"]).to_csv(
            os.path.join(out, "dataset_test_refs.csv"), index=False)
        pd.DataFrame(item_rows, columns=["split", "method", "ref", "dataset", "attack", "rate", "seed", "T"]).to_csv(
            os.path.join(out, "dataset_test_items.csv"), index=False)
        print("wrote dataset_test_refs.csv and dataset_test_items.csv")

    elif args.mode == "frontier":
        from joblib import Parallel, delayed, cpu_count
        f = cfg["frontier"]
        tasks = []
        for split in splits:
            mpath = os.path.join(model_dir(cfg, out), f"{split['split']}.pt")
            if not os.path.exists(mpath):
                sys.exit(f"missing frozen model {mpath}")
            for name in split["test"]:
                d = data[name]
                if d["is_reg"]:
                    continue
                X, y = d["X"], d["y"]
                if len(X) > f["max_rows"]:
                    idx = np.sort(np.random.default_rng(1).choice(len(X), f["max_rows"], replace=False))
                    X, y = X[idx], y[idx]
                for seed in range(f["seeds"]):
                    tasks.append(dict(name=name, split=split["split"], seed=seed, model=mpath,
                                      X=X, y=y, X_te=d["X_te"], y_te=d["y_te"]))
        part = os.path.join(out, "frontier_parts")
        os.makedirs(part, exist_ok=True)
        todo = [t for t in tasks if not os.path.exists(os.path.join(part, f"{t['name']}_s{t['seed']}.csv"))]
        print(f"frontier: {len(tasks)} tasks, {len(todo)} to run")

        def run_one(t):
            try:
                r = _frontier_task(t, cfg, out)
            except Exception as exc:
                r = [dict(dataset=t["name"], split=t["split"], seed=t["seed"], error=f"{type(exc).__name__}: {exc}")]
            pd.DataFrame(r).to_csv(os.path.join(part, f"{t['name']}_s{t['seed']}.csv"), index=False)
            return len(r)
        n_jobs = cfg.get("n_jobs") or max(1, cpu_count(only_physical_cores=True))
        Parallel(n_jobs=n_jobs, verbose=5)(delayed(run_one)(t) for t in sorted(todo, key=lambda t: -t["X"].size))
        df = pd.concat([pd.read_csv(os.path.join(part, f"{t['name']}_s{t['seed']}.csv")) for t in tasks])
        df.to_csv(os.path.join(out, "frontier.csv"), index=False)
        if "method" in df:
            print(df.pivot_table(index=["setting", "lam"], columns="method", values="AUROC").round(3).to_string())
            print(df[df.method == "RPFNet"].groupby(["setting", "lam"], dropna=False)[
                ["mean_a", "dmg_lr", "dmg_rf", "harm_ratio", "spike_ratio"]].mean().round(3).to_string())

    elif args.mode == "adaptive":
        a = cfg["adaptive"]
        split = splits[0]
        detectors = [("RPFNet", train_split(cfg, split, data, cache_root, out)[0])]
        if a.get("adversarial_training"):
            # the same detector trained with adversarial rounds (plan B2), attacked the same way
            acfg = json.loads(json.dumps(cfg))
            acfg["train"]["adversarial"] = a["adversarial_training"]
            detectors.append(("RPFNet+adversarial",
                              train_split(acfg, split, data, cache_root, out, tag="_adv", linear=False)[0]))
        rows = []
        for dname, det in detectors:
            for name in split["test"]:
                if is_reg[name] or len(data[name]["X"]) > a.get("max_n", 3000):
                    continue
                for lam in a["lams"]:
                    for seed in range(a.get("seeds", 1)):
                        Xp, yp, pidx, hist = evasive_attack(
                            data[name]["X"], data[name]["y"], a["rate"],
                            score_fn=lambda Xc, yc, det=det: det.score(Xc, yc)[0],
                            base=a.get("base", "label_flip"), lam=lam, iters=a.get("iters", 30), seed=seed)
                        info = dict(dataset=name, attack=f"evasive_{a.get('base', 'label_flip')}",
                                    family="optimised", rate=a["rate"], seed=seed, lam=lam, detector=dname,
                                    attacker_damage=hist[-1][0], attacker_mean_score=hist[-1][1])
                        r = ev.score_dataset(det, Xp, yp, None, pidx, data[name], info,
                                             baselines=a.get("baselines", []), seed=seed)
                        rows.extend(r)
                        print(f"  {dname} {name} lam={lam}: damage={hist[-1][0]:.3f} "
                              f"AUROC={r[0]['AUROC']:.3f}")
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out, "adaptive.csv"), index=False)
        if len(df):
            print(df[df["method"] == "RPFNet"].pivot_table(index="lam", columns="detector",
                                                           values="AUROC").round(3).to_string())


if __name__ == "__main__":
    main()
