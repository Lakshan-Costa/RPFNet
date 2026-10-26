"""Supervised combiners of the baseline scores under the unseen-datasets protocol."""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Backend"), ROOT]
warnings.filterwarnings("ignore")

ev = None

CFG = os.path.join(ROOT, "configs", "dataset_kfold.json")
OUT = os.path.join(ROOT, "results", "v2", "combiner_check")
RESULTS = os.path.join(ROOT, "results", "v2", "dataset_kfold", "results.csv")
MAX_CLEAN = 200
TREES = dict(max_iter=200, max_depth=6, learning_rate=0.1, random_state=0)


class CacheMiss(RuntimeError):
    pass


def _refuse(*_a, **_k):
    raise CacheMiss("not in the cache of the main run")


def setup():
    global ev
    from RPFNet import evaluation as ev          # noqa: E402
    from run_experiments import new_detector     # noqa: E402
    with open(CFG) as f:
        cfg = json.load(f)
    data = {n: ev.load_named(n, ROOT) for n in cfg["datasets"]}
    for d in data.values():   # same training-row cap as run_experiments.py
        if cfg.get("max_rows") and len(d["X"]) > cfg["max_rows"]:
            idx = np.sort(np.random.default_rng(0).choice(len(d["X"]), cfg["max_rows"], replace=False))
            d["X"], d["y"] = d["X"][idx], d["y"][idx]
            d["y_cont"] = None if d["y_cont"] is None else d["y_cont"][idx]
    cache_root = os.path.join(ROOT, cfg.get("output_dir", "results/v2"), "rpf_cache")
    ex = new_detector(cfg).extractor
    fcache = ev.RPFCache(cache_root, ex)
    for c in (fcache.parts or [fcache]):          # never extract: a miss must stop the run
        c.ex.extract = _refuse
    bcache = ev.BaselineCache(cache_root)
    for reg in (ev.BASELINES, ev.REGRESSION_BASELINES):
        for m in list(reg):
            reg[m] = _refuse
    return cfg, data, fcache, bcache, ex.feature_version


def configs(path=RESULTS):
    """The 5,120 evaluated configurations: those with a valid RPFNet row."""
    r = pd.read_csv(path, low_memory=False)
    r = r[(r.method == "RPFNet") & (r.skipped.astype(str) != "True") & r.error.isna()]
    return r[["dataset", "attack", "family", "rate", "seed"]].drop_duplicates().reset_index(drop=True)


def rank_z(s):
    s = np.asarray(s, float)
    rk = rankdata(s) / len(s)
    q1, med, q3 = np.percentile(s, [25, 50, 75])
    z = (s - med) / (q3 - q1 if q3 > q1 else (np.std(s) or 1.0))
    return rk, np.clip(z, -50, 50)


def features(cfg, data, fcache, bcache, fv, c):
    """Fingerprint, baseline features and truth of one configuration."""
    d = data[c.dataset]
    Xp, yp, pidx, ycp = ev.poison(d, c.attack, float(c.rate), int(c.seed))
    rpf = fcache(("eval", fv, c.dataset, c.attack, float(c.rate), int(c.seed)), Xp, yp, ycp)
    cols = []
    for m in cfg["eval"]["baselines"]:
        if m in ev.REGRESSION_BASELINES and not d["is_reg"]:
            cols += [np.full(len(Xp), np.nan)] * 2
            continue
        tgt = ycp if m in ev.REGRESSION_BASELINES else yp
        sc = bcache.get(m, Xp, tgt, int(c.seed))[0]
        cols += list(rank_z(sc))
    y = np.zeros(len(Xp), bool)
    y[pidx] = True
    return np.asarray(rpf, np.float32), np.stack(cols, 1).astype(np.float32), y


def train_rows(y, rng):
    pos, neg = np.where(y)[0], np.where(~y)[0]
    neg = rng.choice(neg, min(MAX_CLEAN, len(neg)), replace=False)
    idx = np.concatenate([pos, neg])
    w = np.concatenate([np.full(len(pos), 0.5 / len(pos)), np.full(len(neg), 0.5 / len(neg))])
    return idx, w


def fit_models(F, B, y, w):
    w = w * len(w) / w.sum()
    Bf = np.nan_to_num(B, nan=0.0)
    models = {
        "Base-GB": (HistGradientBoostingClassifier(**TREES).fit(B, y, sample_weight=w), "B"),
        "Base-LR": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(
            Bf, y, logisticregression__sample_weight=w), "Bf"),
        "FP-GB": (HistGradientBoostingClassifier(**TREES).fit(F, y, sample_weight=w), "F"),
        "FP+Base-GB": (HistGradientBoostingClassifier(**TREES).fit(np.hstack([F, B]), y, sample_weight=w), "FB"),
    }
    return models


def predict(models, F, B):
    inp = {"B": B, "Bf": np.nan_to_num(B, nan=0.0), "F": F, "FB": np.hstack([F, B])}
    return {m: mod.predict_proba(inp[k])[:, 1] for m, (mod, k) in models.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="read 3 configurations and report")
    ap.add_argument("--summary", action="store_true", help="only summarise an existing scores.csv")
    ap.add_argument("--fair", action="store_true",
                    help="train the combiners without the held-out attacks (eval.unseen_attacks), as RPFNet was; "
                         "writes scores_fair.csv and summary_fair.txt")
    args = ap.parse_args()
    tag = "_fair" if args.fair else ""
    os.makedirs(OUT, exist_ok=True)
    if args.summary:
        summarise(pd.read_csv(os.path.join(OUT, f"scores{tag}.csv")), tag)
        return
    cfg, data, fcache, bcache, fv = setup()
    cf = configs()
    splits = ev.make_splits(cfg, {n: d["is_reg"] for n, d in data.items()})
    fold_of = {n: s["split"] for s in splits for n in s["test"]}
    print(f"{len(cf)} configurations, {len(splits)} folds")

    if args.check:
        for i in np.random.default_rng(0).choice(len(cf), 3, replace=False):
            c = cf.iloc[i]
            t = time.perf_counter()
            try:
                F, B, y = features(cfg, data, fcache, bcache, fv, c)
                print(f"  OK  {c.dataset}/{c.attack}/{c.rate}/{c.seed}: {F.shape[0]} rows, "
                      f"{time.perf_counter() - t:.1f}s")
            except CacheMiss as e:
                print(f"  MISS {c.dataset}/{c.attack}/{c.rate}/{c.seed}: {e}")
        return

    # Pass 1: training rows of every configuration (all poison, <= 200 clean).
    p1 = os.path.join(OUT, "pass1.npz")
    if os.path.exists(p1):
        z = np.load(p1)
        TF, TB, TY, TW, TC = z["F"], z["B"], z["y"], z["w"], z["c"]
    else:
        TF, TB, TY, TW, TC = [], [], [], [], []
        t0 = time.perf_counter()
        for i, c in cf.iterrows():
            F, B, y = features(cfg, data, fcache, bcache, fv, c)
            idx, w = train_rows(y, np.random.default_rng(i))
            TF.append(F[idx]); TB.append(B[idx]); TY.append(y[idx]); TW.append(w)
            TC.append(np.full(len(idx), i, np.int32))
            if (i + 1) % 200 == 0:
                print(f"  pass 1: {i + 1}/{len(cf)} configurations, {time.perf_counter() - t0:.0f}s")
        TF, TB, TY, TW, TC = map(np.concatenate, (TF, TB, TY, TW, TC))
        np.savez_compressed(p1, F=TF, B=TB, y=TY, w=TW, c=TC)
    cfold = cf.dataset.map(fold_of).to_numpy()
    held_rows = cf.attack.isin(cfg_unseen()).to_numpy() if args.fair else np.zeros(len(cf), bool)

    # Pass 2: per fold, train on the other folds and score every row of the held-out configurations.
    rows = []
    for s in splits:
        tr = (cfold[TC] != s["split"]) & ~held_rows[TC]
        t = time.perf_counter()
        models = fit_models(TF[tr], TB[tr], TY[tr], TW[tr])
        print(f"[{s['split']}] trained on {tr.sum()} rows in {time.perf_counter() - t:.0f}s")
        for i in np.where(cfold == s["split"])[0]:
            c = cf.iloc[i]
            F, B, y = features(cfg, data, fcache, bcache, fv, c)
            for m, sc in predict(models, F, B).items():
                rows.append(dict(dataset=c.dataset, attack=c.attack, family=c.family, rate=c.rate,
                                 seed=c.seed, method=m, AUPRC=average_precision_score(y, sc),
                                 AUROC=roc_auc_score(y, sc)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"scores{tag}.csv"), index=False)
    summarise(df, tag)


def cfg_unseen():
    with open(CFG) as f:
        return json.load(f)["eval"].get("unseen_attacks", [])


def summarise(df, tag=""):
    r = pd.read_csv(RESULTS, low_memory=False)
    r = r[r.method.isin(["RPFNet", "ECOD"]) & (r.skipped.astype(str) != "True") & r.error.isna()]
    keys = ["dataset", "attack", "rate", "seed"]
    allr = pd.concat([df, r[keys + ["family", "method", "AUPRC", "AUROC"]]], ignore_index=True)
    allr = allr[allr.set_index(keys).index.isin(df.set_index(keys).index)]
    lines = []
    pooled = allr.groupby("method")[["AUROC", "AUPRC"]].mean().sort_values("AUPRC", ascending=False)
    lines += ["Pooled means over configurations:", pooled.round(3).to_string(), ""]
    held_out = cfg_unseen()
    for tag, sub in (("all attacks", allr), ("attacks RPFNet was trained on", allr[~allr.attack.isin(held_out)])):
        ds = sub.groupby(["method", "dataset"]).AUPRC.mean().unstack(0)
        lines += ["", f"Per dataset, {tag} (95% dataset-bootstrap interval):",
                  "  means: " + ", ".join(f"{m} {v:.3f}" for m, v in ds.mean().sort_values(ascending=False).items())]
        for a, b in [("FP-GB", "Base-GB"), ("FP+Base-GB", "FP-GB"), ("FP+Base-GB", "Base-GB"),
                     ("RPFNet", "Base-GB"), ("FP-GB", "RPFNet"), ("RPFNet", "Base-LR"), ("Base-GB", "ECOD"),
                     ("RPFNet", "ECOD")]:
            dd = (ds[a] - ds[b]).dropna()
            rng = np.random.default_rng(0)
            bs = [dd.to_numpy()[rng.integers(0, len(dd), len(dd))].mean() for _ in range(5000)]
            lines.append(f"  {a} - {b}: {dd.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, "
                         f"{np.percentile(bs, 97.5):+.3f}], higher on {(dd > 0).sum()}/{len(dd)}, "
                         f"Wilcoxon p={wilcoxon(dd).pvalue:.1e}")
    fam = allr.groupby(["method", "family"]).AUPRC.mean().unstack(1)
    lines += ["", "AUPRC by family:", fam.round(3).to_string()]
    text = "\n".join(lines)
    with open(os.path.join(OUT, f"summary{tag}.txt"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    try:
        main()
    except CacheMiss as e:
        sys.exit(f"Stopped: a fingerprint or baseline score was {e}. The code, data or library versions "
                 f"differ from the main run, so this check would not compare the same scores.")
