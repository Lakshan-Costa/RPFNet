import os

import numpy as np
import pandas as pd
import pytest

from RPFNet import evaluation as ev
from RPFNet.RateEstimator import bh_select, empirical_pvalues, storey_rate
from RPFNet.baselines import BASELINES, itrim, trim
from RPFNet.diagnostics import gradient_diagnostics
from RPFNet.Attack import apply_attack
from conftest import dataset


def test_bh_matches_definition():
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(size=90), rng.uniform(0, 1e-3, size=10)])
    flags = bh_select(p, 0.1)
    ps = np.sort(p)
    k = max([i + 1 for i in range(len(p)) if ps[i] <= 0.1 * (i + 1) / len(p)], default=0)
    assert flags.sum() == k and set(np.where(flags)[0]) == set(np.argsort(p)[:k])


def test_pvalues_and_storey():
    null = np.linspace(0, 1, 1001)
    p = empirical_pvalues(np.array([0.0, 0.5, 1.0, 2.0]), null)
    assert p[0] == 1.0 and p[-1] == pytest.approx(1 / 1002) and np.all(np.diff(p) < 0)
    assert abs(storey_rate(np.random.default_rng(0).uniform(size=5000))) < 0.05


def test_threshold_free_and_prf():
    y = np.array([0, 0, 0, 1, 1], dtype=bool)
    m = ev.threshold_free(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9]))
    assert m["AUROC"] == 1.0 and m["AUPRC"] == 1.0 and m["P@k"] == 1.0
    assert ev.prf(y, np.array([0, 0, 1, 1, 0], dtype=bool)) == (0.5, 0.5, 0.5)


def test_folds_and_splits():
    names = [f"c{i}" for i in range(12)] + [f"r{i}" for i in range(5)]
    is_reg = {n: n.startswith("r") for n in names}
    folds = ev.dataset_folds(names, is_reg, 5)
    assert sorted(sum(folds, [])) == sorted(names)
    assert max(sum(is_reg[n] for n in f) for f in folds) == 1
    cfg = {"datasets": names, "protocol": {"scheme": "dataset_kfold", "n_folds": 5},
           "train": {"attacks": ["label_flip", "backdoor"], "attacks_reg": ["target_shift"]},
           "eval": {"attacks": ["label_flip"], "attacks_reg": ["target_shift"], "unseen_attacks": ["influence_lr"]}}
    for s in ev.make_splits(cfg, is_reg):
        assert not set(s["train"]) & set(s["test"])
    cfg["protocol"]["scheme"] = "family_out"
    splits = ev.make_splits(cfg, is_reg)
    assert {s["exclude_family"] for s in splits} == {"label_flip", "backdoor", "regression"}
    s = next(s for s in splits if s["exclude_family"] == "backdoor")
    tr_c, _, te = ev.split_attacks(cfg, s, False)
    assert "backdoor" not in tr_c and te == ["backdoor"]


def test_statistics_on_synthetic_results():
    rng = np.random.default_rng(0)
    rows = []
    for d in range(8):
        for a in range(3):
            for m, shift in [("RPFNet", 0.2), ("A", 0.0), ("B", -0.1)]:
                for seed in range(2):
                    rows.append(dict(dataset=f"d{d}", attack=f"a{a}", rate=0.1, seed=seed, method=m,
                                     AUPRC=0.5 + shift + 0.02 * rng.standard_normal()))
    df = pd.DataFrame(rows)
    t = ev.pooled_tests(df, "AUPRC")
    assert (t["mean_diff"] > 0).all() and (t["p_datasets_holm"] < 0.05).all()
    assert t["n_configs"].iloc[0] == 24          # seeds averaged, not counted twice
    fr = ev.friedman_nemenyi(df, "AUPRC")
    assert list(fr["avg_rank"])[0] == "RPFNet" and fr["friedman_p"] < 0.01
    mean, lo, hi = ev.bootstrap_ci(df, "AUPRC", "A", ref="RPFNet", B=300)
    assert lo < mean < hi and lo > 0


def test_baselines_return_finite_scores():
    X, y, _ = dataset("wine")
    Xp, yp, pidx = apply_attack("label_flip", X, y, 0.1, seed=0)
    for name, fn in BASELINES.items():
        s, f = fn(Xp, yp, 0)
        assert len(s) == len(Xp) and np.isfinite(s).all(), name
        assert f is None or (f.dtype == bool and len(f) == len(Xp)), name
    Xr, yr, yc = dataset("diabetes_reg")
    s, f = trim(Xr, yc, 0)
    assert len(s) == len(Xr) and f.sum() == int(len(Xr) * 0.1)
    Xq, yq, qidx, ycq = apply_attack("target_shift", Xr, yr, 0.1, seed=0, y_cont=yc, return_y_cont=True)
    s, f = itrim(Xq, ycq, 0)
    assert len(s) == len(Xq) and np.isfinite(s).all() and f.dtype == bool
    assert 0 < f.sum() <= int(0.2 * len(Xq)) + 1          # it trims at most eps_max
    assert f[qidx].mean() > 0.8                            # and catches the shifted targets


def test_rate_estimator_flag():
    """probabilities=True uses the threshold path even when the scores happen
    to look uniform, where the automatic guess would treat them as ranks."""
    from RPFNet.RateEstimator import RATE_SCALE, RATE_THRESHOLD, estimate_contamination_rate
    u = np.linspace(0.0005, 0.9995, 1000)
    want = float(np.clip(RATE_SCALE * (u > RATE_THRESHOLD).mean(), 0.001, 0.40))
    assert estimate_contamination_rate(u, probabilities=True) == want
    assert estimate_contamination_rate(u) == 0.001                      # guessed: ranks
    p = np.r_[np.full(950, 0.02), np.full(50, 0.97)]
    assert estimate_contamination_rate(p) == estimate_contamination_rate(p, probabilities=True)


def test_theorems_hold_on_real_data():
    X, y, _ = dataset("breast_cancer")
    for atk in ["label_flip", "backdoor"]:
        Xp, yp, pidx = apply_attack(atk, X, y, 0.1, seed=3)
        d = gradient_diagnostics(Xp, yp, pidx)
        assert d["residual_all"] < 1e-5 and d["residual_P"] < 1e-5
        assert d["thm1_holds"] and d["max_logit_change"] <= d["logit_change_bound"] + 1e-9
        assert d["thm2_direction_holds"] and d["thm2_auroc_holds"]


@pytest.mark.parametrize("name", ["breast_cancer", "wine"])
def test_harm_certificate(name):
    """harm = ||grad F_P(theta_eps)|| equals eps ||mu_Q - mu_P|| up to the solver residual and never exceeds the spectral certificate (binary and softmax proxies)."""
    X, y, _ = dataset(name)
    for atk in ["label_flip", "feat_perturb"]:
        Xp, yp, pidx = apply_attack(atk, X, y, 0.1, seed=1)
        d = gradient_diagnostics(Xp, yp, pidx, test=(X[:60], y[:60]), original=(X, y))
        assert d["cert_holds"] and abs(d["harm"] - d["eps"] * d["gap"]) <= d["residual_all"] + 1e-9
        assert d["harm_ratio"] <= d["spike_ratio"] * (1 + 1e-6) and d["spike_ratio"] >= 1 - d["eps"] - 1e-9
        assert 0 <= d["auprc_blockH"] <= 1 and abs(d["damage_proxy"]) <= 1 and abs(d["damage_rf"]) <= 1


def test_null_columns():
    rng = np.random.default_rng(0)
    null = np.sort(rng.normal(size=5000))
    s = np.r_[rng.normal(size=950), rng.normal(4, 1, size=50)]
    y = np.r_[np.zeros(950, bool), np.ones(50, bool)]
    out = ev.null_columns(s, y, {"": null, "_oof": null}, 0.1)
    p = empirical_pvalues(s, null)
    bh = bh_select(p, 0.1)
    # the columns the frozen runs report keep their definitions
    assert out["n_flag_bh"] == bh.sum() and (out["P_bh"], out["R_bh"], out["F1_bh"]) == ev.prf(y, bh)
    assert out["FDR_bh"] == (bh & ~y).sum() / bh.sum() and out["storey_rate"] == storey_rate(p)
    assert out["F1_bh_oof"] == out["F1_bh"] and out["t_bh"] == pytest.approx(0.1 * bh.sum() / 1000)
    assert "n_flag_by" not in out and out["n_flag_by_oof"] <= out["n_flag_bh_oof"]
    assert out["c1e-02"] == pytest.approx((p[~y] <= 0.01).mean() / 0.01)
    assert 0.5 < out["c3e-02"] < 1.5          # clean scores drawn like the null: c(t) near 1
    small = ev.null_columns(s, y, {"": null[::50]}, 0.1)             # 100 null scores: p >= 1/101
    assert np.isnan(small["c1e-04"]) and np.isnan(small["c3e-03"]) and not np.isnan(small["c1e-02"])
    none = ev.null_columns(s, np.zeros(1000, bool), {"": null}, 0.1)   # all clean: every flag is false
    assert none["FDR_bh"] == float(none["n_flag_bh"] > 0) and none["F1_bh"] == 0


def test_null_parts_keep_groups_and_balance_tasks():
    import run_experiments as rx
    names = ["a", "b", "c", "d", "e", "f", "r1_reg", "r2_reg", "r3_reg",
             "digits", "optdigits", "california_large", "california_reg"]
    data = {n: {"is_reg": n.endswith("_reg")} for n in names}
    parts = rx.null_parts({"groups": {"optdigits": "digits", "california_reg": "california_large"}},
                          names, data)
    assert set(parts) == set(names) and set(parts.values()) == {0, 1, 2}
    assert parts["digits"] == parts["optdigits"] and parts["california_large"] == parts["california_reg"]
    reg = np.bincount([parts[n] for n in names if n.endswith("_reg")], minlength=3)
    assert reg.max() - reg.min() <= 1


def test_evaluate_config_rows(tmp_path):
    from RPFNet.detection import RPFNetPoisonDetector
    X, y, _ = dataset("wine")
    det = RPFNetPoisonDetector(device="cpu", epochs=1, feature_version=2, seed=0)
    det.RPFNet_fit({"wine": (X, y)}, attacks=["label_flip"], rates=[0.1], seeds=1, verbose=False)
    data = ev.load_named("wine")
    cache = ev.RPFCache(str(tmp_path), det.extractor)
    rows = ev.evaluate_config(det, data, "wine", "backdoor", 0.1, 0, baselines=["IsoForest"], cache=cache)
    methods = {r["method"] for r in rows}
    assert {"RPFNet", "IsoForest", "_none", "_oracle", "_clean"} <= methods
    rpf_row = next(r for r in rows if r["method"] == "RPFNet")
    for k in ["AUROC", "AUPRC", "F1_own", "F1_shared", "F1_bh", "FDR_bh", "downstream", "time_s"]:
        assert k in rpf_row
    extra = {"Const": lambda r: (np.arange(len(r), dtype=float), np.zeros(len(r), bool))}
    const = ev.evaluate_config(det, data, "wine", "backdoor", 0.1, 0, cache=cache, extra_scorers=extra)
    assert next(r for r in const if r["method"] == "Const")["F1_own"] == 0   # its own flags are used
    seen = []

    def task_aware(r, reg):
        seen.append(reg)
        s = np.arange(len(r), dtype=float)
        return s, ev.topk_flags(s, 0.1), {"alt": np.ones(len(r), bool)}   # a further flag set
    task_aware.needs_task = True
    rows = ev.evaluate_config(det, data, "wine", "backdoor", 0.1, 0, cache=cache, extra_scorers={"T": task_aware})
    assert seen == [False]
    t = next(r for r in rows if r["method"] == "T")
    p = t["n_poison"] / t["n"]                  # flagging every row: precision p, recall 1
    assert t["F1_alt"] == pytest.approx(2 * p / (1 + p)) and t["est_rate_own"] == pytest.approx(0.1, abs=0.01)
    again = ev.evaluate_config(det, data, "wine", "backdoor", 0.1, 0, baselines=[], cache=cache)
    assert again[0]["AUROC"] == rpf_row["AUROC"]              # cache hit, same result
    assert ev.evaluate_config(det, data, "wine", "backdoor", 0.001, 0)[0]["skipped"]


def test_cache_key_includes_the_data(tmp_path):
    """The same key with different rows (e.g. another max_rows) must not hit."""
    from RPFNet.RPFExtractor import RPFExtractor
    X, y, _ = dataset("wine")
    cache = ev.RPFCache(str(tmp_path), RPFExtractor(5, 15, 3, feature_version=2))
    a = cache(("eval", 2, "wine"), X, y)
    b = cache(("eval", 2, "wine"), X[:100], y[:100])
    assert a.shape[0] == len(X) and b.shape[0] == 100
    assert np.array_equal(cache(("eval", 2, "wine"), X, y), a)


def test_composite_cache_reuses_version2(tmp_path):
    """Version 3 caches its version-2 part where a version-2 run stores it."""
    from RPFNet.effect_blocks import make_extractor
    X, y, _ = dataset("wine")
    c2 = ev.RPFCache(str(tmp_path), make_extractor(2, 5, 15, 3))
    c3 = ev.RPFCache(str(tmp_path), make_extractor(3, 5, 15, 3))
    a = c2(("eval", 2, "wine"), X, y)
    n2 = len(os.listdir(c2.dir))
    b = c3(("eval", 3, "wine"), X, y)
    assert c3.parts[0].dir == c2.dir and len(os.listdir(c2.dir)) == n2   # a cache hit
    assert b.shape == (len(X), c3.ex.dim) and np.array_equal(b[:, :a.shape[1]], a)
    assert np.array_equal(c3(("eval", 3, "wine"), X, y), b)


def test_minp_combiner():
    """Tippett's combination: the smallest p-value decides; Fisher's sum only
    orders rows whose smallest p-value is the same."""
    from RPFNet.combine import MinPCombiner, robust_scale
    c = MinPCombiner(np.tile(np.arange(100.0)[:, None], (1, 2)))    # clean values 0..99
    S = np.array([[200.0, -1.0],    # A: beyond every clean value in score 1, ordinary in score 2
                  [99.0, 99.0],     # B: next-to-extreme in both
                  [200.0, 99.0],    # C: extreme in 1, next-to-extreme in 2
                  [200.0, 200.0],   # D: extreme in both
                  [50.0, 50.0]])    # E: ordinary
    np.testing.assert_allclose(c.pvalues(S)[:2], [[1 / 101, 1.0], [2 / 101, 2 / 101]])
    s = c.score(S)
    assert s[3] > s[2] > s[0] > s[1] > s[4]    # A beats B although B's Fisher sum is larger
    np.testing.assert_allclose(robust_scale(np.full(7, 3.0)), 0.0)
    v = robust_scale(np.arange(101.0))
    assert abs(np.median(v)) < 1e-12 and abs((np.percentile(v, 75) - np.percentile(v, 25)) - 1.349) < 1e-6


def test_rate_model_tracks_the_rate():
    """Trained on datasets with known rates, the estimate follows the true rate
    on new datasets drawn the same way."""
    from RPFNet.combine import MinPCombiner, RateModel
    rng = np.random.default_rng(0)
    clean = rng.normal(size=(5000, 3))

    def poisoned(rate):
        S = rng.normal(size=(1000, 3))
        k = int(1000 * rate)
        S[:k, rng.integers(3)] += 5.0          # one invariant violated by the poison
        return S, rate
    m = RateModel(MinPCombiner(clean), clean).fit([poisoned(r) for r in [0.01, 0.03, 0.05, 0.1, 0.2] * 10])
    est = [m.predict(poisoned(r)[0]) for r in [0.01, 0.05, 0.1, 0.2]]
    assert np.all(np.diff(est) > 0)
    assert all(0.5 < e / r < 2 for e, r in zip(est, [0.01, 0.05, 0.1, 0.2]))


def test_learned_combiner_ignores_a_flooding_signal():
    """Score 2 has naturally heavy tails on clean rows and never marks poison."""
    from sklearn.metrics import average_precision_score
    from RPFNet.combine import LearnedCombiner, MinPCombiner, dataset_evidence
    rng = np.random.default_rng(0)
    clean = rng.normal(size=(5000, 3))

    def dataset(seed):
        r = np.random.default_rng(seed)
        S = r.normal(size=(1000, 3)); y = np.zeros(1000, bool)
        S[r.choice(1000, 50, replace=False), 2] += 4.0            # natural outliers, clean
        idx = r.choice(1000, 50, replace=False); y[idx] = True
        S[idx, r.integers(2)] += 4.0                               # poison in score 0 or 1
        return S, y
    assert dataset_evidence(MinPCombiner(clean).pvalues(dataset(1)[0])).shape == (9,)
    comb = LearnedCombiner(MinPCombiner(clean)).fit([dataset(s) for s in range(40)])
    S, y = dataset(99)
    learned = average_precision_score(y, comb.score(S))
    minp = average_precision_score(y, MinPCombiner(clean).score(S))
    assert learned > 0.9 and learned > minp + 0.2


def test_parallel_precompute_matches_sequential(tmp_path):
    """Fingerprints and baselines written by the parallel precompute are the
    ones the sequential code computes, under the keys it looks up."""
    X, y, _ = dataset("wine")
    Xb, yb, _ = dataset("breast_cancer")
    data = {"wine": dict(X=X, y=y, y_cont=None, is_reg=False),
            "bc": dict(X=Xb, y=yb, y_cont=None, is_reg=False)}
    tasks = [dict(kind="clean", name="wine", attack=None, rate=None, seed=None, X=X, y=y, y_cont=None,
                  is_reg=False, baselines=()),
             dict(kind="pool", name="bc", attack="tree_aware", rate=0.1, seed=0, X=Xb, y=yb, y_cont=None,
                  is_reg=False, baselines=()),
             dict(kind="eval", name="wine", attack="label_flip", rate=0.1, seed=1, X=X, y=y, y_cont=None,
                  is_reg=False, baselines=("ECOD", "IsoForest"))]
    par, seq = str(tmp_path / "par"), str(tmp_path / "seq")
    ev.precompute(tasks, par, (3, 5, 15, 3, 42), n_jobs=2)
    from RPFNet.effect_blocks import make_extractor
    from RPFNet.Attack import attack_seed
    cp, cs = ev.RPFCache(par, make_extractor(3, 5, 15, 3)), ev.RPFCache(seq, make_extractor(3, 5, 15, 3))
    Xp, yp, pidx = apply_attack("tree_aware", Xb, yb, 0.1, seed=attack_seed("tree_aware", 0))
    Xe, ye, pe, _ = ev.poison(data["wine"], "label_flip", 0.1, 1)
    for key, args in [(("clean", 3, "wine"), (X, y)), (("RPFNet", 3, "bc", "tree_aware", 0.1, 0), (Xp, yp)),
                      (("eval", 3, "wine", "label_flip", 0.1, 1), (Xe, ye))]:
        n_before = sum(len(os.listdir(c.dir)) for c in cp.parts)
        a = cp(key, *args)                                     # must be a cache hit
        assert sum(len(os.listdir(c.dir)) for c in cp.parts) == n_before
        np.testing.assert_allclose(a, cs(key, *args), rtol=1e-5, atol=1e-5)
    bp, bs = ev.BaselineCache(par), ev.BaselineCache(seq)
    n_before = len(os.listdir(bp.dir))
    for m in ("ECOD", "IsoForest"):
        s1, f1, _ = bp.get(m, Xe, ye, 1)
        s2, f2, _ = bs.get(m, Xe, ye, 1)
        assert np.array_equal(s1, s2) and ((f1 is None and f2 is None) or np.array_equal(f1, f2))
    assert len(os.listdir(bp.dir)) == n_before
