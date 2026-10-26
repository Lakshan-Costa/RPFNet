import numpy as np
import pytest

import RPFNet.RPFExtractor as RX
from RPFNet.Attack import apply_attack
from scipy.stats import spearmanr

from conftest import summary, dataset

CASES = [("breast_cancer", "label_flip"), ("wine", "label_flip"), ("wine", "backdoor"),
         ("diabetes_reg", "target_shift"), ("moons", "label_flip")]


def _poisoned(name, atk):
    X, y, yc = dataset(name)
    Xp, yp, _ = apply_attack(atk, X, y, fraction=0.1, y_cont=yc)
    return Xp, yp, yc


def _dups():
    """Breast Cancer with its first 40 rows duplicated."""
    X, y, _ = dataset("breast_cancer")
    return np.vstack([X, X[:40]]), np.concatenate([y, y[:40]])


def _hubs():
    """Three points with > 200 reverse neighbours each, which exercise v1's
    sampled two-hop path."""
    rng = np.random.default_rng(2)
    S = rng.normal(size=(2500, 50))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    Xh = np.vstack([np.zeros((3, 50)) + rng.normal(0, 1e-3, (3, 50)), S]).astype(np.float32)
    return Xh, (rng.random(len(Xh)) < 0.5).astype(int)


@pytest.mark.parametrize("name,atk", CASES)
def test_v1_output_unchanged(golden, name, atk):
    """feature_version=1 must reproduce what the shipped model was trained on."""
    Xp, yp, yc = _poisoned(name, atk)
    R = RX.RPFExtractor(5, 15, 3).extract(Xp, yp, y_cont=yc)
    ref = golden[f"rpf|{int(RX.FAISS_AVAILABLE)}|{name}|{atk}"]
    np.testing.assert_allclose(summary(R), ref, rtol=1e-4, atol=1e-4)


def test_v1_duplicates_and_hubs_unchanged(golden):
    f = int(RX.FAISS_AVAILABLE)
    Xp, yp, _ = apply_attack("label_flip", *_dups(), fraction=0.1)
    R = RX.RPFExtractor(5, 15, 3).extract(Xp, yp)
    np.testing.assert_allclose(summary(R), golden[f"rpf|{f}|bc_dups|label_flip"], rtol=1e-4, atol=1e-4)
    R = RX.RPFExtractor(5, 15, 3).extract(*_hubs())
    np.testing.assert_allclose(summary(R), golden[f"rpf|{f}|hubs|none"], rtol=1e-4, atol=1e-4)


def test_deterministic():
    Xp, yp, _ = _poisoned("wine", "label_flip")
    for fv in (1, 2):
        a = RX.RPFExtractor(5, 15, 3, feature_version=fv).extract(Xp, yp)
        b = RX.RPFExtractor(5, 15, 3, feature_version=fv).extract(Xp, yp)
        assert np.array_equal(a, b)


def test_v2_layout():
    ex = RX.RPFExtractor(5, 15, 3, feature_version=2)
    assert ex.dim == 59 == len(ex.feature_names) == len(set(ex.feature_names))
    covered = sorted(i for _, sl in ex.block_names.values() for i in range(sl.start, sl.stop))
    assert covered == list(range(ex.dim))
    for i in RX.RPFExtractor.V2_DROP:
        assert RX.FEATURE_NAMES_V1[i] not in ex.feature_names
    Xp, yp, _ = _poisoned("wine", "backdoor")
    R = ex.extract(Xp, yp)
    assert R.shape == (len(Xp), 59) and np.isfinite(R).all()
    assert set(ex.last_timings) == {"neighbours", "A", "B", "C", "D", "E", "F", "G", "H"}


def test_v2_removes_self_from_neighbours():
    Xd, yd = _dups()
    ex = RX.RPFExtractor(5, 15, 3, feature_version=2)
    ex.extract(Xd, yd)
    I = ex._last_neighbors[0]
    assert not (I == np.arange(len(Xd))[:, None]).any()


def test_v2_block_h_when_gradients_outnumber_rows():
    """The gradient matrix has K (d + 1) columns for K classes, which can
    exceed the number of rows (mfeat_factors: 10 x 217 > 1600)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 60)).astype(np.float32)
    y = rng.integers(0, 3, 120)
    h = RX.RPFExtractor(5, 15, 3, feature_version=2)._block_h(X, y)
    assert h.shape == (120,) and np.ptp(h) > 0


@pytest.mark.skipif(not RX.FAISS_AVAILABLE, reason="faiss not installed")
def test_v2_faiss_matches_sklearn():
    """FAISS computes squared distances in float32 as |x|^2 + |y|^2 - 2<x,y>, with an absolute error of order 1e-7 (|x|^2 + |y|^2), about 1e-6 here."""
    Xp, yp, _ = _poisoned("moons", "label_flip")
    out = {}
    try:
        for f in (False, True):
            RX.FAISS_AVAILABLE = f
            ex = RX.RPFExtractor(5, 15, 3, feature_version=2)
            out[f] = ex.extract(Xp, yp), ex._last_neighbors[1]
    finally:
        RX.FAISS_AVAILABLE = True
    (R_sk, D_sk), (R_fa, D_fa) = out[False], out[True]
    np.testing.assert_allclose(D_fa, D_sk, rtol=1e-4, atol=5e-3)
    for j, name in enumerate(ex.feature_names):
        if np.ptp(R_sk[:, j]) > 0:
            assert spearmanr(R_sk[:, j], R_fa[:, j])[0] > 0.99, name


def test_v3_is_v2_plus_effect_block():
    """Version 3 = the version-2 columns unchanged, then block I, oriented so
    that higher is more suspicious."""
    from sklearn.metrics import roc_auc_score
    from RPFNet.effect_blocks import EFFECT_NAMES, make_extractor
    X, y, _ = dataset("breast_cancer")
    Xp, yp, pidx = apply_attack("label_flip", X, y, fraction=0.1, seed=0)
    v2 = make_extractor(2, 5, 15, 3).extract(Xp, yp)
    ex = make_extractor(3, 5, 15, 3)
    v3 = ex.extract(Xp, yp)
    assert v3.shape == (len(Xp), ex.dim) and ex.dim == v2.shape[1] + len(EFFECT_NAMES)
    assert np.array_equal(v3[:, :v2.shape[1]], v2) and np.isfinite(v3).all()
    assert ex.feature_names[v2.shape[1]:] == EFFECT_NAMES
    assert ex.block_names["I"][1] == slice(v2.shape[1], ex.dim)
    t = np.zeros(len(Xp), bool)
    t[pidx] = True
    for name in ["I_trim_resid", "I_cluster_oof", "I_aum", "I_knn_shapley"]:
        assert roc_auc_score(t, v3[:, ex.feature_names.index(name)]) > 0.8, name
    Xr, yr, yc = dataset("diabetes_reg")
    Xq, yq, qidx, ycq = apply_attack("target_shift", Xr, yr, 0.1, seed=0, y_cont=yc, return_y_cont=True)
    v3r = ex.extract(Xq, yq, y_cont=ycq)
    t = np.zeros(len(Xq), bool)
    t[qidx] = True
    assert roc_auc_score(t, v3r[:, ex.feature_names.index("I_huber_resid")]) > 0.8
