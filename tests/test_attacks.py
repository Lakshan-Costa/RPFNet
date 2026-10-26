import numpy as np
import pytest

from RPFNet.Attack import apply_attack, attack_seed, evasive_attack, _trigger_features
from conftest import dataset, attack_record

EXISTING = ["label_flip", "tree_aware", "catback", "gradient_matching", "feature_collision",
            "clean_label", "backdoor", "grad_flip", "boundary_flip", "gauss_noise", "interpolation",
            "null_feature", "targeted_class", "feat_perturb", "backdoor_heavy", "repr_inversion",
            "dist_shift", "outlier_inject", "feat_dropout", "combo_flip_perturb", "combo_flip_noise",
            "adaptive_blend"]
REG_ATTACKS = ["target_shift", "leverage_attack", "target_flip_extreme", "feat_perturb", "backdoor"]


@pytest.mark.parametrize("atk", EXISTING)
@pytest.mark.parametrize("name", ["breast_cancer", "wine"])
def test_existing_attacks_unchanged(golden, name, atk):
    X, y, _ = dataset(name)
    for seed in (42, 7):
        got = attack_record(*apply_attack(atk, X, y, fraction=0.1, seed=seed))
        ref = golden[f"atk|{name}|{atk}|{seed}"]
        n = int(ref[0])
        assert len(got) == len(ref) and np.array_equal(got[:1 + n], ref[:1 + n])
        np.testing.assert_allclose(got[1 + n:], ref[1 + n:], rtol=1e-6, atol=1e-3)


def test_regression_attacks_unchanged(golden):
    X, y, yc = dataset("diabetes_reg")
    for atk in REG_ATTACKS:
        got = attack_record(*apply_attack(atk, X, y, 0.1, seed=42, y_cont=yc))
        ref = golden[f"atk|diabetes_reg|{atk}|42"]
        n = int(ref[0])
        assert len(got) == len(ref) and np.array_equal(got[:1 + n], ref[:1 + n])
        np.testing.assert_allclose(got[1 + n:], ref[1 + n:], rtol=1e-6, atol=1e-3)


def test_return_y_cont():
    X, y, yc = dataset("diabetes_reg")
    Xp, yp, pidx, ycp = apply_attack("target_shift", X, y, 0.1, y_cont=yc, return_y_cont=True)
    assert np.all(ycp[pidx] != yc[pidx])
    assert np.array_equal(np.delete(ycp, pidx), np.delete(yc, pidx))
    assert apply_attack("label_flip", X, y, 0.1, return_y_cont=True)[3] is None


def test_backdoor_trigger_ignores_rounding_noise():
    Z = np.random.default_rng(0).normal(size=(300, 6))
    Zs = ((Z - Z.mean(0)) / Z.std(0)).astype(np.float32)   # unit variance up to rounding
    assert list(_trigger_features(Zs)) == [0, 1, 2]
    assert list(_trigger_features(Z * [1, 5, 2, 9, 3, 4])) == [3, 1, 5]


def test_attack_seed_is_stable():
    # zlib.crc32 is fixed across processes and platforms, unlike hash(str)
    assert attack_seed("label_flip", 0) == 463
    assert attack_seed("label_flip", 2) == 2463


@pytest.mark.parametrize("atk", ["influence_lr", "subpopulation", "adaptive_dup"])
def test_new_attacks(atk):
    X, y, yc = dataset("wine")
    Xp, yp, pidx = apply_attack(atk, X, y, 0.1, seed=0)
    assert Xp.shape == X.shape and len(pidx) == int(len(X) * 0.1)
    rest = np.setdiff1d(np.arange(len(X)), pidx)
    assert np.array_equal(Xp[rest], X[rest]) and np.array_equal(yp[rest], y[rest])
    if atk == "influence_lr":   # poison stays inside the clean 1st-99th percentile box
        lo, hi = np.percentile(X, [1, 99], axis=0)
        assert (Xp[pidx] >= lo - 1e-5).all() and (Xp[pidx] <= hi + 1e-5).all()
    if atk == "subpopulation":
        assert len(np.unique(yp[pidx])) == 1
    with pytest.raises(ValueError):
        apply_attack(atk, *dataset("diabetes_reg")[:2], 0.1, y_cont=dataset("diabetes_reg")[2])


def test_evasive_attack_objective_never_decreases():
    X, y, _ = dataset("breast_cancer")
    score_fn = lambda Xc, yc: np.abs(Xc).sum(1)          # stand-in detector
    Xp, yp, pidx, hist = evasive_attack(X, y, 0.1, score_fn, lam=1.0, iters=5, seed=0)
    obj = [d - s for d, s in hist]
    assert all(b >= a for a, b in zip(obj, obj[1:]))
    assert Xp.shape == X.shape and len(pidx) == int(len(X) * 0.1)


def test_badnets_relabels_to_one_target():
    """BadNets: trigger stamped and every poisoned row relabelled to one target
    class it did not have; regression: target set to mean + 3 sd."""
    X, y, _ = dataset("wine")
    Xp, yp, pidx = apply_attack("badnets", X, y, 0.1, seed=3)
    trig = _trigger_features(X)
    assert len(pidx) == int(len(X) * 0.1) and len(np.unique(yp[pidx])) == 1
    assert np.all(y[pidx] != yp[pidx]) and np.all(Xp[np.ix_(pidx, trig)] == 4.0)
    rest = np.setdiff1d(np.arange(len(X)), pidx)
    assert np.array_equal(Xp[rest], X[rest]) and np.array_equal(yp[rest], y[rest])
    assert np.all(apply_attack("badnets_heavy", X, y, 0.1, seed=3)[0][np.ix_(pidx, trig)] == 10.0)
    Xr, yr, yc = dataset("diabetes_reg")
    Xq, yq, qidx, ycq = apply_attack("badnets", Xr, yr, 0.1, seed=3, y_cont=yc, return_y_cont=True)
    np.testing.assert_allclose(ycq[qidx], yc.mean() + 3 * yc.std())
    assert np.array_equal(yq, (ycq > 0).astype(int))


def test_surrogate_forest_is_fitted_once_with_the_same_result():
    """tree_aware and null_feature share one memoised forest per dataset; its
    importances are those of a freshly fitted forest."""
    import RPFNet.Attack as A
    from sklearn.ensemble import RandomForestClassifier
    X, y, _ = dataset("wine")
    A._RF_IMPORTANCE.clear()
    a = A.apply_attack("tree_aware", X, y, 0.1, seed=1)
    b = A.apply_attack("null_feature", X, y, 0.1, seed=2)
    assert len(A._RF_IMPORTANCE) == 1
    ref = RandomForestClassifier(random_state=42, n_jobs=-1).fit(X, y).feature_importances_
    assert np.array_equal(next(iter(A._RF_IMPORTANCE.values())), ref)
    assert np.array_equal(A.apply_attack("tree_aware", X, y, 0.1, seed=1)[0], a[0])
