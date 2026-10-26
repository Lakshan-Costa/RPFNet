import os

import numpy as np
import pytest

import RPFNet.RPFExtractor as RX
from RPFNet.Attack import apply_attack
from RPFNet.detection import RPFNetPoisonDetector, HybridEnsembleDetector
from conftest import ROOT, dataset

MODEL = os.path.join(ROOT, "Backend", "RPFNet", "RPFNet_universal.pt")
FUSION = os.path.join(ROOT, "Backend", "RPFNet", "RPFNet_universal_fusion.pt")
CASES = [("breast_cancer", "label_flip"), ("wine", "backdoor"), ("moons", "feat_perturb")]


@pytest.mark.parametrize("name,atk", CASES)
def test_shipped_model_unchanged(golden, name, atk):
    """The bundled checkpoint must give the same scores and picks as before."""
    det = RPFNetPoisonDetector(device="cpu")
    det.load(MODEL)
    assert det.extractor.feature_version == 1 and det.extractor.dim == 61
    hyb = HybridEnsembleDetector(det)
    hyb.load(FUSION)
    X, y, _ = dataset(name)
    Xp, yp, _ = apply_attack(atk, X, y, 0.1)
    f = int(RX.FAISS_AVAILABLE)
    np.testing.assert_allclose(det.score(Xp, yp)[0], golden[f"model|{f}|{name}|scores"],
                               rtol=1e-4, atol=1e-5)
    pred, _ = hyb.predict_adaptive(Xp, yp)
    assert np.array_equal(pred, golden[f"model|{f}|{name}|pred_adaptive"].astype(int))


def test_v2_train_save_load_roundtrip(tmp_path):
    X, y, _ = dataset("wine")
    det = RPFNetPoisonDetector(device="cpu", epochs=2, feature_version=2, seed=0,
                               drop_features=["E_iforest"])
    det.RPFNet_fit({"wine": (X, y)}, attacks=["label_flip"], rates=[0.1], seeds=1, verbose=False)
    assert det.null_scores is not None and np.all(np.diff(det.null_scores) >= 0)
    path = str(tmp_path / "m.pt")
    det.save(path)
    det2 = RPFNetPoisonDetector(device="cpu")        # defaults to v1; load must switch
    det2.load(path)
    assert det2.extractor.feature_version == 2 and det2.drop_features == ["E_iforest"]
    np.testing.assert_allclose(det2.null_scores, det.null_scores)
    Xp, yp, _ = apply_attack("backdoor", X, y, 0.1)
    rpf = det.extractor.extract(Xp, yp)
    np.testing.assert_allclose(det._score_rpf(rpf), det2._score_rpf(rpf), rtol=1e-6)


def test_drop_features_are_ignored():
    det = RPFNetPoisonDetector(device="cpu", feature_version=2, drop_features=["E_iforest"], seed=0)
    X, y, _ = dataset("wine")
    rpf = det.extractor.extract(X, y)
    j = det.extractor.feature_names.index("E_iforest")
    rpf2 = rpf.copy()
    rpf2[:, j] += 100.0
    assert np.array_equal(det._score_rpf(rpf), det._score_rpf(rpf2))
    with pytest.raises(ValueError):
        RPFNetPoisonDetector(device="cpu", drop_features=["not_a_feature"])


def test_context_variant(tmp_path):
    X, y, _ = dataset("wine")
    Xb, yb, _ = dataset("breast_cancer")
    det = RPFNetPoisonDetector(device="cpu", epochs=2, feature_version=2, seed=0, context=True)
    det.RPFNet_fit({"wine": (X, y), "bc": (Xb, yb)}, attacks=["label_flip", "feat_perturb"],
                   rates=[0.1], seeds=1, verbose=False)
    assert det.net.net[0].in_features == 3 * det.extractor.dim
    path = str(tmp_path / "c.pt")
    det.save(path)
    det2 = RPFNetPoisonDetector(device="cpu")
    det2.load(path)
    assert det2.context
    Xp, yp, _ = apply_attack("label_flip", X, y, 0.1)
    rpf = det.extractor.extract(Xp, yp)
    np.testing.assert_allclose(det._score_rpf(rpf), det2._score_rpf(rpf), rtol=1e-6)
    with pytest.raises(ValueError):   # context training needs the per-dataset row counts
        det.train_net(rpf, np.zeros(len(rpf), np.float32), verbose=False, groups=[len(rpf) - 1])



def test_balanced_training_and_adversarial_pool(tmp_path):
    """Balanced, context-aware training on version 3, extended with adversarial_pool (rows, labels and rows per dataset to append to the pool), survives save/load."""
    X, y, _ = dataset("wine")
    Xb, yb, _ = dataset("breast_cancer")
    data = {"wine": (X, y), "bc": (Xb, yb)}
    det = RPFNetPoisonDetector(device="cpu", epochs=2, feature_version=3, seed=0,
                               context=True, balance=True)
    Xpool, ypool = det.build_pool(data, ["label_flip", "feat_perturb"], [0.1], 1, verbose=False)
    det.train_net(Xpool, ypool, verbose=False)
    more = det.adversarial_pool({"wine": (X, y)}, [0.1], iters=2, seed=0)
    assert more is not None and len(more[0]) == len(more[1]) == sum(more[2]) == len(X)
    assert more[0].shape[1] == det.extractor.dim and more[1].sum() == int(len(X) * 0.1)
    det._pool_groups = det._pool_groups + more[2]
    det.reset_net()
    det.train_net(np.concatenate([Xpool, more[0]]), np.concatenate([ypool, more[1]]), verbose=False)
    det._calibrate(data)
    path = str(tmp_path / "b.pt")
    det.save(path)
    det2 = RPFNetPoisonDetector(device="cpu")
    det2.load(path)
    assert det2.balance and det2.context and det2.extractor.feature_version == 3
    rpf = det.extractor.extract(X, y)
    np.testing.assert_allclose(det._score_rpf(rpf), det2._score_rpf(rpf), rtol=1e-6)


def test_tree_scorer_train_save_load(tmp_path):
    """scorer="trees": boosted trees on a balanced sample of the pool; scores
    survive save/load, dropped features are ignored, and poison ranks high."""
    X, y, _ = dataset("wine")
    Xb, yb, _ = dataset("breast_cancer")
    data = {"wine": (X, y), "bc": (Xb, yb)}
    det = RPFNetPoisonDetector(device="cpu", feature_version=2, seed=0, scorer="trees",
                               trees={"max_iter": 50}, drop_features=["E_iforest"])
    Xpool, ypool = det.build_pool(data, ["label_flip", "feat_perturb"], [0.1], 1, verbose=False)
    assert det.train_net(Xpool, ypool, verbose=False) == []
    det._calibrate(data)
    path = str(tmp_path / "t.pt")
    det.save(path)
    det2 = RPFNetPoisonDetector(device="cpu")
    det2.load(path)
    assert det2.scorer == "trees" and det2.trees_params["max_iter"] == 50
    Xp, yp, pidx = apply_attack("feat_perturb", X, y, 0.1, seed=5)
    rpf = det.extractor.extract(Xp, yp)
    s = det._score_rpf(rpf)
    np.testing.assert_allclose(s, det2._score_rpf(rpf))
    j = det.extractor.feature_names.index("E_iforest")
    rpf2 = rpf.copy()
    rpf2[:, j] += 100.0
    assert np.array_equal(det._score_rpf(rpf2), s)
    from sklearn.metrics import roc_auc_score
    t = np.zeros(len(Xp), bool)
    t[pidx] = True
    assert roc_auc_score(t, s) > 0.9
    with pytest.raises(ValueError):
        RPFNetPoisonDetector(device="cpu", scorer="forest")
