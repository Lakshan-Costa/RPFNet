"""Record the exact outputs the golden tests compare against."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import GOLDEN, attack_record, dataset, environment, summary   # noqa: E402
import RPFNet.RPFExtractor as RX                                            # noqa: E402
from RPFNet.Attack import apply_attack                                      # noqa: E402
from RPFNet.detection import RPFNetPoisonDetector, HybridEnsembleDetector   # noqa: E402
from test_attacks import EXISTING, REG_ATTACKS                              # noqa: E402
from test_detection import CASES as MODEL_CASES, MODEL, FUSION              # noqa: E402
from test_extractor import CASES as RPF_CASES, _dups, _hubs, _poisoned      # noqa: E402


def record():
    out = {"env": np.array(json.dumps(environment()))}
    for name in ["breast_cancer", "wine"]:
        X, y, _ = dataset(name)
        for atk in EXISTING:
            for seed in (42, 7):
                out[f"atk|{name}|{atk}|{seed}"] = attack_record(*apply_attack(atk, X, y, fraction=0.1, seed=seed))
    X, y, yc = dataset("diabetes_reg")
    for atk in REG_ATTACKS:
        out[f"atk|diabetes_reg|{atk}|42"] = attack_record(*apply_attack(atk, X, y, 0.1, seed=42, y_cont=yc))

    have_faiss = RX.FAISS_AVAILABLE
    try:
        for f in (False, True) if have_faiss else (False,):
            RX.FAISS_AVAILABLE = f
            for name, atk in RPF_CASES:
                Xp, yp, yc = _poisoned(name, atk)
                out[f"rpf|{int(f)}|{name}|{atk}"] = summary(RX.RPFExtractor(5, 15, 3).extract(Xp, yp, y_cont=yc))
            Xp, yp, _ = apply_attack("label_flip", *_dups(), fraction=0.1)
            out[f"rpf|{int(f)}|bc_dups|label_flip"] = summary(RX.RPFExtractor(5, 15, 3).extract(Xp, yp))
            out[f"rpf|{int(f)}|hubs|none"] = summary(RX.RPFExtractor(5, 15, 3).extract(*_hubs()))
            det = RPFNetPoisonDetector(device="cpu")
            det.load(MODEL)
            hyb = HybridEnsembleDetector(det)
            hyb.load(FUSION)
            for name, atk in MODEL_CASES:
                X, y, _ = dataset(name)
                Xp, yp, _ = apply_attack(atk, X, y, 0.1)
                out[f"model|{int(f)}|{name}|scores"] = np.asarray(det.score(Xp, yp)[0], np.float64)
                out[f"model|{int(f)}|{name}|pred_adaptive"] = np.asarray(hyb.predict_adaptive(Xp, yp)[0], np.float64)
    finally:
        RX.FAISS_AVAILABLE = have_faiss
    np.savez(GOLDEN, **out)
    print(f"wrote {len(out) - 1} outputs to {GOLDEN}\nenvironment: {environment()}")


if __name__ == "__main__":
    record()
