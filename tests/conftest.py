import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Backend"), ROOT]
GOLDEN = os.path.join(ROOT, "tests", "golden", "golden_v1.npz")


def environment():
    """What exact outputs depend on. FAISS (used for datasets over 500 rows)
    changes the fingerprints, so its version is part of it."""
    import platform, scipy, sklearn
    import RPFNet.RPFExtractor as RX
    return {"os": platform.system(), "numpy": np.__version__, "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__, "faiss": RX.faiss.__version__ if hasattr(RX, "faiss") else None}


@pytest.fixture(scope="session")
def golden():
    """Exact outputs written by tests/record_golden.py."""
    g = np.load(GOLDEN)
    rec, now = json.loads(str(g["env"])), environment()
    if now["faiss"] is None:          # only the FAISS-free outputs are compared then
        rec["faiss"] = None
    if rec != now:
        pytest.skip(f"golden outputs were recorded with {rec}, this environment is {now}; "
                    f"record a baseline here with: python tests/record_golden.py")
    return g


def summary(R, seed=0):
    rows = np.random.default_rng(seed).choice(len(R), min(20, len(R)), replace=False)
    return np.vstack([np.quantile(R, [0, 0.1, 0.5, 0.9, 1], axis=0), R[np.sort(rows)]])


def attack_record(Xp, yp, pidx):
    """[n, sorted poison indices, sum(Xp), sum(|Xp|), sum(yp)]"""
    Xp, yp = np.asarray(Xp, np.float64), np.asarray(yp, np.float64)
    return np.concatenate([[len(pidx)], np.sort(pidx), [Xp.sum(), np.abs(Xp).sum(), yp.sum()]])


def dataset(name):
    from RPFNet.Dataset import load_builtin
    r = load_builtin(name)
    return r[0], r[2], r[7]
