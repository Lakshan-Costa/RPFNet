# RPFNet: attack-agnostic poisoning detection for tabular data

Code for the paper *What Can Attack-Agnostic Poisoning Detection Guarantee? A Detector That Transfers Across Tabular Datasets*.

RPFNet maps each row of a dataset to 67 statistics (neighbourhood label agreement, class geometry, model disagreement, influence, robust-fit residuals and others) and scores them with gradient-boosted trees trained on poisoned datasets from other sources. A new dataset is screened without retraining and without knowing the attack or the contamination rate.

## Repository structure

```
run_experiments.py   # runs every experiment from a JSON config
configs/             # experiment configs and the protocols written before the post-freeze runs
Backend/RPFNet/      # fingerprint, detector, attacks, baselines, evaluation
theory_checks/       # scripts that compute the paper's tables and checks from the result files
tests/               # unit and regression tests
results/              # result files of the main runs (see below)
Datasets/            # CSV datasets not available through scikit-learn or OpenML
Frontend/            # optional web interface
```

## Installation

Python 3.10.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt  # exact versions: requirements-lock.txt
```

Datasets are loaded from scikit-learn, OpenML and Hugging Face, or from `Datasets/csv/`.

## Reproducing the paper

Results are written to `results/v2/<config name>/`. Each run resumes where it stopped; if the code or config has changed since, it stops instead of mixing results (add `--fresh` to start over).

```bash
python run_experiments.py --config configs/smoke.json                          # short sanity run
python run_experiments.py --config configs/dataset_kfold.json                  # main results: 5 folds over 32 datasets
python run_experiments.py --config configs/family_out.json                     # attack families left out of training
python run_experiments.py --config configs/dataset_kfold.json --mode ablation  # fingerprint blocks removed one at a time
python run_experiments.py --config configs/dataset_kfold.json --mode stability
python run_experiments.py --config configs/dataset_kfold.json --mode diagnostics   # harm ratio and theory checks
python run_experiments.py --config configs/dataset_kfold.json --mode scaling       # run time
```

`configs/quick.json` is the development run on 18 datasets.

### Post-freeze experiments

These reuse the frozen scorers in `results/v2/dataset_kfold/models` and do not change RPFNet. Their protocols are in `configs/PREREGISTRATION_attacks.md` and `configs/PREREGISTRATION_dataset_test.md`.

```bash
python run_experiments.py --config configs/koh_attacks.json                     # sanitiser-aware attacks
python run_experiments.py --config configs/koh_attacks.json --mode diagnostics
python run_experiments.py --config configs/frontier.json --mode frontier        # white-box adaptive attacker
python run_experiments.py --config configs/dataset_test.json --mode dataset_test   # dataset-level test
```

### Tables and checks

Run from the repository root after the corresponding experiment:

| Script | What it computes |
|---|---|
| `theory_checks/fig_comp.py`, `regime_check.py` | Figure 1 and the two regimes of Proposition 4 |
| `theory_checks/prop4a_simulation.py` | numerical check of Proposition 4(a) |
| `theory_checks/relational_check.py`, `relational_summary.py` | check of Proposition 3 on real attacks |
| `theory_checks/family_out_summary.py`, `regret_summary.py` | attack families left out, regret |
| `theory_checks/damage_strata.py`, `combiner_damage.py` | detection by downstream damage |
| `theory_checks/combiner_check.py [--fair]` | supervised combiners (XGBOD-style) |
| `theory_checks/robustness_summary.py` | robustness of the main comparison |
| `theory_checks/koh_summary.py`, `koh_table.py` | sanitiser-aware attacks |
| `theory_checks/frontier_summary.py`, `frontier_table.py` | adaptive attacker |
| `theory_checks/dataset_test_summary.py` | dataset-level test (Proposition 5) |

## Included results

To stay within the size limit, the archive includes the result files of the main five-fold run
(`results/v2/dataset_kfold`), the sanitiser-aware attacks, the adaptive attacker, the dataset-level test, the
combiner check and the diagnostics, without trained models or caches. The leave-one-family-out and development runs
are omitted; the configs above regenerate them.

## Tests

```bash
pytest tests -q
```

The golden tests compare exact outputs stored in `tests/golden/golden_v1.npz`. These depend on library versions and the OS, so they skip on other environments; `python tests/record_golden.py` records a new baseline.

## Web interface (optional)

```bash
cd Backend && python -m RPFNet.app            # API
cd Frontend/MetaPoisonWeb && npm install && npm run dev
```

## Hardware

The experiments ran on a laptop (Intel i5-11400H, 16 GB RAM, NVIDIA RTX 3050). The main five-fold run takes several hours; the fingerprint cache needs a few GB of disk.

## License

MIT (see `LICENSE`).
