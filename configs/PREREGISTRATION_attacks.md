# Protocol for the two post-freeze experiments (written 5 Oct 2026, before either run)

RPFNet is not changed. Both experiments use the frozen fold scorers of
`results/v2/dataset_kfold/models` (`"models_from": "dataset_kfold"`); a missing model stops the
run instead of retraining. Attack parameters below are fixed now and will not be tuned after
results are seen; any rerun is reported together with the first run.

## B1. Sanitiser-aware attacks (Koh, Steinhardt & Liang 2022) — `configs/koh_attacks.json`

* Attacks (never used in training): `koh_minmax` (improved min-max, concentrated) and
  `koh_influence` (influence attack started from the min-max placement), `Backend/RPFNet/koh_attacks.py`.
* Feasible set per poisoned label: sphere and slab at the 0.90 quantile of the clean rows of that
  class, and the clean 1st–99th percentile box. 10 distinct locations, copies jittered by 0.01 SD.
  Influence attack: 15 projected steps with backtracking on the attacker's held-out clean rows.
* Data: the 26 classification datasets, five folds as in the main benchmark, rates
  0.5/1/5/10/20 %, three evaluation seeds (configurations with fewer than three poisoned rows skipped).
* Methods: every method of Table 2 (RPFNet, Stack, MinP, Net and the 16 baselines).
* Primary metric AUPRC. Per-dataset means; RPFNet against each baseline by Wilcoxon signed-rank,
  Holm-corrected. Reported whatever the outcome, including losses to any baseline.
* Damage: logistic-regression accuracy loss on the clean test split (`_none` vs `_clean` rows),
  and with `--mode diagnostics` (seed 0) the random-forest loss and the harm ratio ρ.

## B2. White-box adaptive attacker — `configs/frontier.json`, `--mode frontier`

* Attacker: `Backend/RPFNet/adaptive_attack.py`. Label flips of 10 % of the rows; each poison row
  moved toward the mean of its 5 nearest clean rows of the flipped label by a_i ∈ {0, ¼, ½, ¾, 1}.
  Objective damage − λ·AUROC(RPFNet), damage = log-loss of a logistic regression on clean rows the
  attacker holds out. Uniform sweep over a, then 20 steps of block coordinate ascent per λ,
  λ ∈ {0, 0.5, 2, 8}. The attacker queries the exact frozen scorer of the dataset's fold.
* Data: every classification test dataset of the five folds, training split subsampled to at most
  2,000 rows, two seeds.
* Reported for every uniform and adaptive solution: AUROC and AUPRC of RPFNet and of CL, k-NN label,
  ECOD, Isolation Forest, SEVER, gradient norm and spectral signatures; accuracy loss of a logistic
  regression and a random forest on the clean test split; harm ratio ρ and spectral ratio.
* Planned analyses: (i) RPFNet AUROC against damage along the frontier; (ii) the share of
  solutions that evade RPFNet (AUROC ≤ 0.7) while costing the logistic regression ≥ 2 accuracy
  points; (iii) for evading solutions, whether ρ < 1/4 (harm within the floor of Proposition 2)
  or the spectral ratio exceeds 1 (visible to the eigenvalue rule).
