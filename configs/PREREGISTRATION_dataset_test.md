# Protocol for the dataset-level test (written 6 Oct 2026, before the run)

RPFNet is not changed. `configs/dataset_test.json`, `--mode dataset_test`.

* Statistic (fixed now): T(D) = mean of the largest max(5, ceil(0.01 n)) row scores of dataset D.
  RPFNet: raw scores (probabilities). Baselines (ECOD, Isolation Forest, gradient norm): scores mapped to
  (s - median) / IQR within the dataset first, because their raw scale differs across datasets.
* References: the clean classification training datasets of each outer fold (K = 20 or 21).
  RPFNet: for each reference j, a scorer trained exactly like the fold's scorer but without j's poisoned
  configurations scores j and every test item (jackknife+; Proposition 6, guarantee <= 2 alpha).
  Baselines: no training, exact conformal p-value (guarantee <= alpha).
* Test items: every classification test dataset of the fold, clean, and every evaluation configuration of
  the main benchmark (12 attacks, 5 rates, 3 seeds; configurations with < 3 poisoned rows skipped).
* Primary level alpha = 0.05. Reported: false alarms on the 26 clean datasets; power by rate and family;
  per-dataset comparison with each baseline (Wilcoxon); gated filter = RPFNet's own flags only where the test
  rejects (F1 on poisoned configurations, flagged clean datasets). Regression is excluded (5 references
  per fold give a smallest p-value of 1/6). Results are reported whatever they show.
