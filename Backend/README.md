**Status: Beta** - APIs may change and detection thresholds are still being refined. Results should not be considered production-stable.

RPFNet is a Python library for detecting data poisoning attacks in tabular datasets. It supports batch analysis, streaming detection, and clean dataset export.

---

## Installation

```bash
pip install rpfnet
pip install ucimlrepo  # only needed for UCI datasets
```

---

## Quick Start

```python
from RPFNet import api

# UCI dataset (by ID)
report = api.analyze('uci', 73)
print(f"Flagged: {report['n_flagged']} / {report['n_rows']}  ({report['pct_flagged']}%)")

# Local CSV
report = api.analyze('csv', 'dataset.csv')

# Remote URL
report = api.analyze('url', 'https://example.com/dataset.csv')
# Google Drive: https://drive.google.com/uc?id=FILE_ID&export=download
```

---

## Report Keys

| Key               | Description                                                     |
| ----------------- | --------------------------------------------------------------- |
| `n_rows`          | Total rows analyzed                                             |
| `n_flagged`       | Rows flagged as potentially poisoned                            |
| `n_clean`         | Rows considered clean                                           |
| `pct_flagged`     | Percentage flagged                                              |
| `estimated_rate`  | Auto-estimated contamination rate                               |
| `mode`            | Scoring mode (`rpfnet_hybrid` or `isolation_forest`)            |
| `scores`          | Per-row anomaly scores (0–1)                                    |
| `flags`           | Per-row binary flags (1 = flagged)                              |
| `flagged_indices` | Row indices of flagged samples                                  |
| `clean_indices`   | Row indices of clean samples                                    |
| `dataframe`       | Original data annotated with `_poison_score` and `_poison_flag` |

---

## Export Clean Data

```python
# Returns original DataFrame with poisoned rows removed
clean_df = api.clean('uci', 73)
clean_df = api.clean('csv', 'dataset.csv')
clean_df = api.clean('url', 'https://example.com/dataset.csv')

print(f"Kept {len(clean_df)} clean rows")
clean_df.to_csv('clean_dataset.csv', index=False)
```

---

## Streaming Detection

For real-time or incremental data pipelines, RPFNet supports row-by-row streaming.

```python
import pandas as pd
from RPFNet import api

# Existing dataset — used as baseline
existing = pd.read_csv('dataset.csv')
api.analyze('stream', existing)

# New rows scored against that baseline
for _, row in new_data.iterrows():
    result = api.analyze('stream', row)

    if result['status'] == 'warming_up':
        continue

    if result['poison_flag']:
        print(f"Poisoned row detected! score={result['score']:.4f}")

# Export clean rows from the buffer
clean_df = api.clean('stream')
print(f"Clean rows in buffer: {len(clean_df)}")

# Utilities
api.analyze('stream')       # check buffer status
api.stream_retrain()        # force retrain after concept drift
api.stream_reset()          # wipe buffer and start fresh
```

### Streaming Result Keys

| Key                  | Description                |
| -------------------- | -------------------------- |
| `status`             | `warming_up`, `active`     |
| `score`              | Anomaly score for this row |
| `threshold`          | Current decision threshold |
| `poison_flag`        | `True` if row is flagged   |
| `n_samples`          | Rows in current buffer     |
| `buffer_pct_flagged` | % flagged in buffer window |

---

## Supported Sources

| Source     | Argument               | Example                             |
| ---------- | ---------------------- | ----------------------------------- |
| `'uci'`    | Dataset ID (int)       | `api.analyze('uci', 73)`            |
| `'csv'`    | File path (str)        | `api.analyze('csv', 'data.csv')`    |
| `'url'`    | URL string (str)       | `api.analyze('url', 'https://...')` |
| `'stream'` | DataFrame / row / None | `api.analyze('stream', df)`         |
