# RPFNet: Relational Invariant Approach to Attack-Agnostic Data Poisoning Detection in Tabular Data

This repository contains the official implementation of **RPFNet**, a relational invariant-based framework for detecting data poisoning in tabular datasets. The method is evaluated across multiple datasets, attack types, and contamination rates, supporting both classification and regression tasks.

---

## Overview

RPFNet is based on the hypothesis that **data poisoning disrupts relational structure** within datasets. Instead of relying on point-wise anomalies, RPFNet constructs a **Relational Poison Fingerprint (RPF)** capturing:

- Neighborhood consistency
- Geometric coherence
- Influence on model behavior
- Density and structural properties

A **meta-learned detector** is trained across multiple datasets to generalize to unseen domains without requiring:

- Known contamination rate
- Known attack type
- Clean validation data

---

## Repository Structure

```
RPFNet/
│── Backend/              # Core model + training pipeline
│── Frontend/             # UI for visualization and interaction
│── Datasets/             # Dataset loaders and preprocessing
|── figures/              # Returns ablation study and other figures
│── results/              # Returns the model training and testing results
│── requirements.txt      # Dependencies
│── README.md
```

---

## Environment Setup

### 1. Open the repository folder

```bash
cd rpfnet
```

---

### 2. Create environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 💻 Compute Requirements

Experiments were conducted on:

GPU: NVIDIA RTX 3050 (CUDA-enabled)
CPU: Intel i5-11400H
RAM: 16GB

Typical requirements:

Disk: ~3–6 GB (including datasets + dependencies)
RAM: 4–8 GB during execution

---

## Datasets

RPFNet supports multiple datasets:

### Classification

Breast Cancer
Wine
Digits
Adult Income
German Credit
Ionosphere
Spambase
Vehicle, Segment, Pendigits, etc.

### Regression

California Housing
Diabetes
Abalone
Friedman1
Synthetic regression datasets

Datasets are loaded via:

`scikit-learn`
`OpenML`
`HuggingFace Datasets`

---

## Poisoning Attacks

The framework evaluates multiple attack types:

Label Flipping
Clean-label attacks
Backdoor injection
Feature perturbation
Distribution shift
Regression-specific attacks

Contamination rates:

```
0.5%, 1%, 5%, 10%, 15%, 20%, 25%
```

---

### Outputs

The system generates:

- F1-score, Precision, Recall
- Per-dataset evaluation results
- Contamination estimates
- Model predictions

---

## Model Details

- Input: 61-dimensional Relational Poison Fingerprint
- Architecture: Fully connected neural network
- Loss: Focal loss (class imbalance handling)
- Optimization: Adam + cosine annealing
- Training: Meta-learning across datasets

---

## Evaluation Protocol

Train/test split: 70/10/20
Poisoning applied only to training set
Evaluation on clean test data
Metrics:

Precision
Recall
F1-score
AUC

Results are reported as:

```
Mean ± standard deviation over 3 random seeds
```

---

## Statistical Significance

- Paired t-tests are used to compare RPFNet with baselines
- Significance threshold: p < 0.05
- Full statistical results are provided in the paper appendix

---

## Reproducibility Notes

To reproduce results:

1. Use the same random seed:

```python
seed = 42
```

2. Ensure dataset subsampling matches paper settings

3. Use identical preprocessing:

   Standard scaling
   One-hot encoding

4. Run experiments using provided scripts

---

## Frontend

To run the UI:

```bash
cd frontend
npm install
npm run dev
```

---

## Limitations

Optimized for tabular datasets
Computational cost increases with dataset size
Lower performance at very low contamination rates (≤1%)

---

## License

This project is for academic and research purposes.

---
