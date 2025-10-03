Telco Customer Churn - LightGBM example
=====================================

This small example trains a LightGBM classifier on the Telco churn dataset
from OpenML (dataset id=44228). The goal is to provide a simple, readable
reference implementation.

Quick start (conda + pip)
-------------------------

1. Create and activate a conda environment:

```bash
conda create -n btaic python=3.12 -y
conda activate btaic
```

2. Install Python dependencies with pip:

```bash
pip install -r requirements.txt
```


3. Run the training script:

```bash
python customer_churn.py
```

Options
-------

Use --help to see script options:

```bash
python customer_churn.py --help
```

Notes
-----

- The script performs minimal preprocessing and is intended to be a starting
  point. For production use, add more robust feature engineering and
  hyperparameter tuning.
- The dataset is fetched from OpenML on first run; subsequent runs use the
  cached copy from sklearn's OpenML cache.


