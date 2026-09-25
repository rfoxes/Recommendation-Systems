"""Globally permuted-label sanity control for the V1 pipeline (tune step, Oct 28). Run: .venv/bin/python -m analysis.permutation_check"""
import numpy as np
import pandas as pd

from src.metrics import evaluate
from src.models import LightGBMModel

from .common import to_md, tune_step, write

SEEDS = range(5)

# --- Step 1: shuffle click across all rows, rebuild every feature, train and score V1 ---
rows = []
for seed in SEEDS:
    train, val, encoder, X_train, X_val = tune_step(label_permutation_seed=seed)
    y_train, y_val = train["click"].to_numpy(), val["click"].to_numpy()
    model = LightGBMModel(encoder.categorical).fit(X_train, y_train, X_val=X_val, y_val=y_val)
    scores = evaluate(y_val, model.predict(X_val), np.full(len(y_val), y_train.mean()), val["publisher_id"].to_numpy())
    rows.append({"permutation_seed": seed, "trees": model.num_rounds, **scores})
table = pd.DataFrame(rows).set_index("permutation_seed")

# --- Step 2: write results ---
md = f"""# Permuted-label control — V1 LightGBM, train Oct 21-27, evaluate Oct 28

- `click` shuffled across all 1,000,000 rows before any feature (history, recent click rates, encodings) is built
- a pipeline without leakage scores AUC ~0.5 and NE >= 1

{to_md(table[['trees', 'logloss', 'NE', 'AUC', 'AUC_within_publisher', 'mean_pred', 'actual_ctr']])}

- mean AUC {table['AUC'].mean():.4f} (min {table['AUC'].min():.4f}, max {table['AUC'].max():.4f})
"""
write("permutation_check", md)
