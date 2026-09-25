"""LightGBM complexity sweep (additive stumps -> large trees) and logistic regression; train Oct 21-27, evaluate Oct 28. Run: .venv/bin/python -m analysis.complexity_sweep"""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import LGBM_PARAMS
from src.models import LightGBMModel, LogisticModel

from .common import OUT, row_logloss, to_md, tune_step, write

NUM_LEAVES = [2, 4, 8, 16, 32, 63, 127, 255]
MAX_ROUNDS = 5000
LR_C = 0.1

train, val, encoder, X_train, X_val = tune_step()
y_train, y_val = train["click"].to_numpy(), val["click"].to_numpy()
constant_val_loss = row_logloss(y_val, np.full(len(y_val), y_train.mean())).mean()


def scores(fitted, fit_seconds, **extra):
    p_train, p_val = fitted.predict(X_train), fitted.predict(X_val)
    val_loss = row_logloss(y_val, p_val).mean()
    return {
        **extra,
        "train_AUC": roc_auc_score(y_train, p_train),
        "val_AUC": roc_auc_score(y_val, p_val),
        "train_logloss": row_logloss(y_train, p_train).mean(),
        "val_logloss": val_loss,
        "val_NE": val_loss / constant_val_loss,
        "fit_seconds": fit_seconds,
    }


# --- Step 1: LightGBM from additive (2 leaves) to large trees ---
rows = []
for num_leaves in NUM_LEAVES:
    model = LightGBMModel(encoder.categorical, params={**LGBM_PARAMS, "num_leaves": num_leaves}, max_rounds=MAX_ROUNDS)
    start = time.perf_counter()
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    rows.append(scores(model, time.perf_counter() - start, model=f"lightgbm {num_leaves} leaves",
                       num_leaves=num_leaves, trees=model.num_rounds))

# --- Step 2: logistic regression reference ---
lr = LogisticModel(encoder.categorical + encoder.decile_columns, C=LR_C)
start = time.perf_counter()
lr.fit(X_train, y_train)
rows.append(scores(lr, time.perf_counter() - start, model=f"logistic regression C={LR_C}", num_leaves=np.nan, trees=np.nan))

sweep = pd.DataFrame(rows).set_index("model")
sweep["train_minus_val_AUC"] = sweep["train_AUC"] - sweep["val_AUC"]

# --- Step 3: figure ---
lgbm_rows = sweep.dropna(subset=["num_leaves"])
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(lgbm_rows["num_leaves"], lgbm_rows["train_AUC"], marker="o", color="#eb6834", label="LightGBM train (Oct 21-27)")
ax.plot(lgbm_rows["num_leaves"], lgbm_rows["val_AUC"], marker="o", color="#2a78d6", label="LightGBM validation (Oct 28)")
ax.axhline(sweep.iloc[-1]["val_AUC"], color="#52514e", linestyle="--", linewidth=1, label="logistic regression validation")
ax.set_xscale("log", base=2)
ax.set_xticks(NUM_LEAVES, [str(n) for n in NUM_LEAVES])
ax.set(xlabel="leaves per tree (2 = additive, no feature combinations)", ylabel="AUC", title="Model complexity vs AUC")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "complexity_sweep.png", dpi=150)
plt.close(fig)

# --- Step 4: write results ---
md = f"""# Complexity sweep — train Oct 21-27 ({len(train):,} rows), validate Oct 28 ({len(val):,} rows)

- same V1 features and settings; only `num_leaves` changes; trees chosen by early stopping on Oct 28 (max {MAX_ROUNDS})
- 2 leaves = each tree splits on one feature once, so the model is additive (no feature combinations)

![complexity sweep](complexity_sweep.png)

{to_md(sweep)}
"""
write("complexity_sweep", md)
