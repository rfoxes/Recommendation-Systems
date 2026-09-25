"""Rare and unseen categorical values: train vocabulary vs val/test coverage at several count thresholds. Run: .venv/bin/python eda/02_rare_values.py"""
import pandas as pd

from common import df_to_md, load_joined, split_labels, write_result

CATEGORICAL = ["banner_pos", "site_id", "site_domain", "site_category", "app_id", "app_domain", "app_category",
               "device_id", "device_ip", "device_model", "device_type", "device_conn_type",
               "C1", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
               "character_id", "safety_tier", "creator_type", "genre", "id"]
THRESHOLDS = [5, 20, 100]
RARE_CUTOFF = 20

df = load_joined()
df["split"] = split_labels(df)
train, val, test = (df[df["split"] == s] for s in ["train", "val", "test"])
train_counts = {col: train[col].value_counts() for col in CATEGORICAL}

# --- Step 1: train vocabulary per threshold ---
vocab_rows = []
for col in CATEGORICAL:
    counts = train_counts[col]
    row = {"column": col, "train_distinct": len(counts), "train_singletons": int((counts == 1).sum())}
    for n in THRESHOLDS:
        row[f"values_ge{n}"] = int((counts >= n).sum())
        row[f"train_rows_in_values_ge{n}"] = counts[counts >= n].sum() / len(train)
    vocab_rows.append(row)
vocab = pd.DataFrame(vocab_rows).set_index("column")

# --- Step 2: val/test rows by how often their value appears in train ---
def coverage(part):
    rows = []
    for col in CATEGORICAL:
        seen_count = part[col].map(train_counts[col]).fillna(0)
        row = {"column": col, "unseen": (seen_count == 0).mean()}
        for n in THRESHOLDS:
            row[f"lt{n}"] = (seen_count < n).mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("column")


val_cov, test_cov = coverage(val), coverage(test)

# --- Step 3: val CTR, rare vs rest ---
ctr_rows = []
for col in CATEGORICAL:
    is_rare = val[col].map(train_counts[col]).fillna(0) < RARE_CUTOFF
    ctr_rows.append({
        "column": col,
        "rare_rows": int(is_rare.sum()),
        "ctr_rare": val.loc[is_rare, "click"].mean(),
        "ctr_rest": val.loc[~is_rare, "click"].mean(),
    })
rare_ctr = pd.DataFrame(ctr_rows).set_index("column")
rare_ctr = rare_ctr[rare_ctr["rare_rows"] > 0]

# --- Step 4: write results ---
md = f"""# 02 — Rare and unseen categorical values

- train: Oct 21-27 ({len(train):,} rows) | val: Oct 28 ({len(val):,}) | test: Oct 29-30 ({len(test):,})
- counts are taken on train only; a val/test value is "unseen" if it never appears in train
- `ltN` = share of rows whose value appears fewer than N times in train (includes unseen)

## 1. Train vocabulary
{df_to_md(vocab)}

## 2. Test rows (Oct 29-30) by train count of their value
{df_to_md(test_cov)}

## 3. Val rows (Oct 28) by train count of their value
{df_to_md(val_cov)}

## 4. Val CTR (Oct 28): rows with train count < {RARE_CUTOFF} vs the rest
{df_to_md(rare_ctr)}
"""
write_result("02_rare_values", md)
