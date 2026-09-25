"""CTR by hour of day and random-vs-temporal split comparison. Run: .venv/bin/python eda/01_temporal_split.py"""
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from common import (FIGURES, RANDOM_SEED, add_user_proxy, apply_category_maps, ctr_table, df_to_md, evaluate,
                    fit_category_maps, load_impressions, split_labels, write_result)

df = add_user_proxy(load_impressions())
df["split"] = split_labels(df)
df["hour_of_day"] = df["ts"].dt.hour

# --- Step 1: CTR by hour of day ---
by_hod = ctr_table(df, "hour_of_day").sort_index()
by_hour = ctr_table(df, "hour")
hourly_ctr_std = by_hour["ctr"].std()
binomial_std = np.sqrt((by_hour["ctr"].mean() * (1 - by_hour["ctr"].mean()) / by_hour["n"]).mean())

fig, ax = plt.subplots(figsize=(8, 4))
ax.fill_between(by_hod.index, by_hod["ci_low"], by_hod["ci_high"], color="#2a78d6", alpha=0.18, linewidth=0)
ax.plot(by_hod.index, by_hod["ctr"], color="#2a78d6", linewidth=2, marker="o", markersize=4)
ax.axhline(df["click"].mean(), color="#52514e", linewidth=1, linestyle="--")
ax.set(title="CTR by hour of day (95% CI), all days pooled", xlabel="hour of day (UTC)", ylabel="CTR", xticks=range(0, 24, 2))
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
FIGURES.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURES / "01_ctr_by_hour_of_day.png", dpi=150)
plt.close(fig)

# --- Step 2: random vs temporal split ---
FEATURES = ["banner_pos", "site_id", "site_domain", "site_category", "app_id", "app_domain", "app_category",
            "device_model", "device_type", "device_conn_type", "C1", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]
PARAMS = {"objective": "binary", "num_leaves": 63, "learning_rate": 0.05, "min_data_in_leaf": 100,
          "num_threads": 3, "seed": RANDOM_SEED, "verbose": -1}
OOV_CHECK = ["site_id", "app_id", "device_model", "C14"]
SPLIT_SEEDS = [RANDOM_SEED, RANDOM_SEED + 1, RANDOM_SEED + 2]
METRICS = ["logloss", "NE", "AUC", "ECE"]


def fit_and_evaluate(train, val, test):
    maps = fit_category_maps(train, FEATURES)
    X = {name: apply_category_maps(part, maps).assign(hour_of_day=part["hour_of_day"])
         for name, part in [("train", train), ("val", val), ("test", test)]}
    dtrain = lgb.Dataset(X["train"], train["click"], categorical_feature=FEATURES)
    dval = lgb.Dataset(X["val"], val["click"], reference=dtrain)
    model = lgb.train(PARAMS, dtrain, num_boost_round=1000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    p = model.predict(X["test"], num_iteration=model.best_iteration)
    y = test["click"].to_numpy()
    return {
        "train_rows": len(train), "val_rows": len(val), "test_rows": len(test),
        "best_iteration": model.best_iteration,
        **evaluate(y, p, base_rate=y.mean()),
        "test_rows_user_in_train": test["user"].isin(train["user"]).mean(),
        **{f"test_unseen_{c}": (X["test"][c] == 0).mean() for c in OOV_CHECK},
    }


def row_split(frame, seed):
    train, rest = train_test_split(frame, test_size=0.2, random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, random_state=seed)
    return train, val, test


def user_split(frame, seed):
    def holdout(part, test_size):
        keep, hold = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
                          .split(part, groups=part["user"]))
        return part.iloc[keep], part.iloc[hold]
    train, rest = holdout(frame, 0.2)
    val, test = holdout(rest, 0.5)
    return train, val, test


records = [{"scheme": "temporal", "split_seed": "-",
            **fit_and_evaluate(*(df[df["split"] == s] for s in ["train", "val", "test"]))}]
for seed in SPLIT_SEEDS:
    records.append({"scheme": "random_row", "split_seed": str(seed), **fit_and_evaluate(*row_split(df, seed))})
    records.append({"scheme": "random_user", "split_seed": str(seed), **fit_and_evaluate(*user_split(df, seed))})
runs = pd.DataFrame(records)
runs.index = runs["scheme"] + " / " + runs["split_seed"]

scheme_stats = runs.groupby("scheme")[METRICS].agg(["mean", "std"])
scheme_stats.columns = [f"{m}_{stat}" for m, stat in scheme_stats.columns]
means = runs.groupby("scheme")[METRICS].mean()
gaps = pd.DataFrame({
    "temporal - random_row": means.loc["temporal"] - means.loc["random_row"],
    "temporal - random_user": means.loc["temporal"] - means.loc["random_user"],
    "random_user - random_row": means.loc["random_user"] - means.loc["random_row"],
}).T

# --- Step 3: write results ---
eval_cols = ["train_rows", "val_rows", "test_rows", "best_iteration", "logloss", "NE", "AUC", "ECE", "mean_pred", "actual_ctr"]
overlap_cols = ["test_rows_user_in_train"] + [f"test_unseen_{c}" for c in OOV_CHECK]

md = f"""# 01 — Hour of day and split comparison

## 1. CTR by hour of day
![CTR by hour of day](figures/01_ctr_by_hour_of_day.png)

- hour-of-day CTR range: {by_hod['ctr'].min():.4f} (hour {by_hod['ctr'].idxmin()}) to {by_hod['ctr'].max():.4f} (hour {by_hod['ctr'].idxmax()})
- std of CTR across the {len(by_hour)} hours: **{hourly_ctr_std:.4f}**; std expected from binomial noise alone: **{binomial_std:.4f}**

{df_to_md(by_hod[['n', 'ctr', 'ci_low', 'ci_high']])}

## 2. Random vs temporal split, identical LightGBM
- features: {', '.join(FEATURES)}, hour_of_day
- categorical codes from fit_category_maps(min_count=20) on each train part; rare/unseen -> 0
- params: num_leaves 63, learning_rate 0.05, min_data_in_leaf 100, early stopping 50 rounds on the val part, max 1000 rounds
- random_row: 80/10/10 rows; random_user: 80/10/10 by user (GroupShuffleSplit); split seeds {', '.join(map(str, SPLIT_SEEDS))}
- temporal: train Oct 21-27 / val Oct 28 / test Oct 29-30
- NE base rate = the test part's own CTR

### Per run
{df_to_md(runs[eval_cols])}

### Test rows whose user / value was not in the train part
{df_to_md(runs[overlap_cols])}

### Per scheme (mean and std over split seeds)
{df_to_md(scheme_stats)}

### Gaps between scheme means
{df_to_md(gaps)}
"""
write_result("01_temporal_split", md)
