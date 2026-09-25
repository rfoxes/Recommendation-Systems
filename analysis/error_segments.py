"""Error analysis of the V1 LightGBM on the validation day (Oct 28), by segment. Run: .venv/bin/python -m analysis.error_segments"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import MIN_COUNT
from src.models import LightGBMModel

from .common import row_logloss, to_md, tune_step, write

# --- Step 1: V1 model, predictions on Oct 28 ---
train, val, encoder, X_train, X_val = tune_step()
y_train, y_val = train["click"].to_numpy(), val["click"].to_numpy()
model = LightGBMModel(encoder.categorical).fit(X_train, y_train, X_val=X_val, y_val=y_val)

val = val.assign(p=model.predict(X_val))
val["loss"] = row_logloss(y_val, val["p"].to_numpy())
val["loss_constant"] = row_logloss(y_val, np.full(len(val), y_train.mean()))
val["improvement"] = val["loss_constant"] - val["loss"]


# --- Step 2: define segments ---
def seen_status(col):
    count = val[col].map(train[col].value_counts()).fillna(0)
    return pd.Series(np.select([count == 0, count < MIN_COUNT],
                               ["new (not in train)", f"rare (1-{MIN_COUNT - 1} train rows)"],
                               f"known (>= {MIN_COUNT} train rows)"), index=val.index)


SEGMENTS = {
    "C14 (creative)": seen_status("C14"),
    "C17": seen_status("C17"),
    "C21": seen_status("C21"),
    "publisher_id": seen_status("publisher_id"),
    "character_id": seen_status("character_id"),
    "device_model": seen_status("device_model"),
    "traffic": val["is_app"].map({1: "app", 0: "site"}),
    "banner_pos": val["banner_pos"].astype(str),
    "hour of day": pd.cut(val["hour_of_day"], [-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"]),
    "safety_tier": val["safety_tier"],
    "publisher size decile": X_val["publisher_id_freq_decile"].map(lambda d: "new" if d == 0 else f"decile {d}"),
}


# --- Step 3: metrics per segment ---
def segment_table(labels):
    rows = []
    for segment, part in val.groupby(labels, observed=True):
        rows.append({
            "segment": segment,
            "rows": len(part),
            "row_share": len(part) / len(val),
            "loss_share": part["loss"].sum() / val["loss"].sum(),
            "improvement_share": part["improvement"].sum() / val["improvement"].sum(),
            "actual_ctr": part["click"].mean(),
            "mean_pred": part["p"].mean(),
            "NE": part["loss"].mean() / part["loss_constant"].mean(),
            "AUC": roc_auc_score(part["click"], part["p"]) if part["click"].nunique() == 2 else np.nan,
        })
    return pd.DataFrame(rows).set_index("segment")


tables = {name: segment_table(labels) for name, labels in SEGMENTS.items()}

# --- Step 4: ranking within each publisher ---
by_publisher = segment_table(val["publisher_id"]).sort_values("rows", ascending=False)
by_publisher["traffic"] = val.groupby("publisher_id")["is_app"].first().map({1: "app", 0: "site"})
rankable = by_publisher.dropna(subset=["AUC"])
rankable = rankable[rankable["rows"] >= 100]
within_publisher_auc = (rankable["AUC"] * rankable["rows"]).sum() / rankable["rows"].sum()

train_code0 = (X_train["C14"] == 0).mean()

# --- Step 5: write results ---
overall_auc = roc_auc_score(y_val, val["p"])
md = f"""# Error analysis by segment — V1 LightGBM, validation day Oct 28

- model: V1 LightGBM trained Oct 21-27 ({len(train):,} rows), {model.num_rounds} trees; evaluated on Oct 28 ({len(val):,} rows)
- overall: AUC {overall_auc:.4f}, NE {val['loss'].mean() / val['loss_constant'].mean():.4f}
- `loss_share`: segment's share of total logloss; `improvement_share`: segment's share of the model's total logloss reduction vs the constant model
- `NE` per segment: model logloss / constant (train CTR) logloss on the segment's rows
- C14 code 0 (rare/unseen): {train_code0:.2%} of training rows vs {(X_val['C14'] == 0).mean():.2%} of Oct 28 rows

""" + "\n\n".join(f"## {name}\n{to_md(table)}" for name, table in tables.items()) + f"""

## Ranking within each publisher
- publishers with >= 100 rows and both classes on Oct 28: {len(rankable):,}, covering {rankable['rows'].sum() / len(val):.2%} of rows
- row-weighted mean AUC within a publisher: **{within_publisher_auc:.4f}** (overall AUC across all rows: {overall_auc:.4f})

### 15 largest publishers
{to_md(by_publisher.head(15)[['traffic', 'rows', 'row_share', 'actual_ctr', 'mean_pred', 'NE', 'AUC']])}
"""
write("error_segments", md)
