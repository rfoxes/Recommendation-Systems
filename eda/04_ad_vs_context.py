"""Which columns describe the ad vs the context, how they depend on each other, and how many candidate ads are live. Run: .venv/bin/python eda/04_ad_vs_context.py"""
import numpy as np
import pandas as pd

from common import SITE_PLACEHOLDER, df_to_md, load_impressions, write_result

AD_COLUMNS = ["banner_pos", "C1", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]
REFERENCE = ["device_type", "device_conn_type", "device_model"]
MIN_CONTEXT_ROWS = 20

df = load_impressions()
is_app = df["site_id"] == SITE_PLACEHOLDER
df["publisher_id"] = df["app_id"].where(is_app, df["site_id"])
df["size"] = df["C15"].astype(str) + "x" + df["C16"].astype(str)
df["hour_idx"] = (df["ts"] - df["ts"].min()) // pd.Timedelta(hours=1)


def entropy(counts):
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def conditional_entropy_ratio(frame, keys, col):
    """H(col | keys) / H(col): 0 = fully fixed by the keys, 1 = keys say nothing about it."""
    joint = frame.groupby(keys + [col], observed=True).size()
    per_key = joint.groupby(level=list(range(len(keys))), observed=True)
    h_given = sum(n.sum() * entropy(n) for _, n in per_key) / len(frame)
    return h_given / entropy(frame[col].value_counts())


def determination(frame, x, y):
    """Row-weighted share of rows whose y equals the most common y for their x (1 = x fully determines y)."""
    counts = frame.groupby([x, y], observed=True).size()
    return float(counts.groupby(level=0, observed=True).max().sum() / len(frame))


# --- Step 1: variation within a context ---
CONTEXTS = {"publisher + hour": ["publisher_id", "hour_idx"],
            "publisher + device_model + hour": ["publisher_id", "device_model", "hour_idx"]}
variation_rows = []
for name, keys in CONTEXTS.items():
    size = df.groupby(keys, observed=True)["id"].transform("size")
    sub = df[size >= MIN_CONTEXT_ROWS]
    distinct = sub.groupby(keys, observed=True)[AD_COLUMNS + REFERENCE].nunique()
    weights = sub.groupby(keys, observed=True).size()
    for col in AD_COLUMNS + REFERENCE:
        if col in keys:
            continue
        variation_rows.append({
            "context": name, "column": col,
            "rows_in_contexts": len(sub),
            "mean_distinct_per_context": float(np.average(distinct[col], weights=weights)),
            "share_rows_in_constant_contexts": float(weights[distinct[col] == 1].sum() / weights.sum()),
            "entropy_ratio_given_context": conditional_entropy_ratio(sub, keys, col),
        })
variation = pd.DataFrame(variation_rows).set_index(["context", "column"])

# --- Step 2: functional dependencies ---
DETERMINERS = ["C14", "C17", "publisher_id", "banner_pos", "device_model"]
dependency = pd.DataFrame({x: {y: determination(df, x, y) for y in AD_COLUMNS + ["size", "publisher_id"] if y != x}
                           for x in DETERMINERS}).T
dependency.index.name = "x determines y ->"

# --- Step 3: banner size ---
sizes = df.groupby("size").agg(rows=("id", "size"), ctr=("click", "mean"), creatives=("C14", "nunique"))
sizes = sizes.sort_values("rows", ascending=False)
size_by = pd.DataFrame({"determination of size": {
    "C14 (creative)": determination(df, "C14", "size"),
    "publisher_id": determination(df, "publisher_id", "size"),
    "publisher_id + banner_pos": determination(df.assign(slot=df["publisher_id"] + "|" + df["banner_pos"].astype(str)), "slot", "size"),
    "device_type": determination(df, "device_type", "size"),
}})

# --- Step 4: candidate pool: distinct creatives live in the previous 24 hours ---
hours = sorted(df["hour_idx"].unique())
live = df.groupby("hour_idx")["C14"].unique()
live_by_size = df.groupby(["hour_idx", "size"])["C14"].unique()
pool_rows = []
for h in hours[24:]:
    window = set(np.concatenate(live.loc[h - 24:h - 1].to_numpy()))
    row = {"hour_idx": h, "creatives_live_24h": len(window)}
    for s in sizes.index[:3]:
        in_size = [live_by_size.get((hh, s), np.array([])) for hh in range(h - 24, h)]
        row[f"creatives_live_24h_size_{s}"] = len(set(np.concatenate(in_size))) if in_size else 0
    pool_rows.append(row)
pool = pd.DataFrame(pool_rows).drop(columns="hour_idx").describe(percentiles=[0.05, 0.5]).T[["min", "5%", "50%", "max"]]

hourly_volume = df.groupby(["hour_idx", "C14"]).size()
top_share = hourly_volume.groupby(level=0).apply(lambda s: s.nlargest(5).sum() / s.sum())

# --- Step 5: write results ---
md = f"""# 04 — Ad vs context columns, dependencies, candidate pool

- contexts with >= {MIN_CONTEXT_ROWS} impressions; `entropy_ratio_given_context` = H(column | context) / H(column): 0 = fixed by the context, 1 = independent of it
- `determination` = row-weighted share of rows whose y equals the most common y for their x (1 = x fully determines y)

## 1. Variation within a context
{df_to_md(variation)}

## 2. Functional dependencies (row = x, column = y)
{df_to_md(dependency)}

## 3. Banner size (C15 x C16)
{df_to_md(sizes.head(10))}

{df_to_md(size_by)}

## 4. Candidate pool: distinct creatives (C14) with an impression in the previous 24 hours, per hour
{df_to_md(pool)}

- share of an hour's impressions taken by its 5 most-shown creatives: median {top_share.median():.3f}, min {top_share.min():.3f}, max {top_share.max():.3f}
"""
write_result("04_ad_vs_context", md)
