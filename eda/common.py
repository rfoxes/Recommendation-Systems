"""Shared paths, split dates, user proxy, encoding and metrics for all EDA scripts."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RAW_IMPRESSIONS = ROOT / "impressions.csv"
RAW_CHARACTERS = ROOT / "characters.csv"
PROCESSED = ROOT / "data" / "processed"
IMPRESSIONS_PARQUET = PROCESSED / "impressions.parquet"
CHARACTERS_PARQUET = PROCESSED / "characters.parquet"
RESULTS = ROOT / "eda" / "results"
FIGURES = RESULTS / "figures"

# Avazu placeholders: no device id / app traffic (site_id) / site traffic (app_id)
NULL_DEVICE_ID = "a99f214a"
SITE_PLACEHOLDER = "85f751fd"
APP_PLACEHOLDER = "ecad2386"

# train: Oct 21-27 | val: Oct 28 | test: Oct 29-30
VAL_START = pd.Timestamp("2014-10-28")
TEST_START = pd.Timestamp("2014-10-29")

RANDOM_SEED = 42


def load_impressions():
    return pd.read_parquet(IMPRESSIONS_PARQUET)


def load_characters():
    return pd.read_parquet(CHARACTERS_PARQUET)


def load_joined():
    """Impressions joined with character attributes (+ genre parsed from the name)."""
    imp = load_impressions()
    ch = load_characters()
    ch["genre"] = ch["character_name"].str.split("_").str[0]
    return imp.merge(ch, on="character_id", how="left", validate="many_to_one")


def add_user_proxy(df):
    """device_id when present, else device_ip + device_model."""
    has_device = df["device_id"] != NULL_DEVICE_ID
    df["user"] = np.where(has_device, "d_" + df["device_id"], "im_" + df["device_ip"] + "_" + df["device_model"])
    return df


def split_labels(df):
    return pd.Series(
        np.select([df["ts"] < VAL_START, df["ts"] < TEST_START], ["train", "val"], "test"),
        index=df.index,
    )


# ---------- categorical encoding ----------

def fit_category_maps(train_df, cols, min_count=20):
    """Values seen >= min_count times in train -> codes 1..K; everything else -> 0."""
    maps = {}
    for c in cols:
        vc = train_df[c].value_counts()
        keep = vc[vc >= min_count].index
        maps[c] = pd.Series(np.arange(1, len(keep) + 1), index=keep)
    return maps


def apply_category_maps(df, maps):
    out = pd.DataFrame(index=df.index)
    for c, m in maps.items():
        out[c] = df[c].map(m).fillna(0).astype("int32")
    return out


# ---------- metrics ----------

def binary_entropy(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def evaluate(y, p, base_rate):
    """logloss, NE (logloss / entropy of base_rate), AUC, ECE (20 bins), mean prediction vs actual CTR."""
    y = np.asarray(y)
    p = np.clip(np.asarray(p), 1e-7, 1 - 1e-7)
    ll = log_loss(y, p)
    return {
        "logloss": ll,
        "NE": ll / binary_entropy(base_rate),
        "AUC": roc_auc_score(y, p),
        "ECE": expected_calibration_error(y, p),
        "mean_pred": p.mean(),
        "actual_ctr": y.mean(),
    }


def expected_calibration_error(y, p, n_bins=20):
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    df = pd.DataFrame({"y": y, "p": p, "b": bins})
    g = df.groupby("b").agg(n=("y", "size"), y=("y", "mean"), p=("p", "mean"))
    return float((g["n"] * (g["y"] - g["p"]).abs()).sum() / len(df))


def wilson_ci(clicks, n, z=1.96):
    """Wilson 95% CI for clicks / n."""
    clicks, n = np.asarray(clicks, dtype=float), np.asarray(n, dtype=float)
    p = clicks / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def ctr_table(df, col, min_n=0):
    """CTR per value of `col` with count and Wilson CI."""
    g = df.groupby(col, observed=True)["click"].agg(n="size", clicks="sum")
    g = g[g["n"] >= min_n]
    g["ctr"] = g["clicks"] / g["n"]
    g["ci_low"], g["ci_high"] = wilson_ci(g["clicks"], g["n"])
    return g.sort_values("n", ascending=False)


# ---------- output ----------

def write_result(name, markdown):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.md"
    path.write_text(markdown)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def df_to_md(df, digits=4):
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: f"{v:.{digits}f}")
        elif pd.api.types.is_integer_dtype(out[c]):
            out[c] = out[c].map(lambda v: f"{v:,}")
    return out.to_markdown()
