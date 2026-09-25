"""Evaluation metrics for click probabilities."""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from .config import EVAL_START, HELD_OUT_START

EPS = 1e-7


def evaluate(y, p, p_constant, publisher=None):
    """
    logloss: quality of the probabilities (lower is better).
    NE: logloss / logloss of the constant model on the same rows (< 1 = better than predicting the training CTR).
    AUC: ranking of clicks above non-clicks. AUC_within_publisher: the same, but only comparing rows of one publisher.
    ECE: mean |predicted - actual| over 20 probability bins.
    """
    y, p, p_constant = np.asarray(y), np.clip(p, EPS, 1 - EPS), np.clip(p_constant, EPS, 1 - EPS)
    logloss = log_loss(y, p, labels=[0, 1])
    out = {
        "rows": len(y),
        "logloss": logloss,
        "NE": logloss / log_loss(y, p_constant, labels=[0, 1]),
        "AUC": roc_auc_score(y, p),
    }
    if publisher is not None:
        out["AUC_within_publisher"] = grouped_auc(y, p, publisher)
    out.update({"ECE": expected_calibration_error(y, p), "mean_pred": float(p.mean()), "actual_ctr": float(y.mean())})
    return out


def grouped_auc(y, p, groups, min_rows=100):
    """Row-weighted mean AUC within groups that have >= min_rows rows and both classes."""
    aucs, weights = [], []
    for _, part in pd.DataFrame({"y": y, "p": p, "g": np.asarray(groups)}).groupby("g", observed=True):
        if len(part) >= min_rows and part["y"].nunique() == 2:
            aucs.append(roc_auc_score(part["y"], part["p"]))
            weights.append(len(part))
    return float(np.average(aucs, weights=weights)) if aucs else float("nan")


def expected_calibration_error(y, p, n_bins=20):
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    g = pd.DataFrame({"y": y, "p": p, "bin": bins}).groupby("bin").agg(n=("y", "size"), y=("y", "mean"), p=("p", "mean"))
    return float((g["n"] * (g["y"] - g["p"]).abs()).sum() / len(y))


def calibration_table(y, p, n_bins=10):
    """Rows split into equal-size bins by predicted probability: mean prediction vs actual CTR per bin."""
    bins = pd.qcut(p, n_bins, labels=False, duplicates="drop")
    return (pd.DataFrame({"y": y, "p": p, "bin": bins}).groupby("bin")
            .agg(rows=("y", "size"), mean_pred=("p", "mean"), actual_ctr=("y", "mean")))


def wilson_ci(clicks, n, z=1.96):
    """Wilson 95% interval for a click rate of clicks / n."""
    clicks, n = np.asarray(clicks, dtype=float), np.asarray(n, dtype=float)
    p = clicks / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def evaluation_periods(ts):
    """Reported periods as {label: row mask}: development blocks, held-out blocks, and the last 24 hours of data."""
    ts = pd.Series(np.asarray(ts))
    end = ts.max() + pd.Timedelta(hours=1)
    last = end - pd.Timedelta(hours=24)
    return {
        f"development ({EVAL_START:%b %d}-{HELD_OUT_START - pd.Timedelta(days=1):%b %d})": (ts < HELD_OUT_START).to_numpy(),
        f"held-out ({HELD_OUT_START:%b %d}-{end - pd.Timedelta(hours=1):%b %d})": (ts >= HELD_OUT_START).to_numpy(),
        f"last 24 hours ({last:%b %d %H:%M}-{end:%b %d %H:%M})": (ts >= last).to_numpy(),
    }
