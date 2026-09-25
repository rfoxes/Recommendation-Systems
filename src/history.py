"""Features from earlier hours only: user history, fatigue, recent click rates and traffic volume."""
import numpy as np
import pandas as pd

from .config import HISTORY_SMOOTHING, NULL_DEVICE_ID, RECENT_WINDOW_HOURS

RECENT_CTR_KEYS = {"publisher": "publisher_id", "creative": "C14", "C17": "C17", "character": "character_id"}


def _hourly_totals(df, keys):
    """Impressions (n) and clicks (c) per key per hour, sorted by key then hour, with running totals."""
    g = (df.groupby(keys + ["hour_idx"], observed=True, sort=False)["click"].agg(n="size", c="sum")
         .reset_index().sort_values(keys + ["hour_idx"], kind="stable").reset_index(drop=True))
    by_key = g.groupby(keys, observed=True, sort=False)
    g["cum_n"], g["cum_c"] = by_key["n"].cumsum(), by_key["c"].cumsum()
    g["last_hour"] = by_key["hour_idx"].shift(1)
    return g


def _running_totals_at(g, keys, hours, lookups=None):
    """Running totals of each key as of the given hours (inclusive); 0 if the key had no earlier rows.
    Keys are taken from `lookups` (default: the rows of g)."""
    lookups = g if lookups is None else lookups
    target = lookups[keys].assign(asof=hours, row=np.arange(len(lookups))).sort_values("asof", kind="stable")
    ref = g[keys + ["hour_idx", "cum_n", "cum_c"]].rename(columns={"hour_idx": "asof"}).sort_values("asof", kind="stable")
    merged = pd.merge_asof(target, ref, on="asof", by=keys, direction="backward").sort_values("row")
    return merged["cum_n"].fillna(0).to_numpy(), merged["cum_c"].fillna(0).to_numpy()


def key_history(df, keys, window=None):
    """
    For each row: impressions and clicks of the row's key in hours strictly before the row's hour
    (all earlier hours, or only the last `window` hours), plus the last earlier hour the key appeared.
    """
    g = _hourly_totals(df, keys)
    n, c = (g["cum_n"] - g["n"]).to_numpy(), (g["cum_c"] - g["c"]).to_numpy()
    if window is not None:
        old_n, old_c = _running_totals_at(g, keys, g["hour_idx"].to_numpy() - window - 1)
        n, c = n - old_n, c - old_c
    g["hist_n"], g["hist_c"] = n, c
    out = df[keys + ["hour_idx"]].merge(g[keys + ["hour_idx", "hist_n", "hist_c", "last_hour"]],
                                        on=keys + ["hour_idx"], how="left", validate="many_to_one")
    return out


def _smoothed_ctr(clicks, impressions, prior):
    return (clicks + HISTORY_SMOOTHING * prior) / (impressions + HISTORY_SMOOTHING)


def _hourly_global(df):
    hours = df.groupby("hour_idx")["click"].agg(n="size", c="sum")
    return hours.reindex(range(hours.index.min(), hours.index.max() + 1), fill_value=0)


def recent_prior(df):
    """Global click rate over the RECENT_WINDOW_HOURS before each hour (smoothing prior), indexed by hour_idx."""
    recent = _hourly_global(df).rolling(RECENT_WINDOW_HOURS, min_periods=1).sum().shift(1)
    return recent["c"] / recent["n"]


def window_totals(df, keys, lookups, window=RECENT_WINDOW_HOURS):
    """
    Impressions and clicks of each lookup's key in the `window` hours strictly before its hour_idx.
    `lookups` need not be rows of df (e.g. candidate ads that were never shown in that hour).
    """
    g = _hourly_totals(df, keys)
    hours = lookups["hour_idx"].to_numpy()
    n_end, c_end = _running_totals_at(g, keys, hours - 1, lookups)
    n_start, c_start = _running_totals_at(g, keys, hours - window - 1, lookups)
    return n_end - n_start, c_end - c_start


def recent_click_rate(df, keys, lookups):
    """Smoothed click rate of each lookup's key over the RECENT_WINDOW_HOURS before its hour (same as the *_ctr_24h features)."""
    n, c = window_totals(df, keys, lookups)
    return _smoothed_ctr(c, n, recent_prior(df).reindex(lookups["hour_idx"]).to_numpy())


def add_history_features(df):
    df["hour_idx"] = (df["ts"] - pd.Timestamp(0)) // pd.Timedelta(hours=1)
    has_device = df["device_id"] != NULL_DEVICE_ID
    df["user"] = ("d:" + df["device_id"]).where(has_device, "ip:" + df["device_ip"] + "_" + df["device_model"])

    # Global click rate before each hour: smoothing prior for the per-key rates.
    hours = _hourly_global(df)
    prior_all = (hours["c"].cumsum().shift(1) / hours["n"].cumsum().shift(1)).reindex(df["hour_idx"]).to_numpy()
    prior_recent = recent_prior(df).reindex(df["hour_idx"]).to_numpy()

    # --- user history ---
    user = key_history(df, ["user"])
    df["user_prior_impressions"] = user["hist_n"].to_numpy()
    df["user_prior_clicks"] = user["hist_c"].to_numpy()
    df["user_prior_ctr"] = _smoothed_ctr(user["hist_c"].to_numpy(), user["hist_n"].to_numpy(), prior_all)
    df["user_hours_since_last"] = (df["hour_idx"] - user["last_hour"].to_numpy()).astype("float64")
    df["user_seen_before"] = (df["user_prior_impressions"] > 0).astype("int8")

    # --- fatigue ---
    df["user_impressions_24h"] = key_history(df, ["user"], window=RECENT_WINDOW_HOURS)["hist_n"].to_numpy()
    df["user_character_prior_impressions"] = key_history(df, ["user", "character_id"])["hist_n"].to_numpy()

    # --- recent click rates ---
    for name, col in RECENT_CTR_KEYS.items():
        recent_key = key_history(df, [col], window=RECENT_WINDOW_HOURS)
        df[f"{name}_ctr_24h"] = _smoothed_ctr(recent_key["hist_c"].to_numpy(), recent_key["hist_n"].to_numpy(), prior_recent)

    # --- traffic volume ---
    df["publisher_impressions_prev_hour"] = key_history(df, ["publisher_id"], window=1)["hist_n"].to_numpy()
    df["total_impressions_prev_hour"] = hours["n"].shift(1).reindex(df["hour_idx"]).to_numpy()
    return df
