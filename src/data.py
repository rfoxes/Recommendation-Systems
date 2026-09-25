"""Load impressions + characters and derive the base columns used by the features."""
import numpy as np
import pandas as pd

from .config import (APP_PLACEHOLDER, CACHE_DIR, NULL_DEVICE_ID, RAW_CHARACTERS, RAW_IMPRESSIONS,
                     SITE_PLACEHOLDER)
from .history import add_history_features

CHARACTER_COLUMNS = ["character_id", "safety_tier", "creator_type", "genre", "num_interactions", "created_at"]


def load_raw():
    """Raw tables, cached as parquet after the first CSV read."""
    imp_path, ch_path = CACHE_DIR / "impressions.parquet", CACHE_DIR / "characters.parquet"
    if imp_path.exists() and ch_path.exists():
        return pd.read_parquet(imp_path), pd.read_parquet(ch_path)
    imp = pd.read_csv(RAW_IMPRESSIONS, dtype={"id": str, "hour": str})
    imp["ts"] = pd.to_datetime(imp["hour"], format="%y%m%d%H")
    ch = pd.read_csv(RAW_CHARACTERS, parse_dates=["created_at"])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    imp.to_parquet(imp_path, index=False)
    ch.to_parquet(ch_path, index=False)
    return imp, ch


def build_frame(label_permutation_seed=None):
    """
    One row per impression with character attributes and derived columns.
    label_permutation_seed: shuffle `click` across all rows before any feature is derived (sanity control only).
    """
    imp, ch = load_raw()
    if label_permutation_seed is not None:
        imp["click"] = np.random.default_rng(label_permutation_seed).permutation(imp["click"].to_numpy())
    ch = ch.assign(genre=ch["character_name"].str.split("_").str[0])
    df = imp.merge(ch[CHARACTER_COLUMNS], on="character_id", how="left", validate="many_to_one")

    # Every row is either app traffic (site_* are placeholders) or site traffic (app_* are placeholders).
    is_app = df["site_id"] == SITE_PLACEHOLDER
    assert ((df["app_id"] == APP_PLACEHOLDER) == ~is_app).all()
    df["is_app"] = is_app.astype("int8")
    for part in ["id", "domain", "category"]:
        df[f"publisher_{part}"] = df[f"app_{part}"].where(is_app, df[f"site_{part}"])

    df["has_device_id"] = (df["device_id"] != NULL_DEVICE_ID).astype("int8")
    df["hour_of_day"] = df["ts"].dt.hour
    df["day"] = df["ts"].dt.normalize()
    df["character_age_days"] = (df["day"] - df["created_at"]).dt.days
    return add_history_features(df)
