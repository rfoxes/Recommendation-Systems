"""Paths, dates, feature lists and model settings for the CTR pipeline."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_IMPRESSIONS = ROOT / "impressions.csv"
RAW_CHARACTERS = ROOT / "characters.csv"
CACHE_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

NULL_DEVICE_ID = "a99f214a"
SITE_PLACEHOLDER = "85f751fd"  # site_id on app traffic
APP_PLACEHOLDER = "ecad2386"   # app_id on site traffic

# Walk-forward: retrain every RETRAIN_HOURS on all earlier rows, score the next block. Oct 21-23 are warm-up (training only).
# Design choices were made on blocks before HELD_OUT_START; later blocks are held out. Tree count and FM passes: tuned on Oct 28.
EVAL_START = pd.Timestamp("2014-10-24")
HELD_OUT_START = pd.Timestamp("2014-10-29")
EVAL_DAYS = list(pd.date_range(EVAL_START, "2014-10-30", freq="D"))
TEST_DAYS = [d for d in EVAL_DAYS if d >= HELD_OUT_START]
TUNE_VAL_DAY = pd.Timestamp("2014-10-28")

CATEGORICAL = [
    "publisher_id", "publisher_domain", "publisher_category",
    "banner_pos", "device_type", "device_conn_type", "device_model",
    "C1", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
    "character_id", "safety_tier", "creator_type", "genre",
]
FREQ_DECILE = ["publisher_id", "publisher_domain", "device_model", "C14", "C17", "character_id"]
FLAGS = ["is_app", "has_device_id", "user_seen_before"]
HISTORY = [
    "user_prior_impressions", "user_prior_clicks", "user_prior_ctr", "user_hours_since_last",
    "user_impressions_24h", "user_character_prior_impressions",
    "publisher_ctr_24h", "creative_ctr_24h", "C17_ctr_24h", "character_ctr_24h",
    "publisher_impressions_prev_hour", "total_impressions_prev_hour",
]
NUMERIC = ["hour_of_day", "num_interactions", "character_age_days", "conversation_turn", "session_msg_count"] + HISTORY
MIN_COUNT = 20

# History features use only hours strictly before the row's hour.
RECENT_WINDOW_HOURS = 24
HISTORY_SMOOTHING = 20

SEED = 42

LGBM_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 16,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "seed": SEED,
    "deterministic": True,
    "num_threads": 8,
    "verbose": -1,
}
LGBM_MAX_ROUNDS = 2000
LGBM_EARLY_STOPPING_ROUNDS = 50

LR_C_GRID = [0.01, 0.1, 1.0]
LR_ONE_HOT_NUMERIC = ["hour_of_day"]
LR_LOG_SCALED = [
    "num_interactions", "character_age_days", "conversation_turn", "session_msg_count",
    "user_prior_impressions", "user_prior_clicks", "user_hours_since_last", "user_impressions_24h",
    "user_character_prior_impressions", "publisher_impressions_prev_hour", "total_impressions_prev_hour",
]
LR_RATES = ["user_prior_ctr", "publisher_ctr_24h", "creative_ctr_24h", "C17_ctr_24h", "character_ctr_24h"]
LR_NEVER_SEEN_HOURS = 240  # stands in for a missing user_hours_since_last

# Factorization machine (tuned on Oct 24-25, analysis/fm_equal_protocol.py); blended 50/50 in log-odds with LightGBM
FM_MIN_COUNT = 2
FM_NUMERIC_BINS = 50
FM_DENSE = LR_RATES
FM_PARAMS = {"dim": 16, "l2": 0.01, "lr": 0.001, "batch_size": 4096}
FM_MAX_EPOCHS = 10
FM_THREADS = 8

# V1 is retrained at these hours of every day on everything before them
RETRAIN_HOURS = [0, 6, 12, 18]

# Goal 2: candidate ranking
N_CANDIDATES = 5
CANDIDATE_WINDOW_HOURS = 24  # a candidate creative ran on the same publisher and slot size within this window
AD_ATTRIBUTES = ["C15", "C16", "C17", "C18", "C19", "C21"]  # determined by the creative (C14); everything else is the opportunity
SAFETY_TIERS = ["sfw", "suggestive", "mature"]
# Simulated brand-safety setting per advertiser (C17), since the data has no such labels: share of advertisers accepting up to each tier
ADVERTISER_MAX_TIER_SHARES = {"sfw": 0.60, "suggestive": 0.25, "mature": 0.15}
FATIGUE_HARD_CAP = 3  # same creative per user per 24h
EXPLORATION_SHARE = 0.10  # max share of an hour's opportunities where a toss-up goes to the least-known candidate
RETRIEVAL_POOL = 10  # retrieval returns up to this many eligible creatives in order; ranking takes the first N whose advertiser has budget left

# Goals 3-4: adaptation and exploration layer on top of the Goal 2 ranking
GRADUATION_IMPRESSIONS = 1000  # a creative is cold until it has this many impressions in the last GRADUATION_WINDOW_HOURS
GRADUATION_WINDOW_HOURS = 72
COLD_EXPLORATION_SHARE = 0.05  # max share of an hour's opportunities where a cold creative is shown instead of the leader
DOMINANT_COHORTS = 5  # cohorts (genre x safety tier) with the most impressions in the last 24h
