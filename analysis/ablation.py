"""Leave-one-group-out ablation of the V1 LightGBM; train Oct 21-27, evaluate Oct 28. Run: .venv/bin/python -m analysis.ablation"""
import time

import numpy as np
import pandas as pd

from src.config import LGBM_PARAMS, SEED
from src.metrics import evaluate
from src.models import LightGBMModel

from .common import to_md, tune_step, write

GROUPS = {
    "publisher": ["publisher_id", "publisher_domain", "publisher_category", "is_app"],
    "ad (banner_pos, C1, C14-C21)": ["banner_pos", "C1", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"],
    "device": ["device_type", "device_conn_type", "device_model", "has_device_id"],
    "character": ["character_id", "safety_tier", "creator_type", "genre", "num_interactions", "character_age_days"],
    "conversation": ["conversation_turn", "session_msg_count"],
    "hour of day": ["hour_of_day"],
    "frequency deciles": ["publisher_id_freq_decile", "publisher_domain_freq_decile", "device_model_freq_decile",
                          "C14_freq_decile", "C17_freq_decile", "character_id_freq_decile"],
    "user history": ["user_prior_impressions", "user_prior_clicks", "user_prior_ctr", "user_hours_since_last",
                     "user_seen_before"],
    "fatigue": ["user_impressions_24h", "user_character_prior_impressions"],
    "recent click rates": ["publisher_ctr_24h", "creative_ctr_24h", "C17_ctr_24h", "character_ctr_24h"],
    "traffic volume": ["publisher_impressions_prev_hour", "total_impressions_prev_hour"],
}
NOISE_SEEDS = [SEED, SEED + 1, SEED + 2]

train, val, encoder, X_train, X_val = tune_step()
y_train, y_val = train["click"].to_numpy(), val["click"].to_numpy()
p_constant = np.full(len(y_val), y_train.mean())
publisher = val["publisher_id"].to_numpy()
assert sorted(sum(GROUPS.values(), [])) == sorted(encoder.columns)


def fit_and_score(drop=(), seed=SEED):
    keep = [c for c in encoder.columns if c not in drop]
    model = LightGBMModel([c for c in encoder.categorical if c in keep], params={**LGBM_PARAMS, "seed": seed})
    start = time.perf_counter()
    model.fit(X_train[keep], y_train, X_val=X_val[keep], y_val=y_val)
    scores = evaluate(y_val, model.predict(X_val[keep]), p_constant, publisher)
    return {"features": len(keep), "trees": model.num_rounds, **scores, "fit_seconds": time.perf_counter() - start}


# --- Step 1: full model, 3 seeds (noise level) ---
full = pd.DataFrame([fit_and_score(seed=s) for s in NOISE_SEEDS], index=[f"seed {s}" for s in NOISE_SEEDS])
reference = full.iloc[0]
noise = full[["logloss", "NE", "AUC", "AUC_within_publisher"]].agg(["std", lambda s: s.max() - s.min()])
noise.index = ["std across seeds", "max - min across seeds"]

# --- Step 2: drop one group at a time ---
rows = {name: fit_and_score(drop=cols) for name, cols in GROUPS.items()}
ablation = pd.DataFrame(rows).T
for metric in ["logloss", "NE", "AUC", "AUC_within_publisher"]:
    ablation[f"delta_{metric}"] = ablation[metric] - reference[metric]
ablation = ablation.sort_values("delta_logloss", ascending=False)

# --- Step 3: write results ---
cols = ["features", "trees", "logloss", "delta_logloss", "NE", "delta_NE", "AUC", "delta_AUC",
        "AUC_within_publisher", "delta_AUC_within_publisher", "fit_seconds"]
md = f"""# Ablation — V1 LightGBM (16 leaves), train Oct 21-27 ({len(train):,} rows), evaluate Oct 28 ({len(val):,} rows)

- each row removes one feature group and retrains (trees chosen by early stopping on Oct 28)
- `delta_*` = ablated model minus full model (seed {SEED}); positive delta_logloss / negative delta_AUC = the group helps
- noise level: full model retrained with seeds {', '.join(map(str, NOISE_SEEDS))}

## Full model, noise level
{to_md(full[['features', 'trees', 'logloss', 'NE', 'AUC', 'AUC_within_publisher']])}

{to_md(noise)}

## Leave one group out (sorted: most useful group first)
{to_md(ablation[cols])}

## Groups
""" + "\n".join(f"- **{name}**: {', '.join(cols)}" for name, cols in GROUPS.items()) + "\n"
write("ablation", md)
