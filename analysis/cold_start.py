"""
Goal 3: simulated brand-new characters (what V1 knows before any clicks), which attributes carry that signal, and
graduation curves for characters, creatives and users. Judged on Oct 26, 27, 28. Run: .venv/bin/python -m analysis.cold_start
"""
import json
import re

import numpy as np
import pandas as pd

from src.config import CATEGORICAL, RESULTS_DIR, SEED
from src.data import build_frame, load_raw
from src.features import FeatureEncoder
from src.history import add_history_features
from src.metrics import evaluate
from src.models import LightGBMModel, V1Model

from .common import to_md, write

JUDGE_DAYS = [pd.Timestamp("2014-10-26"), pd.Timestamp("2014-10-27"), pd.Timestamp("2014-10-28")]
HOLDOUT_SHARE = 0.10
ADD_BACK = [0, 10, 50, 200, "all"]
CHARACTER_ATTRIBUTES = ["genre", "safety_tier", "creator_type"]
DESCRIPTION = ["archetype", "trait_1", "trait_2"]
METRICS = ["rows", "logloss", "NE", "AUC", "AUC_within_publisher", "ECE", "mean_pred", "actual_ctr"]


def description_features(characters):
    """Description = genre-specific template + two personality traits from a shared list: parse them apart."""
    words = characters["character_description"].str.lower().str.findall(r"[a-z\-]+")
    genre = characters["character_name"].str.split("_").str[0]
    genres_per_word = pd.DataFrame({"word": words, "genre": genre}).explode("word").groupby("word")["genre"].nunique()
    counts = words.explode().value_counts()
    traits = set(genres_per_word[(genres_per_word >= 8) & (counts.reindex(genres_per_word.index) >= 100)].index) - {"a", "the", "and"}
    found = words.map(lambda ws: [w for w in ws if w in traits] + ["none", "none"])
    pattern = r"\b(" + "|".join(re.escape(t) for t in sorted(traits, key=len, reverse=True)) + r")\b"
    return pd.DataFrame({
        "character_id": characters["character_id"],
        "archetype": characters["character_description"].str.lower().str.replace(pattern, "_", regex=True),
        "trait_1": found.str[0], "trait_2": found.str[1],
    })


def simulate(df, day, held_out, keep_last):
    """
    The data as it would look if each held-out character had only its `keep_last` most recent impressions before `day`
    (0 = brand new at the start of `day`; "all" = real history). History features are recomputed from what remains;
    brand-new and add-back characters get age 0 and popularity 0 on `day`.
    """
    if keep_last == "all":
        return df
    before = df[(df["ts"] < day) & df["character_id"].isin(held_out)]
    recency = before.sort_values("ts", ascending=False, kind="stable").groupby("character_id").cumcount()
    sim = add_history_features(df.drop(index=recency[recency >= keep_last].index).copy())
    new_rows = (sim["day"] == day) & sim["character_id"].isin(held_out)
    sim.loc[new_rows, ["character_age_days", "num_interactions"]] = 0
    return sim


def segment_metrics(rows, p, base_rate):
    y = rows["click"].to_numpy()
    if len(y) == 0 or y.min() == y.max():
        return {"rows": len(y)}
    return evaluate(y, p, np.full(len(y), base_rate), rows["publisher_id"].to_numpy())


def main():
    tuning = json.loads((RESULTS_DIR / "v1" / "metrics.json").read_text())["tuning"]
    df = build_frame().merge(description_features(load_raw()[1]), on="character_id", how="left", validate="many_to_one")
    characters = np.sort(df["character_id"].unique())
    held_out = set(np.random.default_rng(SEED).choice(characters, int(HOLDOUT_SHARE * len(characters)), replace=False))

    graduation, natural, ablation = [], [], []
    for day in JUDGE_DAYS:
        for keep in ADD_BACK:
            # --- Step 1: brand-new characters and the add-back graduation curve (V1) ---
            sim = simulate(df, day, held_out, keep)
            train, test = sim[sim["ts"] < day], sim[sim["day"] == day]
            model = V1Model(tuning["lightgbm_rounds"], tuning["factorization_machine_epochs"]).fit(train)
            p = model.predict_components(test)[2]
            held = test["character_id"].isin(held_out).to_numpy()
            base = train["click"].mean()
            for segment, mask in [("held-out characters", held), ("other characters", ~held)]:
                graduation.append({"day": str(day.date()), "impressions_before": str(keep), "segment": segment,
                                   **segment_metrics(test[mask], p[mask], base)})

            if keep == "all":
                # --- Step 2: natural curves for creatives and users (normal V1, all history) ---
                creative_seen = test["C14"].map(train["C14"].value_counts()).fillna(0)
                buckets = {
                    "creative: training impressions": pd.cut(creative_seen, [-1, 0, 19, 99, 999, 1e9],
                                                             labels=["0", "1-19", "20-99", "100-999", "1000+"]),
                    "user: prior impressions": pd.cut(test["user_prior_impressions"], [-1, 0, 1, 4, 19, 1e9],
                                                      labels=["0", "1", "2-4", "5-19", "20+"]),
                }
                for entity, bucket in buckets.items():
                    for level in bucket.cat.categories:
                        mask = (bucket == level).to_numpy()
                        natural.append({"day": str(day.date()), "entity": entity, "bucket": level,
                                        **segment_metrics(test[mask], p[mask], base)})

            if keep == 0:
                # --- Step 3: which attributes carry signal for brand-new characters (LightGBM component) ---
                variants = {"V1 attributes": CATEGORICAL}
                variants.update({f"without {a}": [c for c in CATEGORICAL if c != a] for a in CHARACTER_ATTRIBUTES})
                variants["without genre, tier, creator"] = [c for c in CATEGORICAL if c not in CHARACTER_ATTRIBUTES]
                variants["with description (archetype, traits)"] = CATEGORICAL + DESCRIPTION
                for name, categorical in variants.items():
                    encoder = FeatureEncoder(categorical=categorical).fit(train)
                    lgbm = LightGBMModel(encoder.categorical, num_rounds=tuning["lightgbm_rounds"]).fit(
                        encoder.transform(train), train["click"].to_numpy())
                    p_cold = lgbm.predict(encoder.transform(test[held]))
                    ablation.append({"day": str(day.date()), "variant": name,
                                     **segment_metrics(test[held], p_cold, base)})
            print(f"done {day.date()} keep={keep}", flush=True)

    # --- Step 4: summaries ---
    graduation = pd.DataFrame(graduation)
    curve = graduation.groupby(["segment", "impressions_before"], sort=False)[METRICS].mean()
    reference = graduation[(graduation["impressions_before"] == "all") & (graduation["segment"] == "held-out characters")].set_index("day")
    held_rows = graduation[graduation["segment"] == "held-out characters"].copy()
    for metric in ["logloss", "AUC", "AUC_within_publisher"]:
        held_rows[f"{metric}_minus_warm"] = held_rows[metric] - held_rows["day"].map(reference[metric])
    curve_gap = held_rows.groupby("impressions_before", sort=False)[
        ["logloss_minus_warm", "AUC_minus_warm", "AUC_within_publisher_minus_warm"]].agg(["mean", "min", "max"])
    curve_gap.columns = [f"{m} ({s} over days)" for m, s in curve_gap.columns]

    ablation = pd.DataFrame(ablation)
    ref = ablation[ablation["variant"] == "V1 attributes"].set_index("day")
    for metric in ["logloss", "AUC"]:
        ablation[f"{metric}_delta"] = ablation[metric] - ablation["day"].map(ref[metric])
    ablation_summary = ablation.groupby("variant", sort=False)[["logloss", "logloss_delta", "AUC", "AUC_delta", "AUC_within_publisher"]].mean()
    ablation_summary["logloss worse on all 3 days"] = ablation.groupby("variant", sort=False)["logloss_delta"].apply(lambda d: bool((d > 0).all()))
    ablation_summary["logloss better on all 3 days"] = ablation.groupby("variant", sort=False)["logloss_delta"].apply(lambda d: bool((d < 0).all()))

    natural = pd.DataFrame(natural)
    natural_summary = natural.groupby(["entity", "bucket"], sort=False)[METRICS].mean()

    # --- Step 5: write results ---
    n_held = len(held_out)
    md = f"""# Cold start — simulated brand-new characters and graduation curves (judged on Oct 26, 27, 28)

- {n_held} characters ({HOLDOUT_SHARE:.0%}) held out. For each judging day D: the held-out characters keep only their N most recent
  impressions before D (N = 0: brand new at the start of D; "all": their real history); every history feature is recomputed
  from what remains; brand-new / add-back characters get age 0 and popularity 0 on D. V1 (LightGBM + FM) is trained on
  everything before D and scored on D.
- metrics are means over the three days; NE is relative to predicting the training CTR

## 1. Graduation curve for characters: accuracy on the held-out characters by how many impressions they had before D
{to_md(curve)}

### Held-out characters, gap to having their full history (same rows)
{to_md(curve_gap)}

## 2. Which attributes carry signal for a brand-new character (LightGBM component, held-out characters, N = 0)
{to_md(ablation_summary)}

## 3. Natural curves (normal V1 with all history): creatives by training impressions, users by prior impressions
{to_md(natural_summary)}

## Per day
{to_md(graduation.set_index(['day', 'impressions_before', 'segment']))}

{to_md(ablation.set_index(['day', 'variant']))}
"""
    write("cold_start", md)


if __name__ == "__main__":
    main()
