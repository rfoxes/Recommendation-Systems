"""
Goals 3-4: the Goal 2 ranking vs the same ranking with the adaptation and exploration layer, on identical candidates,
every day from Oct 24 (walk-forward). Every layer setting is a fixed decision or measured on earlier days (nothing tuned here).
Also writes the sample output (results/sample_output.md) from the final system = Goal 2 + layer.
Uses the V1 settings from results/v1/metrics.json. Run: .venv/bin/python -m src.run_layer
"""
import json

import numpy as np
import pandas as pd

from .config import (COLD_EXPLORATION_SHARE, DOMINANT_COHORTS, EVAL_DAYS, GRADUATION_IMPRESSIONS,
                     GRADUATION_WINDOW_HOURS, RESULTS_DIR, SEED, TEST_DAYS)
from .data import build_frame
from .metrics import evaluate, evaluation_periods
from .ranking import (Layer, Pacer, add_layer_features, add_slot_size, campaign_repeat_penalty, dominant_cohorts,
                      rank_day, repeat_penalty)
from .run_ranking import day_rng, score_day
from .run_v1 import to_md
from .sample_output import write_sample_output

OUT = RESULTS_DIR / "layer"
GOAL2, LAYER = "Goal 2 ranking", "Goal 2 + Goals 3-4 layer"
PER_DAY_METRICS = ["ads served", "mean predicted CTR of served ads",
                   "dominant cohorts: share of ads on each cohort-day's top-5 campaigns",
                   "dominant cohorts: user saw the same campaign in the last 24h", "served ads that are cold creatives"]
N_BOOT = 200


def top5_concentration(picks):
    """Share of a cohort-day's served ads that went to its 5 most-served campaigns (weighted mean over cohort-days)."""
    counts = picks.groupby(["day", "cohort", "C17"]).size()
    per_cohort_day = counts.groupby(level=[0, 1])
    return float(per_cohort_day.apply(lambda s: s.nlargest(5).sum()).sum() / counts.sum())


def paired_differences(per_opp, users):
    """Mean layer-minus-Goal-2 difference per served opportunity, with a 95% interval from resampling whole users."""
    user_idx = pd.factorize(users)[0]
    diffs = {
        "mean predicted CTR of served ads (pp)": 100 * (per_opp[("p_v1", LAYER)] - per_opp[("p_v1", GOAL2)]),
        "same creative seen in last 24h (pp of ads)": 100 * ((per_opp[("user_creative_exposures_24h", LAYER)] > 0).astype(float)
                                                            - (per_opp[("user_creative_exposures_24h", GOAL2)] > 0)),
        "same campaign seen in last 24h (pp of ads)": 100 * ((per_opp[("user_campaign_exposures_24h", LAYER)] > 0).astype(float)
                                                            - (per_opp[("user_campaign_exposures_24h", GOAL2)] > 0)),
        "cold creative served (pp of ads)": 100 * ((per_opp[("creative_impressions_72h", LAYER)] < GRADUATION_IMPRESSIONS).astype(float)
                                                  - (per_opp[("creative_impressions_72h", GOAL2)] < GRADUATION_IMPRESSIONS)),
    }
    boot_rng = np.random.default_rng(SEED)
    weights = [boot_rng.poisson(1.0, user_idx.max() + 1)[user_idx] for _ in range(N_BOOT)]
    paired = pd.DataFrame({name: {"difference": d.mean(),
                                  "ci_low": np.quantile([np.average(d, weights=w) for w in weights], 0.025),
                                  "ci_high": np.quantile([np.average(d, weights=w) for w in weights], 0.975)}
                           for name, d in diffs.items()}).T
    return paired


def policy_metrics(ranked):
    picks = ranked[ranked["final_rank"] == 1]
    eligible = ranked[ranked["blocked"] == ""]
    leader = eligible.loc[eligible.groupby("opportunity")["z"].idxmax()].set_index("opportunity")
    explored = picks[picks["explored"]]
    dominant = picks[picks["dominant_cohort"]]
    cold = picks["creative_impressions_72h"] < GRADUATION_IMPRESSIONS
    matched = picks[picks["is_logged"]]
    return {
        "opportunities": ranked["opportunity"].nunique(),
        "ads served": len(picks),
        "mean predicted CTR of served ads": picks["p_v1"].mean(),
        "replay: picks that equal the ad actually shown": len(matched),
        "replay: actual CTR of those": matched["click"].mean(),
        "served ads: user saw the same creative in the last 24h": (picks["user_creative_exposures_24h"] > 0).mean(),
        "served ads: user saw the same campaign in the last 24h": (picks["user_campaign_exposures_24h"] > 0).mean(),
        "served ads in dominant cohorts": len(dominant),
        "dominant cohorts: user saw the same campaign in the last 24h": (dominant["user_campaign_exposures_24h"] > 0).mean(),
        "dominant cohorts: share of ads on each cohort-day's top-5 campaigns": top5_concentration(dominant),
        "all cohorts: share of ads on each cohort-day's top-5 campaigns": top5_concentration(picks),
        "served ads that are cold creatives": cold.mean(),
        "distinct cold creatives served": picks.loc[cold, "C14"].nunique(),
        "explored picks": len(explored),
        "explored picks: mean predicted CTR": explored["p_v1"].mean(),
        "explored picks: mean predicted CTR of the leader they replaced": leader.loc[explored["opportunity"], "p_v1"].mean(),
        "dominant-cohort toss-ups resolved to the least-shown campaign": int(picks["diversified"].sum()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tuning = json.loads((RESULTS_DIR / "v1" / "metrics.json").read_text())["tuning"]
    df = add_slot_size(build_frame())
    df["cohort"] = df["genre"] + " / " + df["safety_tier"]

    # --- Step 1: same candidates and scores for both policies; the layer's settings come from the training days ---
    ranked_all, settings = [], {}
    for day in EVAL_DAYS:
        rng = day_rng(day)
        rows, history, train, _ = score_day(df, day, tuning, rng)
        rows = add_layer_features(rows, history)
        creative_ratio, campaign_ratio = repeat_penalty(train), campaign_repeat_penalty(train)
        layer = Layer(campaign_ratio, dominant_cohorts(history))
        dominant_keys = {f"{h}|{c}" for h, cohorts in layer.dominant_by_hour.items() for c in cohorts}
        settings[str(day.date())] = {"creative repeat penalty": creative_ratio, "campaign repeat penalty": campaign_ratio,
                                     "dominant cohorts at the first hour": ", ".join(sorted(
                                         layer.dominant_by_hour[int(rows["hour_idx"].min())]))}
        for name, ranked in [(GOAL2, rank_day(rows, Pacer(history, day), creative_ratio, rng)),
                             (LAYER, rank_day(rows, Pacer(history, day), creative_ratio, rng, layer=layer))]:
            key = ranked["hour_idx"].astype(str) + "|" + ranked["cohort"]
            ranked_all.append(ranked.assign(policy=name, dominant_cohort=key.isin(dominant_keys).to_numpy()))
        print(f"ranked {day.date()}", flush=True)
    ranked = pd.concat(ranked_all, ignore_index=True)
    ranked["day_label"] = ranked["day"].dt.strftime("%Y-%m-%d")
    periods = evaluation_periods(ranked["ts"])

    # --- Step 2: model accuracy on the impressions actually shown (the layer does not change the model) ---
    shown = ranked[(ranked["policy"] == GOAL2) & ranked["is_logged"]]
    accuracy = {}
    for label, part in [(k, shown[m]) for k, m in evaluation_periods(shown["ts"]).items()] + list(shown.groupby("day_label")):
        for model in ["p_lightgbm", "p_fm", "p_v1"]:
            accuracy[(label, model)] = evaluate(part["click"], part[model].to_numpy(), part["train_ctr"].to_numpy(),
                                                part["publisher_id"].to_numpy())
    accuracy = pd.DataFrame(accuracy).T

    # --- Step 3: policy comparison by period and per day ---
    comparison = pd.DataFrame({(label, policy): policy_metrics(ranked[m & (ranked["policy"] == policy).to_numpy()])
                               for label, m in periods.items() for policy in [GOAL2, LAYER]})
    per_day = {}
    for day, part in ranked.groupby("day_label"):
        for policy in [GOAL2, LAYER]:
            m = policy_metrics(part[part["policy"] == policy])
            for metric in PER_DAY_METRICS:
                per_day.setdefault(day, {})[f"{metric} ({'layer' if policy == LAYER else 'Goal 2'})"] = m[metric]
    per_day = pd.DataFrame(per_day).T
    held_out = periods[next(k for k in periods if k.startswith("held-out"))]
    reasons = (ranked[held_out & (ranked["final_rank"] == 1).to_numpy()].groupby("policy")["pick_reason"]
               .value_counts(normalize=True).unstack(0).fillna(0))

    # --- Step 4: paired differences (layer minus Goal 2) with 95% intervals, resampling whole users, by period ---
    picks = ranked[ranked["final_rank"] == 1]
    per_opp_all = picks.pivot_table(index="opportunity", columns="policy",
                                    values=["p_v1", "user_creative_exposures_24h", "user_campaign_exposures_24h",
                                            "creative_impressions_72h"], aggfunc="first").dropna()
    opp_info = ranked.drop_duplicates("opportunity").set_index("opportunity")[["user", "ts"]]
    paired = {}
    for label, mask in evaluation_periods(opp_info.loc[per_opp_all.index, "ts"]).items():
        paired[label] = paired_differences(per_opp_all[mask], opp_info.loc[per_opp_all.index[mask], "user"])
    paired = pd.concat(paired)

    # --- Step 5: write results ---
    md = f"""# Goals 3-4 — adaptation and exploration layer vs the Goal 2 ranking (walk-forward from Oct 24)

- every day from Oct 24 is replayed with V1 retrained every 6 hours; development = days we made design choices on,
  held-out = days kept back, last 24 hours = the most-trained models
- identical candidates and V1 scores for both policies; each policy runs its own hour-by-hour budget pacing
- layer (reorders only, never skips an ad):
  - cold creative = fewer than {GRADUATION_IMPRESSIONS:,} impressions in the last {GRADUATION_WINDOW_HOURS} h; its band is widened for
    data scarcity; shown first if its optimistic end reaches the leader's estimate, up to {COLD_EXPLORATION_SHARE:.0%} of an hour's
    opportunities (replaces the Goal 2 least-known toss-up rule)
  - repeat penalty also for the same campaign seen in the last 24h
  - toss-ups in the {DOMINANT_COHORTS} dominant cohorts (most impressions in the last 24h) go to the campaign with the smallest share
    of that cohort's recent impressions
- the model is unchanged by the layer; its accuracy is reported once, on the impressions actually shown

## Settings, measured on the days before each day
{to_md(pd.DataFrame(settings).T)}

## 1. Model accuracy (impressions actually shown)
{to_md(accuracy)}

## 2. Serving outcomes by period (absolute)
{to_md(comparison)}

### Per day
{to_md(per_day)}

## 3. Paired difference, layer minus Goal 2, by period (opportunities served by both; 95% interval over users)
{to_md(paired)}

## 4. Why each ad won (held-out)
{to_md(reasons)}
"""
    (OUT / "metrics.md").write_text(md)
    print(md)
    print(write_sample_output(ranked[ranked["policy"] == LAYER], RESULTS_DIR / "sample_output.md", TEST_DAYS[0]))


if __name__ == "__main__":
    main()
