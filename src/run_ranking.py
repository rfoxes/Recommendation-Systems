"""
Goal 2: for every impression opportunity from Oct 24 on, rank N candidate ads with V1 + business rules, then evaluate by
period (development, held-out, last 24 hours) and per day.
Uses the V1 settings from results/v1/metrics.json (run src.run_v1 first). Run: .venv/bin/python -m src.run_ranking
"""
import json
import time

import numpy as np
import pandas as pd

from .config import EVAL_DAYS, FATIGUE_HARD_CAP, N_CANDIDATES, RESULTS_DIR, SEED, TEST_DAYS
from .data import build_frame
from .features import logit
from .metrics import evaluation_periods, wilson_ci
from .models import V1Model, retrain_blocks
from .ranking import (TIER_RANK, Pacer, add_slot_size, advertiser_max_tier, candidate_rows, creative_catalog,
                      generate_candidates, rank_day, repeat_penalty)
from .run_v1 import to_md

OUT = RESULTS_DIR / "ranking"
RANKERS = {"V1 (blend) - used for ranking": "p_v1", "LightGBM alone": "p_lightgbm", "FM alone": "p_fm",
           "benchmark, not used: recent creative CTR only": "creative_ctr_24h", "benchmark, not used: random": "random"}
V1_RANKER, RANDOM_RANKER = "V1 (blend) - used for ranking", "benchmark, not used: random"


def day_rng(day):
    """Each day's replay gets its own random stream, so a day's results don't depend on which other days were run."""
    return np.random.default_rng([SEED, day.dayofyear])


def score_day(df, day, tuning, rng):
    """
    Retrieval candidates for every opportunity on `day` and their V1 scores; V1 is retrained every 6 hours on all
    earlier rows. Returns (candidate rows, history up to the end of `day`, rows before `day`, timings).
    """
    history, train = df[df["ts"] < day + pd.Timedelta(days=1)], df[df["ts"] < day]
    opportunities = df[df["day"] == day]
    start = time.perf_counter()
    catalog = creative_catalog(history)
    creative_tier_rank = pd.Series(advertiser_max_tier(catalog["C17"]), index=catalog.index).map(TIER_RANK)
    candidates = generate_candidates(opportunities, history, creative_tier_rank, rng)
    rows = candidate_rows(opportunities, candidates, catalog, history)
    assemble_s = time.perf_counter() - start
    logged = rows[rows["is_logged"]]
    assert np.allclose(logged["creative_ctr_24h"], opportunities.loc[logged["opportunity"], "creative_ctr_24h"])

    fit_s = score_s = 0.0
    for cutoff, first_hour, end_hour in retrain_blocks(day):
        block = ((rows["hour_of_day"] >= first_hour) & (rows["hour_of_day"] < end_hour)).to_numpy()
        if not block.any():
            continue
        block_train = df[df["ts"] < cutoff]
        start = time.perf_counter()
        model = V1Model(tuning["lightgbm_rounds"], tuning["factorization_machine_epochs"]).fit(block_train)
        fit_s += time.perf_counter() - start
        start = time.perf_counter()
        for col, p in zip(["p_lightgbm", "p_fm", "p_v1"], model.predict_components(rows[block])):
            rows.loc[block, col] = p
        rows.loc[block, "train_ctr"] = block_train["click"].mean()
        score_s += time.perf_counter() - start
    timings = {"day": str(day.date()), "opportunities": len(opportunities), "candidates": len(rows),
               "train_v1_seconds": fit_s, "candidates_and_features_seconds": assemble_s, "scoring_seconds": score_s}
    return rows, history, train, timings


def summarize(ranked):
    """Tables for one set of opportunities: shown-ad CTR by rank, candidate spread, rules, no-ad, pick reasons, predicted CTR."""
    opportunities_n = ranked["opportunity"].nunique()

    # does each ranker put the ad that was actually shown at the right place?
    rank_tables, rank_summary = {}, {}
    for name, col in RANKERS.items():
        position = ranked.groupby("opportunity")[col].rank(ascending=False, method="first")
        shown = ranked.loc[ranked["is_logged"]].assign(position=position[ranked["is_logged"]].astype(int))
        table = shown.groupby("position")["click"].agg(opportunities="size", clicks="sum")
        table["ctr"] = table["clicks"] / table["opportunities"]
        table["ci_low"], table["ci_high"] = wilson_ci(table["clicks"], table["opportunities"])
        rank_tables[name] = table.drop(columns="clicks")
        rank_summary[name] = {"CTR when shown ad ranked 1st": table.loc[1, "ctr"],
                              f"CTR when shown ad ranked {N_CANDIDATES}th": table.loc[N_CANDIDATES, "ctr"],
                              "difference (pp)": 100 * (table.loc[1, "ctr"] - table.loc[N_CANDIDATES, "ctr"])}

    # how confident is the model's ordering (before rules)?
    by_score = ranked.sort_values(["opportunity", "p_v1"], ascending=[True, False])
    by_score["order"] = by_score.groupby("opportunity").cumcount()
    first = by_score[by_score["order"] == 0].set_index("opportunity")
    second = by_score[by_score["order"] == 1].set_index("opportunity").reindex(first.index)
    gap_pp = 100 * (first["p_v1"] - second["p_v1"])
    model_tossup = ((logit(first["p_v1"].to_numpy()) - first["u"].to_numpy())
                    <= (logit(second["p_v1"].to_numpy()) + second["u"].to_numpy()))
    spread = {
        "candidates per opportunity (min)": ranked.groupby("opportunity").size().min(),
        "top-1 minus top-2 predicted CTR, median (pp)": gap_pp.median(),
        "top-1 minus top-2 predicted CTR, 90th pct (pp)": gap_pp.quantile(0.9),
        "best minus worst candidate, median (pp)": 100 * (ranked.groupby("opportunity")["p_v1"].max()
                                                          - ranked.groupby("opportunity")["p_v1"].min()).median(),
        "uncertainty half-band u, median (log-odds)": ranked["u"].median(),
        "share of opportunities that are toss-ups (model only)": float(np.mean(model_tossup)),
    }

    # business rules and the final pick
    picks = ranked[ranked["final_rank"] == 1].set_index("opportunity")
    by_opp = ranked.groupby("opportunity")
    counts = {
        "all opportunities": opportunities_n,
        "fewer than 5 candidates reached the model": int((by_opp.size() < N_CANDIDATES).sum()),
        "shown ad blocked by the safety gate": int((ranked.loc[ranked["is_logged"], "blocked"] == "safety").sum()),
        ">= 1 candidate at the fatigue cap": int(by_opp["blocked"].apply(lambda b: (b == "fatigue_cap").any()).sum()),
        ">= 1 candidate given the repeat penalty": int(by_opp["repeat_penalty"].apply(lambda r: (r < 1).any()).sum()),
        ">= 1 candidate paced down": int(by_opp["pacing"].apply(lambda m: (m < 1).any()).sum()),
        "shown ad's advertiser out of budget": int((ranked.loc[ranked["is_logged"], "blocked"] == "budget").sum()),
        "no eligible candidate (no ad served)": opportunities_n - len(picks),
        "picks that were toss-ups": int(picks["tossup"].sum()),
        "picks that explored": int(picks["explored"].sum()),
        "final pick differs from the model-only top candidate": int((picks["C14"] != first["C14"].reindex(picks.index)).sum()),
    }
    shown = ranked[ranked["is_logged"]].assign(no_ad=lambda d: ~d["opportunity"].isin(picks.index))
    no_ad = shown.groupby("safety_tier")["no_ad"].mean()
    reasons = picks["pick_reason"].value_counts(normalize=True)

    # model-estimated predicted CTR of the pick vs the ad actually shown (same opportunities)
    shown_p = ranked.loc[ranked["is_logged"]].set_index("opportunity")["p_v1"].reindex(picks.index)
    lift = {"mean predicted CTR, ad actually shown": shown_p.mean(),
            "mean predicted CTR, our pick": picks["p_v1"].mean(),
            "mean predicted CTR, model-only top candidate (same opportunities)": first["p_v1"].reindex(picks.index).mean(),
            "our pick minus ad actually shown (pp)": 100 * (picks["p_v1"].mean() - shown_p.mean())}
    return {"rank_summary": pd.DataFrame(rank_summary).T, "rank_tables": rank_tables, "spread": spread, "counts": counts,
            "no_ad": no_ad, "reasons": reasons, "lift": lift}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tuning = json.loads((RESULTS_DIR / "v1" / "metrics.json").read_text())["tuning"]
    df = add_slot_size(build_frame())

    # --- Step 1: per day from Oct 24: candidates, features, V1 scores (retrained every 6 hours), rules ---
    ranked_days, timings, penalties = [], [], {}
    for day in EVAL_DAYS:
        rng = day_rng(day)
        rows, history, train, timing = score_day(df, day, tuning, rng)
        penalties[str(day.date())] = repeat_penalty(train)
        start = time.perf_counter()
        ranked = rank_day(rows, Pacer(history, day), penalties[str(day.date())], rng)
        timing["rules_and_ordering_seconds"] = time.perf_counter() - start
        ranked["random"] = rng.random(len(ranked))
        ranked_days.append(ranked)
        timings.append(timing)
        print(f"ranked {day.date()}", flush=True)
    ranked = pd.concat(ranked_days, ignore_index=True)
    ranked[ranked["day"].isin(TEST_DAYS)].drop(columns=["random"]).to_parquet(OUT / "ranked_candidates.parquet", index=False)

    # --- Step 2: the same evaluation by period (development, held-out, last 24 hours) and per day ---
    periods = {label: summarize(ranked[mask]) for label, mask in evaluation_periods(ranked["ts"]).items()}
    held_out_label = next(k for k in periods if k.startswith("held-out"))
    days = {str(d.date()): summarize(g) for d, g in ranked.groupby("day")}

    def per_day_row(s):
        v1, rand = s["rank_summary"].loc[V1_RANKER], s["rank_summary"].loc[RANDOM_RANKER]
        c = s["counts"]
        return {"opportunities": c["all opportunities"],
                "V1: shown-ad CTR at rank 1": v1.iloc[0], f"V1: at rank {N_CANDIDATES}": v1.iloc[1],
                "random: at rank 1": rand.iloc[0], f"random: at rank {N_CANDIDATES}": rand.iloc[1],
                "toss-ups (model only)": s["spread"]["share of opportunities that are toss-ups (model only)"],
                "no ad served": c["no eligible candidate (no ad served)"] / c["all opportunities"],
                "predicted CTR, our pick": s["lift"]["mean predicted CTR, our pick"],
                "predicted CTR, ad actually shown": s["lift"]["mean predicted CTR, ad actually shown"]}

    per_day = pd.DataFrame({d: per_day_row(s) for d, s in days.items()}).T
    rank_summary = pd.concat({label: s["rank_summary"] for label, s in periods.items()})
    spread = pd.DataFrame({label: s["spread"] for label, s in periods.items()})
    counts = pd.DataFrame({label: s["counts"] for label, s in periods.items()})
    shares = counts / counts.loc["all opportunities"]
    rules = pd.concat({"opportunities": counts, "share": shares}, axis=1)
    no_ad = pd.DataFrame({label: s["no_ad"] for label, s in periods.items()})
    reasons = pd.DataFrame({label: s["reasons"] for label, s in periods.items()}).fillna(0)
    lift = pd.DataFrame({label: s["lift"] for label, s in periods.items()})

    # --- Step 3: write results ---
    timing = pd.DataFrame(timings).set_index("day")
    timing["ms_per_opportunity (features + scoring + rules)"] = 1000 * (
        timing["candidates_and_features_seconds"] + timing["scoring_seconds"] + timing["rules_and_ordering_seconds"]
    ) / timing["opportunities"]
    md = f"""# Goal 2 — Candidate ranking, walk-forward from Oct 24 ({ranked['opportunity'].nunique():,} opportunities, up to {N_CANDIDATES} candidates each)

- each day is replayed hour by hour with V1 retrained every 6 hours on all earlier rows; Oct 21-23 are warm-up only;
  development = days we made design choices on, held-out = days kept back, last 24 hours = the most-trained models
- candidates: the ad actually shown + up to {N_CANDIDATES - 1} creatives from advertisers that accept the character's safety tier, that ran on
  the same publisher and slot size in the previous 24h (topped up with the same size from any publisher), drawn in proportion
  to their impressions in that window
- retrieval returns up to 10 such creatives in order; hour by hour, advertisers whose budget is already spent are skipped and
  the model receives the shown ad + the first {N_CANDIDATES - 1} remaining
- the ranking always uses V1 (pointwise: sort by predicted click probability); other rows in section 1 are comparisons only
- rules: safety gate (simulated advertiser max tier), fatigue (repeat penalty per day = {', '.join(f'{d}: {p:.3f}' for d, p in penalties.items())};
  hard cap {FATIGUE_HARD_CAP} per user per creative per 24h), budget pacing (budget = advertiser's previous-day volume)
- uncertainty band: V1 log-odds +/- half the LightGBM-FM gap; toss-up = runner-up band overlaps the leader's

## 1. Click rate of the ad actually shown, by where each ranker placed it
{to_md(rank_summary)}

### Per day
{to_md(per_day)}

### Held-out, by position
""" + "\n".join(f"#### {name}\n{to_md(t)}\n" for name, t in periods[held_out_label]["rank_tables"].items()) + f"""
## 2. How far apart are the candidates? (V1, before rules)
{to_md(spread)}

## 3. Business rules and the final pick
{to_md(rules)}

### No ad served, share by character safety tier
{to_md(no_ad)}

### Why each ad won (share of picks)
{to_md(reasons)}

## 4. Model-estimated predicted CTR (not observed clicks)
{to_md(lift)}

## 5. Timing
{to_md(timing)}
"""
    (OUT / "metrics.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
