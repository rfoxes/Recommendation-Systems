"""
Accuracy pass, kept item: V1 retrained every 6 hours vs daily, judged on Oct 26, 27, 28 (test days untouched).
Rule: lower logloss than daily on all three days and mean gain > 0.0007. Run: .venv/bin/python -m analysis.retrain_frequency
"""
import json

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR, RETRAIN_HOURS
from src.data import build_frame
from src.metrics import evaluate
from src.models import V1Model, retrain_blocks

from .common import to_md, write

JUDGE_DAYS = [pd.Timestamp("2014-10-26"), pd.Timestamp("2014-10-27"), pd.Timestamp("2014-10-28")]
NOISE_LOGLOSS = 0.0007
DAILY, EVERY_6H = "daily (00h)", "every 6 hours (00, 06, 12, 18h)"
SCHEDULES = {DAILY: [0], EVERY_6H: RETRAIN_HOURS}
LATER_HOURS = 6  # hours 00-05 are scored by the same model under both schedules


def main():
    tuning = json.loads((RESULTS_DIR / "v1" / "metrics.json").read_text())["tuning"]
    rounds, epochs = tuning["lightgbm_rounds"], tuning["factorization_machine_epochs"]
    df = build_frame()

    rows = []
    for day in JUDGE_DAYS:
        target = df[df["day"] == day]
        later = (target["hour_of_day"] >= LATER_HOURS).to_numpy()
        p_const = np.full(len(target), df.loc[df["ts"] < day, "click"].mean())
        models = {}
        for schedule, hours in SCHEDULES.items():
            # --- Step 1: each block of the day is scored by V1 trained on all rows before the block starts ---
            p = np.empty(len(target))
            for cutoff, first_hour, end_hour in retrain_blocks(day, hours):
                if cutoff not in models:
                    models[cutoff] = V1Model(rounds, epochs).fit(df[df["ts"] < cutoff])
                block = ((target["hour_of_day"] >= first_hour) & (target["hour_of_day"] < end_hour)).to_numpy()
                p[block] = models[cutoff].predict_components(target[block])[2]

            # --- Step 2: metrics on the whole day and on the hours where the schedules differ ---
            for hours_scored, mask in [("00-23h", np.ones(len(target), bool)), ("06-23h", later)]:
                part = target[mask]
                rows.append({"day": str(day.date()), "hours": hours_scored, "schedule": schedule,
                             **evaluate(part["click"].to_numpy(), p[mask], p_const[mask], part["publisher_id"].to_numpy())})
        print(f"done {day.date()}", flush=True)

    # --- Step 3: compare with daily retraining ---
    results = pd.DataFrame(rows)
    key = results["day"] + "|" + results["hours"]
    reference = results[results["schedule"] == DAILY].assign(key=key).set_index("key")
    for metric in ["logloss", "AUC", "AUC_within_publisher", "ECE"]:
        results[f"delta_{metric}"] = results[metric] - key.map(reference[metric])
    summary = results.groupby(["hours", "schedule"], sort=False)[
        ["logloss", "delta_logloss", "NE", "AUC", "delta_AUC", "AUC_within_publisher", "delta_AUC_within_publisher",
         "ECE", "delta_ECE"]].mean()
    summary["better on all 3 days"] = results.groupby(["hours", "schedule"], sort=False)["delta_logloss"].apply(
        lambda d: bool((d < 0).all()))
    summary["passes rule"] = summary["better on all 3 days"] & (summary["delta_logloss"] < -NOISE_LOGLOSS)

    md = f"""# Retrain frequency — V1 retrained every 6 hours vs daily, judged on Oct 26, 27, 28 (test days untouched)

- each block of day D is scored by V1 (LightGBM {rounds} rounds, FM {epochs} passes) trained on all rows before the block starts
- rule: lower logloss than daily retraining on all three days and mean improvement > {NOISE_LOGLOSS}
- hours 00-05 are scored by the same model under both schedules; 06-23h shows where they differ

## Mean over the three days (delta = schedule minus daily)
{to_md(summary)}

## Per day
{to_md(results.set_index(['day', 'hours', 'schedule']))}
"""
    write("retrain_frequency", md)


if __name__ == "__main__":
    main()
