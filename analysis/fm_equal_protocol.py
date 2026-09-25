"""Equal-protocol round: factorization machine (properly tuned) and LightGBM both trained exactly like V1 — iterations chosen
on earlier days, then retrained on all days before the scored day, no holdout, no calibration. Judged on Oct 26, 27, 28.
Run: .venv/bin/python -m analysis.fm_equal_protocol"""
import itertools
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT, SEED
from src.data import build_frame
from src.features import FeatureEncoder, FMEncoder, logit
from src.metrics import evaluate
from src.models import LightGBMModel

from .common import to_md, write

TUNE_DAYS = [pd.Timestamp("2014-10-24"), pd.Timestamp("2014-10-25")]
JUDGE_DAYS = [pd.Timestamp("2014-10-26"), pd.Timestamp("2014-10-27"), pd.Timestamp("2014-10-28")]
NOISE_LOGLOSS = 0.0007
FM_GRID = {"dim": [8, 16, 32], "l2": [1e-3, 1e-2, 5e-2], "lr": [1e-3, 3e-3]}
FM_FIXED = {"batch_size": 4096, "max_epochs": 10}
FM_MIN_COUNT = 2
PARALLEL_JOBS, THREADS_PER_JOB = 3, 4


def save_inputs(path, encoder, train, stop=None, target=None):
    """Encode rows for the FM and save them for the worker process (empty stop/target allowed)."""
    arrays = {}
    for name, rows in [("train", train), ("stop", stop), ("eval", target)]:
        if rows is None:
            rows = train.iloc[:0]
        arrays[f"idx_{name}"], arrays[f"dense_{name}"] = encoder.transform(rows)
    np.savez(path, y_train=train["click"].to_numpy(), y_stop=(stop if stop is not None else train.iloc[:0])["click"].to_numpy(),
             **arrays)


def run_jobs(jobs):
    """jobs: list of (key, inputs path, settings). Runs PARALLEL_JOBS FM processes at a time; returns {key: outputs}."""
    pending, running, results = list(jobs), [], {}
    while pending or running:
        while pending and len(running) < PARALLEL_JOBS:
            key, inputs, settings = pending.pop(0)
            outputs = inputs.with_name(f"out_{abs(hash(key))}.npz")
            command = [sys.executable, "-m", "src.fm", "fit", str(inputs), str(outputs),
                       json.dumps({"seed": SEED, "threads": THREADS_PER_JOB, **settings})]
            running.append((key, subprocess.Popen(command, cwd=ROOT), outputs))
        for job in list(running):
            key, process, outputs = job
            if process.poll() is not None:
                if process.returncode != 0:
                    raise RuntimeError(f"FM job {key} failed with code {process.returncode}")
                results[key] = dict(np.load(outputs))
                running.remove(job)
                print(f"finished {key}", flush=True)
        time.sleep(0.5)
    return results


def main():
    df = build_frame()
    configs = [dict(zip(FM_GRID, values)) for values in itertools.product(*FM_GRID.values())]
    with tempfile.TemporaryDirectory(prefix="fm_") as tmp:
        workdir = Path(tmp)

        # --- Step 1: tune on Oct 24-25: train < D, early stopping on D (like V1's tuning on Oct 28) ---
        jobs, lgbm_tuning = [], []
        for day in TUNE_DAYS:
            train, stop = df[df["ts"] < day], df[df["day"] == day]
            inputs = workdir / f"tune_{day:%m%d}.npz"
            save_inputs(inputs, FMEncoder(min_count=FM_MIN_COUNT).fit(train), train, stop)
            jobs += [((str(day.date()), i), inputs, {**FM_FIXED, **c}) for i, c in enumerate(configs)]
            encoder = FeatureEncoder().fit(train)
            lgbm = LightGBMModel(encoder.categorical).fit(encoder.transform(train), train["click"].to_numpy(),
                                                          X_val=encoder.transform(stop), y_val=stop["click"].to_numpy())
            lgbm_tuning.append({"day": str(day.date()), "rounds": lgbm.num_rounds,
                                "logloss": evaluate(stop["click"], lgbm.predict(encoder.transform(stop)),
                                                    lgbm.predict(encoder.transform(stop)))["logloss"]})
        tuned = run_jobs(jobs)
        tuning = pd.DataFrame([{"day": day, "config": i, **configs[i], "epochs": float(out["epochs"]),
                                "logloss": evaluate(df.loc[df["day"] == pd.Timestamp(day), "click"], out["p_stop"],
                                                    out["p_stop"])["logloss"]}
                               for (day, i), out in tuned.items()])
        by_config = tuning.pivot_table(index=["config", *FM_GRID], columns="day", values="logloss")
        by_config["mean"] = by_config.mean(axis=1)
        best_id = int(by_config["mean"].idxmin()[0])
        best = configs[best_id]
        fm_epochs = float(tuning.loc[tuning["config"] == best_id, "epochs"].mean())
        lgbm_tuning = pd.DataFrame(lgbm_tuning).set_index("day")
        lgbm_rounds = int(round(lgbm_tuning["rounds"].mean()))

        # --- Step 2: judge on Oct 26-28: both retrained on all days before D with fixed iterations ---
        jobs, lgbm_preds, days = [], {}, {}
        for day in JUDGE_DAYS:
            train, target = df[df["ts"] < day], df[df["day"] == day]
            days[day] = (train, target)
            inputs = workdir / f"judge_{day:%m%d}.npz"
            save_inputs(inputs, FMEncoder(min_count=FM_MIN_COUNT).fit(train), train, target=target)
            jobs.append((str(day.date()), inputs, {**FM_FIXED, **best, "fixed_epochs": fm_epochs,
                                                    "max_epochs": fm_epochs + 1}))
            encoder = FeatureEncoder().fit(train)
            start = time.perf_counter()
            lgbm = LightGBMModel(encoder.categorical, num_rounds=lgbm_rounds).fit(encoder.transform(train),
                                                                                  train["click"].to_numpy())
            lgbm_preds[day] = (lgbm.predict(encoder.transform(target)), time.perf_counter() - start)
        judged = run_jobs(jobs)

    rows = []
    for day in JUDGE_DAYS:
        train, target = days[day]
        y = target["click"].to_numpy()
        p_const, publisher = np.full(len(y), train["click"].mean()), target["publisher_id"].to_numpy()
        p_lgbm, lgbm_seconds = lgbm_preds[day]
        out = judged[str(day.date())]
        p_fm = out["p_eval"]
        p_blend = 1 / (1 + np.exp(-(logit(p_lgbm) + logit(p_fm)) / 2))
        for name, p, seconds in [("V1 LightGBM", p_lgbm, lgbm_seconds), ("FM", p_fm, float(out["seconds"])),
                                 ("Blend V1 + FM", p_blend, lgbm_seconds + float(out["seconds"]))]:
            rows.append({"day": str(day.date()), "model": name, **evaluate(y, p, p_const, publisher),
                         "train_seconds": seconds})

    # --- Step 3: compare with V1 ---
    results = pd.DataFrame(rows)
    reference = results[results["model"] == "V1 LightGBM"].set_index("day")
    for metric in ["logloss", "AUC", "AUC_within_publisher"]:
        results[f"delta_{metric}"] = results[metric] - results["day"].map(reference[metric])
    mean = results.groupby("model", sort=False)[["logloss", "delta_logloss", "NE", "AUC", "delta_AUC",
                                                  "AUC_within_publisher", "delta_AUC_within_publisher", "ECE",
                                                  "train_seconds"]].mean()
    mean["better_on_all_days"] = results.groupby("model", sort=False)["delta_logloss"].apply(lambda d: bool((d < 0).all()))
    mean["passes_rule"] = mean["better_on_all_days"] & (mean["delta_logloss"] < -NOISE_LOGLOSS)

    # --- Step 4: write results ---
    cols = ["rows", "logloss", "delta_logloss", "NE", "AUC", "delta_AUC", "AUC_within_publisher",
            "delta_AUC_within_publisher", "ECE", "mean_pred", "actual_ctr", "train_seconds"]
    md = f"""# Equal-protocol round — FM and V1 LightGBM trained exactly like V1, judged on Oct 26, 27, 28

- tuning (Oct 24, 25): train on days before D, early stopping on D; FM grid {FM_GRID}, {FM_FIXED}, IDs kept down to {FM_MIN_COUNT} rows
- chosen FM: {best}, trained for {fm_epochs:.2f} epochs (mean best epoch of that setting); LightGBM: {lgbm_rounds} rounds (mean best)
- judging (Oct 26, 27, 28): both retrained on all days before D with those fixed iterations; no holdout, no calibration
- blend: average of the two models' logits
- rule: better = lower logloss than V1 on all three days and mean improvement > {NOISE_LOGLOSS}

## Mean over the three judging days (delta vs V1)
{to_md(mean)}

## Per day
{to_md(results.set_index(['day', 'model']).reindex(columns=cols))}

## LightGBM tuning
{to_md(lgbm_tuning)}

## FM tuning (logloss at the best epoch)
{to_md(by_config)}
"""
    write("fm_equal_protocol", md)


if __name__ == "__main__":
    main()
