"""
V1 = blend of LightGBM and a factorization machine (50/50 in log-odds). Tune on Oct 28, then walk forward from Oct 24:
retrain every 6 hours on all earlier rows and score the next 6 hours; constant and logistic regression are baselines.
Run: .venv/bin/python -m src.run_v1
"""
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import EVAL_DAYS, LR_C_GRID, RESULTS_DIR, TUNE_VAL_DAY
from .data import build_frame
from .features import FeatureEncoder
from .metrics import calibration_table, evaluate, evaluation_periods
from .models import ConstantModel, FactorizationMachineModel, LightGBMModel, LogisticModel, blend, retrain_blocks

OUT = RESULTS_DIR / "v1"
MODEL_NAMES = ["constant", "logistic_regression", "lightgbm", "factorization_machine", "v1_blend"]


def to_md(df):
    """Markdown table: whole-number columns as integers, other floats to 4 decimals."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            whole = (out[col].dropna() % 1 == 0).all()
            out[col] = out[col].map(lambda v: f"{v:,.0f}" if whole else f"{v:.4f}")
    return out.to_markdown()


def encode(train, test):
    encoder = FeatureEncoder().fit(train)
    return encoder, encoder.transform(train), encoder.transform(test)


def timed_fit(model, X, y, **kwargs):
    start = time.perf_counter()
    model.fit(X, y, **kwargs)
    return time.perf_counter() - start


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_frame()

    # --- Step 1: tune on Oct 21-27, validate on Oct 28 ---
    train, val = df[df["ts"] < TUNE_VAL_DAY], df[df["day"] == TUNE_VAL_DAY]
    encoder, X_train, X_val = encode(train, val)
    y_train, y_val = train["click"].to_numpy(), val["click"].to_numpy()

    constant = ConstantModel().fit(X_train, y_train)
    p_const_val = constant.predict(X_val)

    lgbm = LightGBMModel(encoder.categorical).fit(X_train, y_train, X_val=X_val, y_val=y_val)
    best_rounds = lgbm.num_rounds
    fm = FactorizationMachineModel().fit(train, stop=val)
    best_epochs = fm.num_epochs
    p_lgbm_val, p_fm_val = lgbm.predict(X_val), fm.predict(val)

    lr_grid = {}
    for C in LR_C_GRID:
        lr = LogisticModel(encoder.categorical + encoder.decile_columns, C=C).fit(X_train, y_train)
        lr_grid[C] = (lr, evaluate(y_val, lr.predict(X_val), p_const_val)["logloss"])
    best_C = min(lr_grid, key=lambda c: lr_grid[c][1])

    val_publisher = val["publisher_id"].to_numpy()
    tune_metrics = pd.DataFrame({
        "constant": evaluate(y_val, p_const_val, p_const_val, val_publisher),
        "logistic_regression": evaluate(y_val, lr_grid[best_C][0].predict(X_val), p_const_val, val_publisher),
        "lightgbm": evaluate(y_val, p_lgbm_val, p_const_val, val_publisher),
        "factorization_machine": evaluate(y_val, p_fm_val, p_const_val, val_publisher),
        "v1_blend": evaluate(y_val, blend(p_lgbm_val, p_fm_val), p_const_val, val_publisher),
    }).T

    # --- Step 2: walk-forward from Oct 24: every 6 hours, retrain on all rows before that hour, predict the next 6 hours ---
    predictions, timings, importances = [], [], {}
    for day in EVAL_DAYS:
        for cutoff, first_hour, end_hour in retrain_blocks(day):
            train = df[df["ts"] < cutoff]
            test = df[(df["day"] == day) & (df["hour_of_day"] >= first_hour) & (df["hour_of_day"] < end_hour)]
            if test.empty:
                continue
            block = f"{day.date()} {first_hour:02d}h"
            encoder, X_train, X_test = encode(train, test)
            y_train = train["click"].to_numpy()
            models = [ConstantModel(),
                      LogisticModel(encoder.categorical + encoder.decile_columns, C=best_C),
                      LightGBMModel(encoder.categorical, num_rounds=best_rounds)]
            block_pred = test[["id", "ts", "day", "click", "publisher_id"]].assign(block=block, train_rows=len(train))
            for model in models:
                fit_s = timed_fit(model, X_train, y_train)
                block_pred[model.name] = model.predict(X_test)
                timings.append({"model": model.name, "fit_seconds": fit_s})
            importances[block] = models[-1].feature_importance()

            fm = FactorizationMachineModel(num_epochs=best_epochs)
            start = time.perf_counter()
            fm.fit(train)
            timings.append({"model": fm.name, "fit_seconds": time.perf_counter() - start})
            block_pred[fm.name] = fm.predict(test)
            block_pred["v1_blend"] = blend(block_pred["lightgbm"].to_numpy(), block_pred[fm.name].to_numpy())
            predictions.append(block_pred)
            print(f"scored {block} ({len(test):,} rows, trained on {len(train):,})", flush=True)
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_parquet(OUT / "predictions.parquet", index=False)

    # --- Step 3: metrics by period (rows pooled), per day, per block ---
    def metrics_for(rows, models=MODEL_NAMES):
        return pd.DataFrame({m: evaluate(rows["click"], rows[m].to_numpy(), rows["constant"].to_numpy(),
                                         rows["publisher_id"].to_numpy())
                             for m in models}).T

    periods = evaluation_periods(pred["ts"])
    by_period = pd.concat({label: metrics_for(pred[mask]) for label, mask in periods.items()})
    per_day = pd.concat({str(d.date()): metrics_for(g) for d, g in pred.groupby("day")})
    v1_per_day = pd.DataFrame({str(d.date()): {"trained_on_rows": g["train_rows"].min(), **evaluate(
        g["click"], g["v1_blend"].to_numpy(), g["constant"].to_numpy(), g["publisher_id"].to_numpy())}
        for d, g in pred.groupby("day")}).T
    v1_per_block = pd.DataFrame({b: {"trained_on_rows": g["train_rows"].iloc[0], **evaluate(
        g["click"], g["v1_blend"].to_numpy(), g["constant"].to_numpy(), g["publisher_id"].to_numpy())}
        for b, g in pred.groupby("block")}).T
    block_spread = {}
    for label, mask in periods.items():
        blocks = v1_per_block.loc[pred.loc[mask, "block"].unique()]
        for metric in ["AUC", "AUC_within_publisher", "logloss"]:
            block_spread[(label, metric)] = {"row-weighted mean": np.average(blocks[metric], weights=blocks["rows"]),
                                             "min": blocks[metric].min(), "max": blocks[metric].max(),
                                             "blocks": len(blocks)}
    block_spread = pd.DataFrame(block_spread).T

    held_out = pred[periods[next(k for k in periods if k.startswith("held-out"))]]
    calibration = {m: calibration_table(held_out["click"].to_numpy(), held_out[m].to_numpy())
                   for m in ["logistic_regression", "lightgbm", "factorization_machine", "v1_blend"]}
    calibration_wide = pd.concat({m: c[["mean_pred", "actual_ctr"]] for m, c in calibration.items()}, axis=1)
    calibration_wide.columns = [f"{m}: {stat}" for m, stat in calibration_wide.columns]
    calibration_wide.index = calibration_wide.index + 1
    calibration_wide.index.name = "prediction decile"
    importance = pd.DataFrame(importances)
    importance = pd.DataFrame({"mean over blocks": importance.mean(axis=1),
                               f"last block ({importance.columns[-1]})": importance.iloc[:, -1]})
    importance = importance.sort_values("mean over blocks", ascending=False)
    timing = pd.DataFrame(timings).groupby("model", sort=False)["fit_seconds"].agg(["mean", "max"])
    timing.columns = ["mean fit seconds", "max fit seconds"]

    # --- Step 4: write results ---
    fig, ax = plt.subplots(figsize=(5.5, 5))
    lim = [0, max(c["mean_pred"].max() for c in calibration.values()) * 1.05]
    ax.plot(lim, lim, color="#999999", linestyle="--", linewidth=1, label="perfect calibration")
    for m, color in [("logistic_regression", "#eb6834"), ("lightgbm", "#2a78d6"), ("factorization_machine", "#8a5cd6"),
                     ("v1_blend", "#1baf7a")]:
        ax.plot(calibration[m]["mean_pred"], calibration[m]["actual_ctr"], marker="o", color=color, label=m)
    ax.set(xlabel="mean predicted CTR (per decile of predictions)", ylabel="actual CTR", xlim=lim, ylim=lim,
           title="Calibration, held-out Oct 29-30 (walk-forward)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "calibration.png", dpi=150)
    plt.close(fig)

    summary = {
        "tuning": {"lightgbm_rounds": best_rounds, "factorization_machine_epochs": best_epochs,
                   "logistic_regression_C": best_C,
                   "logistic_regression_val_logloss_by_C": {str(c): v for c, (_, v) in lr_grid.items()}},
        "by_period": {label: by_period.loc[label].to_dict(orient="index") for label in periods},
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))

    md = f"""# V1 results — walk-forward, retrained every 6 hours

- V1 = blend of LightGBM and the factorization machine (50/50 in log-odds); constant and logistic regression are baselines
- every 6 hours (00, 06, 12, 18h) all models are retrained on all rows before that hour and predict the next 6 hours
- Oct 21-23 are warm-up only (the first model trains on 3 days); {pred['block'].nunique()} blocks are scored, Oct 24 00h to Oct 30 00h
- development = blocks we made design choices on; held-out = blocks kept back for the final numbers; last 24 hours = the
  most-trained models; metrics pool all rows of a period (each row counts once)
- tree count and FM passes picked once on Oct 28 (train Oct 21-27, {len(df[df['ts'] < TUNE_VAL_DAY]):,} rows): LightGBM {best_rounds} rounds,
  FM {best_epochs:.2f} passes; logistic regression C: {best_C} (Oct 28 logloss by C: {', '.join(f'{c}: {v:.4f}' for c, (_, v) in lr_grid.items())})

## 1. By period (rows pooled)
{to_md(by_period)}

## 2. V1 per day
{to_md(v1_per_day)}

## 3. V1 per 6-hour block
{to_md(v1_per_block)}

### Spread across blocks
{to_md(block_spread)}

## 4. All models per day
{to_md(per_day)}

## 5. Tuning day (Oct 28: train Oct 21-27, early stopping on Oct 28)
{to_md(tune_metrics)}

## 6. Calibration, held-out (10 equal-size bins by prediction)
![calibration](calibration.png)

{to_md(calibration_wide)}

## 7. LightGBM feature importance (share of total gain)
{to_md(importance)}

## 8. Training time per retrain
{to_md(timing)}
"""
    (OUT / "metrics.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
