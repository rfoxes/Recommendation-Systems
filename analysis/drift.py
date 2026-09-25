"""Goal 4: temporal shifts in click behavior, character cohort mix (genre x safety tier), novelty, fatigue and feature distributions, on the design days only (before the held-out days). Run: .venv/bin/python -m analysis.drift"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr

from src.config import HELD_OUT_START
from src.data import build_frame
from src.history import key_history
from src.metrics import wilson_ci

from .common import OUT, to_md, write

PSI_COLUMNS = ["cohort", "publisher_id", "C17", "C14", "device_type", "device_conn_type", "banner_pos", "is_app"]

df = build_frame()
df = df[df["ts"] < HELD_OUT_START].copy()
df["cohort"] = df["genre"] + " / " + df["safety_tier"]
df["day_label"] = df["day"].dt.strftime("%m-%d")
full = df


def psi(expected, actual, eps=1e-4):
    e, a = expected.align(actual, fill_value=0)
    e, a = (e / e.sum()).clip(lower=eps), (a / a.sum()).clip(lower=eps)
    return float(((a - e) * np.log(a / e)).sum())


def top_k(series, k=50):
    top = series.value_counts().index[:k]
    return series.where(series.isin(top), "other").astype(str)


def ctr_table(frame, by):
    t = frame.groupby(by, observed=True)["click"].agg(impressions="size", clicks="sum")
    t["ctr"] = t["clicks"] / t["impressions"]
    t["ci_low"], t["ci_high"] = wilson_ci(t["clicks"], t["impressions"])
    return t.drop(columns="clicks")


# --- Step 1: click behavior over time, overall and per cohort ---
daily = ctr_table(df, "day_label")
cohort_share = full.groupby("cohort").size().sort_values(ascending=False) / len(full)
dominant = cohort_share.index[:5].tolist()
cohort_ctr = full.pivot_table(index="cohort", columns="day_label", values="click", aggfunc="mean").loc[cohort_share.index]
cohort_drift = []
for cohort, part in full.groupby("cohort"):
    table = pd.crosstab(part["day_label"], part["click"])
    daily_ctr = part.groupby("day_label")["click"].mean()
    cohort_drift.append({"cohort": cohort, "impressions": len(part), "share": len(part) / len(full),
                         "ctr": part["click"].mean(), "min_day_ctr": daily_ctr.min(), "max_day_ctr": daily_ctr.max(),
                         "p_value_ctr_differs_by_day": chi2_contingency(table)[1]})
cohort_drift = pd.DataFrame(cohort_drift).set_index("cohort").sort_values("impressions", ascending=False)

# --- Step 2: character mix over time ---
share_by_day = (full.groupby(["cohort", "day_label"]).size() / full.groupby("day_label").size()).unstack().loc[cohort_share.index]
first_day = full["day_label"].min()
mix_psi = pd.Series({d: psi(full.loc[full["day_label"] == first_day, "cohort"].value_counts(),
                            full.loc[full["day_label"] == d, "cohort"].value_counts())
                     for d in sorted(full["day_label"].unique())}, name="cohort mix PSI vs first day")

per_char_day = full.groupby(["day_label", "character_id"]).size().unstack(fill_value=0)
days = per_char_day.index.tolist()
churn = pd.DataFrame([{"from": a, "to": b, "spearman_character_impressions": spearmanr(per_char_day.loc[a], per_char_day.loc[b])[0]}
                      for a, b in zip(days[:-1], days[1:])]).set_index(["from", "to"])
top100_share = per_char_day.apply(lambda row: row.nlargest(100).sum() / row.sum(), axis=1).rename("share of impressions from top-100 characters")
new_char_share = full.groupby("day_label").apply(
    lambda part: (part["created_at"] >= pd.Timestamp("2014-10-21")).mean()).rename("share from characters created Oct 21+")
mix = pd.concat([mix_psi, top100_share, new_char_share], axis=1)

# --- Step 3: novelty ---
df["character_age_bucket"] = pd.cut(df["character_age_days"], [-1, 7, 30, 90, 180, 10_000],
                                    labels=["0-7", "8-30", "31-90", "91-180", "181+"])
character_novelty = ctr_table(df, "character_age_bucket")
first_seen = df.groupby("C14")["hour_idx"].transform("min")
uncensored = first_seen > df["hour_idx"].min() + 24  # creatives first seen after the first day (their true start is visible)
creative_age = (df["hour_idx"] - first_seen)[uncensored]
creative_novelty = ctr_table(df[uncensored].assign(
    creative_age=pd.cut(creative_age, [-1, 5, 23, 47, 71, 10_000], labels=["0-5 h", "6-23 h", "day 2", "day 3", "day 4+"])),
    "creative_age")

# --- Step 4: fatigue per cohort ---
exposures = key_history(df, ["user", "cohort"], window=24)["hist_n"].to_numpy()
df["cohort_exposures_24h"] = pd.cut(exposures, [-1, 0, 1, 4, 1e9], labels=["0", "1", "2-4", "5+"])
returning = df[df["user_seen_before"] == 1]
fatigue = ctr_table(returning, "cohort_exposures_24h")
last_day = full["day_label"].max()
dominant_last_day = full.loc[full["day_label"] == last_day, "cohort"].value_counts().index[:5].tolist()
fatigue_dominant = returning[returning["cohort"].isin(dominant_last_day)].pivot_table(
    index="cohort", columns="cohort_exposures_24h", values="click", aggfunc="mean", observed=True).loc[dominant_last_day]
seen_creative = key_history(df, ["user", "C14"], window=24)["hist_n"].to_numpy() > 0
repeat_ad = ctr_table(df.assign(saw_this_ad_24h=seen_creative)[df["user_seen_before"] == 1], "saw_this_ad_24h")
concentration = (full.groupby(["cohort", "day_label", "C14"]).size().groupby(level=[0, 1])
                 .apply(lambda s: s.nlargest(5).sum() / s.sum()).groupby(level=0).mean()
                 .rename("share of a cohort-day's impressions on its top-5 creatives").loc[cohort_share.index])

# --- Step 5: feature distribution shift, each day vs the first day ---
shift = pd.DataFrame({col: {d: psi(top_k(full[col]).loc[full["day_label"] == first_day].value_counts(),
                                   top_k(full[col]).loc[full["day_label"] == d].value_counts())
                            for d in sorted(full["day_label"].unique())} for col in PSI_COLUMNS})

# --- Step 6: figure and results ---
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(daily.index, daily["ctr"], color="#0b0b0b", linewidth=2.5, label="all traffic")
for cohort, color in zip(dominant, ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#c9a227"]):
    ax.plot(cohort_ctr.columns, cohort_ctr.loc[cohort], color=color, marker="o", markersize=3, label=cohort)
ax.set(xlabel="day (design days only)", ylabel="CTR", title="Daily CTR, all traffic and the 5 largest cohorts")
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "drift_daily_ctr.png", dpi=150)
plt.close(fig)

md = f"""# Drift — temporal shifts, cohorts (genre x safety tier), novelty, fatigue

- design days only (Oct 21-28); the held-out days (Oct 29-30) are excluded
- cohort = genre x safety tier ({df['cohort'].nunique()} cohorts)
- PSI (population stability index): < 0.1 small, 0.1-0.25 moderate, > 0.25 large shift

## 1. Click behavior over time
![daily CTR](drift_daily_ctr.png)

{to_md(daily)}

### Per cohort (sorted by traffic): does CTR differ by day beyond noise?
{to_md(cohort_drift)}

- cohorts whose CTR differs by day at p < 0.01: {int((cohort_drift['p_value_ctr_differs_by_day'] < 0.01).sum())} of {len(cohort_drift)}

### Daily CTR of the 10 largest cohorts
{to_md(cohort_ctr.head(10))}

## 2. Character mix over time
### Share of impressions per cohort by day (10 largest)
{to_md(share_by_day.head(10))}

{to_md(mix)}

### Day-to-day churn of character popularity (Spearman correlation of impressions per character)
{to_md(churn)}

## 3. Novelty
### CTR by character age at impression (days since created_at)
{to_md(character_novelty)}

### CTR by creative age (hours since first seen; creatives first seen after the first day only)
{to_md(creative_novelty)}

## 4. Fatigue per cohort (returning users)
### CTR by the user's impressions with the same cohort in the previous 24h
{to_md(fatigue)}

### Same, for the 5 largest cohorts on the last design day ({last_day}), i.e. the dominant cohorts going into the held-out days
{to_md(fatigue_dominant)}

### Returning users: CTR by whether they saw this creative in the previous 24h
{to_md(repeat_ad)}

### Exposure concentration (mean over days)
{to_md(concentration.to_frame())}

## 5. Feature distribution shift (PSI of each day vs Oct 21; top-50 values + other)
{to_md(shift)}
"""
write("drift", md)
