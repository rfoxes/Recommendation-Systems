"""Missing values, duplicates, placeholder values, impossible values and num_interactions sanity checks. Run: .venv/bin/python eda/03_anomalies.py"""
import pandas as pd

from common import APP_PLACEHOLDER, NULL_DEVICE_ID, SITE_PLACEHOLDER, df_to_md, load_characters, load_joined, write_result

df = load_joined()
ch = load_characters()
WINDOW_START, WINDOW_END = df["ts"].min(), df["ts"].max()

# --- Step 1: missing values and duplicates ---
basic = pd.DataFrame({
    "rows": [len(df), len(ch)],
    "cells_missing": [int(df.drop(columns=["genre"]).isna().sum().sum()), int(ch.isna().sum().sum())],
    "duplicate_keys": [int(df["id"].duplicated().sum()), int(ch["character_id"].duplicated().sum())],
    "fully_duplicate_rows_excl_key": [int(df.drop(columns=["id"]).duplicated().sum()),
                                      int(ch.drop(columns=["character_id"]).duplicated().sum())],
}, index=["impressions", "characters"])

# --- Step 2: placeholder values ---
PLACEHOLDERS = {
    f"device_id == {NULL_DEVICE_ID}": df["device_id"] == NULL_DEVICE_ID,
    f"site_id == {SITE_PLACEHOLDER}": df["site_id"] == SITE_PLACEHOLDER,
    f"app_id == {APP_PLACEHOLDER}": df["app_id"] == APP_PLACEHOLDER,
    "both site and app placeholder": (df["site_id"] == SITE_PLACEHOLDER) & (df["app_id"] == APP_PLACEHOLDER),
    "neither site nor app placeholder": (df["site_id"] != SITE_PLACEHOLDER) & (df["app_id"] != APP_PLACEHOLDER),
    "C20 == -1": df["C20"] == -1,
}
placeholder_rows = []
for name, mask in PLACEHOLDERS.items():
    placeholder_rows.append({
        "check": name,
        "rows": int(mask.sum()),
        "share": mask.mean(),
        "ctr_when_true": df.loc[mask, "click"].mean() if mask.any() else float("nan"),
        "ctr_when_false": df.loc[~mask, "click"].mean(),
    })
placeholders = pd.DataFrame(placeholder_rows).set_index("check")

site_traffic = df["app_id"] == APP_PLACEHOLDER
app_traffic = df["site_id"] == SITE_PLACEHOLDER
companion = pd.DataFrame({
    "app traffic (site_id placeholder)": [df.loc[app_traffic, c].nunique() for c in ["site_domain", "site_category"]]
                                         + [df.loc[app_traffic, c].nunique() for c in ["app_domain", "app_category"]],
    "site traffic (app_id placeholder)": [df.loc[site_traffic, c].nunique() for c in ["site_domain", "site_category"]]
                                         + [df.loc[site_traffic, c].nunique() for c in ["app_domain", "app_category"]],
}, index=["site_domain distinct", "site_category distinct", "app_domain distinct", "app_category distinct"])

# --- Step 3: impossible values ---
df["character_age_days"] = (df["ts"].dt.normalize() - df["created_at"]).dt.days
before_creation = df["character_age_days"] < 0
impossible = pd.DataFrame([
    {"check": "impression before character created_at", "rows": int(before_creation.sum()),
     "characters_affected": df.loc[before_creation, "character_id"].nunique()},
    {"check": "conversation_turn > session_msg_count", "rows": int((df["conversation_turn"] > df["session_msg_count"]).sum()),
     "characters_affected": df.loc[df["conversation_turn"] > df["session_msg_count"], "character_id"].nunique()},
    {"check": "conversation_turn < 1", "rows": int((df["conversation_turn"] < 1).sum()),
     "characters_affected": df.loc[df["conversation_turn"] < 1, "character_id"].nunique()},
    {"check": "session_msg_count < 1", "rows": int((df["session_msg_count"] < 1).sum()),
     "characters_affected": df.loc[df["session_msg_count"] < 1, "character_id"].nunique()},
    {"check": "num_interactions < 0", "rows": int((df["num_interactions"] < 0).sum()),
     "characters_affected": int((ch["num_interactions"] < 0).sum())},
]).set_index("check")

created_in_window = ch["created_at"] >= WINDOW_START.normalize()
age_summary = df["character_age_days"].describe(percentiles=[0.01, 0.1, 0.5, 0.9]).to_frame("character_age_days_at_impression")
before_detail = (df.loc[before_creation, "character_age_days"].describe().to_frame("days (negative = before creation)")
                 if before_creation.any() else pd.DataFrame())

# --- Step 4: num_interactions vs what happened in the window ---
per_char = df.groupby("character_id").agg(impressions=("click", "size"), clicks=("click", "sum"))
per_char = ch.set_index("character_id").join(per_char, how="left").fillna({"impressions": 0, "clicks": 0})
per_char["ctr"] = per_char["clicks"] / per_char["impressions"].where(per_char["impressions"] > 0)
per_char["age_at_window_end_days"] = (WINDOW_END.normalize() - per_char["created_at"]).dt.days
per_char["created_in_window"] = per_char["created_at"] >= WINDOW_START.normalize()

enough = per_char[per_char["impressions"] >= 100]
spearman = pd.DataFrame({
    "spearman_with_num_interactions": [
        per_char["num_interactions"].corr(per_char["impressions"], method="spearman"),
        per_char["num_interactions"].corr(per_char["clicks"], method="spearman"),
        enough["num_interactions"].corr(enough["ctr"], method="spearman"),
        per_char["num_interactions"].corr(per_char["age_at_window_end_days"], method="spearman"),
    ],
    "characters": [len(per_char), len(per_char), len(enough), len(per_char)],
}, index=["impressions in window", "clicks in window", "CTR in window (chars with >= 100 impressions)",
          "age at window end (days)"])

by_cohort = per_char.groupby("created_in_window").agg(
    characters=("num_interactions", "size"),
    num_interactions_median=("num_interactions", "median"),
    num_interactions_mean=("num_interactions", "mean"),
    impressions_median=("impressions", "median"),
)
by_cohort.index = by_cohort.index.map({True: "created Oct 21-29", False: "created before Oct 21"})

per_char["age_bucket"] = pd.cut(per_char["age_at_window_end_days"], [-1, 9, 30, 90, 180, 365],
                                labels=["0-9 (in window)", "10-30", "31-90", "91-180", "181-365"])
by_age = per_char.groupby("age_bucket", observed=True).agg(
    characters=("num_interactions", "size"),
    num_interactions_median=("num_interactions", "median"),
    impressions_median=("impressions", "median"),
    ctr_mean=("ctr", "mean"),
)

# --- Step 5: write results ---
md = f"""# 03 — Anomalies and placeholders

## 1. Missing values and duplicates
{df_to_md(basic)}

## 2. Placeholder values
{df_to_md(placeholders)}

### Companion columns on each traffic type (distinct values)
{df_to_md(companion)}

## 3. Impossible values
{df_to_md(impossible)}

- characters created inside the window (Oct 21-29): {int(created_in_window.sum())}

### Character age at impression (days)
{df_to_md(age_summary)}

{("### Impressions before creation: days" + chr(10) + df_to_md(before_detail)) if before_creation.any() else ""}

## 4. num_interactions vs activity in the window
{df_to_md(spearman)}

### By creation cohort
{df_to_md(by_cohort)}

### By character age at window end
{df_to_md(by_age)}
"""
write_result("03_anomalies", md)
