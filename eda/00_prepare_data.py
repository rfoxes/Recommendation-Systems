"""Convert raw CSVs to parquet and write a dataset overview. Run: .venv/bin/python eda/00_prepare_data.py"""
import pandas as pd

from common import (CHARACTERS_PARQUET, IMPRESSIONS_PARQUET, PROCESSED, RAW_CHARACTERS, RAW_IMPRESSIONS,
                    df_to_md, write_result)

# --- Step 1: read + parse + save ---
imp = pd.read_csv(RAW_IMPRESSIONS, dtype={"id": str, "hour": str})
imp["ts"] = pd.to_datetime(imp["hour"], format="%y%m%d%H")
ch = pd.read_csv(RAW_CHARACTERS, parse_dates=["created_at"])
PROCESSED.mkdir(parents=True, exist_ok=True)
imp.to_parquet(IMPRESSIONS_PARQUET, index=False)
ch.to_parquet(CHARACTERS_PARQUET, index=False)

# --- Step 2: overview ---
by_day = imp.groupby(imp["ts"].dt.date)["click"].agg(rows="size", ctr="mean")
by_day["hours_covered"] = imp.groupby(imp["ts"].dt.date)["hour"].nunique()
card = pd.DataFrame({
    "distinct_values": imp.nunique(),
    "top_value": [imp[c].value_counts().index[0] for c in imp.columns],
    "top_share": [imp[c].value_counts(normalize=True).iloc[0] for c in imp.columns],
}).drop(index=["id", "ts"])

# --- Step 3: join integrity ---
missing_chars = (~imp["character_id"].isin(ch["character_id"])).sum()
unused_chars = (~ch["character_id"].isin(imp["character_id"])).sum()

md = f"""# 00 — Dataset overview

- impressions: **{len(imp):,}** rows, {imp['hour'].nunique()} distinct hours, {imp['ts'].min()} → {imp['ts'].max()}
- overall CTR: **{imp['click'].mean():.4f}**
- characters: **{len(ch):,}** rows; created_at {ch['created_at'].min().date()} → {ch['created_at'].max().date()}
- impressions whose character_id is missing from characters.csv: {missing_chars}; characters never shown: {unused_chars}

## Volume and CTR per day
{df_to_md(by_day)}

Note: Oct 30 only covers hours 00–05.

## Cardinality per column
{df_to_md(card)}

## Characters: categorical columns
{df_to_md(ch['safety_tier'].value_counts().to_frame())}

{df_to_md(ch['creator_type'].value_counts().to_frame())}

{df_to_md(ch['character_name'].str.split('_').str[0].value_counts().rename('genre (name prefix)').to_frame())}
"""
write_result("00_prepare_data", md)
