"""Shared setup for post-V1 analyses: the V1 tuning step (train Oct 21-27, validate Oct 28)."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import TUNE_VAL_DAY
from src.data import build_frame
from src.features import FeatureEncoder

OUT = Path(__file__).resolve().parent / "results"


def tune_step(label_permutation_seed=None):
    """Raw train/val rows, fitted encoder and encoded matrices for the V1 tuning step."""
    df = build_frame(label_permutation_seed)
    train, val = df[df["ts"] < TUNE_VAL_DAY], df[df["day"] == TUNE_VAL_DAY]
    encoder = FeatureEncoder().fit(train)
    return train, val, encoder, encoder.transform(train), encoder.transform(val)


def row_logloss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def to_md(df):
    """Markdown table: whole-number columns as integers, other floats to 4 decimals."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            whole = (out[col].dropna() % 1 == 0).all()
            out[col] = out[col].map(lambda v: f"{v:,.0f}" if whole else f"{v:.4f}")
    return out.to_markdown()


def write(name, markdown):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.md"
    path.write_text(markdown)
    print(f"wrote {path}")
