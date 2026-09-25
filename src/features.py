"""Feature encodings fitted on training rows only: V1 trees (FeatureEncoder) and the factorization machine (FMEncoder)."""
import numpy as np
import pandas as pd

from .config import (CATEGORICAL, FLAGS, FM_DENSE, FM_MIN_COUNT, FM_NUMERIC_BINS, FREQ_DECILE, MIN_COUNT,
                     NUMERIC)


class FeatureEncoder:
    """
    Categorical: values with >= min_count training rows get codes 1..K; rare and unseen values get 0.
    Frequency decile: 10 bins holding ~10% of training rows each, by how often the value appears in training
    (1 = long tail, 10 = most common); unseen values get 0.
    """

    def __init__(self, categorical=CATEGORICAL, freq_decile=FREQ_DECILE, flags=FLAGS, numeric=NUMERIC,
                 min_count=MIN_COUNT):
        self.categorical = categorical
        self.freq_decile = freq_decile
        self.flags = flags
        self.numeric = numeric
        self.min_count = min_count

    @property
    def decile_columns(self):
        return [f"{col}_freq_decile" for col in self.freq_decile]

    @property
    def columns(self):
        return self.categorical + self.decile_columns + self.flags + self.numeric

    def fit(self, train):
        self.codes_, self.counts_, self.decile_edges_ = {}, {}, {}
        for col in self.categorical:
            counts = train[col].value_counts()
            kept = counts.index[counts >= self.min_count]
            self.codes_[col] = pd.Series(np.arange(1, len(kept) + 1, dtype="int32"), index=kept)
        for col in self.freq_decile:
            counts = train[col].value_counts()
            row_freq = train[col].map(counts).to_numpy()
            self.counts_[col] = counts
            self.decile_edges_[col] = np.unique(np.quantile(row_freq, np.linspace(0.1, 0.9, 9)))
        return self

    def transform(self, df):
        out = {}
        for col in self.categorical:
            out[col] = df[col].map(self.codes_[col]).fillna(0).astype("int32").to_numpy()
        for col in self.freq_decile:
            freq = df[col].map(self.counts_[col]).fillna(0).to_numpy()
            decile = np.searchsorted(self.decile_edges_[col], freq, side="right") + 1
            out[f"{col}_freq_decile"] = np.where(freq == 0, 0, decile).astype("int8")
        for col in self.flags + self.numeric:
            out[col] = df[col].to_numpy()
        return pd.DataFrame(out, index=df.index)[self.columns]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class FMEncoder:
    """
    Encoding for the factorization machine.
    Categorical + flags: codes for values with >= min_count training rows, rest 0.
    Numerics except click rates: 0 NULL, 1 ZERO, 2.. quantile bins of the nonzero training values.
    Click rates: dense standardized logits clipped to [-5, 5], missing -> 0.
    """

    def __init__(self, min_count=FM_MIN_COUNT, bins=FM_NUMERIC_BINS, dense=FM_DENSE):
        self.codes = FeatureEncoder(freq_decile=[], numeric=[], min_count=min_count)
        self.bins = bins
        self.dense = dense
        self.binned = [c for c in NUMERIC if c not in dense]

    def fit(self, train):
        self.codes.fit(train)
        self.edges_ = {}
        for col in self.binned:
            nonzero = train[col].dropna()
            self.edges_[col] = np.unique(np.quantile(nonzero[nonzero != 0], np.linspace(0, 1, self.bins + 1)[1:-1]))
        rates = logit(train[self.dense].to_numpy(dtype="float64"))
        self.mean_, self.std_ = np.nanmean(rates, axis=0), np.nanstd(rates, axis=0)
        return self

    def transform(self, rows):
        """Returns (value ids: int32 rows x categorical fields, dense: float32 rows x dense fields)."""
        codes = self.codes.transform(rows).to_numpy(dtype="int32")
        binned = []
        for col, edges in self.edges_.items():
            values = rows[col].to_numpy(dtype="float64")
            bins = np.searchsorted(edges, values, side="right") + 2
            binned.append(np.select([np.isnan(values), values == 0], [0, 1], bins))
        dense = (logit(rows[self.dense].to_numpy(dtype="float64")) - self.mean_) / self.std_
        return np.column_stack([codes] + binned).astype("int32"), np.clip(np.nan_to_num(dense), -5, 5).astype("float32")
