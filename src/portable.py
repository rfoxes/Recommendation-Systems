"""V1 scoring from exported files (python -m src.export), without pandas or PyTorch.

The reference implementation for serving: it reproduces V1Model.predict_components exactly, and the export
checks that on real rows. A row is a plain dict of raw feature values keyed by manifest["input_columns"];
missing keys and None count as missing values (unseen categories, no history).
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .serving import NumpyFM


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def blend(p_lightgbm, p_fm):
    return 1 / (1 + np.exp(-(logit(p_lightgbm) + logit(p_fm)) / 2))


def _lookup(spec):
    return dict(zip(spec["values"], spec["codes"]))


def _number(value):
    return np.nan if value is None else float(value)


class PortableV1:
    def __init__(self, directory):
        directory = Path(directory)
        enc = json.loads((directory / "encoders.json").read_text())
        self.booster = lgb.Booster(model_file=str(directory / "lightgbm.txt"))
        self.fm = NumpyFM(directory / "fm_weights.npz")

        trees = enc["lightgbm"]
        self.categorical, self.flags, self.numeric = trees["categorical"], trees["flags"], trees["numeric"]
        self.codes = {col: _lookup(spec) for col, spec in trees["codes"].items()}
        self.freq = {col: (_lookup(spec["counts"]), np.asarray(spec["decile_edges"]))
                     for col, spec in trees["frequency_deciles"].items()}

        fm = enc["factorization_machine"]
        self.fm_codes = {col: _lookup(spec) for col, spec in fm["codes"].items()}
        self.fm_flags = fm["flags"]
        self.fm_bins = {col: np.asarray(edges) for col, edges in fm["bin_edges"].items()}
        self.fm_dense = fm["dense"]
        self.fm_mean, self.fm_std = np.asarray(fm["dense_mean"]), np.asarray(fm["dense_std"])

    def lightgbm_matrix(self, rows):
        out = []
        for row in rows:
            values = [self.codes[col].get(row.get(col), 0) for col in self.categorical]
            for col, (counts, edges) in self.freq.items():
                freq = counts.get(row.get(col), 0)
                values.append(0 if freq == 0 else int(np.searchsorted(edges, freq, side="right")) + 1)
            values += [_number(row.get(col)) for col in self.flags + self.numeric]
            out.append(values)
        return np.asarray(out, dtype="float64")

    def fm_arrays(self, rows):
        idx, dense = [], []
        for row in rows:
            ids = [codes.get(row.get(col), 0) for col, codes in self.fm_codes.items()]
            ids += [int(row.get(col) or 0) for col in self.fm_flags]
            for col, edges in self.fm_bins.items():
                value = _number(row.get(col))
                ids.append(0 if np.isnan(value) else 1 if value == 0 else int(np.searchsorted(edges, value, side="right")) + 2)
            idx.append(ids)
            dense.append([_number(row.get(col)) for col in self.fm_dense])
        rates = (logit(np.asarray(dense, dtype="float64")) - self.fm_mean) / self.fm_std
        return np.asarray(idx, dtype="int32"), np.clip(np.nan_to_num(rates), -5, 5).astype("float32")

    def predict_components(self, rows):
        """(LightGBM probability, factorization machine probability, V1 blend probability), one per row."""
        p_lightgbm = self.booster.predict(self.lightgbm_matrix(rows))
        p_fm = self.fm.predict(*self.fm_arrays(rows))
        return p_lightgbm, p_fm, blend(p_lightgbm, p_fm)
