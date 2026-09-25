"""Serving-time scoring without PyTorch: the factorization machine as plain numpy arithmetic on exported weights."""
import numpy as np


class NumpyFM:
    """Factorization machine forward pass on weights exported by `python -m src.fm export` (same math as src/fm.py)."""

    def __init__(self, path):
        w = np.load(path)
        self.embedding, self.linear = w["embedding"], w["linear"]
        self.dense_embedding, self.dense_linear = w["dense_embedding"], w["dense_linear"]
        self.bias = float(w["bias"][0])
        self.sizes, self.offsets = w["sizes"], w["offsets"]

    def predict(self, idx, dense):
        ids = np.minimum(idx, self.sizes - 1) + self.offsets
        v = np.concatenate([self.embedding[ids], dense[:, :, None] * self.dense_embedding[None]], axis=1)
        logit = (self.bias + self.linear[ids].sum(1) + (dense * self.dense_linear).sum(1)
                 + 0.5 * (v.sum(1) ** 2 - (v ** 2).sum(1)).sum(1))
        return 1 / (1 + np.exp(-logit))
