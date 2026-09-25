"""
Factorization machine (PyTorch), always run as its own process: PyTorch and LightGBM crash when sharing one process on macOS.
Run: python -m src.fm fit <inputs.npz> <outputs.npz> <settings json>
     python -m src.fm predict <model.pt> <inputs.npz> <outputs.npz>
     python -m src.fm export <model.pt> <weights.npz>   (plain arrays for serving without PyTorch, see src/serving.py)
"""
import json
import math
import sys
import time

import numpy as np
import torch

EVALS_PER_EPOCH = 4
PATIENCE = 2


class FactorizationMachine(torch.nn.Module):
    """
    Categorical fields: one weight + one embedding per value. Dense fields: one weight + one embedding per field,
    scaled by the value. logit = bias + linear terms + sum over all field pairs of <v_i, v_j>.
    """

    def __init__(self, n_values, n_dense, dim):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.linear = torch.nn.Embedding(n_values, 1)
        self.embedding = torch.nn.Embedding(n_values, dim)
        self.dense_linear = torch.nn.Parameter(torch.zeros(n_dense))
        self.dense_embedding = torch.nn.Parameter(torch.randn(n_dense, dim) * 0.01)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, idx, dense):
        v = torch.cat([self.embedding(idx), dense.unsqueeze(-1) * self.dense_embedding], dim=1)
        pairwise = 0.5 * (v.sum(1).pow(2) - v.pow(2).sum(1)).sum(1)
        linear = self.linear(idx).sum(dim=(1, 2)) + (dense * self.dense_linear).sum(1)
        return self.bias + linear + pairwise, v


def _to_tensors(idx, dense, sizes, offsets):
    """Unseen value ids beyond the training vocabulary map to the field's last known id (always the 'other' code 0 in practice)."""
    return (torch.as_tensor(np.minimum(idx, sizes - 1) + offsets, dtype=torch.long),
            torch.as_tensor(dense, dtype=torch.float32))


def _predict(model, idx, dense):
    if len(idx) == 0:
        return np.array([])
    model.eval()
    with torch.no_grad():
        logits = [model(i, d)[0] for i, d in zip(idx.split(65536), dense.split(65536))]
        return torch.sigmoid(torch.cat(logits)).numpy()


def _logloss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_fm(idx_train, dense_train, y_train, idx_stop, dense_stop, y_stop, idx_eval, dense_eval,
           seed, dim, lr, l2, batch_size, max_epochs, fixed_epochs=None, threads=8, save_path=None):
    """
    idx_*: int arrays (rows x categorical fields) of value ids; dense_*: float arrays (rows x dense fields).
    fixed_epochs=None: early stopping on the `stop` rows, checked EVALS_PER_EPOCH times per epoch.
    fixed_epochs=x: train exactly x epochs (rounded to quarter epochs), no early stopping.
    Returns predictions on stop and eval rows, epochs trained, seconds. Saves the model to save_path if given.
    """
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    start = time.perf_counter()
    sizes = idx_train.max(axis=0) + 1
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    x_train, d_train = _to_tensors(idx_train, dense_train, sizes, offsets)
    t_train = torch.as_tensor(y_train, dtype=torch.float32)
    x_stop, d_stop = _to_tensors(idx_stop, dense_stop, sizes, offsets)
    x_eval, d_eval = _to_tensors(idx_eval, dense_eval, sizes, offsets)

    model = FactorizationMachine(int(sizes.sum()), dense_train.shape[1], dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    target_steps = None if fixed_epochs is None else max(1, round(fixed_epochs * EVALS_PER_EPOCH))
    best_loss, best_state, best_step, bad, step, done = np.inf, None, 0, 0, 0, False
    for _ in range(math.ceil(max_epochs)):
        batches = torch.randperm(len(x_train)).split(batch_size)
        check_every = max(1, len(batches) // EVALS_PER_EPOCH)
        for i, batch in enumerate(batches, 1):
            model.train()
            logit, v = model(x_train[batch], d_train[batch])
            loss = loss_fn(logit, t_train[batch]) + l2 * v.pow(2).sum() / len(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % check_every:
                continue
            step += 1
            if target_steps is not None:
                done = step >= target_steps
            else:
                stop_loss = _logloss(y_stop, _predict(model, x_stop, d_stop))
                if stop_loss < best_loss:
                    best_loss, best_step, bad = stop_loss, step, 0
                    best_state = {k: t.clone() for k, t in model.state_dict().items()}
                else:
                    bad += 1
                done = bad >= PATIENCE
            if done:
                break
        if done:
            break
    if target_steps is None:
        model.load_state_dict(best_state)
        step = best_step
    if save_path is not None:
        torch.save({"state": model.state_dict(), "sizes": sizes, "offsets": offsets, "n_dense": dense_train.shape[1],
                    "dim": dim}, save_path)
    return _predict(model, x_stop, d_stop), _predict(model, x_eval, d_eval), step / EVALS_PER_EPOCH, time.perf_counter() - start


def predict_saved(model_path, idx, dense):
    saved = torch.load(model_path, weights_only=False)
    model = FactorizationMachine(int(saved["sizes"].sum()), saved["n_dense"], saved["dim"])
    model.load_state_dict(saved["state"])
    return _predict(model, *_to_tensors(idx, dense, saved["sizes"], saved["offsets"]))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "fit":
        data = np.load(sys.argv[2])
        keys = ["idx_train", "dense_train", "y_train", "idx_stop", "dense_stop", "y_stop", "idx_eval", "dense_eval"]
        p_stop, p_eval, epochs, seconds = fit_fm(*(data[k] for k in keys), **json.loads(sys.argv[4]))
        np.savez(sys.argv[3], p_stop=p_stop, p_eval=p_eval, epochs=epochs, seconds=seconds)
    elif mode == "predict":
        data = np.load(sys.argv[3])
        np.savez(sys.argv[4], p=predict_saved(sys.argv[2], data["idx"], data["dense"]))
    elif mode == "export":
        saved = torch.load(sys.argv[2], weights_only=False)
        state = {k: v.numpy() for k, v in saved["state"].items()}
        np.savez(sys.argv[3], embedding=state["embedding.weight"], linear=state["linear.weight"][:, 0],
                 dense_embedding=state["dense_embedding"], dense_linear=state["dense_linear"], bias=state["bias"],
                 sizes=saved["sizes"], offsets=saved["offsets"])
    else:
        raise SystemExit(f"unknown mode {mode!r}")
