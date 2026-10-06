"""
Train V1 on every row and export it for serving: .venv/bin/python -m src.export [out_dir]   (default: artifacts/v1)

Writes, for a serving process without pandas or PyTorch (see src/portable.py):
  lightgbm.txt        LightGBM booster, text format
  fm_weights.npz      factorization machine weights (src/serving.py format)
  encoders.json       category codes, frequency deciles, numeric bins and rate scaling for both models
  golden_sample.json  real rows with V1's predictions, so any serving implementation can prove it matches
  manifest.json       feature schema, training window, tuning, source commit and file checksums
Uses the tuned settings in results/v1/metrics.json (run src.run_v1 first).
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CATEGORICAL, FLAGS, NUMERIC, RESULTS_DIR, ROOT, SEED
from .data import build_frame
from .models import V1Model
from .portable import PortableV1

GOLDEN_ROWS = 300
MAX_PREDICTION_DIFF = 1e-5  # numpy FM vs PyTorch FM float differences
INPUT_COLUMNS = list(dict.fromkeys(CATEGORICAL + FLAGS + NUMERIC))


def _codes_spec(series):
    return {"values": series.index.tolist(), "codes": series.to_numpy().tolist()}


def _encoders(model):
    trees, fm = model.encoder_, model.fm_.encoder_
    return {
        "lightgbm": {
            "categorical": trees.categorical,
            "flags": trees.flags,
            "numeric": trees.numeric,
            "columns": trees.columns,
            "codes": {col: _codes_spec(trees.codes_[col]) for col in trees.categorical},
            "frequency_deciles": {
                col: {"counts": _codes_spec(trees.counts_[col]), "decile_edges": trees.decile_edges_[col].tolist()}
                for col in trees.freq_decile
            },
        },
        "factorization_machine": {
            "codes": {col: _codes_spec(fm.codes.codes_[col]) for col in fm.codes.categorical},
            "flags": fm.codes.flags,
            "bin_edges": {col: edges.tolist() for col, edges in fm.edges_.items()},
            "dense": fm.dense,
            "dense_mean": fm.mean_.tolist(),
            "dense_std": fm.std_.tolist(),
        },
    }


def _records(frame):
    """Rows as plain JSON-able dicts: NaN -> None, numpy scalars -> Python numbers (full precision)."""
    return [{k: (None if pd.isna(v) else v.item() if hasattr(v, "item") else v) for k, v in row.items()}
            for row in frame.to_dict("records")]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    tuning = json.loads((RESULTS_DIR / "v1" / "metrics.json").read_text())["tuning"]
    df = build_frame()
    model = V1Model(tuning["lightgbm_rounds"], tuning["factorization_machine_epochs"]).fit(df)

    model.lightgbm_.booster_.save_model(str(out_dir / "lightgbm.txt"), num_iteration=model.lightgbm_rounds)
    model.fm_.export_weights(out_dir / "fm_weights.npz")
    (out_dir / "encoders.json").write_text(json.dumps(_encoders(model)))

    # Golden sample: V1's own predictions on real rows; the exported files must reproduce them.
    sample = df.sample(GOLDEN_ROWS, random_state=SEED)
    rows = _records(sample[INPUT_COLUMNS])
    expected = model.predict_components(sample)
    actual = PortableV1(out_dir).predict_components(rows)
    diff = max(float(np.abs(e - a).max()) for e, a in zip(expected, actual))
    assert diff < MAX_PREDICTION_DIFF, f"exported model disagrees with V1 by {diff:.2e}"
    (out_dir / "golden_sample.json").write_text(json.dumps({
        "rows": rows,
        "p_lightgbm": expected[0].tolist(), "p_fm": expected[1].tolist(), "p_v1": expected[2].tolist(),
        "tolerance": MAX_PREDICTION_DIFF,
    }))

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "src"], cwd=ROOT, capture_output=True, text=True).stdout)
    files = ["lightgbm.txt", "fm_weights.npz", "encoders.json", "golden_sample.json"]
    manifest = {
        "model": "v1: LightGBM + factorization machine, blended 50/50 in log-odds",
        "input_columns": INPUT_COLUMNS,
        "categorical": CATEGORICAL, "flags": FLAGS, "numeric": NUMERIC,
        "training": {"rows": len(df), "click_rate": float(df["click"].mean()),
                     "from": str(df["ts"].min()), "to": str(df["ts"].max())},
        "tuning": {"lightgbm_rounds": tuning["lightgbm_rounds"],
                   "factorization_machine_epochs": tuning["factorization_machine_epochs"]},
        "golden_sample_max_diff": diff,
        "source_commit": commit + ("-dirty" if dirty else ""),
        "files": {name: {"sha256": _sha256(out_dir / name), "bytes": (out_dir / name).stat().st_size} for name in files},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"exported V1 to {out_dir} (golden sample max diff {diff:.1e})")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts" / "v1")
