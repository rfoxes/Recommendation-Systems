"""
Per-request serving latency for 5 candidates, one process, no PyTorch: feature rows, LightGBM, FM (numpy on exported
weights), blend + score. 2,000 real requests from the Oct 29 ranking replay. Run: .venv/bin/python -m analysis.latency
(needs results/v1/metrics.json and results/ranking/ranked_candidates.parquet)
"""
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR, TEST_DAYS
from src.data import build_frame
from src.features import logit
from src.models import V1Model
from src.serving import NumpyFM

from .common import to_md, write

N_REQUESTS = 2000

# --- Step 1: V1 trained on the days before Oct 29; FM weights exported to plain arrays ---
tuning = json.loads((RESULTS_DIR / "v1" / "metrics.json").read_text())["tuning"]
df = build_frame()
model = V1Model(tuning["lightgbm_rounds"], tuning["factorization_machine_epochs"]).fit(df[df["ts"] < TEST_DAYS[0]])
with tempfile.TemporaryDirectory() as tmp:
    weights = Path(tmp) / "fm_weights.npz"
    model.fm_.export_weights(weights)
    fm = NumpyFM(weights)

candidates = pd.read_parquet(RESULTS_DIR / "ranking" / "ranked_candidates.parquet")
candidates = candidates[candidates["day"] == TEST_DAYS[0]]
requests = [group for _, group in candidates.groupby("opportunity")][:N_REQUESTS]

# --- Step 2: the numpy FM gives the same predictions as the PyTorch FM ---
check = pd.concat(requests[:200])
max_diff = float(np.abs(fm.predict(*model.fm_.encoder_.transform(check)) - model.fm_.predict(check)).max())
assert max_diff < 1e-5, max_diff

# --- Step 3: time each stage per request ---
stages = {name: [] for name in ["LightGBM feature rows", "LightGBM trees", "FM feature rows", "FM arithmetic (numpy)",
                                "blend + score", "total"]}
for request in requests:
    t0 = time.perf_counter()
    X = model.encoder_.transform(request)
    t1 = time.perf_counter()
    p_lightgbm = model.lightgbm_.predict(X)
    t2 = time.perf_counter()
    idx, dense = model.fm_.encoder_.transform(request)
    t3 = time.perf_counter()
    p_fm = fm.predict(idx, dense)
    t4 = time.perf_counter()
    l_lightgbm, l_fm = logit(p_lightgbm), logit(p_fm)
    z = (l_lightgbm + l_fm) / 2
    u = np.abs(l_lightgbm - l_fm) / 2
    order = np.argsort(-z)
    t5 = time.perf_counter()
    for name, seconds in zip(stages, [t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t5 - t0]):
        stages[name].append(1000 * seconds)

table = pd.DataFrame({name: {"p50 ms": np.percentile(t, 50), "p95 ms": np.percentile(t, 95),
                             "p99 ms": np.percentile(t, 99), "max ms": np.max(t)} for name, t in stages.items()}).T

# --- Step 4: write results ---
md = f"""# Serving latency per request (5 candidates, {N_REQUESTS:,} real requests from Oct 29, one process, no PyTorch)

- V1 trained on the days before Oct 29; FM weights exported to plain arrays (`python -m src.fm export`) and scored with numpy
- numpy FM vs PyTorch FM, largest difference in predicted probability: {max_diff:.1e}
- not measured here: network, feature-store lookups, retrieval (see docs/recording_notes.md, Goal 5)

{to_md(table)}
"""
write("latency", md)
