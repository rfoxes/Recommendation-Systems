# CTR prediction and ad ranking for AI companion chat

A click model, an ad ranker with business rules, and a cold-start and fatigue layer, built on `impressions.csv` and `characters.csv`.

## Setup

Python 3.14. Put `impressions.csv` and `characters.csv` in the repo root, then:

```bash
python3 -m venv .venv
.venv/bin/pip3 install -r requirements.txt
```

## Run

```bash
.venv/bin/python -m src.run_v1        # Goal 1: click model          -> results/v1/
.venv/bin/python -m src.run_ranking   # Goal 2: ranking              -> results/ranking/
.venv/bin/python -m src.run_layer     # Goals 3-4 + sample output    -> results/layer/, results/sample_output.md
```

Run `src.run_v1` first: the other two steps use the model settings it tunes.

Each step takes about 6 minutes, because it replays a whole week offline: the models are retrained every 6 hours on all earlier data and predict the next 6 hours, and Oct 29–30 are held out. Serving a single request is about 7 ms (`analysis/latency.py`).

## Experiments

Each script in `eda/` and `analysis/` backs one design decision and writes its results to `eda/results/` or `analysis/results/`. They're standalone, and some read the pipeline's outputs, so run them after the pipeline. For example:

```bash
.venv/bin/python eda/01_temporal_split.py
.venv/bin/python -m analysis.error_segments
```

## Layout

- **`src/`: the pipeline.**
  - Feature extraction: `data.py`, `history.py`, `features.py`.
  - Models: `models.py`, `fm.py`, `serving.py`.
  - Evaluation: `metrics.py`.
  - Ranking logic: `ranking.py`.
  - Pipeline steps: `run_*.py`, plus `sample_output.py`.
- **`eda/`, `analysis/`:** the experiments behind each decision.
- **`results/`:** metrics and the sample output.
- **`docs/recording_notes.md`:** the walkthrough script.
# Recommendation-Systems
