# Serving latency per request (5 candidates, 2,000 real requests from Oct 29, one process, no PyTorch)

- V1 trained on the days before Oct 29; FM weights exported to plain arrays (`python -m src.fm export`) and scored with numpy
- numpy FM vs PyTorch FM, largest difference in predicted probability: 2.7e-07
- not measured here: network, feature-store lookups, retrieval (see docs/recording_notes.md, Goal 5)

|                       |   p50 ms |   p95 ms |   p99 ms |   max ms |
|:----------------------|---------:|---------:|---------:|---------:|
| LightGBM feature rows |   2.581  |   2.8129 |   3.2365 |   8.2912 |
| LightGBM trees        |   0.2624 |   0.3383 |   0.3736 |   0.5745 |
| FM feature rows       |   2.418  |   2.7492 |   3.0401 |  10.6957 |
| FM arithmetic (numpy) |   0.0213 |   0.0347 |   0.0395 |   0.095  |
| blend + score         |   0.009  |   0.0125 |   0.0144 |   0.0827 |
| total                 |   5.3252 |   5.8485 |   6.8773 |  13.6478 |
