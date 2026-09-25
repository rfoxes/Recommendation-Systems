# Permuted-label control — V1 LightGBM, train Oct 21-27, evaluate Oct 28

- `click` shuffled across all 1,000,000 rows before any feature (history, recent click rates, encodings) is built
- a pipeline without leakage scores AUC ~0.5 and NE >= 1

|   permutation_seed |   trees |   logloss |     NE |    AUC |   AUC_within_publisher |   mean_pred |   actual_ctr |
|-------------------:|--------:|----------:|-------:|-------:|-----------------------:|------------:|-------------:|
|                  0 |       1 |    0.4702 | 1      | 0.4989 |                 0.4984 |      0.1806 |       0.1792 |
|                  1 |       5 |    0.4707 | 0.9999 | 0.504  |                 0.5032 |      0.1808 |       0.1796 |
|                  2 |       1 |    0.4744 | 1      | 0.4996 |                 0.4993 |      0.1799 |       0.182  |
|                  3 |       1 |    0.4725 | 1      | 0.5004 |                 0.4998 |      0.1807 |       0.1807 |
|                  4 |       1 |    0.4712 | 1      | 0.4982 |                 0.4984 |      0.1802 |       0.1799 |

- mean AUC 0.5002 (min 0.4982, max 0.5040)
