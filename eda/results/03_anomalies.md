# 03 — Anomalies and placeholders

## 1. Missing values and duplicates
|             |      rows |   cells_missing |   duplicate_keys |   fully_duplicate_rows_excl_key |
|:------------|----------:|----------------:|-----------------:|--------------------------------:|
| impressions | 1,000,000 |               0 |                0 |                               2 |
| characters  |     5,000 |               0 |                0 |                               0 |

## 2. Placeholder values
| check                            |    rows |   share |   ctr_when_true |   ctr_when_false |
|:---------------------------------|--------:|--------:|----------------:|-----------------:|
| device_id == a99f214a            | 821,801 |  0.8218 |          0.1839 |           0.1646 |
| site_id == 85f751fd              | 355,851 |  0.3559 |          0.1339 |           0.2061 |
| app_id == ecad2386               | 644,149 |  0.6441 |          0.2061 |           0.1339 |
| both site and app placeholder    |       0 |  0      |        nan      |           0.1804 |
| neither site nor app placeholder |       0 |  0      |        nan      |           0.1804 |
| C20 == -1                        | 468,253 |  0.4683 |          0.2006 |           0.1627 |

### Companion columns on each traffic type (distinct values)
|                        |   app traffic (site_id placeholder) |   site traffic (app_id placeholder) |
|:-----------------------|------------------------------------:|------------------------------------:|
| site_domain distinct   |                                   1 |                               2,905 |
| site_category distinct |                                   1 |                                  21 |
| app_domain distinct    |                                 208 |                                   1 |
| app_category distinct  |                                  27 |                                   1 |

## 3. Impossible values
| check                                  |   rows |   characters_affected |
|:---------------------------------------|-------:|----------------------:|
| impression before character created_at |      0 |                     0 |
| conversation_turn > session_msg_count  |      0 |                     0 |
| conversation_turn < 1                  |      0 |                     0 |
| session_msg_count < 1                  |      0 |                     0 |
| num_interactions < 0                   |      0 |                     0 |

- characters created inside the window (Oct 21-29): 243

### Character age at impression (days)
|       |   character_age_days_at_impression |
|:------|-----------------------------------:|
| count |                             1e+06  |
| mean  |                           146.902  |
| std   |                            86.2789 |
| min   |                             0      |
| 1%    |                             2      |
| 10%   |                            25      |
| 50%   |                           147      |
| 90%   |                           266      |
| max   |                           302      |



## 4. num_interactions vs activity in the window
|                                               |   spearman_with_num_interactions |   characters |
|:----------------------------------------------|---------------------------------:|-------------:|
| impressions in window                         |                           0.7887 |        5,000 |
| clicks in window                              |                           0.7818 |        5,000 |
| CTR in window (chars with >= 100 impressions) |                           0.0212 |        2,345 |
| age at window end (days)                      |                           0.1011 |        5,000 |

### By creation cohort
| created_in_window     |   characters |   num_interactions_median |   num_interactions_mean |   impressions_median |
|:----------------------|-------------:|--------------------------:|------------------------:|---------------------:|
| created before Oct 21 |        4,757 |                       191 |                590.917  |                   68 |
| created Oct 21-29     |          243 |                        14 |                 52.9588 |                   35 |

### By character age at window end
| age_bucket      |   characters |   num_interactions_median |   impressions_median |   ctr_mean |
|:----------------|-------------:|--------------------------:|---------------------:|-----------:|
| 0-9 (in window) |          243 |                        14 |                 35   |     0.1819 |
| 10-30           |          343 |                       178 |                 95   |     0.1782 |
| 31-90           |          935 |                       200 |                 91   |     0.1815 |
| 91-180          |        1,455 |                       194 |                 89   |     0.1804 |
| 181-365         |        2,024 |                       188 |                 62.5 |     0.1802 |
