# 00 — Dataset overview

- impressions: **1,000,000** rows, 222 distinct hours, 2014-10-21 00:00:00 → 2014-10-30 05:00:00
- overall CTR: **0.1804**
- characters: **5,000** rows; created_at 2014-01-01 → 2014-10-29
- impressions whose character_id is missing from characters.csv: 0; characters never shown: 3

## Volume and CTR per day
| ts         |    rows |    ctr |   hours_covered |
|:-----------|--------:|-------:|----------------:|
| 2014-10-21 | 111,065 | 0.1792 |              24 |
| 2014-10-22 | 143,836 | 0.1666 |              24 |
| 2014-10-23 | 104,452 | 0.1916 |              24 |
| 2014-10-24 |  89,378 | 0.1843 |              24 |
| 2014-10-25 |  90,218 | 0.1925 |              24 |
| 2014-10-26 | 103,948 | 0.195  |              24 |
| 2014-10-27 |  86,788 | 0.1955 |              24 |
| 2014-10-28 | 142,909 | 0.1654 |              24 |
| 2014-10-29 | 104,450 | 0.1727 |              24 |
| 2014-10-30 |  22,956 | 0.1655 |               6 |

Note: Oct 30 only covers hours 00–05.

## Cardinality per column
|                   |   distinct_values | top_value   |   top_share |
|:------------------|------------------:|:------------|------------:|
| hour              |               222 | 14102209    |      0.0122 |
| click             |                 2 | 0           |      0.8196 |
| banner_pos        |                 7 | 0           |      0.7158 |
| site_id           |             2,663 | 85f751fd    |      0.3559 |
| site_domain       |             2,905 | c4e18dd6    |      0.3693 |
| site_category     |                21 | 50e219e0    |      0.4044 |
| app_id            |             3,139 | ecad2386    |      0.6441 |
| app_domain        |               208 | 7801e8d9    |      0.6807 |
| app_category      |                27 | 07d7df22    |      0.6529 |
| device_id         |           152,548 | a99f214a    |      0.8218 |
| device_ip         |           547,394 | 6b9769f2    |      0.0052 |
| device_model      |             5,174 | 8a4875bd    |      0.0607 |
| device_type       |                 4 | 1           |      0.9205 |
| device_conn_type  |                 4 | 0           |      0.8643 |
| C1                |                 7 | 1005        |      0.9163 |
| C14               |             2,133 | 4687        |      0.0248 |
| C15               |                 8 | 320         |      0.9329 |
| C16               |                 9 | 50          |      0.9433 |
| C17               |               400 | 1722        |      0.1188 |
| C18               |                 4 | 0           |      0.4106 |
| C19               |                65 | 35          |      0.3092 |
| C20               |               162 | -1          |      0.4683 |
| C21               |                57 | 23          |      0.2179 |
| character_id      |             4,997 | dfcea2a88c  |      0.0014 |
| conversation_turn |               113 | 1           |      0.2201 |
| session_msg_count |               139 | 1           |      0.0799 |

## Characters: categorical columns
| safety_tier   |   count |
|:--------------|--------:|
| sfw           |   2,936 |
| suggestive    |   1,302 |
| mature        |     762 |

| creator_type   |   count |
|:---------------|--------:|
| community      |   4,508 |
| official       |     492 |

| character_name   |   genre (name prefix) |
|:-----------------|----------------------:|
| horror           |                   533 |
| historical       |                   514 |
| romance          |                   513 |
| slice            |                   504 |
| comedic          |                   502 |
| mystery          |                   494 |
| mentor           |                   494 |
| anime            |                   489 |
| sci              |                   482 |
| fantasy          |                   475 |
