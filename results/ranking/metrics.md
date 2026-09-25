# Goal 2 — Candidate ranking, walk-forward from Oct 24 (640,647 opportunities, up to 5 candidates each)

- each day is replayed hour by hour with V1 retrained every 6 hours on all earlier rows; Oct 21-23 are warm-up only;
  development = days we made design choices on, held-out = days kept back, last 24 hours = the most-trained models
- candidates: the ad actually shown + up to 4 creatives from advertisers that accept the character's safety tier, that ran on
  the same publisher and slot size in the previous 24h (topped up with the same size from any publisher), drawn in proportion
  to their impressions in that window
- retrieval returns up to 10 such creatives in order; hour by hour, advertisers whose budget is already spent are skipped and
  the model receives the shown ad + the first 4 remaining
- the ranking always uses V1 (pointwise: sort by predicted click probability); other rows in section 1 are comparisons only
- rules: safety gate (simulated advertiser max tier), fatigue (repeat penalty per day = 2014-10-24: 0.746, 2014-10-25: 0.722, 2014-10-26: 0.707, 2014-10-27: 0.717, 2014-10-28: 0.723, 2014-10-29: 0.751, 2014-10-30: 0.758;
  hard cap 3 per user per creative per 24h), budget pacing (budget = advertiser's previous-day volume)
- uncertainty band: V1 log-odds +/- half the LightGBM-FM gap; toss-up = runner-up band overlaps the leader's

## 1. Click rate of the ad actually shown, by where each ranker placed it
|                                                                                                |   CTR when shown ad ranked 1st |   CTR when shown ad ranked 5th |   difference (pp) |
|:-----------------------------------------------------------------------------------------------|-------------------------------:|-------------------------------:|------------------:|
| ('development (Oct 24-Oct 28)', 'V1 (blend) - used for ranking')                               |                         0.2024 |                         0.1513 |            5.1175 |
| ('development (Oct 24-Oct 28)', 'LightGBM alone')                                              |                         0.2017 |                         0.1511 |            5.0613 |
| ('development (Oct 24-Oct 28)', 'FM alone')                                                    |                         0.2021 |                         0.1508 |            5.1256 |
| ('development (Oct 24-Oct 28)', 'benchmark, not used: recent creative CTR only')               |                         0.2022 |                         0.1479 |            5.4303 |
| ('development (Oct 24-Oct 28)', 'benchmark, not used: random')                                 |                         0.1813 |                         0.1912 |           -0.9918 |
| ('held-out (Oct 29-Oct 30)', 'V1 (blend) - used for ranking')                                  |                         0.2059 |                         0.1287 |            7.7138 |
| ('held-out (Oct 29-Oct 30)', 'LightGBM alone')                                                 |                         0.206  |                         0.1289 |            7.7156 |
| ('held-out (Oct 29-Oct 30)', 'FM alone')                                                       |                         0.2052 |                         0.1392 |            6.5983 |
| ('held-out (Oct 29-Oct 30)', 'benchmark, not used: recent creative CTR only')                  |                         0.1958 |                         0.1349 |            6.0932 |
| ('held-out (Oct 29-Oct 30)', 'benchmark, not used: random')                                    |                         0.178  |                         0.1706 |            0.7408 |
| ('last 24 hours (Oct 29 06:00-Oct 30 06:00)', 'V1 (blend) - used for ranking')                 |                         0.2072 |                         0.1252 |            8.1996 |
| ('last 24 hours (Oct 29 06:00-Oct 30 06:00)', 'LightGBM alone')                                |                         0.2075 |                         0.1257 |            8.182  |
| ('last 24 hours (Oct 29 06:00-Oct 30 06:00)', 'FM alone')                                      |                         0.2061 |                         0.1358 |            7.0361 |
| ('last 24 hours (Oct 29 06:00-Oct 30 06:00)', 'benchmark, not used: recent creative CTR only') |                         0.198  |                         0.1331 |            6.483  |
| ('last 24 hours (Oct 29 06:00-Oct 30 06:00)', 'benchmark, not used: random')                   |                         0.1778 |                         0.1695 |            0.823  |

### Per day
|            |   opportunities |   V1: shown-ad CTR at rank 1 |   V1: at rank 5 |   random: at rank 1 |   random: at rank 5 |   toss-ups (model only) |   no ad served |   predicted CTR, our pick |   predicted CTR, ad actually shown |
|:-----------|----------------:|-----------------------------:|----------------:|--------------------:|--------------------:|------------------------:|---------------:|--------------------------:|-----------------------------------:|
| 2014-10-24 |          89,378 |                       0.2095 |          0.1471 |              0.1787 |              0.1926 |                  0.6973 |         0.1197 |                    0.2101 |                             0.1925 |
| 2014-10-25 |          90,218 |                       0.2243 |          0.1355 |              0.1988 |              0.193  |                  0.5628 |         0.2278 |                    0.2022 |                             0.192  |
| 2014-10-26 |         103,948 |                       0.2183 |          0.1461 |              0.196  |              0.1923 |                  0.5979 |         0.208  |                    0.2029 |                             0.1972 |
| 2014-10-27 |          86,788 |                       0.2248 |          0.162  |              0.1943 |              0.2014 |                  0.6445 |         0.15   |                    0.204  |                             0.1922 |
| 2014-10-28 |         142,909 |                       0.1706 |          0.1672 |              0.1614 |              0.1776 |                  0.443  |         0.3885 |                    0.1808 |                             0.1692 |
| 2014-10-29 |         104,450 |                       0.203  |          0.1268 |              0.1808 |              0.1709 |                  0.6619 |         0.1258 |                    0.1849 |                             0.1722 |
| 2014-10-30 |          22,956 |                       0.2367 |          0.136  |              0.1571 |              0.1693 |                  0.6953 |         0.0001 |                    0.1934 |                             0.1691 |

### Held-out, by position
#### V1 (blend) - used for ranking
|   position |   opportunities |    ctr |   ci_low |   ci_high |
|-----------:|----------------:|-------:|---------:|----------:|
|          1 |           40555 | 0.2059 |   0.202  |    0.2098 |
|          2 |           22399 | 0.1784 |   0.1735 |    0.1835 |
|          3 |           23436 | 0.1599 |   0.1553 |    0.1647 |
|          4 |           21448 | 0.1504 |   0.1457 |    0.1553 |
|          5 |           19568 | 0.1287 |   0.1241 |    0.1335 |

#### LightGBM alone
|   position |   opportunities |    ctr |   ci_low |   ci_high |
|-----------:|----------------:|-------:|---------:|----------:|
|          1 |           42320 | 0.206  |   0.2022 |    0.2099 |
|          2 |           23911 | 0.1718 |   0.1671 |    0.1766 |
|          3 |           21981 | 0.164  |   0.1592 |    0.169  |
|          4 |           21148 | 0.1456 |   0.141  |    0.1505 |
|          5 |           18046 | 0.1289 |   0.1241 |    0.1339 |

#### FM alone
|   position |   opportunities |    ctr |   ci_low |   ci_high |
|-----------:|----------------:|-------:|---------:|----------:|
|          1 |           40810 | 0.2052 |   0.2013 |    0.2091 |
|          2 |           22810 | 0.1759 |   0.171  |    0.1809 |
|          3 |           22575 | 0.1555 |   0.1508 |    0.1603 |
|          4 |           21263 | 0.1489 |   0.1442 |    0.1537 |
|          5 |           19948 | 0.1392 |   0.1345 |    0.1441 |

#### benchmark, not used: recent creative CTR only
|   position |   opportunities |    ctr |   ci_low |   ci_high |
|-----------:|----------------:|-------:|---------:|----------:|
|          1 |           40342 | 0.1958 |   0.192  |    0.1997 |
|          2 |           22812 | 0.1801 |   0.1751 |    0.1851 |
|          3 |           23218 | 0.1641 |   0.1594 |    0.1689 |
|          4 |           21274 | 0.1578 |   0.153  |    0.1628 |
|          5 |           19760 | 0.1349 |   0.1302 |    0.1397 |

#### benchmark, not used: random
|   position |   opportunities |    ctr |   ci_low |   ci_high |
|-----------:|----------------:|-------:|---------:|----------:|
|          1 |           37666 | 0.178  |   0.1741 |    0.1819 |
|          2 |           24942 | 0.1702 |   0.1656 |    0.1749 |
|          3 |           22825 | 0.1671 |   0.1623 |    0.1719 |
|          4 |           21504 | 0.1668 |   0.1619 |    0.1718 |
|          5 |           20469 | 0.1706 |   0.1655 |    0.1758 |

## 2. How far apart are the candidates? (V1, before rules)
|                                                       |   development (Oct 24-Oct 28) |   held-out (Oct 29-Oct 30) |   last 24 hours (Oct 29 06:00-Oct 30 06:00) |
|:------------------------------------------------------|------------------------------:|---------------------------:|--------------------------------------------:|
| candidates per opportunity (min)                      |                        1      |                     1      |                                      1      |
| top-1 minus top-2 predicted CTR, median (pp)          |                        0.9587 |                     1.0235 |                                      1.0756 |
| top-1 minus top-2 predicted CTR, 90th pct (pp)        |                        4.7267 |                     4.5268 |                                      4.7287 |
| best minus worst candidate, median (pp)               |                        3.0124 |                     4.2958 |                                      4.16   |
| uncertainty half-band u, median (log-odds)            |                        0.1053 |                     0.1107 |                                      0.1122 |
| share of opportunities that are toss-ups (model only) |                        0.5738 |                     0.6679 |                                      0.6399 |

## 3. Business rules and the final pick
|                                                      |   ('opportunities', 'development (Oct 24-Oct 28)') |   ('opportunities', 'held-out (Oct 29-Oct 30)') |   ('opportunities', 'last 24 hours (Oct 29 06:00-Oct 30 06:00)') |   ('share', 'development (Oct 24-Oct 28)') |   ('share', 'held-out (Oct 29-Oct 30)') |   ('share', 'last 24 hours (Oct 29 06:00-Oct 30 06:00)') |
|:-----------------------------------------------------|---------------------------------------------------:|------------------------------------------------:|-----------------------------------------------------------------:|-------------------------------------------:|----------------------------------------:|---------------------------------------------------------:|
| all opportunities                                    |                                             513241 |                                          127406 |                                                           103499 |                                     1      |                                  1      |                                                   1      |
| fewer than 5 candidates reached the model            |                                             199078 |                                           24480 |                                                            24452 |                                     0.3879 |                                  0.1921 |                                                   0.2363 |
| shown ad blocked by the safety gate                  |                                             166038 |                                           42528 |                                                            35285 |                                     0.3235 |                                  0.3338 |                                                   0.3409 |
| >= 1 candidate at the fatigue cap                    |                                              11077 |                                            3507 |                                                             3084 |                                     0.0216 |                                  0.0275 |                                                   0.0298 |
| >= 1 candidate given the repeat penalty              |                                              45623 |                                           12805 |                                                            10647 |                                     0.0889 |                                  0.1005 |                                                   0.1029 |
| >= 1 candidate paced down                            |                                             385903 |                                          102885 |                                                            85401 |                                     0.7519 |                                  0.8075 |                                                   0.8251 |
| shown ad's advertiser out of budget                  |                                             105617 |                                           14715 |                                                            14698 |                                     0.2058 |                                  0.1155 |                                                   0.142  |
| no eligible candidate (no ad served)                 |                                             121402 |                                           13138 |                                                            13135 |                                     0.2365 |                                  0.1031 |                                                   0.1269 |
| picks that were toss-ups                             |                                             291328 |                                           82645 |                                                            64644 |                                     0.5676 |                                  0.6487 |                                                   0.6246 |
| picks that explored                                  |                                              49712 |                                           12729 |                                                            10340 |                                     0.0969 |                                  0.0999 |                                                   0.0999 |
| final pick differs from the model-only top candidate |                                             180003 |                                           52523 |                                                            41695 |                                     0.3507 |                                  0.4122 |                                                   0.4029 |

### No ad served, share by character safety tier
| safety_tier   |   development (Oct 24-Oct 28) |   held-out (Oct 29-Oct 30) |   last 24 hours (Oct 29 06:00-Oct 30 06:00) |
|:--------------|------------------------------:|---------------------------:|--------------------------------------------:|
| mature        |                        0.5137 |                     0.2721 |                                      0.3343 |
| sfw           |                        0.0822 |                     0.0338 |                                      0.0416 |
| suggestive    |                        0.4178 |                     0.155  |                                      0.1903 |

### Why each ad won (share of picks)
| pick_reason                              |   development (Oct 24-Oct 28) |   held-out (Oct 29-Oct 30) |   last 24 hours (Oct 29 06:00-Oct 30 06:00) |
|:-----------------------------------------|------------------------------:|---------------------------:|--------------------------------------------:|
| toss-up -> highest score                 |                        0.6166 |                     0.6119 |                                      0.6009 |
| clear winner                             |                        0.2565 |                     0.2767 |                                      0.2846 |
| toss-up -> explore least-known candidate |                        0.1269 |                     0.1114 |                                      0.1144 |

## 4. Model-estimated predicted CTR (not observed clicks)
|                                                                   |   development (Oct 24-Oct 28) |   held-out (Oct 29-Oct 30) |   last 24 hours (Oct 29 06:00-Oct 30 06:00) |
|:------------------------------------------------------------------|------------------------------:|---------------------------:|--------------------------------------------:|
| mean predicted CTR, ad actually shown                             |                        0.1881 |                     0.1715 |                                      0.1693 |
| mean predicted CTR, our pick                                      |                        0.1995 |                     0.1866 |                                      0.1848 |
| mean predicted CTR, model-only top candidate (same opportunities) |                        0.2141 |                     0.2001 |                                      0.1988 |
| our pick minus ad actually shown (pp)                             |                        1.1359 |                     1.5056 |                                      1.551  |

## 5. Timing
| day        |   opportunities |   candidates |   train_v1_seconds |   candidates_and_features_seconds |   scoring_seconds |   rules_and_ordering_seconds |   ms_per_opportunity (features + scoring + rules) |
|:-----------|----------------:|-------------:|-------------------:|----------------------------------:|------------------:|-----------------------------:|--------------------------------------------------:|
| 2014-10-24 |           89378 |       890578 |            19.8139 |                            7.4371 |            4.1494 |                       1.081  |                                            0.1417 |
| 2014-10-25 |           90218 |       899136 |            22.0859 |                            7.8272 |            4.0736 |                       1.0448 |                                            0.1435 |
| 2014-10-26 |          103948 |      1037046 |            26.5076 |                            9.1932 |            4.6303 |                       1.2398 |                                            0.1449 |
| 2014-10-27 |           86788 |       867240 |            29.6312 |                            8.5268 |            3.9833 |                       1.0975 |                                            0.1568 |
| 2014-10-28 |          142909 |      1428047 |            32.0769 |                           10.9187 |            5.0615 |                       1.3421 |                                            0.1212 |
| 2014-10-29 |          104450 |      1043867 |            36.9283 |                            8.913  |            4.4432 |                       1.4075 |                                            0.1413 |
| 2014-10-30 |           22956 |       229436 |             9.831  |                            3.2657 |            0.991  |                       0.4356 |                                            0.2044 |
