# 04 — Ad vs context columns, dependencies, candidate pool

- contexts with >= 20 impressions; `entropy_ratio_given_context` = H(column | context) / H(column): 0 = fixed by the context, 1 = independent of it
- `determination` = row-weighted share of rows whose y equals the most common y for their x (1 = x fully determines y)

## 1. Variation within a context
|                                                         |   rows_in_contexts |   mean_distinct_per_context |   share_rows_in_constant_contexts |   entropy_ratio_given_context |
|:--------------------------------------------------------|-------------------:|----------------------------:|----------------------------------:|------------------------------:|
| ('publisher + hour', 'banner_pos')                      |            726,995 |                      1.0495 |                            0.9505 |                        0.0337 |
| ('publisher + hour', 'C1')                              |            726,995 |                      1      |                            1      |                        0      |
| ('publisher + hour', 'C14')                             |            726,995 |                     15.1052 |                            0.0212 |                        0.3482 |
| ('publisher + hour', 'C15')                             |            726,995 |                      1.1732 |                            0.8298 |                        0.1251 |
| ('publisher + hour', 'C16')                             |            726,995 |                      1.1616 |                            0.8394 |                        0.1137 |
| ('publisher + hour', 'C17')                             |            726,995 |                      6.3353 |                            0.1064 |                        0.212  |
| ('publisher + hour', 'C18')                             |            726,995 |                      2.355  |                            0.1771 |                        0.3329 |
| ('publisher + hour', 'C19')                             |            726,995 |                      3.8166 |                            0.1417 |                        0.2439 |
| ('publisher + hour', 'C20')                             |            726,995 |                      8.5761 |                            0.0276 |                        0.4755 |
| ('publisher + hour', 'C21')                             |            726,995 |                      3.8951 |                            0.1446 |                        0.2261 |
| ('publisher + hour', 'device_type')                     |            726,995 |                      1.0103 |                            0.9897 |                        0.0364 |
| ('publisher + hour', 'device_conn_type')                |            726,995 |                      1.8771 |                            0.1888 |                        0.4459 |
| ('publisher + hour', 'device_model')                    |            726,995 |                     78.5662 |                            0.0041 |                        0.6129 |
| ('publisher + device_model + hour', 'banner_pos')       |            173,787 |                      1.0087 |                            0.9913 |                        0.0074 |
| ('publisher + device_model + hour', 'C1')               |            173,787 |                      1      |                            1      |                        0      |
| ('publisher + device_model + hour', 'C14')              |            173,787 |                      6.2493 |                            0.0764 |                        0.3115 |
| ('publisher + device_model + hour', 'C15')              |            173,787 |                      1.0017 |                            0.9983 |                        0.0115 |
| ('publisher + device_model + hour', 'C16')              |            173,787 |                      1.0003 |                            0.9997 |                        0.002  |
| ('publisher + device_model + hour', 'C17')              |            173,787 |                      1.8313 |                            0.525  |                        0.0778 |
| ('publisher + device_model + hour', 'C18')              |            173,787 |                      1.3802 |                            0.6712 |                        0.0957 |
| ('publisher + device_model + hour', 'C19')              |            173,787 |                      1.5728 |                            0.621  |                        0.0834 |
| ('publisher + device_model + hour', 'C20')              |            173,787 |                      3.8408 |                            0.1111 |                        0.3731 |
| ('publisher + device_model + hour', 'C21')              |            173,787 |                      1.549  |                            0.6122 |                        0.0718 |
| ('publisher + device_model + hour', 'device_type')      |            173,787 |                      1.0016 |                            0.9984 |                        0.0024 |
| ('publisher + device_model + hour', 'device_conn_type') |            173,787 |                      1.4304 |                            0.7718 |                        0.1517 |

## 2. Functional dependencies (row = x, column = y)
| x determines y ->   |   banner_pos |     C1 |    C15 |    C16 |      C17 |    C18 |    C19 |    C20 |    C21 |   size |   publisher_id |      C14 |
|:--------------------|-------------:|-------:|-------:|-------:|---------:|-------:|-------:|-------:|-------:|-------:|---------------:|---------:|
| C14                 |       0.8626 | 0.9552 | 1      | 1      |   1      | 0.9942 | 0.9283 | 0.645  | 1      | 1      |         0.4761 | nan      |
| C17                 |       0.8548 | 0.9515 | 0.976  | 0.9814 | nan      | 0.9931 | 0.9185 | 0.6427 | 1      | 0.9739 |         0.4643 |   0.4943 |
| publisher_id        |       0.9922 | 1      | 0.9803 | 0.9877 |   0.4401 | 0.7117 | 0.5884 | 0.5431 | 0.5436 | 0.9802 |       nan      |   0.2305 |
| banner_pos          |     nan      | 0.9177 | 0.9329 | 0.9443 |   0.1358 | 0.4166 | 0.3452 | 0.4683 | 0.2187 | 0.9312 |         0.2269 |   0.0368 |
| device_model        |       0.7678 | 0.9821 | 0.9355 | 0.9456 |   0.3016 | 0.6103 | 0.4547 | 0.541  | 0.4249 | 0.9329 |         0.323  |   0.1522 |

## 3. Banner size (C15 x C16)
| size     |    rows |    ctr |   creatives |
|:---------|--------:|-------:|------------:|
| 320x50   | 930,256 | 0.1694 |       1,517 |
| 300x250  |  44,072 | 0.4288 |         142 |
| 300x50   |  13,021 | 0.1661 |         253 |
| 216x36   |   7,865 | 0.1307 |          82 |
| 320x480  |   2,629 | 0.2237 |          47 |
| 728x90   |   1,908 | 0.0802 |          54 |
| 120x20   |      88 | 0.0341 |          23 |
| 480x320  |      63 | 0.2857 |           9 |
| 1024x768 |      60 | 0.3167 |           3 |
| 768x1024 |      38 | 0.5    |           3 |

|                           |   determination of size |
|:--------------------------|------------------------:|
| C14 (creative)            |                  1      |
| publisher_id              |                  0.9802 |
| publisher_id + banner_pos |                  0.9803 |
| device_type               |                  0.9303 |

## 4. Candidate pool: distinct creatives (C14) with an impression in the previous 24 hours, per hour
|                                 |   min |     5% |   50% |   max |
|:--------------------------------|------:|-------:|------:|------:|
| creatives_live_24h              |   594 | 605.55 |   720 |  1016 |
| creatives_live_24h_size_320x50  |   466 | 472.85 |   564 |   733 |
| creatives_live_24h_size_300x250 |    34 |  36    |    50 |    70 |
| creatives_live_24h_size_300x50  |    40 |  43    |    51 |   139 |

- share of an hour's impressions taken by its 5 most-shown creatives: median 0.231, min 0.134, max 0.748
