# Ablation — V1 LightGBM (16 leaves), train Oct 21-27 (729,685 rows), evaluate Oct 28 (142,909 rows)

- each row removes one feature group and retrains (trees chosen by early stopping on Oct 28)
- `delta_*` = ablated model minus full model (seed 42); positive delta_logloss / negative delta_AUC = the group helps
- noise level: full model retrained with seeds 42, 43, 44

## Full model, noise level
|         |   features |   trees |   logloss |     NE |    AUC |   AUC_within_publisher |
|:--------|-----------:|--------:|----------:|-------:|-------:|-----------------------:|
| seed 42 |         46 |     189 |    0.3961 | 0.8807 | 0.7409 |                 0.6168 |
| seed 43 |         46 |     235 |    0.3954 | 0.8792 | 0.7416 |                 0.6188 |
| seed 44 |         46 |     226 |    0.3956 | 0.8797 | 0.7412 |                 0.6184 |

|                        |   logloss |     NE |    AUC |   AUC_within_publisher |
|:-----------------------|----------:|-------:|-------:|-----------------------:|
| std across seeds       |    0.0003 | 0.0007 | 0.0004 |                  0.001 |
| max - min across seeds |    0.0007 | 0.0015 | 0.0007 |                  0.002 |

## Leave one group out (sorted: most useful group first)
|                              |   features |   trees |   logloss |   delta_logloss |     NE |   delta_NE |    AUC |   delta_AUC |   AUC_within_publisher |   delta_AUC_within_publisher |   fit_seconds |
|:-----------------------------|-----------:|--------:|----------:|----------------:|-------:|-----------:|-------:|------------:|-----------------------:|-----------------------------:|--------------:|
| character                    |         40 |     155 |    0.4022 |          0.0061 | 0.8943 |     0.0137 | 0.7266 |     -0.0143 |                 0.5651 |                      -0.0517 |        2.264  |
| publisher                    |         42 |     166 |    0.3993 |          0.0033 | 0.8879 |     0.0072 | 0.733  |     -0.0079 |                 0.6056 |                      -0.0112 |        2.5387 |
| recent click rates           |         42 |     226 |    0.3968 |          0.0007 | 0.8823 |     0.0016 | 0.7397 |     -0.0012 |                 0.6198 |                       0.0029 |        3.0741 |
| device                       |         42 |     174 |    0.3968 |          0.0007 | 0.8823 |     0.0016 | 0.7386 |     -0.0023 |                 0.61   |                      -0.0068 |        2.7522 |
| ad (banner_pos, C1, C14-C21) |         36 |     169 |    0.3962 |          0.0001 | 0.8809 |     0.0002 | 0.7406 |     -0.0003 |                 0.6153 |                      -0.0016 |        2.658  |
| hour of day                  |         45 |     232 |    0.3959 |         -0.0002 | 0.8801 |    -0.0005 | 0.7408 |     -0.0001 |                 0.6163 |                      -0.0006 |        3.2674 |
| conversation                 |         44 |     200 |    0.3958 |         -0.0003 | 0.88   |    -0.0006 | 0.7412 |      0.0003 |                 0.6181 |                       0.0013 |        2.8893 |
| user history                 |         41 |     242 |    0.3958 |         -0.0003 | 0.88   |    -0.0006 | 0.7409 |      0      |                 0.6166 |                      -0.0002 |        3.3603 |
| frequency deciles            |         40 |     223 |    0.3957 |         -0.0004 | 0.8799 |    -0.0008 | 0.7412 |      0.0004 |                 0.616  |                      -0.0009 |        3.0666 |
| fatigue                      |         44 |     230 |    0.3956 |         -0.0005 | 0.8796 |    -0.001  | 0.7414 |      0.0006 |                 0.6187 |                       0.0019 |        3.3139 |
| traffic volume               |         44 |     227 |    0.3956 |         -0.0005 | 0.8795 |    -0.0012 | 0.7413 |      0.0004 |                 0.6184 |                       0.0016 |        3.1522 |

## Groups
- **publisher**: publisher_id, publisher_domain, publisher_category, is_app
- **ad (banner_pos, C1, C14-C21)**: banner_pos, C1, C14, C15, C16, C17, C18, C19, C20, C21
- **device**: device_type, device_conn_type, device_model, has_device_id
- **character**: character_id, safety_tier, creator_type, genre, num_interactions, character_age_days
- **conversation**: conversation_turn, session_msg_count
- **hour of day**: hour_of_day
- **frequency deciles**: publisher_id_freq_decile, publisher_domain_freq_decile, device_model_freq_decile, C14_freq_decile, C17_freq_decile, character_id_freq_decile
- **user history**: user_prior_impressions, user_prior_clicks, user_prior_ctr, user_hours_since_last, user_seen_before
- **fatigue**: user_impressions_24h, user_character_prior_impressions
- **recent click rates**: publisher_ctr_24h, creative_ctr_24h, C17_ctr_24h, character_ctr_24h
- **traffic volume**: publisher_impressions_prev_hour, total_impressions_prev_hour
