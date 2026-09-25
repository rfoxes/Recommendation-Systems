# Sample output: how the system ranks ads for two chat moments (test day Oct 29 2014)

**How to read this.** When an ad slot opens in a chat, the system:
1. takes 5 candidate ads: the ad the original 2014 system actually showed (always included, even if our
   rules would not have chosen it, because it is the only ad whose real click we know), plus 4 others
   that fit the slot, ran on the same site/app recently, and whose advertiser accepts the character's safety tier;
2. predicts each one's click probability with **V1**, our click model (LightGBM and a factorization machine, averaged);
3. applies the business rules, which lower the score of some ads (below);
4. shows the ad with the highest score. The data has no bids, so ads are ranked by adjusted click probability alone.

**What is real and what is simulated.** The ad, user, website/app, slot and click come from the real 2014 ad log. The
character, its safety tier and the conversation are synthetic columns that came with the data. The advertisers' safety
settings and daily budgets are simulated by us (the data has neither).

**Columns**
- **ad / campaign:** anonymized IDs. The data doesn't name its columns; we measured that column C14 behaves like the ad
  creative (each value always comes with one C17 value and one banner size) and C17 like its campaign.
- **V1 click prob.:** the model's predicted chance of a click for this user, character, conversation and ad.
- **rules applied:** each rule multiplies the ad's click **odds** (p / (1 − p)), not the probability, so a score falls a
  little less than the multiplier suggests:
  - *budget pacing:* a campaign that has used more of its daily budget than the time of day allows is slowed down
    (×1 = on schedule; smaller = further ahead);
  - *user saw this ad:* odds ×0.75 if the user saw this exact ad in the last 24 h; otherwise *user saw this
    campaign:* odds ×0.84 if they saw another ad from its campaign. Both were measured on earlier days,
    apply once however many views there were, and don't stack. Separately, an ad the user saw 3 times
    in 24 h is removed.
- **score:** the V1 probability after the rules. It is a ranking number, not a click chance; the highest score wins.
  Scores are computed from unrounded values, so recomputing from the rounded table can differ by 0.1 point.
- **score range:** a band around the score: how much V1's two internal models disagree, widened for ads with little
  data. It is a heuristic, not a confidence interval. If the runner-up's range overlaps the winner's, it's a toss-up.
- **advertiser accepts:** the most mature character tier the advertiser allows (sfw < suggestive < mature).
- **ad's impressions (72 h):** ads with fewer than 1,000 are marked "new": their range is
  wider, and on up to 5% of an hour's ad slots a new ad whose optimistic end reaches the leader is tried out
  first. Neither pick below is a try-out: both win on score.

We picked these two moments on purpose to show the mechanics; they are not evidence of accuracy. Over all 640,647 ad
slots of the test days, `results/ranking/metrics.md` shows how often the real click agrees with V1's order,
and `results/layer/metrics.md` shows the effect of the new-ad and fatigue layer.

## Example 1: our pick is the ad the user actually clicked

- **The moment:** A first-time user is on message 6 of a 12-message session with a **suggestive mystery** character, on an app (publisher 9c13b419), 320x50 banner slot, Oct 29 03:00.
- **Our pick:** ad 22953. V1 predicts 36.1%; after budget pacing (odds ×0.60), its score is 25.3%.
- **Why it wins:** highest score, and its range (22.7%–28.2%) sits entirely above the runner-up's (20.0%–22.6%): a clear winner.
- **What really happened in 2014:** the original system showed ad 22953 (also our pick), and the user **clicked**. Clicks are known only for that ad; the others were never shown.

| our rank         |    ad |   campaign | 2014 log           |   V1 click prob. | rules applied              |   score |   score range | advertiser accepts   |   user saw (ad / campaign, 24 h) |   ad's impressions (72 h) |
|:-----------------|------:|-----------:|:-------------------|-----------------:|:---------------------------|--------:|--------------:|:---------------------|---------------------------------:|--------------------------:|
| **1 = our pick** | 22953 |       2654 | shown, **clicked** |            36.1% | budget pacing (odds ×0.60) |   25.3% | 22.7% – 28.2% | up to suggestive     |                            0 / 0 |                   254 new |
| 2                | 20633 |       2374 |                    |            34.6% | budget pacing (odds ×0.51) |   21.3% | 20.0% – 22.6% | up to mature         |                            0 / 0 |                     2,196 |
| 3                | 22948 |       2654 |                    |            28.8% | budget pacing (odds ×0.60) |   19.6% | 17.5% – 21.8% | up to suggestive     |                            0 / 0 |                   240 new |
| 4                | 22957 |       2655 |                    |            32.8% | budget pacing (odds ×0.27) |   11.8% |  9.7% – 14.2% | up to suggestive     |                            0 / 0 |                   102 new |
| 5                | 22960 |       2655 |                    |            27.2% | budget pacing (odds ×0.27) |    9.3% |  7.6% – 11.4% | up to suggestive     |                            0 / 0 |                    86 new |

## Example 2: the user saw the model's favourite in the last 24 h, so an ad they haven't seen wins

- **The moment:** A returning user is on message 4 of a 7-message session with a **sfw mentor** character, on a website (publisher 1fbe01fe), 320x50 banner slot, Oct 29 01:00.
- **Our pick:** ad 22261. V1 predicts 28.6%; after budget pacing (odds ×0.89), its score is 26.2%.
- **Why it wins:** highest score, and its range (24.0%–28.6%) sits entirely above the runner-up's (11.8%–15.9%): a clear winner.
- **Why not ad 22676** (V1 29.7%): its campaign is spending ahead of its daily budget schedule, so budget pacing multiplies its odds by 0.50 and this user already saw it in the last 24 h (on earlier days such repeats were clicked at 0.75 times the rate of a first view), so its odds are multiplied by 0.75, which lowers its score to 13.7%, so it ranks 2.
- **What really happened in 2014:** the original system showed ad 22676 (our rank 2), and the user **did not click**. Clicks are known only for that ad; the others were never shown.

| our rank                              |    ad |   campaign | 2014 log           |   V1 click prob. | rules applied                                                   |   score |   score range | advertiser accepts   |   user saw (ad / campaign, 24 h) |   ad's impressions (72 h) |
|:--------------------------------------|------:|-----------:|:-------------------|-----------------:|:----------------------------------------------------------------|--------:|--------------:|:---------------------|---------------------------------:|--------------------------:|
| **1 = our pick**                      | 22261 |       2545 |                    |            28.6% | budget pacing (odds ×0.89)                                      |   26.2% | 24.0% – 28.6% | up to suggestive     |                            0 / 0 |                   617 new |
| 2                                     | 22676 |       2616 | shown, not clicked |            29.7% | budget pacing (odds ×0.50), user saw this ad (odds ×0.75)       |   13.7% | 11.8% – 15.9% | up to sfw            |                            2 / 2 |                     3,534 |
| 3                                     | 23165 |       2668 |                    |            27.1% | budget pacing (odds ×0.34), user saw this campaign (odds ×0.84) |    9.6% |  8.0% – 11.4% | up to mature         |                            0 / 2 |                   177 new |
| 4                                     | 23170 |       2668 |                    |            22.2% | budget pacing (odds ×0.34), user saw this campaign (odds ×0.84) |    7.5% |   6.3% – 8.9% | up to mature         |                            0 / 2 |                   165 new |
| removed: user saw it 3+ times in 24 h | 20108 |       2299 |                    |            27.8% | user saw this ad (odds ×0.75)                                   |       — |             — | up to sfw            |                          15 / 15 |                    10,288 |
