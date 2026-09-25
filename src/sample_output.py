"""Sample output: how the ranker orders the candidate ads for two chat moments, written for a first-time reader."""
import numpy as np
import pandas as pd

from .config import (COLD_EXPLORATION_SHARE, DOMINANT_COHORTS, FATIGUE_HARD_CAP, GRADUATION_IMPRESSIONS,
                     GRADUATION_WINDOW_HOURS, N_CANDIDATES, SAFETY_TIERS)

NEAR_TIE = 0.01  # a lower-ranked ad within 1 point of the pick's V1 probability gets its own "why not" line


def pct(p):
    return f"{100 * p:.1f}%"


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def first_match(opportunities, conditions):
    """Lowest opportunity id meeting all conditions, dropping the last condition until one matches."""
    for k in range(len(conditions), 0, -1):
        mask = np.logical_and.reduce([c.to_numpy() for c in conditions[:k]])
        if mask.any():
            return opportunities.index[mask].min()


def select_examples(ranked):
    """Example 1: our pick is the ad the user actually clicked. Example 2: a repeat penalty overrides the model's favourite."""
    shown = ranked[ranked["is_logged"]].set_index("opportunity")
    eligible = ranked[ranked["blocked"] == ""]
    top = eligible.loc[eligible.groupby("opportunity")["p_v1"].idxmax()].set_index("opportunity").reindex(shown.index)
    pick = ranked[ranked["final_rank"] == 1].set_index("opportunity").reindex(shown.index)
    full = ranked.groupby("opportunity").size().reindex(shown.index) == N_CANDIDATES
    plain_pick = pick["pick_reason"].isin(["clear winner", "toss-up -> highest score"])
    typical = top["p_v1"] >= 0.12
    example_1 = first_match(shown, [
        full, shown["final_rank"] == 1, shown["click"] == 1, shown["C14"] == top["C14"], typical,
        shown["safety_tier"] != "sfw", pick["pick_reason"] == "clear winner", shown["conversation_turn"] >= 3])
    example_2 = first_match(shown, [
        full, (top["user_creative_exposures_24h"] > 0) & (top["final_rank"] != 1), plain_pick, typical,
        shown["C14"] == top["C14"], top["final_rank"] == 2, shown["click"] == 0, shown["safety_tier"] == "sfw"])
    return example_1, example_2


def effects(c):
    """Each rule that lowered this candidate, as text."""
    parts = []
    if c["pacing"] < 1:
        parts.append(f"budget pacing (odds ×{c['pacing']:.2f})")
    if c["repeat_penalty"] < 1:
        seen = "this ad" if c["user_creative_exposures_24h"] > 0 else "this campaign"
        parts.append(f"user saw {seen} (odds ×{c['repeat_penalty']:.2f})")
    return parts


def why_lower(c):
    """Plain-language reason a candidate's score is below its V1 prediction."""
    reasons = []
    if c["pacing"] < 1:
        reasons.append(f"its campaign is spending ahead of its daily budget schedule, so budget pacing multiplies its "
                       f"odds by {c['pacing']:.2f}")
    if c["repeat_penalty"] < 1 and c["user_creative_exposures_24h"] > 0:
        reasons.append(f"this user already saw it in the last 24 h (on earlier days such repeats were clicked at "
                       f"{c['repeat_penalty']:.2f} times the rate of a first view), so its odds are multiplied by "
                       f"{c['repeat_penalty']:.2f}")
    elif c["repeat_penalty"] < 1:
        reasons.append(f"this user saw another ad from the same campaign in the last 24 h, so its odds are multiplied "
                       f"by {c['repeat_penalty']:.2f}")
    return " and ".join(reasons)


def candidate_table(ex):
    ex = ex.assign(order=ex["final_rank"].fillna(99)).sort_values(["order", "p_v1"], ascending=[True, False])
    tier_names = np.array(SAFETY_TIERS)
    rows = []
    for _, c in ex.iterrows():
        removed = c["blocked"] != ""
        if c["blocked"] == "safety":
            rank = f"removed: advertiser doesn't accept {c['safety_tier']} characters"
        elif c["blocked"] == "fatigue_cap":
            rank = f"removed: user saw it {FATIGUE_HARD_CAP}+ times in 24 h"
        elif c["blocked"] == "budget":
            rank = "removed: daily budget spent"
        else:
            rank = "**1 = our pick**" if c["final_rank"] == 1 else str(int(c["final_rank"]))
        rows.append({
            "our rank": rank,
            "ad": int(c["C14"]),
            "campaign": int(c["C17"]),
            "2014 log": ("shown, **clicked**" if c["click"] else "shown, not clicked") if c["is_logged"] else "",
            "V1 click prob.": pct(c["p_v1"]),
            "rules applied": ", ".join(effects(c)) or "none",
            "score": "—" if removed else pct(sigmoid(c["z"])),
            "score range": "—" if removed else f"{pct(sigmoid(c['z'] - c['u']))} – {pct(sigmoid(c['z'] + c['u']))}",
            "advertiser accepts": f"up to {tier_names[int(c['advertiser_tier_rank'])]}",
            "user saw (ad / campaign, 24 h)": f"{int(c['user_creative_exposures_24h'])} / {int(c['user_campaign_exposures_24h'])}",
            f"ad's impressions ({GRADUATION_WINDOW_HOURS} h)": f"{int(c['creative_impressions_72h']):,}" + (" new" if c["cold"] else ""),
        })
    align = ["left", "right", "right", "left", "right", "left", "right", "right", "left", "right", "right"]
    return pd.DataFrame(rows).to_markdown(index=False, colalign=align)


def describe(ex, number, title):
    ctx = ex[ex["is_logged"]].iloc[0]
    pick = ex[ex["final_rank"] == 1].iloc[0]
    eligible = ex[ex["blocked"] == ""]
    runner_up = eligible[eligible["final_rank"] == 2]
    user = "a returning user" if ctx["user_seen_before"] else "a first-time user"
    where = "an app" if ctx["is_app"] else "a website"
    lines = [
        f"## Example {number}: {title}",
        "",
        f"- **The moment:** {user.capitalize()} is on message {ctx['conversation_turn']} of a "
        f"{ctx['session_msg_count']}-message session with a **{ctx['safety_tier']} {ctx['genre']}** character, on "
        f"{where} (publisher {ctx['publisher_id']}), {ctx['size']} banner slot, {ctx['ts']:%b %d %H:00}.",
    ]
    pick_line = f"- **Our pick:** ad {int(pick['C14'])}. V1 predicts {pct(pick['p_v1'])}"
    pick_line += (f"; after {' and '.join(effects(pick))}, its score is {pct(sigmoid(pick['z']))}." if effects(pick)
                  else "; no rule changes it.")
    lines.append(pick_line)
    if len(runner_up):
        second = runner_up.iloc[0]
        pick_range = f"{pct(sigmoid(pick['z'] - pick['u']))}–{pct(sigmoid(pick['z'] + pick['u']))}"
        second_range = f"{pct(sigmoid(second['z'] - second['u']))}–{pct(sigmoid(second['z'] + second['u']))}"
        if pick["pick_reason"] == "clear winner":
            lines.append(f"- **Why it wins:** highest score, and its range ({pick_range}) sits entirely above the "
                         f"runner-up's ({second_range}): a clear winner.")
        elif pick["pick_reason"] == "toss-up -> highest score":
            lines.append(f"- **Why it wins:** highest score, but its range ({pick_range}) overlaps the runner-up's "
                         f"({second_range}), so it's a toss-up. We still show the higher score; a toss-up only means "
                         f"little is lost if the model is wrong. (In the {DOMINANT_COHORTS} character cohorts users see "
                         f"most, a toss-up would instead go to the campaign that cohort has seen least.)")
        else:
            lines.append(f"- **Why it wins:** {pick['pick_reason']}.")
    beaten = eligible[(eligible["final_rank"] != 1) & (eligible["p_v1"] >= pick["p_v1"] - NEAR_TIE)]
    for _, c in beaten.sort_values("p_v1", ascending=False).iterrows():
        lines.append(f"- **Why not ad {int(c['C14'])}** (V1 {pct(c['p_v1'])}): {why_lower(c)}, which lowers its "
                     f"score to {pct(sigmoid(c['z']))}, so it ranks {int(c['final_rank'])}.")
    if ctx["hour_of_day"] == 0 and (ex["pacing"] == 1).all():
        lines.append("- **No budget pacing here:** at 00:00 the day's budgets have just reset, so no campaign is ahead "
                     "of schedule yet.")
    status = ("also our pick" if ctx["final_rank"] == 1 else
              f"removed by our rules ({ctx['blocked']})" if ctx["blocked"] else f"our rank {int(ctx['final_rank'])}")
    lines += [
        f"- **What really happened in 2014:** the original system showed ad {int(ctx['C14'])} ({status}), and the "
        f"user **{'clicked' if ctx['click'] else 'did not click'}**. Clicks are known only for that ad; the others "
        f"were never shown.",
        "",
        candidate_table(ex),
        "",
    ]
    return "\n".join(lines)


def write_sample_output(ranked, path, day):
    """ranked: the ranked candidates of the full ranker (Goal 2 rules + Goals 3-4 layer) on the test days."""
    slots = ranked["opportunity"].nunique()
    ranked = ranked[ranked["day"] == day]
    example_1, example_2 = select_examples(ranked)
    ex1, ex2 = ranked[ranked["opportunity"] == example_1], ranked[ranked["opportunity"] == example_2]
    tried_out = any(e.loc[e["final_rank"] == 1, "explored"].any() for e in (ex1, ex2))
    seen_ad = ranked["user_creative_exposures_24h"] > 0
    penalty_ad = ranked.loc[seen_ad, "repeat_penalty"].max()
    penalty_campaign = ranked.loc[~seen_ad & (ranked["user_campaign_exposures_24h"] > 0), "repeat_penalty"].max()
    md = f"""# Sample output: how the system ranks ads for two chat moments (test day {day:%b %d %Y})

**How to read this.** When an ad slot opens in a chat, the system:
1. takes {N_CANDIDATES} candidate ads: the ad the original 2014 system actually showed (always included, even if our
   rules would not have chosen it, because it is the only ad whose real click we know), plus {N_CANDIDATES - 1} others
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
  - *user saw this ad:* odds ×{penalty_ad:.2f} if the user saw this exact ad in the last 24 h; otherwise *user saw this
    campaign:* odds ×{penalty_campaign:.2f} if they saw another ad from its campaign. Both were measured on earlier days,
    apply once however many views there were, and don't stack. Separately, an ad the user saw {FATIGUE_HARD_CAP} times
    in 24 h is removed.
- **score:** the V1 probability after the rules. It is a ranking number, not a click chance; the highest score wins.
  Scores are computed from unrounded values, so recomputing from the rounded table can differ by 0.1 point.
- **score range:** a band around the score: how much V1's two internal models disagree, widened for ads with little
  data. It is a heuristic, not a confidence interval. If the runner-up's range overlaps the winner's, it's a toss-up.
- **advertiser accepts:** the most mature character tier the advertiser allows (sfw < suggestive < mature).
- **ad's impressions ({GRADUATION_WINDOW_HOURS} h):** ads with fewer than {GRADUATION_IMPRESSIONS:,} are marked "new": their range is
  wider, and on up to {COLD_EXPLORATION_SHARE:.0%} of an hour's ad slots a new ad whose optimistic end reaches the leader is tried out
  first. {"Neither pick below is a try-out: both win on score." if not tried_out else "One pick below is a try-out; its line says so."}

We picked these two moments on purpose to show the mechanics; they are not evidence of accuracy. Over all {slots:,} ad
slots of the test days, `results/ranking/metrics.md` shows how often the real click agrees with V1's order,
and `results/layer/metrics.md` shows the effect of the new-ad and fatigue layer.

{describe(ex1, 1, "our pick is the ad the user actually clicked")}
{describe(ex2, 2, "the user saw the model's favourite in the last 24 h, so an ad they haven't seen wins")}"""
    path.write_text(md)
    return md
