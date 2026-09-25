"""Goal 2: simulated retrieval, candidate features, business rules and uncertainty-aware ordering.
Goals 3-4: an optional adaptation and exploration layer on top (cold-creative exploration, fatigue-aware reordering)."""
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (AD_ATTRIBUTES, ADVERTISER_MAX_TIER_SHARES, CANDIDATE_WINDOW_HOURS, COLD_EXPLORATION_SHARE,
                     DOMINANT_COHORTS, EXPLORATION_SHARE, FATIGUE_HARD_CAP, GRADUATION_IMPRESSIONS,
                     GRADUATION_WINDOW_HOURS, HISTORY_SMOOTHING, N_CANDIDATES, RETRIEVAL_POOL, SAFETY_TIERS)
from .features import logit
from .history import key_history, recent_click_rate, window_totals

TIER_RANK = {tier: i for i, tier in enumerate(SAFETY_TIERS)}


def add_slot_size(df):
    df["size"] = df["C15"].astype(str) + "x" + df["C16"].astype(str)
    return df


# ---------- simulated retrieval ----------

def creative_catalog(history):
    """Attributes of each creative (C14): its most common value of each attribute (C14 determines them in the data)."""
    out = {}
    for col in AD_ATTRIBUTES:
        counts = history.groupby(["C14", col]).size()
        out[col] = counts.groupby(level=0).idxmax().map(lambda pair: pair[1])
    return pd.DataFrame(out)


def generate_candidates(opportunities, history, creative_tier_rank, rng, n=RETRIEVAL_POOL, window=CANDIDATE_WINDOW_HOURS):
    """
    Simulated retrieval layer. For each opportunity: the ad actually shown (kept so the ranking can be checked against its
    real click) plus up to n - 1 other creatives, in retrieval order (`draw_order`), that
      - belong to an advertiser that accepts the character's safety tier (hard eligibility filter, known before ranking),
      - ran on the same publisher with the same slot size in the previous `window` hours (topped up with the same size on
        any publisher when there are too few),
      - are drawn in proportion to their impressions in that window (active campaigns are retrieved more often).
    creative_tier_rank: highest character safety tier (as TIER_RANK) each creative's advertiser accepts, indexed by C14.
    """
    impressions = history.groupby(["publisher_id", "size", "hour_idx", "C14"]).size().rename("n").reset_index()
    impressions["tier_rank"] = creative_tier_rank.reindex(impressions["C14"]).to_numpy()

    def arrays(g):
        return g["hour_idx"].to_numpy(), g["C14"].to_numpy(), g["n"].to_numpy(), g["tier_rank"].to_numpy()

    by_slot = {key: arrays(g) for key, g in impressions.groupby(["publisher_id", "size"])}
    by_size = {key: arrays(g) for key, g in impressions.groupby("size")}
    size_pools = {}

    def live_pool(entry, hour, min_tier):
        """Eligible creatives live in the window and their impression counts."""
        if entry is None:
            return np.array([], dtype="int64"), np.array([])
        hours, creatives, counts, tiers = entry
        mask = (hours >= hour - window) & (hours < hour) & (tiers >= min_tier)
        ids, inverse = np.unique(creatives[mask], return_inverse=True)
        return ids, np.bincount(inverse, weights=counts[mask], minlength=len(ids))

    def draw(pool, k, exclude):
        ids, weights = pool
        keep = ~np.isin(ids, exclude)
        ids, weights = ids[keep], weights[keep]
        k = min(k, len(ids))
        return rng.choice(ids, k, replace=False, p=weights / weights.sum()) if k else ids[:0]

    opportunity_ids, creatives, is_logged = [], [], []
    for (publisher, size, hour), group in opportunities.groupby(["publisher_id", "size", "hour_idx"], sort=False):
        local_pools = {}
        tiers = group["safety_tier"].map(TIER_RANK).to_numpy()
        for opportunity, logged, tier in zip(group.index, group["C14"].to_numpy(), tiers):
            if tier not in local_pools:
                local_pools[tier] = live_pool(by_slot.get((publisher, size)), hour, tier)
            others = draw(local_pools[tier], n - 1, [logged])
            if len(others) < n - 1:
                if (size, hour, tier) not in size_pools:
                    size_pools[(size, hour, tier)] = live_pool(by_size.get(size), hour, tier)
                others = np.concatenate([others, draw(size_pools[(size, hour, tier)], n - 1 - len(others),
                                                      np.append(others, logged))])
            picked = np.append(logged, others)
            opportunity_ids.append(np.full(len(picked), opportunity))
            creatives.append(picked)
    creatives = np.concatenate(creatives)
    opportunity_ids = np.concatenate(opportunity_ids)
    draw_order = pd.Series(opportunity_ids).groupby(opportunity_ids).cumcount().to_numpy()
    return pd.DataFrame({"opportunity": opportunity_ids, "C14": creatives, "draw_order": draw_order,
                         "is_logged": draw_order == 0})


def candidate_rows(opportunities, candidates, catalog, history):
    """
    One row per candidate: the opportunity's context (publisher, slot, device, character, conversation, user history)
    with the candidate creative's attributes and ad-dependent features swapped in. The logged creative keeps the
    attributes it was actually shown with. Ad-dependent features use only hours before the opportunity's hour.
    """
    rows = opportunities.loc[candidates["opportunity"].to_numpy()].reset_index(drop=True)
    is_logged = candidates["is_logged"].to_numpy()
    rows["C14"] = candidates["C14"].to_numpy()
    for col in AD_ATTRIBUTES:
        swapped = catalog[col].reindex(rows["C14"]).to_numpy()
        rows[col] = np.where(is_logged, rows[col].to_numpy(), swapped).astype("int64")
    rows["creative_ctr_24h"] = recent_click_rate(history, ["C14"], rows)
    rows["C17_ctr_24h"] = recent_click_rate(history, ["C17"], rows)
    rows["creative_impressions_24h"] = window_totals(history, ["C14"], rows)[0]
    rows["user_creative_exposures_24h"] = window_totals(history, ["user", "C14"], rows)[0]
    rows["opportunity"] = candidates["opportunity"].to_numpy()
    rows["draw_order"] = candidates["draw_order"].to_numpy()
    rows["is_logged"] = is_logged
    return rows


# ---------- business rules ----------

def advertiser_max_tier(advertisers):
    """
    Simulated brand-safety setting per advertiser (C17): the highest character safety tier it accepts.
    The data has no such labels, so each advertiser gets a stable pseudo-random tier with ADVERTISER_MAX_TIER_SHARES.
    """
    unique = pd.unique(np.asarray(advertisers))
    u = np.array([zlib.crc32(str(a).encode()) / 2**32 for a in unique])
    tiers = np.array(SAFETY_TIERS)[np.searchsorted(np.cumsum(list(ADVERTISER_MAX_TIER_SHARES.values())), u, side="right")]
    return pd.Series(tiers, index=unique).reindex(advertisers).to_numpy()


def repeat_penalty(history):
    """Click rate of a creative the user already saw in the previous 24h, relative to a returning user's first exposure to it."""
    exposures = key_history(history, ["user", "C14"], window=CANDIDATE_WINDOW_HOURS)["hist_n"].to_numpy()
    returning = history["user_seen_before"].to_numpy() == 1
    click = history["click"].to_numpy()
    return float(click[exposures > 0].mean() / click[returning & (exposures == 0)].mean())


def campaign_repeat_penalty(history):
    """
    Click rate when the user saw this campaign (C17) in the previous 24h but not this creative, relative to a returning
    user's first exposure to the campaign.
    """
    campaign = key_history(history, ["user", "C17"], window=CANDIDATE_WINDOW_HOURS)["hist_n"].to_numpy()
    creative = key_history(history, ["user", "C14"], window=CANDIDATE_WINDOW_HOURS)["hist_n"].to_numpy()
    returning = history["user_seen_before"].to_numpy() == 1
    click = history["click"].to_numpy()
    return float(click[(campaign > 0) & (creative == 0)].mean() / click[returning & (campaign == 0)].mean())


class Pacer:
    """
    Daily impression budget per advertiser (C17) = its volume on the previous day (median budget for new advertisers).
    Budgets are expected to be spent along the day's traffic curve: an advertiser ahead of that pace has its score scaled
    by expected / delivered, and one that has delivered its whole budget is removed. Delivery is updated hour by hour.
    """

    def __init__(self, history, day):
        previous = history[history["day"] == day - pd.Timedelta(days=1)]
        self.budget = previous.groupby("C17").size().astype("float64")
        self.default_budget = float(self.budget.median())
        traffic = history[history["day"] < day].groupby("hour_of_day").size()
        self.share_by_end_of_hour = (traffic / traffic.sum()).cumsum()
        self.delivered = pd.Series(dtype="float64")

    def adjust(self, advertisers, hour_of_day):
        """(score multiplier, budget exhausted) per advertiser at the start of this hour."""
        budget = self.budget.reindex(advertisers).fillna(self.default_budget).to_numpy()
        delivered = self.delivered.reindex(advertisers).fillna(0).to_numpy()
        expected = budget * self.share_by_end_of_hour.loc[hour_of_day]
        multiplier = np.where(delivered > expected, expected / np.maximum(delivered, 1), 1.0)
        return multiplier, delivered >= budget

    def record(self, advertisers):
        self.delivered = self.delivered.add(pd.Series(advertisers).value_counts(), fill_value=0)


# ---------- Goals 3-4 layer ----------

@dataclass(frozen=True)
class Layer:
    """Adaptation and exploration layer; rank_day(..., layer=None) is the Goal 2 policy."""
    campaign_repeat_ratio: float
    dominant_by_hour: dict  # hour_idx -> set of dominant cohorts


def dominant_cohorts(history, k=DOMINANT_COHORTS, window=CANDIDATE_WINDOW_HOURS):
    """For each hour: the k cohorts (genre x safety tier) with the most impressions in the previous `window` hours."""
    counts = history.groupby(["hour_idx", "cohort"]).size().unstack(fill_value=0)
    counts = counts.reindex(range(counts.index.min(), counts.index.max() + 2), fill_value=0)
    recent = counts.rolling(window, min_periods=1).sum().shift(1).dropna()
    return {hour: set(row.nlargest(k).index) for hour, row in recent.iterrows()}


def add_layer_features(rows, history):
    """Per candidate, from hours before the opportunity: the creative's impressions in the graduation window, the user's
    exposures to this campaign in the last 24h, and the campaign's share of this cohort's impressions in the last 24h."""
    rows["creative_impressions_72h"] = window_totals(history, ["C14"], rows, window=GRADUATION_WINDOW_HOURS)[0]
    rows["user_campaign_exposures_24h"] = window_totals(history, ["user", "C17"], rows)[0]
    cohort_campaign = window_totals(history, ["cohort", "C17"], rows)[0]
    cohort_total = window_totals(history, ["cohort"], rows)[0]
    rows["cohort_campaign_share_24h"] = np.divide(cohort_campaign, cohort_total, out=np.zeros(len(rows)),
                                                  where=cohort_total > 0)
    return rows


# ---------- ordering ----------

def rank_hour(cands, pacer, repeat_ratio, rng, n=N_CANDIDATES, layer=None):
    """
    Rank the candidates of one hour's opportunities.
    Retrieval first skips advertisers whose budget is already spent (known before ranking): the model receives the shown ad
    plus the first n - 1 other candidates, in retrieval order, whose advertiser can still serve.
    Blocked: advertiser does not accept the character's safety tier, fatigue cap reached, or budget exhausted.
    Score z = logit(V1) + log(pacing multiplier) + log(repeat penalty if the user saw this creative in the last 24h).
    Uncertainty band z +/- u, u = half the gap between LightGBM and the FM in log-odds (V1 sits in the middle).
    Toss-up: the runner-up's band overlaps the leader's. In a toss-up, while this hour's exploration budget lasts, the
    tied candidate with the least recent data (fewest impressions in the last 24h) goes first, if it is not the leader.

    With the Goals 3-4 layer (reorders only, never skips an ad):
    - repeat penalty also when the user saw the same campaign (not this creative) in the last 24h;
    - band widened for data scarcity: u = sqrt(u_model^2 + u_data^2), u_data = 1 / sqrt((n + k) p (1 - p)) with n the
      creative's impressions in the last 72h and k the smoothing prior; a creative is cold until n reaches 1,000;
    - cold creative shown first if its optimistic end reaches the leader's estimate (z + u >= z_leader): the one with the
      highest optimistic end, most promising opportunities first, up to COLD_EXPLORATION_SHARE of the hour's opportunities
      (replaces the least-known toss-up rule);
    - in a toss-up in a dominant cohort, the tied candidate whose campaign has the smallest share of the cohort's last-24h
      impressions goes first (spreads ads within the cohorts users see most).
    """
    c = cands.sort_values(["opportunity", "draw_order"])
    hour_of_day = int(c["hour_of_day"].iloc[0])
    _, spent = pacer.adjust(c["C17"], hour_of_day)
    c = c[c["is_logged"].to_numpy() | ~spent]
    c = c[c.groupby("opportunity").cumcount() < n].copy()
    c["pacing"], exhausted = pacer.adjust(c["C17"], hour_of_day)
    c["repeat_penalty"] = np.where(c["user_creative_exposures_24h"] > 0, repeat_ratio, 1.0)
    if layer is not None:
        c["repeat_penalty"] = np.where((c["user_creative_exposures_24h"] == 0) & (c["user_campaign_exposures_24h"] > 0),
                                       layer.campaign_repeat_ratio, c["repeat_penalty"])
    c["blocked"] = np.select(
        [c["character_tier_rank"] > c["advertiser_tier_rank"], c["user_creative_exposures_24h"] >= FATIGUE_HARD_CAP,
         exhausted], ["safety", "fatigue_cap", "budget"], "")
    c["z"] = logit(c["p_v1"].to_numpy()) + np.log(c["pacing"]) + np.log(c["repeat_penalty"])
    c["u"] = np.abs(logit(c["p_lightgbm"].to_numpy()) - logit(c["p_fm"].to_numpy())) / 2
    if layer is not None:
        p = c["p_v1"].to_numpy()
        u_data = 1 / np.sqrt((c["creative_impressions_72h"].to_numpy() + HISTORY_SMOOTHING) * p * (1 - p))
        c["u"] = np.sqrt(c["u"] ** 2 + u_data ** 2)
        c["cold"] = c["creative_impressions_72h"] < GRADUATION_IMPRESSIONS

    ok = c[c["blocked"] == ""].sort_values(["opportunity", "z"], ascending=[True, False])
    ok["order"] = ok.groupby("opportunity").cumcount()
    leader_low = (ok["z"] - ok["u"]).where(ok["order"] == 0).groupby(ok["opportunity"]).transform("max")
    ok["tied"] = (ok["z"] + ok["u"]) >= leader_low
    tossups = ok.loc[(ok["order"] == 1) & ok["tied"], "opportunity"].to_numpy()
    diversify_pick = ok.index[:0]
    if layer is None:
        # In each toss-up, the least-known tied candidate; exploring only changes something when it is not the leader.
        least_known = (ok[ok["opportunity"].isin(tossups) & ok["tied"]]
                       .sort_values(["opportunity", "creative_impressions_24h", "z"], ascending=[True, True, False])
                       .groupby("opportunity").head(1))
        least_known = least_known[least_known["order"] != 0]
        n_explore = int(EXPLORATION_SHARE * c["opportunity"].nunique())
        chosen = set(rng.permutation(least_known["opportunity"].to_numpy())[:n_explore])
        explore_pick = least_known[least_known["opportunity"].isin(chosen)].index
    else:
        leader_z = ok["z"].where(ok["order"] == 0).groupby(ok["opportunity"]).transform("max")
        upper = ok["z"] + ok["u"]
        qualifies = ok["cold"] & (ok["order"] != 0) & (upper >= leader_z)
        best_cold = (ok[qualifies].assign(upper=upper[qualifies], margin=(upper - leader_z)[qualifies])
                     .sort_values(["opportunity", "upper"], ascending=[True, False]).groupby("opportunity").head(1))
        n_explore = int(COLD_EXPLORATION_SHARE * c["opportunity"].nunique())
        explore_pick = best_cold.nlargest(n_explore, "margin").index
        explored_opps = ok.loc[explore_pick, "opportunity"]
        dominant = ok["cohort"].isin(layer.dominant_by_hour.get(int(c["hour_idx"].iloc[0]), set()))
        spread = (ok[ok["opportunity"].isin(tossups) & ~ok["opportunity"].isin(explored_opps) & dominant & ok["tied"]]
                  .sort_values(["opportunity", "cohort_campaign_share_24h", "z"], ascending=[True, True, False])
                  .groupby("opportunity").head(1))
        diversify_pick = spread[spread["order"] != 0].index
    ok["priority"] = np.where(ok.index.isin(explore_pick) | ok.index.isin(diversify_pick), -1, ok["order"])
    ok = ok.sort_values(["opportunity", "priority"])
    ok["final_rank"] = ok.groupby("opportunity").cumcount() + 1

    c["final_rank"] = ok["final_rank"].reindex(c.index)
    c["tossup"] = c["opportunity"].isin(tossups)
    c["explored"] = c.index.isin(explore_pick)
    c["diversified"] = c.index.isin(diversify_pick)
    explore_reason = ("toss-up -> explore least-known candidate" if layer is None
                      else "cold creative -> explore (its optimistic end reaches the leader)")
    c["pick_reason"] = np.select(
        [c["final_rank"] != 1, c["explored"], c["diversified"], c["tossup"]],
        ["", explore_reason, "dominant-cohort toss-up -> least-shown campaign", "toss-up -> highest score"],
        "clear winner")
    pacer.record(c.loc[c["final_rank"] == 1, "C17"].to_numpy())
    return c


def rank_day(cands, pacer, repeat_ratio, rng, layer=None):
    """Rank hour by hour so budget pacing sees the delivery of earlier hours. layer=None is the Goal 2 policy."""
    cands = cands.assign(character_tier_rank=cands["safety_tier"].map(TIER_RANK).to_numpy(),
                         advertiser_tier_rank=pd.Series(advertiser_max_tier(cands["C17"])).map(TIER_RANK).to_numpy())
    return pd.concat([rank_hour(group, pacer, repeat_ratio, rng, layer=layer)
                      for _, group in cands.groupby("hour_idx", sort=True)])
