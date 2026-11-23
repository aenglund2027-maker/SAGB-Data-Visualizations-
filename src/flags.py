"""
"Ranking was out of date" flags for Giant Killer opportunities (Graph 4).

A ranking-based expectation treats a player's current ranking as his true
level. Two common situations break that, and they are flagged per match:

  comeback : the underdog's best ranking in the previous 3 years was top-30
             AND at least 3x better than his current ranking (e.g. Djokovic
             ranked 22 after elbow surgery). Uses only PAST rankings.
  riser    : the underdog was aged <= 23 AND his ranking ~1 year later was at
             least 2x better than now (his ranking hadn't caught up with his
             level yet). Uses FUTURE rankings - fine for a descriptive label,
             never used as a model feature.

Ranking history comes from the pre-match rankings in the match files (every
appearance gives a dated ranking), so no extra data is needed.
All thresholds live in config.py.
"""
import numpy as np
import pandas as pd

from src import config


def ranking_history(clean):
    """One row per player per appearance: date and pre-match ranking (all retained matches)."""
    h = pd.concat([
        clean[["tourney_date", "winner_id", "winner_rank"]].set_axis(["date", "player_id", "rank"], axis=1),
        clean[["tourney_date", "loser_id", "loser_rank"]].set_axis(["date", "player_id", "rank"], axis=1),
    ]).dropna()
    return h.drop_duplicates(["player_id", "date"]).sort_values(["player_id", "date"]).reset_index(drop=True)


def flag_opportunities(opps, clean):
    """
    opps: underdog opportunities with dog_id, tourney_date, dog_rank, dog_age.
    Adds dog_peak_prev3y, dog_rank_next_year, comeback, riser.
    """
    hist = ranking_history(clean)
    by_player = {pid: (g["date"].to_numpy(), g["rank"].to_numpy()) for pid, g in hist.groupby("player_id")}
    lookback = np.timedelta64(config.FLAG_COMEBACK_LOOKBACK_DAYS, "D")
    year, slack = np.timedelta64(365, "D"), np.timedelta64(120, "D")

    peak = np.full(len(opps), np.nan)
    nxt = np.full(len(opps), np.nan)
    for i, (pid, date) in enumerate(zip(opps["dog_id"].to_numpy(), opps["tourney_date"].to_numpy())):
        dates, ranks = by_player[pid]
        lo = np.searchsorted(dates, date - lookback, side="left")
        hi = np.searchsorted(dates, date, side="left")          # strictly earlier tournaments
        if hi > lo:
            peak[i] = ranks[lo:hi].min()
        j = np.searchsorted(dates, date + year, side="left")    # first appearance >= 1 year later
        if j < len(dates) and dates[j] <= date + year + slack:
            nxt[i] = ranks[j]

    out = opps.copy()
    out["dog_peak_prev3y"] = peak
    out["dog_rank_next_year"] = nxt
    out["comeback"] = ((out["dog_peak_prev3y"] <= config.FLAG_COMEBACK_MAX_PEAK)
                       & (out["dog_rank"] >= config.FLAG_COMEBACK_RATIO * out["dog_peak_prev3y"]))
    out["riser"] = ((out["dog_age"] <= config.FLAG_RISER_MAX_AGE)
                    & (out["dog_rank_next_year"] * config.FLAG_RISER_RATIO <= out["dog_rank"]))
    return out


def player_flags(flagged):
    """Share of each player's opportunities that are comeback / riser matches, and the resulting label."""
    g = flagged.groupby("dog_id")
    f = pd.DataFrame({"comeback_share": g["comeback"].mean(), "riser_share": g["riser"].mean()})
    f["flag"] = np.select(
        [f["comeback_share"] >= config.FLAG_PLAYER_SHARE, f["riser_share"] >= config.FLAG_PLAYER_SHARE],
        ["comeback", "riser"], default="none")
    # A player can qualify for both; the larger share wins (reported in the table either way)
    both = (f["comeback_share"] >= config.FLAG_PLAYER_SHARE) & (f["riser_share"] >= config.FLAG_PLAYER_SHARE)
    f.loc[both, "flag"] = np.where(f.loc[both, "riser_share"] > f.loc[both, "comeback_share"], "riser", "comeback")
    return f
