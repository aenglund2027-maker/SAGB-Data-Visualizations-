"""
Favorite / underdog assignment and the upset label.

The definition is a single switch (config.UPSET_DEFINITION), so the whole
pipeline can move between ranking-based and Elo-based upsets without code
changes.

  "rank" (current): favorite = player with the BETTER pre-match ATP ranking,
                    i.e. the numerically SMALLER rank (No. 20 is favored over No. 70).
  "elo":            favorite = player with the higher pre-match Elo.

upset = 1 if the underdog won, 0 if the favorite won, NaN if no favorite can
be defined (missing ranking under the "exclude" policy, or an exact tie).

Regardless of the definition, BOTH strength measures are stored from the
favorite's point of view:
  rank_diff      = dog_rank - fav_rank                (ranking places)
  log_rank_diff  = ln(dog_rank) - ln(fav_rank)        (relative ranking gap)
  elo_diff       = fav_elo - dog_elo                  (can be negative under "rank":
                                                       the rank favorite may have the lower Elo)
  elo_prob_dog   = Elo-formula probability that the underdog wins
"""
import numpy as np
import pandas as pd

from src import config
from src.elo import expected_score

# Per-player columns copied to fav_/dog_ form (if present in the frame)
PLAYER_FIELDS = ["id", "name", "rank", "rank_points", "elo_pre", "elo_n", "age", "hand", "ht"]


def effective_ranks(df, policy=None):
    """Pre-match ranks after applying the missing-rank policy."""
    policy = policy or config.MISSING_RANK_POLICY
    w, l = df["winner_rank"].copy(), df["loser_rank"].copy()
    if policy == "unranked_as_worst":
        w, l = w.fillna(config.UNRANKED_RANK_VALUE), l.fillna(config.UNRANKED_RANK_VALUE)
    elif policy != "exclude":
        raise ValueError(f"unknown MISSING_RANK_POLICY {policy!r}")
    return w, l


def assign_favorite_underdog(df, definition=None, policy=None):
    definition = definition or config.UPSET_DEFINITION
    df = df.copy()
    w_rank, l_rank = effective_ranks(df, policy)

    if definition == "rank":
        label_ok = w_rank.notna() & l_rank.notna() & (w_rank != l_rank)
        winner_is_fav = w_rank < l_rank            # smaller number = better ranking
    elif definition == "elo":
        label_ok = df["winner_elo_pre"] != df["loser_elo_pre"]
        winner_is_fav = df["winner_elo_pre"] > df["loser_elo_pre"]
    else:
        raise ValueError(f"unknown UPSET_DEFINITION {definition!r}")

    df["upset_definition"] = definition
    df["upset_label_ok"] = label_ok
    df["winner_is_fav"] = winner_is_fav
    df["upset"] = np.where(label_ok, (~winner_is_fav).astype(float), np.nan)

    for f in PLAYER_FIELDS:
        wc, lc = f"winner_{f}", f"loser_{f}"
        if wc in df.columns:
            df[f"fav_{f}"] = np.where(winner_is_fav, df[wc], df[lc])
            df[f"dog_{f}"] = np.where(winner_is_fav, df[lc], df[wc])
    # Effective ranks (after the missing-rank policy) for the strength features
    df["fav_rank_eff"] = np.where(winner_is_fav, w_rank, l_rank)
    df["dog_rank_eff"] = np.where(winner_is_fav, l_rank, w_rank)

    df["rank_diff"] = df["dog_rank_eff"] - df["fav_rank_eff"]
    df["log_rank_diff"] = np.log(df["dog_rank_eff"]) - np.log(df["fav_rank_eff"])
    if "winner_elo_pre" in df.columns:
        df["elo_diff"] = df["fav_elo_pre"] - df["dog_elo_pre"]
        df["elo_prob_dog"] = expected_score(df["dog_elo_pre"], df["fav_elo_pre"])

    # Any profile columns (winner_prof_* / loser_prof_*) are mirrored too
    for wc in [c for c in df.columns if c.startswith("winner_prof_") or c == "winner_profile_ok"]:
        base = wc[len("winner_"):]
        df[f"fav_{base}"] = np.where(winner_is_fav, df[wc], df[f"loser_{base}"])
        df[f"dog_{base}"] = np.where(winner_is_fav, df[f"loser_{base}"], df[wc])
    return df
