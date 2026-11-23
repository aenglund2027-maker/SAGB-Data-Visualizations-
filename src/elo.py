"""
Step 2: chronological pre-match Elo ratings (overall, not surface-specific).

Implementation: the FiveThirtyEight tennis Elo, which Kovalchik (2016,
"Searching for the GOAT of tennis win prediction") found to be the best-
performing published Elo variant for ATP matches.

  Expected score for A:  E_A = 1 / (1 + 10 ** ((Elo_B - Elo_A) / 400))
  Update after match:    Elo_A_new = Elo_A + K_A * (S_A - E_A)
                         where S_A = 1 if A won, 0 if A lost
  Player-specific K:     K_A = 250 / (n_A + 5) ** 0.4
                         n_A = number of completed matches A had played before this one

Why a decaying K? New players start at 1500 with no information, so their
rating should move fast (K = 131 for a debut match). Established players have
stable ratings (K ~ 30 after 100 matches, ~ 20 after 300). Because each player
uses their own K, rating changes are not exactly zero-sum.

New players: everyone starts at ELO_START (1500) the first time they appear.
Since the dataset starts in Jan 2014, every player (including Djokovic, Nadal,
Federer) starts at 1500 then, so 2014 is treated as a burn-in season.

Only COMPLETED matches update Elo. Retirements/defaults are skipped (they
mostly reflect injury, not skill). The pre-match rating is still recorded for
every match, completed or not, so nothing downstream is missing an Elo.
"""
import numpy as np
import pandas as pd

from src import config


def expected_score(elo_a, elo_b):
    """P(A beats B) under the standard Elo logistic curve."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def k_factor(n_prev_matches):
    return config.ELO_K_NUMERATOR / (n_prev_matches + config.ELO_K_OFFSET) ** config.ELO_K_SHAPE


def compute_elo(df):
    """
    Walk through matches in chronological order (df must be sorted by
    match_order) and attach each player's rating BEFORE the match.

    Adds columns:
      winner_elo_pre, loser_elo_pre     rating before this match
      winner_elo_n, loser_elo_n         completed matches played before this one
      winner_elo_post, loser_elo_post   rating after (for verification only; never a feature)
    """
    assert df["match_order"].is_monotonic_increasing, "matches must be chronologically sorted"

    rating = {}   # player_id -> current Elo
    n_played = {}  # player_id -> completed matches so far

    out = {k: np.empty(len(df)) for k in
           ["winner_elo_pre", "loser_elo_pre", "winner_elo_n", "loser_elo_n",
            "winner_elo_post", "loser_elo_post"]}

    winners = df["winner_id"].to_numpy()
    losers = df["loser_id"].to_numpy()
    completed = df["completed"].to_numpy()

    for i in range(len(df)):
        w, l = winners[i], losers[i]
        r_w = rating.get(w, config.ELO_START)
        r_l = rating.get(l, config.ELO_START)
        n_w = n_played.get(w, 0)
        n_l = n_played.get(l, 0)

        # --- record PRE-match state (this is what features may use) ---
        out["winner_elo_pre"][i], out["loser_elo_pre"][i] = r_w, r_l
        out["winner_elo_n"][i], out["loser_elo_n"][i] = n_w, n_l

        # --- update only after completed matches ---
        if completed[i]:
            e_w = expected_score(r_w, r_l)  # winner's pre-match win probability
            r_w_new = r_w + k_factor(n_w) * (1.0 - e_w)
            r_l_new = r_l + k_factor(n_l) * (0.0 - (1.0 - e_w))
            rating[w], rating[l] = r_w_new, r_l_new
            n_played[w], n_played[l] = n_w + 1, n_l + 1
        else:
            r_w_new, r_l_new = r_w, r_l

        out["winner_elo_post"][i], out["loser_elo_post"][i] = r_w_new, r_l_new

    df = df.copy()
    for k, v in out.items():
        df[k] = v
    df["winner_elo_n"] = df["winner_elo_n"].astype(int)
    df["loser_elo_n"] = df["loser_elo_n"].astype(int)
    return df


def add_burn_in_flag(df):
    """Matches in the burn-in season (ratings not yet meaningful)."""
    df = df.copy()
    df["elo_burn_in"] = df["tourney_date"] < pd.Timestamp(config.ELO_BURN_IN_END)
    return df
