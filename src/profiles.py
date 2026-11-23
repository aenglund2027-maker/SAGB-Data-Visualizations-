"""
Step 3: leakage-free pre-match serve/return profiles (the SHARED seven variables).

These seven definitions are shared with my partner's PCA/clustering work, so
they must stay exactly as written here.

For each player in each match, using ONLY that player's earlier matches:
  - window: matches in the previous PROFILE_WINDOW_DAYS (52 weeks) whose
    match_order is strictly smaller than the current match (earlier rounds of
    the same tournament count; the current match never does)
  - only completed matches with valid stats (stats_ok) contribute

Rates are POINT-WEIGHTED: Sackmann counts are summed over the window, then
divided. Sackmann column names (per player): svpt = serve points, ace, df,
1stIn = first serves in, 1stWon = points won on 1st serve, 2ndWon = points won
on 2nd serve (double faults are 2nd-serve points lost).

  1. ace_rate        = ace / svpt
  2. df_rate         = df / svpt
  3. first_in_pct    = 1stIn / svpt
  4. first_won_pct   = 1stWon / 1stIn
  5. second_won_pct  = 2ndWon / (svpt - 1stIn)
  6. spw             = (1stWon + 2ndWon) / svpt                      (service points won %)
  7. rpw             = (opp_svpt - opp_1stWon - opp_2ndWon) / opp_svpt  (return points won %)
     where opp_* are the OPPONENT's serve counts in the same matches: points the
     opponent lost on serve are points this player won on return.

Why 52 weeks? A regular tour player accumulates ~40 matches (~3,000 serve
points) in that time, which pins spw down to about +/-1 percentage point, while
still tracking changes in form and style. Players with fewer than
PROFILE_MIN_MATCHES matches in the window are flagged profile_ok = False (not dropped).
"""
import numpy as np
import pandas as pd

# Raw counts accumulated per player-match ("opp_" = the opponent's serve)
COUNT_COLS = ["svpt", "ace", "df", "first_in", "first_won", "second_won",
              "opp_svpt", "opp_first_won", "opp_second_won"]

# name -> (numerator, denominator), evaluated on window sums
PROFILE_STATS = {
    "ace_rate":       ("ace", "svpt"),
    "df_rate":        ("df", "svpt"),
    "first_in_pct":   ("first_in", "svpt"),
    "first_won_pct":  ("first_won", "first_in"),
    "second_won_pct": ("second_won", "svpt - first_in"),
    "spw":            ("first_won + second_won", "svpt"),
    "rpw":            ("opp_svpt - opp_first_won - opp_second_won", "opp_svpt"),
}
PROFILE_COLS = [f"prof_{k}" for k in PROFILE_STATS]

PROFILE_WINDOW_DAYS = 364
PROFILE_MIN_MATCHES = 10


def build_player_match_table(df):
    """
    One row per player per match with the player's own serve counts and the
    opponent's. `contributes` marks rows whose stats may enter FUTURE profiles.
    """
    sides = []
    for me, p, o in (("winner", "w", "l"), ("loser", "l", "w")):
        sides.append(pd.DataFrame({
            "match_id": df["match_id"],
            "match_order": df["match_order"],
            "tourney_date": df["tourney_date"],
            "player_id": df[f"{me}_id"],
            "player_name": df[f"{me}_name"],
            "contributes": df["completed"] & df["stats_ok"],
            "svpt": df[f"{p}_svpt"], "ace": df[f"{p}_ace"], "df": df[f"{p}_df"],
            "first_in": df[f"{p}_1stIn"], "first_won": df[f"{p}_1stWon"], "second_won": df[f"{p}_2ndWon"],
            "opp_svpt": df[f"{o}_svpt"], "opp_first_won": df[f"{o}_1stWon"], "opp_second_won": df[f"{o}_2ndWon"],
        }))
    long = pd.concat(sides, ignore_index=True)
    long.loc[~long["contributes"], COUNT_COLS] = 0     # non-contributing rows add nothing
    long[COUNT_COLS] = long[COUNT_COLS].fillna(0)
    return long.sort_values(["player_id", "match_order"]).reset_index(drop=True)


def add_window_sums(long, window_days=PROFILE_WINDOW_DAYS):
    """
    For each player-match row, sum the counts over the player's PREVIOUS matches
    within the window. Per player: cum[i] = sum of rows 0..i-1 (a leading zero
    row makes it strictly-before). The window start j is the first earlier row
    with tourney_date > current date - window. Window sum = cum[i] - cum[j].
    """
    sums = np.zeros((len(long), len(COUNT_COLS)))
    n_stat = np.zeros(len(long), dtype=int)
    counts = long[COUNT_COLS].to_numpy(dtype=float)
    contrib = long["contributes"].to_numpy(dtype=int)
    dates = long["tourney_date"].to_numpy()

    for _, idx in long.groupby("player_id", sort=False).indices.items():
        c = np.vstack([np.zeros(len(COUNT_COLS)), np.cumsum(counts[idx], axis=0)])
        cn = np.concatenate([[0], np.cumsum(contrib[idx])])
        d = dates[idx]
        start = np.searchsorted(d, d - np.timedelta64(window_days, "D"), side="right")
        pos = np.arange(len(idx))
        sums[idx] = c[pos] - c[start]
        n_stat[idx] = cn[pos] - cn[start]

    out = long[["match_id", "match_order", "tourney_date", "player_id", "player_name"]].copy()
    out[[f"w_{c}" for c in COUNT_COLS]] = sums
    out["prof_n_matches"] = n_stat
    return out


def compute_rates(win):
    """Turn window sums into the seven rates. Zero denominators -> NaN."""
    env = {c: win[f"w_{c}"] for c in COUNT_COLS}
    for name, (num, den) in PROFILE_STATS.items():
        n, d = eval(num, {}, env), eval(den, {}, env)
        win[f"prof_{name}"] = n / d.where(d > 0)
    win["prof_serve_points"] = win["w_svpt"]
    win["prof_return_points"] = win["w_opp_svpt"]
    win["profile_ok"] = win["prof_n_matches"] >= PROFILE_MIN_MATCHES
    return win.drop(columns=[f"w_{c}" for c in COUNT_COLS])


def build_profiles(df, window_days=PROFILE_WINDOW_DAYS):
    return compute_rates(add_window_sums(build_player_match_table(df), window_days))


def attach_profiles(df, prof):
    """Merge pre-match profiles onto matches as winner_prof_* and loser_prof_*."""
    cols = PROFILE_COLS + ["prof_n_matches", "prof_serve_points", "prof_return_points", "profile_ok"]
    keyed = prof.set_index(["match_id", "player_id"])[cols]
    out = df.reset_index(drop=True).copy()
    for side in ("winner", "loser"):
        idx = pd.MultiIndex.from_arrays([out["match_id"], out[f"{side}_id"]])
        block = keyed.reindex(idx).reset_index(drop=True)
        block.columns = [f"{side}_{c}" for c in block.columns]
        out = pd.concat([out, block], axis=1)
    return out
