"""
Step 3: leakage-free pre-match profiles (seven shared variables), with verification.

Input:  data/processed/atp_matches_elo.csv
Output: data/processed/player_profiles_long.csv   (one row per player per match; shareable with partner)
        data/processed/atp_matches_profiles.csv   (one row per match, winner_prof_* / loser_prof_*)
Run from the project root:  python3 run_03_profiles.py
"""
import numpy as np
import pandas as pd

from src import config
from src.profiles import (PROFILE_COLS, PROFILE_MIN_MATCHES, PROFILE_STATS, PROFILE_WINDOW_DAYS,
                          attach_profiles, build_profiles)

PROFILES_LONG_PATH = config.PROCESSED_DIR / "player_profiles_long.csv"
PROFILE_MATCHES_PATH = config.PROCESSED_DIR / "atp_matches_profiles.csv"
pd.set_option("display.width", 160)


def brute_force_profile(df, player_id, match_order, date):
    """Recompute all seven stats the slow, obvious way, straight from the match table."""
    prior = df[(df["match_order"] < match_order)
               & (df["tourney_date"] > date - pd.Timedelta(days=PROFILE_WINDOW_DAYS))
               & df["completed"] & df["stats_ok"]
               & ((df["winner_id"] == player_id) | (df["loser_id"] == player_id))]
    won = prior["winner_id"] == player_id
    own = lambda s: np.where(won, prior[f"w_{s}"], prior[f"l_{s}"]).sum()
    opp = lambda s: np.where(won, prior[f"l_{s}"], prior[f"w_{s}"]).sum()
    svpt, first_in, o_svpt = own("svpt"), own("1stIn"), opp("svpt")
    div = lambda a, b: a / b if b else np.nan
    return {
        "n": len(prior),
        "ace_rate": div(own("ace"), svpt),
        "df_rate": div(own("df"), svpt),
        "first_in_pct": div(first_in, svpt),
        "first_won_pct": div(own("1stWon"), first_in),
        "second_won_pct": div(own("2ndWon"), svpt - first_in),
        "spw": div(own("1stWon") + own("2ndWon"), svpt),
        "rpw": div(o_svpt - opp("1stWon") - opp("2ndWon"), o_svpt),
    }


def verify_brute_force(df, prof, n_samples=300, seed=0):
    sample = prof.sample(n_samples, random_state=seed)
    worst, n_bad = 0.0, 0
    for _, r in sample.iterrows():
        bf = brute_force_profile(df, r["player_id"], r["match_order"], r["tourney_date"])
        n_bad += bf["n"] != r["prof_n_matches"]
        for k in PROFILE_STATS:
            a, b = bf[k], r[f"prof_{k}"]
            if not (np.isnan(a) and np.isnan(b)):
                worst = max(worst, abs(a - b))
    print(f"Brute-force check on {n_samples} random player-matches: "
          f"{n_bad} match-count mismatches, max abs difference across all 7 stats = {worst:.2e}")

    r = prof[(prof["player_name"] == "Carlos Alcaraz") & (prof["tourney_date"] == "2023-07-03")].iloc[-1]
    bf = brute_force_profile(df, r["player_id"], r["match_order"], r["tourney_date"])
    print(f"\nWorked example: Alcaraz entering the 2023 Wimbledon final "
          f"(window = matches after {(r['tourney_date'] - pd.Timedelta(days=PROFILE_WINDOW_DAYS)).date()}, "
          f"before this match)")
    print(f"  {'matches':<15s} pipeline={r['prof_n_matches']:<8d} brute force={bf['n']}")
    for k in PROFILE_STATS:
        print(f"  {k:<15s} pipeline={r[f'prof_{k}']:.4f}   brute force={bf[k]:.4f}")


def verify_no_leakage(df):
    """Corrupt one match's stats; only profiles AFTER that match may change."""
    target = df[(df["winner_name"] == "Novak Djokovic") & df["stats_ok"] & (df["season"] == 2023)].iloc[0]
    base = build_profiles(df).set_index(["match_id", "player_id"])
    corrupted = df.copy()
    row = corrupted["match_id"] == target["match_id"]
    corrupted.loc[row, "w_ace"] = corrupted.loc[row, "w_svpt"]
    corrupted.loc[row, "l_2ndWon"] = 0
    new = build_profiles(corrupted).set_index(["match_id", "player_id"])

    changed = (base[PROFILE_COLS].fillna(-1) != new[PROFILE_COLS].fillna(-1)).any(axis=1)
    earliest = base.loc[changed, "match_order"].min()
    print(f"\nPerturbation test: corrupted {target['tourney_name']} {target['round']} "
          f"({target['winner_name']} d. {target['loser_name']}, match_order {target['match_order']})")
    print(f"  profile rows changed: {changed.sum()}; earliest changed match_order = {earliest} -> "
          f"{'OK, no leakage' if earliest > target['match_order'] else 'LEAKAGE!'}")
    at = (target["match_id"], target["winner_id"])
    print(f"  Djokovic ace_rate AT corrupted match: before={base.loc[at, 'prof_ace_rate']:.4f} "
          f"after={new.loc[at, 'prof_ace_rate']:.4f} (must be identical)")
    first = base.reset_index().sort_values("match_order").groupby("player_id").head(1)
    print(f"  every player's first match has 0 prior matches: {(first['prof_n_matches'] == 0).all()}")


def distribution_report(prof):
    ok = prof[prof["profile_ok"] & (prof["tourney_date"] >= config.ELO_BURN_IN_END)]
    print(f"\nProfile distributions (player-matches with profile_ok, 2015+; n={len(ok):,}):")
    desc = ok[PROFILE_COLS].describe(percentiles=[0.05, 0.5, 0.95]).T
    print(desc[["mean", "std", "min", "5%", "50%", "95%", "max"]].round(3).to_string())
    print(f"NaNs among profile_ok rows: {int(ok[PROFILE_COLS].isna().sum().sum())}")
    print("\nCorrelations among the seven variables:")
    c = ok[PROFILE_COLS].corr()
    c.index = c.columns = list(PROFILE_STATS)
    print(c.round(2).to_string())

    print(f"\nPlayer-match rows by prior stat matches in window (all seasons):")
    print(pd.cut(prof["prof_n_matches"], [-1, 0, 4, 9, 19, 39, 1000],
                 labels=["0", "1-4", "5-9", "10-19", "20-39", "40+"]).value_counts().sort_index().to_string())


def sanity_leaders(prof, date="2025-11-15", min_matches=25):
    d = pd.Timestamp(date)
    snap = prof[prof["tourney_date"] <= d].sort_values("match_order").groupby("player_id").tail(1)
    snap = snap[(snap["prof_n_matches"] >= min_matches) & (snap["tourney_date"] > d - pd.Timedelta(days=60))]
    print(f"\nLeaders as of {d.date()} (>= {min_matches} matches in window, active last 60 days; {len(snap)} players):")
    for stat in PROFILE_STATS:
        for asc in (False, True):
            top = snap.sort_values(f"prof_{stat}", ascending=asc).head(4)
            print(f"  {('lowest ' if asc else 'highest ') + stat:<22s} "
                  + ", ".join(f"{n} {v:.3f}" for n, v in zip(top["player_name"], top[f"prof_{stat}"])))


if __name__ == "__main__":
    df = pd.read_csv(config.ELO_MATCHES_PATH, parse_dates=["tourney_date"]).sort_values("match_order").reset_index(drop=True)
    prof = build_profiles(df)
    print("=" * 70)
    print(f"PROFILES: previous {PROFILE_WINDOW_DAYS} days, point-weighted; profile_ok = >= {PROFILE_MIN_MATCHES} matches")
    print("=" * 70)
    print("Formulas (window sums of Sackmann counts):")
    for k, (num, den) in PROFILE_STATS.items():
        print(f"  {k:<15s} = ({num}) / ({den})")
    print(f"\nPlayer-match rows: {len(prof):,} (2 per match x {len(df):,} matches)")

    print("\n--- Leakage / correctness verification ---")
    verify_brute_force(df, prof)
    verify_no_leakage(df)
    distribution_report(prof)
    sanity_leaders(prof)

    m = attach_profiles(df, prof)
    prof.to_csv(PROFILES_LONG_PATH, index=False)
    m.to_csv(PROFILE_MATCHES_PATH, index=False)
    print(f"\nSaved -> {PROFILES_LONG_PATH}\nSaved -> {PROFILE_MATCHES_PATH}")
