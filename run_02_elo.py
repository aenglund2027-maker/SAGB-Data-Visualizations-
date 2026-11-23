"""
Step 2: compute chronological pre-match Elo and verify it.

Input:  data/processed/atp_matches_clean.csv  (from run_01_load_clean.py)
Output: data/processed/atp_matches_elo.csv
Run from the project root:  python3 run_02_elo.py
"""
import numpy as np
import pandas as pd

from src import config
from src.elo import add_burn_in_flag, compute_elo, expected_score, k_factor
from src.upsets import assign_favorite_underdog


def verify_by_hand(df, player_name, n_matches=3, start_index=0):
    """
    Recompute a player's consecutive Elo updates step by step and check
    (a) the stored post-match rating matches the hand calculation and
    (b) the next match's PRE rating equals the previous match's POST rating.
    """
    is_w = df["winner_name"] == player_name
    is_l = df["loser_name"] == player_name
    pm = df[(is_w | is_l) & df["completed"]].iloc[start_index:start_index + n_matches + 1]

    print(f"\n--- {player_name}: matches {start_index + 1}-{start_index + n_matches} ---")
    prev_post = None
    for _, m in pm.iloc[:n_matches].iterrows():
        won = m["winner_name"] == player_name
        me, opp = ("winner", "loser") if won else ("loser", "winner")
        r, r_opp = m[f"{me}_elo_pre"], m[f"{opp}_elo_pre"]
        n = m[f"{me}_elo_n"]
        e = expected_score(r, r_opp)
        k = k_factor(n)
        s = 1.0 if won else 0.0
        hand_post = r + k * (s - e)
        print(f"{m['tourney_date'].date()} {m['tourney_name']:<22s} {m['round']:<4s} vs {m[f'{opp}_name']:<22s} "
              f"{'W' if won else 'L'}\n"
              f"   pre={r:7.1f}  opp_pre={r_opp:7.1f}  n_prev={n:3d}  "
              f"E = 1/(1+10^(({r_opp:.1f}-{r:.1f})/400)) = {e:.3f}  "
              f"K = 250/({n}+5)^0.4 = {k:.1f}\n"
              f"   post = {r:.1f} + {k:.1f}*({s:.0f}-{e:.3f}) = {hand_post:.1f}   "
              f"stored post = {m[f'{me}_elo_post']:.1f}  "
              f"{'OK' if np.isclose(hand_post, m[f'{me}_elo_post']) else 'MISMATCH'}")
        if prev_post is not None:
            print(f"   continuity: this pre ({r:.1f}) == previous post ({prev_post:.1f}) -> "
                  f"{'OK' if np.isclose(r, prev_post) else 'MISMATCH'}")
        prev_post = m[f"{me}_elo_post"]


def global_continuity_check(df):
    """For every player, each match's pre-Elo must equal their previous match's post-Elo."""
    long = pd.concat([
        df[["match_order", "winner_id", "winner_elo_pre", "winner_elo_post"]]
        .set_axis(["match_order", "pid", "pre", "post"], axis=1),
        df[["match_order", "loser_id", "loser_elo_pre", "loser_elo_post"]]
        .set_axis(["match_order", "pid", "pre", "post"], axis=1),
    ]).sort_values(["pid", "match_order"])
    long["prev_post"] = long.groupby("pid")["post"].shift()
    first = long["prev_post"].isna()
    bad = ~np.isclose(long.loc[~first, "pre"], long.loc[~first, "prev_post"])
    print(f"\nGlobal continuity check: {(~first).sum():,} player-matches checked, {bad.sum()} mismatches")
    print(f"All first appearances start at {config.ELO_START}: "
          f"{np.allclose(long.loc[first, 'pre'], config.ELO_START)}")


def elo_diagnostics(df):
    # Diagnostic view with the ELO favorite (independent of the project's upset definition)
    comp = assign_favorite_underdog(df[df["completed"]], definition="elo")
    comp = comp[comp["upset_label_ok"]].copy()
    comp["elo_prob_fav"] = 1 - comp["elo_prob_dog"]
    comp["elo_prob_upset"] = comp["elo_prob_dog"]
    p = comp["elo_prob_fav"].clip(1e-6, 1 - 1e-6)
    y = 1 - comp["upset"]  # 1 = favorite won

    # ATP-ranking comparison: does the better-ranked player win?
    has_rank = comp["winner_rank"].notna() & comp["loser_rank"].notna()
    comp["rank_fav_won"] = np.where(has_rank, comp["winner_rank"] < comp["loser_rank"], np.nan)

    stats = pd.DataFrame({
        "matches": comp.groupby("season").size(),
        "elo_accuracy": (comp["elo_prob_fav"] > 0.5).eq(y == 1).groupby(comp["season"]).mean(),
        "rank_accuracy": comp.groupby("season")["rank_fav_won"].mean(),
        "upset_rate": comp.groupby("season")["upset"].mean(),
        "mean_pred_upset": comp.groupby("season")["elo_prob_upset"].mean(),
        "brier": ((p - y) ** 2).groupby(comp["season"]).mean(),
        "log_loss": (-(y * np.log(p) + (1 - y) * np.log(1 - p))).groupby(comp["season"]).mean(),
    }).round(3)
    print("\nElo predictive sanity check by season (completed, non-tied matches):")
    print("(burn-in 2014 should look worst; Elo should beat 'better ATP rank wins' once warmed up)")
    print(stats.to_string())

    post = comp[~comp["elo_burn_in"]]
    print("\nCalibration, 2015+ (predicted favorite win prob vs observed):")
    bins = pd.cut(post["elo_prob_fav"], [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    cal = post.groupby(bins, observed=True).agg(
        n=("upset", "size"), predicted=("elo_prob_fav", "mean"),
        observed=("upset", lambda u: 1 - u.mean()))
    print(cal.round(3).to_string())


def top_players(df, date, n=10):
    """Highest Elo ratings among players active in the previous 12 months, as of `date`."""
    d = pd.Timestamp(date)
    past = df[df["tourney_date"] < d]
    long = pd.concat([
        past[["tourney_date", "match_order", "winner_name", "winner_elo_post"]].set_axis(["date", "o", "name", "elo"], axis=1),
        past[["tourney_date", "match_order", "loser_name", "loser_elo_post"]].set_axis(["date", "o", "name", "elo"], axis=1),
    ]).sort_values("o")
    last = long.groupby("name").tail(1)
    last = last[last["date"] > d - pd.Timedelta(days=365)]
    print(f"\nTop {n} Elo as of {d.date()} (active in prior 12 months):")
    print(last.nlargest(n, "elo")[["name", "elo"]].round(0).to_string(index=False))


if __name__ == "__main__":
    df = pd.read_csv(config.CLEAN_MATCHES_PATH, parse_dates=["tourney_date"])
    df = df.sort_values("match_order").reset_index(drop=True)

    df = compute_elo(df)
    df = add_burn_in_flag(df)

    # ---------------- verification ----------------
    print("=" * 70)
    print("ELO VERIFICATION")
    print("=" * 70)
    print(f"Players rated: {pd.concat([df.winner_id, df.loser_id]).nunique():,}")
    verify_by_hand(df, "Novak Djokovic", n_matches=3)                       # first matches of 2014 (from 1500)
    verify_by_hand(df, "Carlos Alcaraz", n_matches=3)                       # debut: high K
    verify_by_hand(df, "Jannik Sinner", n_matches=2, start_index=150)        # established: low K
    global_continuity_check(df)

    comp = df[df["completed"]]
    tie = comp["winner_elo_pre"] == comp["loser_elo_pre"]
    print(f"\nExact Elo ties among completed matches: {tie.sum()} "
          f"(both players debuting: {(tie & (comp['winner_elo_n'] == 0) & (comp['loser_elo_n'] == 0)).sum()})")
    print(f"Matches in burn-in season (before {config.ELO_BURN_IN_END}): {comp['elo_burn_in'].sum():,}")

    elo_diagnostics(df)
    top_players(df, "2015-01-01")
    top_players(df, "2025-11-15")

    df.to_csv(config.ELO_MATCHES_PATH, index=False)
    print(f"\nSaved -> {config.ELO_MATCHES_PATH}")
