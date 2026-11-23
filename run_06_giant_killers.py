"""
Step 6 (Graph 4): expected vs. actual upset wins for each player.

Primary expectation: config.GK_BASELINE (log rank + surface), out-of-sample
expanding-window predictions, then season re-centered. The Elo baseline is
computed alongside as a comparison only.

Input:  data/processed/atp_model_dataset.csv
Output: data/processed/gk_match_predictions.csv  (one row per match)
        data/processed/giant_killers.csv         (one row per player)
Run from the project root:  python3 run_06_giant_killers.py
"""
import pandas as pd

from src import config
from src.giant_killers import giant_killer_table, recenter_by_season
from src.models import BASELINES, evaluate, expanding_window_predictions

pd.set_option("display.width", 180)
SHOW = ["player", "opportunities", "upset_wins", "expected_wins", "gk_score", "gk_per_100", "se", "z", "mean_fav_rank"]
PRIMARY = config.GK_BASELINE
COMPARISON = "elo"

if __name__ == "__main__":
    # Graph 4 does not need profiles, so it uses ALL labelled matches (thin-profile players included)
    d = pd.read_csv(config.PROCESSED_DIR / "atp_model_dataset.csv", parse_dates=["tourney_date"], low_memory=False).copy()
    for name in (PRIMARY, COMPARISON):
        d[f"p_{name}_raw"] = expanding_window_predictions(d, BASELINES[name], config.GK_FIRST_PRED_SEASON,
                                                          config.MODEL_FIRST_SEASON)
    pm = d.dropna(subset=[f"p_{PRIMARY}_raw"]).copy()
    for name in (PRIMARY, COMPARISON):
        pm[f"p_{name}"] = recenter_by_season(pm[f"p_{name}_raw"], pm["upset"], pm["season"]).to_numpy()

    print("=" * 70)
    print(f"GIANT KILLERS — upset definition {config.UPSET_DEFINITION!r}; expected wins from {PRIMARY} + surface")
    print(f"out-of-sample expanding-window predictions {config.GK_FIRST_PRED_SEASON}-{pm['season'].max()}, season re-centered")
    print("=" * 70)
    print(f"Matches with predictions: {len(pm):,}; distinct underdogs: {pm['dog_id'].nunique():,}")

    print("\nRe-centering check, by season (re-centered expected must equal actual):")
    s = pm.groupby("season").agg(matches=("upset", "size"), actual=("upset", "sum"),
                                 expected_raw=(f"p_{PRIMARY}_raw", "sum"), expected_recentered=(f"p_{PRIMARY}", "sum"))
    s["mean_shift_pp"] = 100 * (s["expected_recentered"] - s["expected_raw"]) / s["matches"]
    print(s.round(1).to_string())
    for col in (f"p_{PRIMARY}_raw", f"p_{PRIMARY}"):
        e = evaluate(pm["upset"], pm[col])
        print(f"  {col:<18s} log loss {e['log_loss']:.4f}  Brier {e['brier']:.4f}  AUC {e['auc']:.4f}")
    rank_corr = pm[f"p_{PRIMARY}_raw"].corr(pm[f"p_{PRIMARY}"], method="spearman")
    print(f"  Spearman correlation raw vs re-centered probabilities: {rank_corr:.4f}")

    print("\nOpportunities per player (distribution):")
    opp = pm.groupby("dog_id").size()
    print(pd.cut(opp, [0, 9, 19, 29, 49, 99, 1000]).value_counts().sort_index().to_string())
    print(f"Players with >= {config.GK_MIN_OPPORTUNITIES} opportunities: {(opp >= config.GK_MIN_OPPORTUNITIES).sum()}")

    t = giant_killer_table(pm, f"p_{PRIMARY}", config.GK_MIN_OPPORTUNITIES)
    t_raw = giant_killer_table(pm, f"p_{PRIMARY}_raw", config.GK_MIN_OPPORTUNITIES)
    t_cmp = giant_killer_table(pm, f"p_{COMPARISON}", config.GK_MIN_OPPORTUNITIES)
    print(f"\nSum of gk_score over ALL players: raw {(pm['upset'] - pm[f'p_{PRIMARY}_raw']).sum():+.1f}, "
          f"re-centered {(pm['upset'] - pm[f'p_{PRIMARY}']).sum():+.1f}")

    print(f"\n--- PRIMARY: {PRIMARY} + surface, re-centered (min {config.GK_MIN_OPPORTUNITIES} opportunities) ---")
    print("Top 20:")
    print(t.head(20)[SHOW].round(2).to_string())
    print("Bottom 5:")
    print(t.tail(5)[SHOW].round(2).to_string())
    print(f"\nLeaders with z > 2 (unlikely to be luck alone): {(t['z'] > 2).sum()} of {len(t)}; "
          f"z < -2: {(t['z'] < -2).sum()}  (with {len(t)} players, ~{0.023 * len(t):.0f} per tail expected by chance)")

    print("\nEffect of re-centering on the top 10:")
    print(f"  raw:         {', '.join(t_raw.head(10)['player'])}")
    print(f"  re-centered: {', '.join(t.head(10)['player'])}")
    print(f"\nSensitivity of the top 10 to the minimum-opportunity cutoff:")
    for cut in (20, 30, 50, 100):
        print(f"  min {cut:>3d}: {', '.join(giant_killer_table(pm, f'p_{PRIMARY}', cut).head(10)['player'])}")
    print(f"\nComparison only — {COMPARISON} baseline (re-centered) top 10: {', '.join(t_cmp.head(10)['player'])}")

    out = t[SHOW + ["actual_rate", "expected_rate", "first_season", "last_season"]].join(
        t_raw[["expected_wins", "gk_score"]], rsuffix="_raw").join(
        t_cmp[["expected_wins", "gk_score", "z"]], rsuffix=f"_{COMPARISON}")
    pm[["match_id", "tourney_date", "season", "tourney_name", "surface", "round", "fav_id", "fav_name", "fav_rank",
        "dog_id", "dog_name", "dog_rank", "upset", f"p_{PRIMARY}_raw", f"p_{PRIMARY}",
        f"p_{COMPARISON}_raw", f"p_{COMPARISON}"]].to_csv(config.PROCESSED_DIR / "gk_match_predictions.csv", index=False)
    out.to_csv(config.PROCESSED_DIR / "giant_killers.csv")
    print(f"\nSaved -> data/processed/gk_match_predictions.csv, data/processed/giant_killers.csv")
