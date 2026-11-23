"""
Step 4: build the modeling dataset for Graph 3 (and the match pool for Graph 4).

Input:  data/processed/atp_matches_profiles.csv
Output: data/processed/atp_model_dataset.csv
        one row per labelled match (2015-2025), favorite/underdog oriented, with:
          upset label (definition from config.UPSET_DEFINITION)
          rank_diff, log_rank_diff, elo_diff, elo_prob_dog
          surface dummies, fav_/dog_ seven-variable profiles, d_* = dog minus fav
          flags: both_profiles_ok, graph3_sample
Run from the project root:  python3 run_04_model_dataset.py
"""
import numpy as np
import pandas as pd

from src import config
from src.profiles import PROFILE_STATS
from src.upsets import assign_favorite_underdog

MODEL_DATASET_PATH = config.PROCESSED_DIR / "atp_model_dataset.csv"
pd.set_option("display.width", 160)


def step(label, before, after):
    print(f"  {label:<62s} {before:>7,} -> {after:>7,}  (removed {before - after:,})")


if __name__ == "__main__":
    m = pd.read_csv(config.PROCESSED_DIR / "atp_matches_profiles.csv", parse_dates=["tourney_date"], low_memory=False)
    print("=" * 70)
    print(f"UPSET DEFINITION: {config.UPSET_DEFINITION!r}   missing-rank policy: {config.MISSING_RANK_POLICY!r}")
    print("=" * 70)

    # ---------------- missing rankings ----------------
    comp = m[m["completed"]]
    miss = comp["winner_rank"].isna() | comp["loser_rank"].isna()
    print(f"\nCompleted matches with a missing pre-match ranking: {miss.sum()} of {len(comp):,} ({miss.mean():.2%})")
    unranked = pd.concat([
        comp.loc[comp["winner_rank"].isna(), ["winner_name", "winner_entry"]].set_axis(["name", "entry"], axis=1),
        comp.loc[comp["loser_rank"].isna(), ["loser_name", "loser_entry"]].set_axis(["name", "entry"], axis=1)])
    print("  entry type of the unranked player (WC = wildcard, PR = protected ranking, Q = qualifier):")
    print("  " + unranked["entry"].fillna("(direct/none)").value_counts().to_string().replace("\n", "\n  "))
    print("  most frequent unranked players:", unranked["name"].value_counts().head(6).to_dict())

    # ---------------- favorite / underdog + upset ----------------
    m = assign_favorite_underdog(m)

    # ---------------- filters ----------------
    print("\nFilters (matches before -> after):")
    n0 = len(m)
    d = m[m["completed"]]
    step("completed matches only (drop retirements/defaults)", n0, len(d))
    n = len(d); d = d[d["season"] >= config.MODEL_FIRST_SEASON]
    step(f"season >= {config.MODEL_FIRST_SEASON} (2014 = Elo/profile warm-up)", n, len(d))
    n = len(d); d = d[d["upset_label_ok"]]
    step("upset label defined (both players ranked, ranks differ)", n, len(d))
    n_labelled = len(d)
    both = d["winner_profile_ok"] & d["loser_profile_ok"]
    print(f"  {'-> of these, both players have profile_ok (Graph 3 sample)':<62s} {n_labelled:>7,} -> {both.sum():>7,}  "
          f"(flagged, not removed: {n_labelled - both.sum():,})")

    d = d.copy()
    d["both_profiles_ok"] = both.values
    d["graph3_sample"] = d["both_profiles_ok"]
    d["upset"] = d["upset"].astype(int)
    d["surface_clay"] = (d["surface"] == "Clay").astype(int)
    d["surface_grass"] = (d["surface"] == "Grass").astype(int)
    for k in PROFILE_STATS:
        d[f"d_{k}"] = d[f"dog_prof_{k}"] - d[f"fav_prof_{k}"]   # underdog minus favorite

    # ---------------- verification of the upset label ----------------
    print("\n" + "=" * 70)
    print("UPSET LABEL VERIFICATION")
    print("=" * 70)
    if config.UPSET_DEFINITION == "rank":
        expected = (d["winner_rank"] > d["loser_rank"]).astype(int)   # winner had the worse (larger) rank
        print(f"upset == (winner_rank > loser_rank) for every row: {(expected == d['upset']).all()}")
        print(f"fav_rank < dog_rank for every row: {(d['fav_rank'] < d['dog_rank']).all()}")
        print(f"rank ties encountered: {(m['completed'] & (m['winner_rank'] == m['loser_rank'])).sum()}")
    ex = pd.concat([d[d["upset"] == 1].sample(4, random_state=1), d[d["upset"] == 0].sample(3, random_state=1)])
    print("\nExamples:")
    print(ex[["tourney_date", "tourney_name", "round", "winner_name", "winner_rank", "loser_name", "loser_rank",
              "fav_name", "dog_name", "upset"]].to_string(index=False))

    print("\nUpset rate by season (rank-defined):")
    by = d.groupby("season").agg(matches=("upset", "size"), upset_rate=("upset", "mean"),
                                 graph3_matches=("graph3_sample", "sum"))
    by["graph3_upset_rate"] = d[d["graph3_sample"]].groupby("season")["upset"].mean()
    print(by.round(3).to_string())

    agree = (d["elo_diff"] > 0).mean()
    print(f"\nRank favorite also has the higher Elo in {agree:.1%} of matches "
          f"(the two definitions disagree on {1 - agree:.1%}).")
    print("Upset rate by rank-gap band (sanity: bigger gaps -> fewer upsets):")
    print(d.groupby(pd.cut(d["log_rank_diff"], [0, .25, .5, 1, 1.5, 2, 10]), observed=True)["upset"]
          .agg(["size", "mean"]).round(3).to_string())

    print("\nGraph 3 sample vs excluded (profiles thin): not a random subset")
    print(d.groupby("graph3_sample").agg(n=("upset", "size"), upset_rate=("upset", "mean"),
                                         median_fav_rank=("fav_rank", "median"),
                                         median_dog_rank=("dog_rank", "median")).round(3).to_string())
    print(f"  excluded because favorite profile thin: {(~d['fav_profile_ok'].astype(bool)).sum():,}, "
          f"underdog thin: {(~d['dog_profile_ok'].astype(bool)).sum():,}")

    # ---------------- feature validation on TRAINING seasons only ----------------
    g3 = d[d["graph3_sample"]]
    train = g3[g3["season"].isin(config.TRAIN_SEASONS)]
    print("\n" + "=" * 70)
    print(f"PROFILE FEATURE CHECKS — training seasons only ({min(config.TRAIN_SEASONS)}-{max(config.TRAIN_SEASONS)}, "
          f"n={len(train):,}); test seasons untouched")
    print("=" * 70)
    print("Mean underdog-minus-favorite gap, upsets vs non-upsets:")
    cmp = train.groupby("upset")[[f"d_{k}" for k in PROFILE_STATS]].mean().T
    cmp.columns = ["favorite_won", "upset"]
    print(cmp.round(4).to_string())
    for k in ("spw", "rpw"):
        q = pd.qcut(train[f"d_{k}"], 5)
        print(f"\nUpset rate by quintile of d_{k} (dog - fav):")
        print(train.groupby(q, observed=True)["upset"].agg(["size", "mean"]).round(3).to_string())

    d.to_csv(MODEL_DATASET_PATH, index=False)
    print(f"\nSaved -> {MODEL_DATASET_PATH}  ({len(d):,} rows, {d['graph3_sample'].sum():,} in Graph 3 sample)")
    print("\nColumns:", ", ".join(c for c in d.columns if c.startswith(("fav_", "dog_", "d_")) or c in
                                 ("upset", "upset_definition", "rank_diff", "log_rank_diff", "elo_diff", "elo_prob_dog",
                                  "surface_clay", "surface_grass", "both_profiles_ok", "graph3_sample")))
