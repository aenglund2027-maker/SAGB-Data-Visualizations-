"""
Step 1: download (if needed), load, and clean ATP main-tour matches 2014-2026.

Cleaning philosophy: flag rather than delete. Rows are only removed when they
are out of scope (Davis Cup, exhibitions) or were never played (walkovers).
Retirements and matches with missing stats stay in the file with flags so later
steps can decide how to use them.
"""
import urllib.request

import numpy as np
import pandas as pd

from src import config


# ---------------------------------------------------------------------------
# Download / load
# ---------------------------------------------------------------------------
def download_raw_files(force=False):
    """Download any missing yearly ATP match files from the pinned archive."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for fname in config.RAW_FILES:
        dest = config.RAW_DIR / fname
        if dest.exists() and not force:
            continue
        url = f"{config.HF_BASE_URL}/{fname}"
        print(f"  downloading {fname} ...")
        urllib.request.urlretrieve(url, dest)


def load_raw_matches():
    """Read all yearly files into one DataFrame, tagging each row with its source file."""
    frames = []
    for fname in config.RAW_FILES:
        df = pd.read_csv(config.RAW_DIR / fname, low_memory=False)
        df["source_file"] = fname
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(raw):,} raw rows from {len(frames)} files")
    print(raw.groupby("source_file").size().rename("rows").to_string(), "\n")
    return raw


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------
def classify_outcome(score):
    """
    Label how the match ended, based on the score string.

    completed  - played to completion
    retirement - a player retired mid-match ("RET")
    walkover   - match never started ("W/O", "Walkover")
    default    - a player was defaulted ("DEF", "Def.")
    abandoned  - match abandoned / not finished ("ABD", "Played and abandoned")
    """
    s = str(score)
    if "W/O" in s or "walkover" in s.lower():
        return "walkover"
    if "RET" in s:
        return "retirement"
    if "DEF" in s or "Def" in s:
        return "default"
    if "ABD" in s or "abandon" in s.lower():
        return "abandoned"
    if s in ("nan", ""):
        return "unknown"
    return "completed"


def add_stats_validity_flag(df):
    """
    stats_ok = True when all 18 serve-stat columns are present AND pass basic
    internal-consistency checks. Rows that fail are kept; only the flag changes.
    """
    has_all = df[config.W_STATS + config.L_STATS].notna().all(axis=1)
    ok = has_all.copy()
    for p in ("w", "l"):
        svpt, first_in = df[f"{p}_svpt"], df[f"{p}_1stIn"]
        ok &= svpt > 0                                        # no zero-point matches
        ok &= first_in <= svpt                                # can't make more 1st serves than serve points
        ok &= df[f"{p}_1stWon"] <= first_in                   # 1st-serve points won <= 1st serves in
        ok &= df[f"{p}_2ndWon"] <= (svpt - first_in)          # 2nd-serve points won <= 2nd-serve points
        ok &= df[f"{p}_bpSaved"] <= df[f"{p}_bpFaced"]        # can't save more BPs than faced
        ok &= df[f"{p}_ace"] + df[f"{p}_df"] <= svpt
    df["stats_present"] = has_all
    df["stats_ok"] = ok.fillna(False)
    return df


def sort_chronologically(df):
    """
    Sackmann only provides tourney_date (the Monday the event starts), so the
    exact match day is unknown. Within a tournament we order by round, then by
    match_num (which increases through the draw). Across tournaments we order by
    start date. This guarantees each player's matches are processed in the
    order they were actually played, which is what leakage-free features need.
    """
    df["round_order"] = df["round"].map(config.ROUND_ORDER)
    unknown_rounds = df["round_order"].isna().sum()
    if unknown_rounds:
        print(f"  WARNING: {unknown_rounds} rows with unrecognized round labels")
    df = df.sort_values(["tourney_date", "tourney_id", "round_order", "match_num"]).reset_index(drop=True)
    df["match_order"] = np.arange(len(df))  # global chronological index
    return df


def report(step, before, after):
    print(f"  {step:<55s} {before:>7,} -> {after:>7,}  (removed {before - after:,})")


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------
def clean_matches(raw):
    df = raw.copy()
    print("Cleaning steps (rows before -> after):")

    # 1. Exact duplicate matches (same tournament, players, round, score)
    n = len(df)
    df = df.drop_duplicates(subset=["tourney_id", "winner_id", "loser_id", "round", "score"])
    report("1. drop duplicate match records", n, len(df))

    # 2. Restrict to the individual ATP main tour
    n = len(df)
    df = df[~df["tourney_level"].isin(config.EXCLUDED_LEVELS)]
    report("2a. drop Davis Cup (tourney_level D)", n, len(df))
    n = len(df)
    df = df[~df["tourney_name"].isin(config.EXCLUDED_TOURNAMENTS)]
    report("2b. drop Laver Cup + Next Gen Finals", n, len(df))

    # 3. Outcome type; walkovers were never played, so they carry no information
    df["outcome_type"] = df["score"].apply(classify_outcome)
    print("\n  Outcome types before removing walkovers:")
    print("  " + df["outcome_type"].value_counts().to_string().replace("\n", "\n  "))
    n = len(df)
    df = df[df["outcome_type"] != "walkover"]
    report("3. drop walkovers", n, len(df))
    # Retirements/defaults/abandoned are KEPT but flagged; they are excluded
    # from Elo updates, profiles, and modeling (outcome mostly reflects injury).
    df["completed"] = df["outcome_type"] == "completed"

    # 4. Dates and identifiers
    df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(int).astype(str), format="%Y%m%d")
    df["year"] = df["tourney_date"].dt.year
    # Season label from the file (a few events dated late Dec belong to next season)
    df["season"] = df["source_file"].str.extract(r"(\d{4})").astype(int)
    df["match_id"] = df["tourney_id"].astype(str) + "-" + df["match_num"].astype(str)
    assert df["match_id"].is_unique, "match_id not unique"

    # 5. Stats validity flag (no rows removed)
    df = add_stats_validity_flag(df)

    # 6. Chronological ordering
    df = sort_chronologically(df)

    print(f"\nFinal cleaned dataset: {len(df):,} matches "
          f"({df['completed'].sum():,} completed, {(~df['completed']).sum():,} incomplete)\n")
    return df


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def describe_clean_data(df):
    print("=" * 70)
    print("COLUMNS")
    print("=" * 70)
    print(", ".join(df.columns))

    print("\n" + "=" * 70)
    print("MATCHES BY SEASON (all retained / completed / completed with valid stats)")
    print("=" * 70)
    by_year = df.groupby("season").agg(
        all_matches=("match_id", "size"),
        completed=("completed", "sum"),
        completed_stats_ok=("stats_ok", lambda s: (s & df.loc[s.index, "completed"]).sum()),
        first_date=("tourney_date", "min"),
        last_date=("tourney_date", "max"),
    )
    print(by_year.to_string())

    print("\n" + "=" * 70)
    print("SURFACES (completed matches)")
    print("=" * 70)
    comp = df[df["completed"]]
    print(pd.crosstab(comp["season"], comp["surface"].fillna("MISSING"), margins=True).to_string())

    print("\n" + "=" * 70)
    print("TOURNAMENT LEVELS (completed matches)")
    print("=" * 70)
    print(comp["tourney_level"].value_counts().to_string())

    print("\n" + "=" * 70)
    print("MISSINGNESS IN KEY SERVE-STAT COLUMNS (completed matches)")
    print("=" * 70)
    miss = comp[config.W_STATS + config.L_STATS].isna().mean().mul(100).round(2)
    print("% missing per column:")
    print(miss.to_string())
    print(f"\nrows with ALL stats present: {comp['stats_present'].mean():.2%}")
    print(f"rows passing consistency checks (stats_ok): {comp['stats_ok'].mean():.2%}")
    print(f"present but failing checks: {(comp['stats_present'] & ~comp['stats_ok']).sum()}")

    print("\n% of completed matches WITHOUT valid stats, by season and level:")
    print((~comp["stats_ok"]).groupby([comp["season"], comp["tourney_level"]]).mean()
          .mul(100).round(1).unstack().to_string())

    print("\nMissing ATP rankings (completed matches):")
    print(comp[["winner_rank", "loser_rank"]].isna().sum().to_string())
