"""
Project-wide settings: paths, years, data source, and analysis constants.

Every decision that might change later (years, which events count, Elo
parameters) lives here so it is easy to find and document.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

CLEAN_MATCHES_PATH = PROCESSED_DIR / "atp_matches_clean.csv"
ELO_MATCHES_PATH = PROCESSED_DIR / "atp_matches_elo.csv"

# ---------------------------------------------------------------------------
# Data source: Jeff Sackmann ATP archive (mirrored on Hugging Face),
# pinned to a specific commit so the raw data never silently changes.
# ---------------------------------------------------------------------------
HF_COMMIT = "2598fe8a8b89796d49e157084af47c749e8bb4ab"
HF_BASE_URL = (
    "https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive/"
    f"resolve/{HF_COMMIT}/atp"
)

FIRST_YEAR = 2014
LAST_YEAR = 2025  # 2026 excluded by project scope (file may exist in data/raw but is not loaded)

# Only the tour-level main-draw files: atp_matches_YYYY.csv
# (NOT *_qual_chall_*, *_futures_*, *_doubles_*, WTA, or Match Charting data)
RAW_FILES = [f"atp_matches_{y}.csv" for y in range(FIRST_YEAR, LAST_YEAR + 1)]

# ---------------------------------------------------------------------------
# Scope of "ATP main tour" matches
# ---------------------------------------------------------------------------
# tourney_level codes in the Sackmann files:
#   G = Grand Slam, M = Masters 1000, A = other tour-level (250/500, ATP Cup,
#   United Cup, Laver Cup), F = Tour finals (ATP Finals + Next Gen Finals),
#   O = Olympics, D = Davis Cup
EXCLUDED_LEVELS = {"D"}  # Davis Cup: team event, ~50% missing stats, many non-tour players
EXCLUDED_TOURNAMENTS = {
    "Laver Cup",        # exhibition format, no ranking points, match-tiebreak deciders
    "Next Gen Finals",  # experimental Fast4 scoring distorts serve/return stats
}

# Order of rounds within a tournament (used to sort matches chronologically,
# because Sackmann only gives the tournament start date, not the match date).
# RR sits between R16 and QF: in the ATP Finals RR precedes SF, and in team
# events (ATP Cup / United Cup) RR precedes QF.
ROUND_ORDER = {
    "R128": 1, "R64": 2, "R32": 3, "R16": 4, "RR": 4.5,
    "QF": 5, "SF": 6, "BR": 6.5, "F": 7,
}

# The 18 raw serve-stat columns (9 per player)
STAT_SUFFIXES = ["ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced"]
W_STATS = [f"w_{s}" for s in STAT_SUFFIXES]
L_STATS = [f"l_{s}" for s in STAT_SUFFIXES]

# ---------------------------------------------------------------------------
# Elo settings (FiveThirtyEight-style tennis Elo; see src/elo.py)
# ---------------------------------------------------------------------------
ELO_START = 1500.0
ELO_K_NUMERATOR = 250.0
ELO_K_OFFSET = 5.0
ELO_K_SHAPE = 0.4

# All ratings start at 1500 in Jan 2014, so early ratings are not yet
# meaningful. Matches before this date still update Elo but should be
# excluded from modeling.
ELO_BURN_IN_END = "2015-01-01"

# ---------------------------------------------------------------------------
# Upset definition (see src/upsets.py). Change here to switch the whole pipeline.
#   "rank": favorite = better (numerically lower) pre-match ATP ranking   <- current
#   "elo":  favorite = higher pre-match Elo
# ---------------------------------------------------------------------------
UPSET_DEFINITION = "rank"

# Matches where a player has no pre-match ATP ranking (unranked wildcards,
# protected-ranking entries, etc.):
#   "exclude":           no favorite can be defined -> upset label left missing   <- current
#   "unranked_as_worst": treat missing rank as UNRANKED_RANK_VALUE
MISSING_RANK_POLICY = "exclude"
UNRANKED_RANK_VALUE = 2000

# ---------------------------------------------------------------------------
# Modeling (Graph 3)
# ---------------------------------------------------------------------------
MODEL_FIRST_SEASON = 2015            # 2014 = Elo/profile warm-up
TRAIN_SEASONS = list(range(2015, 2023))   # 2015-2022
TEST_SEASONS = list(range(2023, 2026))    # 2023-2025

# ---------------------------------------------------------------------------
# Giant Killers (Graph 4)
# ---------------------------------------------------------------------------
GK_BASELINE = "log_rank"             # which baseline model supplies expected upset wins
GK_FIRST_PRED_SEASON = 2016          # expanding-window predictions need >= 1 prior training season
GK_MIN_OPPORTUNITIES = 30            # minimum matches as the lower-ranked player (configurable)

# Giant Killer "ranking was out of date" flags (descriptive only; see src/flags.py)
FLAG_COMEBACK_LOOKBACK_DAYS = 3 * 365   # look for the player's best rank in the previous 3 years
FLAG_COMEBACK_MAX_PEAK = 30             # ... which must be top-30
FLAG_COMEBACK_RATIO = 3.0               # ... and at least 3x better than his current rank
FLAG_RISER_MAX_AGE = 23                 # riser: aged <= 23 at the match
FLAG_RISER_RATIO = 2.0                  # ... and rank ~1 year later at least 2x better
FLAG_PLAYER_SHARE = 0.30                # player flagged if >= 30% of his opportunities carry the flag
