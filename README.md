# Can Playing-Style Matchups Predict ATP Tennis Upsets Better Than Rankings Alone?

My part: **Graph 3** (baseline vs. playing-style model) and **Graph 4** (Giant Killers).
My partner owns Graphs 1–2 (PCA map, style-matchup heatmap) and the shared style clusters.

## Layout

```
data/raw/                 Sackmann ATP main-tour files (pinned HF commit); 2014-2025 are loaded
data/processed/           pipeline outputs (CSV)
src/config.py             years, scope, upset definition, splits, Elo + Giant Killer settings
src/data_loading.py       download / load / clean
src/elo.py                chronological pre-match Elo
src/profiles.py           the seven shared profile variables (leakage-free, 52-week window)
src/upsets.py             favorite/underdog + upset label (switchable: rank or Elo)
src/models.py             baselines, matchup-model scaffold, evaluation, expanding-window predictions
src/giant_killers.py      actual - expected upset wins
src/flags.py              comeback / riser flags for Giant Killers (descriptive)
src/style_labels.py       hook for merging the partner's cluster labels (no clustering here)

run_01_load_clean.py      -> atp_matches_clean.csv
run_02_elo.py             -> atp_matches_elo.csv            (+ Elo verification)
run_03_profiles.py        -> player_profiles_long.csv, atp_matches_profiles.csv (+ leakage checks)
run_04_model_dataset.py   -> atp_model_dataset.csv          (+ upset-label verification, filter counts)
run_05_baselines.py       -> baseline_test_predictions.csv  (Graph 3 baselines)
run_06_giant_killers.py   -> gk_match_predictions.csv, giant_killers.csv (Graph 4)
run_07_matchup.py         -> matchup_test_predictions.csv, graph3_results.csv, graph3_bootstrap.csv (Graph 3)
run_08_figures.py         -> outputs/figures/graph3_model_comparison.{png,svg}, graph4_giant_killers.{png,svg}, graph4_chart_data.csv
```

Run all steps in order from the project root (about a minute total):
`for s in run_0*.py; do python3 $s; done`

## Data source and scope

- **Source:** Jeff Sackmann / Tennis Abstract ATP data (CC BY-NC-SA 4.0), via the HF
  mirror `Aneeshers/tennis-sackmann-archive` at commit `2598fe8a`.
- **Files:** only `atp_matches_YYYY.csv` for **2014–2025**. No 2026, WTA, Challenger,
  Futures, qualifying, doubles, or Match Charting data. (`atp_matches_2026.csv` sits in
  `data/raw/` from an earlier step but is never loaded.)

## Decision log

### Cleaning (33,517 raw rows -> 30,518 retained -> 29,626 completed)
| Step | Removed | Reason |
|---|---|---|
| Duplicate records | 2 | identical Davis Cup rows |
| Davis Cup (`tourney_level D`) | 2,702 | team event, ~half missing stats, missing surfaces, many non-tour players |
| Laver Cup, Next Gen Finals | 88 | exhibition / experimental scoring |
| Walkovers | 207 | never played |
| Retirements (885) and defaults (7) | kept, flagged `completed=False` | excluded from Elo updates, profiles, and models |

- **Match order:** Sackmann gives only the tournament start date. Matches are ordered
  by date, then round (R128 → … → F, RR before QF), then `match_num`, giving `match_order`.
- **Stats flag:** `stats_ok` means all serve columns are present and pass consistency
  checks (99.4% of completed matches). The main gap is the 2021 Tokyo Olympics (64 matches).

### Upset definition (`src/upsets.py`, switch = `config.UPSET_DEFINITION`)
- **Current: `"rank"`.** The favorite is the player with the better (numerically smaller)
  pre-match ATP ranking. **upset = 1** when the worse-ranked player wins.
  Verified on every row: `upset == (winner_rank > loser_rank)`. No rank ties occur.
- **Missing rankings** (`config.MISSING_RANK_POLICY = "exclude"`): 81 completed matches
  (0.27%) have an unranked player, mostly wildcards and protected-ranking entries. Their label is
  left missing and they are reported. The alternative `"unranked_as_worst"` treats them as rank 2000.
- **Elo is NOT the upset definition.** It is kept as a separate strength measure. The rank
  favorite also has the higher Elo in 81.9% of matches.
- **Strength columns** (favorite's point of view): `rank_diff` = dog_rank − fav_rank,
  `log_rank_diff` = ln(dog_rank) − ln(fav_rank), `elo_diff` = fav_elo − dog_elo
  (negative when the rank favorite has the lower Elo), `elo_prob_dog`.

### Elo (`src/elo.py`)
- FiveThirtyEight tennis Elo: start at 1500; `Elo += K·(result − expected)`;
  `K = 250 / (n_prev + 5)^0.4` (K=131 on debut, ≈33 after 150 matches). Overall Elo, not surface Elo.
- Only completed matches update Elo. Each match stores the pre-match rating.
- Everyone starts at 1500 in Jan 2014, so **2014 is a warm-up season** and is excluded from models.
- Raw Elo probabilities are overconfident. The fitted logistic baseline re-calibrates them.

### The seven shared profile variables (`src/profiles.py`)
All are computed from **window sums** of Sackmann counts over the player's previous **52
weeks**, using matches strictly before the current one (completed, `stats_ok` only):

| Column | Formula |
|---|---|
| `prof_ace_rate` | ace / svpt |
| `prof_df_rate` | df / svpt |
| `prof_first_in_pct` | 1stIn / svpt |
| `prof_first_won_pct` | 1stWon / 1stIn |
| `prof_second_won_pct` | 2ndWon / (svpt − 1stIn)  (double faults count as 2nd-serve points lost) |
| `prof_spw` | (1stWon + 2ndWon) / svpt |
| `prof_rpw` | (opp_svpt − opp_1stWon − opp_2ndWon) / opp_svpt  (opponent's serve columns, same matches) |

- **Pooling:** rates are point-weighted (counts summed, then divided), not averages of per-match percentages.
- **Thin history:** `profile_ok` = ≥10 stat matches in the window. Counts are stored, so the threshold can change.
- **Verified:**
  - brute-force recomputation of all seven stats on 300 random player-matches matches exactly;
  - corrupting one match changes only later profiles;
  - every player's first match has an empty profile.
- **Not adjusted for surface or opponent quality** (yet).
- **Shareable:** `data/processed/player_profiles_long.csv` (player_id, date, match_order,
  seven stats, counts) is the file to share with my partner so both analyses use identical definitions.

### Modeling dataset (`atp_model_dataset.csv`)
| Filter | Matches |
|---|---|
| Retained matches 2014–2025 | 30,518 |
| Completed | 29,626 |
| Season ≥ 2015 | 27,138 |
| Upset label defined (both ranked) | **27,058** (Graph 4 pool) |
| Both players `profile_ok` | **20,607** (Graph 3 sample, flagged `graph3_sample`) |

The Graph 3 sample is **not random**. Excluded matches mostly involve an underdog with a thin
tour history (median underdog rank 169 vs. 65) and have fewer upsets (33.1% vs. 36.1%).
All Graph 3 models are compared on the same rows.

### Graph 3 — baselines (train 2015–2022, test 2023–2025, logistic regression)
| Baseline (+ surface) | Test log loss | Brier | AUC |
|---|---|---|---|
| constant (training upset rate) | 0.6541 | 0.2307 | 0.500 |
| raw rank difference | 0.6514 | 0.2291 | 0.558 |
| log rank difference | 0.6304 | 0.2206 | 0.621 |
| Elo difference | 0.6213 | 0.2168 | 0.648 |
| Elo + log rank | 0.6197 | 0.2161 | 0.651 |

- **Ranking scale:** the raw rank difference is barely better than a constant because the ranking
  scale is very non-linear. The log gap fixes most of that.
- **Elo vs ranking:** Elo is clearly better than log rank (bootstrap 95% CI for the log-loss gain:
  0.004–0.014). Surface effects are negligible in every baseline.

### Graph 3 — matchup model results (`run_07_matchup.py`)
Three nested models per strength measure, all on the same 6,052 test matches:
baseline → **additive** (+ both players' seven profile variables) → **matchup**
(+ dog_rpw×fav_spw, dog_spw×fav_rpw, dog_rpw×fav_2nd, which were specified before test evaluation).

| Model | Log loss | Brier | AUC |
|---|---|---|---|
| log rank + surface | 0.6304 | 0.2206 | 0.621 |
| + profiles (additive) | 0.6220 | 0.2171 | 0.645 |
| + profiles + interactions | 0.6219 | 0.2170 | 0.646 |
| Elo + surface | 0.6213 | 0.2168 | 0.648 |
| + profiles (additive) | 0.6184 | 0.2156 | 0.654 |
| + profiles + interactions | 0.6183 | 0.2156 | 0.654 |

**Findings:**
1. **Profiles help beyond strength.** Log-loss gain 0.0084 over log rank (95% CI 0.0049–0.0124)
   and 0.0029 over Elo (0.0009–0.0050). Profiles beat the baseline in 8/8 (rank) and 7/8 (Elo)
   rolling-origin seasons. The gain is about 3x larger over rankings, which fits rankings lagging true ability.
2. **Interactions add nothing.** Matchup vs additive: +0.0001 (CI −0.0002 to 0.0005) for rank,
   +0.00003 (−0.0003 to 0.0004) for Elo. Interactions help in only 3/8 rolling seasons, and their
   coefficients are tiny.
3. **The gain is overall quality, not style** (post-hoc diagnostic). Two gaps, underdog-minus-favorite
   serve points won and return points won, capture virtually all of it. The other 12 columns add
   0.0003 (rank) / 0.0007 (Elo), with CIs including 0.
4. **Where the gain shows up:** mostly on hard and grass courts and in big rank gaps (clay: +0.0016).
5. **Calibration:** the Elo-based models are a little overconfident at the top (predicted 0.65 → observed 0.60).

**Takeaway:** serve/return statistics carry real information that rankings (and, less so, Elo) miss.
But in this data it behaves like a better strength measure, not a rock-paper-scissors matchup effect.
Note: an earlier exploratory run (Elo-defined upsets, 2026 included) saw the 2023–25 test years before
this evaluation. The feature set was not changed afterwards.

### Graph 4 — Giant Killers (`src/giant_killers.py`)
- **Opportunity:** a labelled match entered as the lower-ranked player (2016–2025, profiles not required).
- **Expected upset wins:** the sum of out-of-sample predicted probabilities. Each season is
  predicted by a model trained only on earlier seasons (expanding window from 2015).
- **Score:** `gk_score` = actual − expected. Also reported: `gk_per_100`, `se = sqrt(Σp(1−p))`, and `z`.
- **Settings:** minimum opportunities = `config.GK_MIN_OPPORTUNITIES` (default 30). Expected wins are
  computed under both the log-rank and Elo baselines (`config.GK_BASELINE`).
- **Season re-centering** (`recenter_by_season`): each season's predicted probabilities are shifted
  by a constant on the log-odds scale so that expected upsets = actual upsets in that season. This
  removes the drift from training on earlier, lower-upset seasons (raw total excess +438). Scores now sum to 0,
  and match ordering is unchanged (Spearman 0.99). Because it uses same-season totals, it is descriptive, not a forecast.
- **Primary baseline: log rank + surface** (consistent with the rank-based upset definition). Elo is kept only as a comparison.
- **Interpretation caveat:** a rank-based expectation rewards players whose ranking understated
  their level: injury comebacks (del Potro, Wawrinka, Murray, and Djokovic/Nadal with 32–37 opportunities)
  and fast risers (Alcaraz, Sinner, Mensik, Draper). The chart should say so, or flag these players.
- **Significance:** 25 of 257 qualifying players have z > 2 (vs ~6 expected by chance), and 15 have z < −2.

### Publication figures (`run_08_figures.py`, PNG + SVG in `outputs/figures/`)
- **Graph 3:** AUC per model (0.50 = chance) on the 2023-25 test set, grouped by starting point (ranking, Elo):
  baseline (gray), + profiles (blue), + profiles + interactions (light blue). Side columns give the change in AUC
  vs the row above, labeled "clear gain" only when the paired-bootstrap 95% interval is entirely above zero,
  plus log loss and Brier for every model. Every value was re-computed from `matchup_test_predictions.csv` and matches.
- **Graph 4:** top 20 by Giant Killer score (ranked among all 257 qualifiers; the top 20 is identical with no
  minimum). Each row shows actual vs expected upset wins and opportunities. Whiskers = score ± 1.96·se, described
  as an *approximate expected-variation range*, not a 95% CI: it reflects match-to-match chance given the
  probabilities and ignores model-estimation uncertainty. Rows whose range includes 0 are marked (Fritz, Struff).
  Table totals re-computed from `gk_match_predictions.csv` (`p_log_rank`) match to floating-point precision.
- **Flags** (`src/flags.py`, thresholds in config):
  - *comeback* = best rank in prior 3 years was top-30 and at least 3x better than current;
  - *riser* = age ≤23 and rank ~1 year later at least 2x better (uses future rankings: descriptive only;
    undercounted in 2025, which has no next-year ranking);
  - a player is flagged when ≥30% of his opportunities qualify.
  40 comeback and 19 riser players among 257 qualifiers.
- **Robustness:** using only unflagged matches, the leaders are Brooksby, Kokkinakis, Kyrgios, Fritz,
  Rinderknech, Hanfmann and Fucsovics (`gk_score_unflagged_only` in `graph4_chart_data.csv`).
