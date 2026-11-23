"""
Graph 4: Giant Killer Score = actual upset wins - expected upset wins.

An "opportunity" is any completed, labelled match a player enters as the
underdog (lower-ranked player under the current upset definition). Expected
upset wins = sum of the baseline model's predicted upset probabilities over
those matches. Predictions are out-of-sample: each season is predicted by a
model trained only on earlier seasons (see models.expanding_window_predictions).

Uncertainty: if each opportunity is an independent coin flip with probability
p_i, the actual win count has variance sum p_i (1 - p_i). So
  se    = sqrt(sum p_i (1 - p_i))
  z     = score / se   (roughly: |z| > 2 is unlikely to be pure luck)
This ignores uncertainty in the model itself, so treat it as a guide, not a test.
"""
import numpy as np
import pandas as pd


def recenter_by_season(probs, outcomes, seasons, tol=1e-10):
    """
    Season re-centering: shift each season's predictions by a constant on the
    log-odds scale so that expected upsets = actual upsets in that season.

    Why: expanding-window models trained on earlier (lower-upset) seasons
    under-predict later seasons, which inflates every player's score. After
    re-centering, scores measure performance relative to a typical underdog in
    the same season, and they sum to zero within each season. A log-odds shift
    keeps every probability in (0, 1) and preserves the ranking of matches.

    This uses each season's total upset count, so the re-centered
    probabilities are descriptive (fine for Giant Killers), not forecasts.
    """
    probs = pd.Series(probs, dtype=float)
    logit = np.log(probs / (1 - probs))
    out = pd.Series(np.nan, index=probs.index)
    for s in pd.unique(seasons):
        mask = (seasons == s).to_numpy()
        target = outcomes[mask].sum()
        lo, hi = -5.0, 5.0
        # bisection for the shift c with sum(sigmoid(logit + c)) = target
        while hi - lo > tol:
            c = (lo + hi) / 2
            if (1 / (1 + np.exp(-(logit[mask] + c)))).sum() < target:
                lo = c
            else:
                hi = c
        out[mask] = 1 / (1 + np.exp(-(logit[mask] + (lo + hi) / 2)))
    return out


def giant_killer_table(matches, prob_col, min_opportunities):
    """
    matches: one row per match with dog_id, dog_name, upset (1 = underdog won)
             and a predicted upset probability column `prob_col`.
    Returns one row per player with at least `min_opportunities` underdog matches.
    """
    m = matches.dropna(subset=[prob_col])
    g = m.groupby("dog_id")
    t = pd.DataFrame({
        "player": g["dog_name"].last(),
        "opportunities": g.size(),
        "upset_wins": g["upset"].sum(),
        "expected_wins": g[prob_col].sum(),
        "variance": g[prob_col].apply(lambda p: (p * (1 - p)).sum()),
        "mean_fav_rank": g["fav_rank"].mean(),
        "first_season": g["season"].min(),
        "last_season": g["season"].max(),
    })
    t["gk_score"] = t["upset_wins"] - t["expected_wins"]
    t["gk_per_100"] = 100 * t["gk_score"] / t["opportunities"]   # extra upsets per 100 underdog matches
    t["actual_rate"] = t["upset_wins"] / t["opportunities"]
    t["expected_rate"] = t["expected_wins"] / t["opportunities"]
    t["se"] = np.sqrt(t["variance"])
    t["z"] = t["gk_score"] / t["se"]
    t = t.drop(columns="variance")
    return t[t["opportunities"] >= min_opportunities].sort_values("gk_score", ascending=False)
