"""
Graph 3 modeling: logistic regressions for P(upset), chronological validation.

Baseline candidates (strength + surface), so the article can use rankings,
Elo, or both:
  rank_raw : rank_diff (dog_rank - fav_rank, ranking places)
  log_rank : log_rank_diff = ln(dog_rank) - ln(fav_rank)
             (the ranking scale is very non-linear: No.1 vs No.10 is a far
              bigger gap than No.101 vs No.110, which a raw difference treats as equal)
  elo      : elo_diff (fav_elo - dog_elo; can be negative when the rank favorite has the lower Elo)
  elo_rank : elo_diff + log_rank_diff
Each also includes surface dummies (Hard = reference).

The MATCHUP model (baseline + seven profile variables + interactions) is
defined here but is NOT evaluated on the test seasons until the profile
features have been signed off.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.profiles import PROFILE_STATS

SURFACE = ["surface_clay", "surface_grass"]
BASELINES = {
    "rank_raw": ["rank_diff"] + SURFACE,
    "log_rank": ["log_rank_diff"] + SURFACE,
    "elo": ["elo_diff"] + SURFACE,
    "elo_rank": ["elo_diff", "log_rank_diff"] + SURFACE,
}

# ---------------------------------------------------------------------------
# Matchup model scaffold (not yet evaluated)
# ---------------------------------------------------------------------------
PROFILE_FEATURES = [f"{side}_prof_{k}" for side in ("dog", "fav") for k in PROFILE_STATS]
# Interactions between the two players' profiles, built from training-period z-scores
INTERACTIONS = {
    "ix_dog_rpw_x_fav_spw": ("dog_prof_rpw", "fav_prof_spw"),        # returner vs server
    "ix_dog_spw_x_fav_rpw": ("dog_prof_spw", "fav_prof_rpw"),        # server vs returner
    "ix_dog_rpw_x_fav_2nd": ("dog_prof_rpw", "fav_prof_second_won_pct"),  # returner vs weak 2nd serve
}


def additive_features(baseline):
    """Baseline + each player's own seven profile variables (no cross-player terms)."""
    return BASELINES[baseline] + PROFILE_FEATURES


def matchup_features(baseline):
    """Additive model + interactions between the two players' profiles."""
    return additive_features(baseline) + list(INTERACTIONS)


def add_interactions(train, test):
    """Products of z-scores using TRAINING means/SDs only (no test-period information)."""
    train, test = train.copy(), test.copy()
    cols = sorted({c for pair in INTERACTIONS.values() for c in pair})
    mu, sd = train[cols].mean(), train[cols].std()
    for d in (train, test):
        z = (d[cols] - mu) / sd
        for name, (a, b) in INTERACTIONS.items():
            d[name] = z[a] * z[b]
    return train, test


# ---------------------------------------------------------------------------
# Fitting / evaluation
# ---------------------------------------------------------------------------
def chronological_split(d, train_seasons, test_seasons):
    return d[d["season"].isin(train_seasons)], d[d["season"].isin(test_seasons)]


def fit_logit(train, features):
    # Mild L2 (C=1) on standardized inputs: ~unpenalized with >10k rows, but keeps
    # correlated profile terms stable once the matchup model is added.
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    model.fit(train[features], train["upset"])
    return model


def evaluate(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {"n": len(y), "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p),
            "auc": roc_auc_score(y, p), "accuracy": np.mean((p > 0.5) == y),
            "mean_pred": p.mean(), "upset_rate": y.mean()}


def calibration_table(y, p, bins=(0, .1, .2, .3, .4, .5, .6, 1)):
    df = pd.DataFrame({"y": y, "p": p})
    return (df.groupby(pd.cut(df["p"], bins), observed=True)
            .agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")))


def coefficients(model, features):
    """Log-odds change per 1 SD of each feature, plus the per-unit coefficient."""
    lr, sc = model.named_steps["logisticregression"], model.named_steps["standardscaler"]
    return pd.DataFrame({"coef_per_sd": lr.coef_[0], "coef_per_unit": lr.coef_[0] / sc.scale_}, index=features)


def paired_bootstrap(y, p_a, p_b, n_boot=2000, seed=0):
    """95% interval for the improvement of model b over model a (positive = b better)."""
    rng = np.random.default_rng(seed)
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    ll = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p))
    d_ll, d_br = ll(p_a) - ll(p_b), (p_a - y) ** 2 - (p_b - y) ** 2
    boots = {"log_loss": [], "brier": [], "auc": []}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        boots["log_loss"].append(d_ll[i].mean())
        boots["brier"].append(d_br[i].mean())
        boots["auc"].append(roc_auc_score(y[i], p_b[i]) - roc_auc_score(y[i], p_a[i]))
    point = {"log_loss": d_ll.mean(), "brier": d_br.mean(), "auc": roc_auc_score(y, p_b) - roc_auc_score(y, p_a)}
    return pd.DataFrame({k: {"improvement": point[k], "ci_low": np.percentile(v, 2.5),
                             "ci_high": np.percentile(v, 97.5)} for k, v in boots.items()}).T


def expanding_window_predictions(d, features, first_pred_season, first_train_season):
    """
    Out-of-sample predictions for every season >= first_pred_season: season Y is
    predicted by a model trained only on seasons first_train_season .. Y-1.
    Used for Graph 4 expected upset wins.
    """
    out = []
    for y in sorted(d["season"].unique()):
        if y < first_pred_season:
            continue
        train = d[(d["season"] >= first_train_season) & (d["season"] < y)]
        test = d[d["season"] == y]
        model = fit_logit(train, features)
        out.append(pd.Series(model.predict_proba(test[features])[:, 1], index=test.index))
    return pd.concat(out)
