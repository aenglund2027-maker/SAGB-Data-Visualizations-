"""
Step 7 (Graph 3, part 2): does player-profile / matchup information improve
out-of-sample upset prediction beyond strength + surface?

For each strength measure (log rank, Elo) three nested models are compared on
the SAME Graph 3 matches:
  baseline : strength + surface
  additive : baseline + both players' seven profile variables
  matchup  : additive + 3 pre-specified profile interactions
             (dog_rpw x fav_spw, dog_spw x fav_rpw, dog_rpw x fav_second_won_pct)

Train 2015-2022, test 2023-2025, plus a rolling-origin robustness check.

Input:  data/processed/atp_model_dataset.csv
Output: data/processed/matchup_test_predictions.csv
        data/processed/graph3_results.csv     (headline metrics for the chart)
Run from the project root:  python3 run_07_matchup.py
"""
import numpy as np
import pandas as pd

from src import config
from src.models import (BASELINES, INTERACTIONS, add_interactions, additive_features, calibration_table,
                        chronological_split, coefficients, evaluate, fit_logit, matchup_features, paired_bootstrap)

pd.set_option("display.width", 180)
STRENGTHS = ["log_rank", "elo"]


def model_specs(strength):
    return {"baseline": BASELINES[strength], "additive": additive_features(strength),
            "matchup": matchup_features(strength)}


def fit_predict_all(train, test):
    train, test = add_interactions(train, test)
    models, preds = {}, {}
    for s in STRENGTHS:
        for kind, feats in model_specs(s).items():
            key = f"{s}:{kind}"
            models[key] = fit_logit(train, feats)
            preds[key] = models[key].predict_proba(test[feats])[:, 1]
    return models, preds, test


def per_match_logloss(y, p):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


if __name__ == "__main__":
    d = pd.read_csv(config.PROCESSED_DIR / "atp_model_dataset.csv", parse_dates=["tourney_date"], low_memory=False)
    g3 = d[d["graph3_sample"]].copy()
    train, test = chronological_split(g3, config.TRAIN_SEASONS, config.TEST_SEASONS)
    print("=" * 70)
    print(f"GRAPH 3: train {config.TRAIN_SEASONS[0]}-{config.TRAIN_SEASONS[-1]} (n={len(train):,}), "
          f"test {config.TEST_SEASONS[0]}-{config.TEST_SEASONS[-1]} (n={len(test):,}); upset definition {config.UPSET_DEFINITION!r}")
    print("=" * 70)
    n_missing = g3[additive_features("elo")].isna().sum().sum()
    print(f"Missing feature values in Graph 3 sample: {n_missing}")

    models, preds, test_ix = fit_predict_all(train, test)
    y = test_ix["upset"].to_numpy()
    const = np.full(len(y), train["upset"].mean())

    rows = {"constant (no information)": evaluate(y, const)}
    for k, p in preds.items():
        rows[k] = evaluate(y, p)
    res = pd.DataFrame(rows).T
    # Reader-friendly: share of the constant->model improvement relative to the baseline's improvement
    for s in STRENGTHS:
        base_gain = res.loc["constant (no information)", "log_loss"] - res.loc[f"{s}:baseline", "log_loss"]
        for kind in ("baseline", "additive", "matchup"):
            gain = res.loc["constant (no information)", "log_loss"] - res.loc[f"{s}:{kind}", "log_loss"]
            res.loc[f"{s}:{kind}", "skill_vs_baseline"] = gain / base_gain
    print("\nOUT-OF-SAMPLE (test). skill_vs_baseline = log-loss gain over 'no information', relative to the baseline (1.00)")
    print(res[["n", "log_loss", "brier", "auc", "accuracy", "mean_pred", "skill_vs_baseline"]].round(4).to_string())

    print("\nPaired bootstrap on test (2,000 resamples). improvement > 0 = second model better; 95% CI")
    boot_rows = []
    for s in STRENGTHS:
        for a, b in (("baseline", "additive"), ("baseline", "matchup"), ("additive", "matchup")):
            bt = paired_bootstrap(y, preds[f"{s}:{a}"], preds[f"{s}:{b}"])
            print(f"\n  [{s}] {b} vs {a}:")
            print("  " + bt.round(5).to_string().replace("\n", "\n  "))
            for metric, r in bt.iterrows():
                boot_rows.append({"strength": s, "comparison": f"{b} vs {a}", "metric": metric, **r.to_dict()})

    print("\nCOEFFICIENTS (per 1 SD; positive = raises upset probability). Fit on training seasons.")
    coef = pd.concat({k: coefficients(models[k], model_specs(k.split(':')[0])[k.split(':')[1]])["coef_per_sd"]
                      for k in models if not k.endswith("baseline")}, axis=1)
    print(coef.round(3).to_string())

    print("\nCALIBRATION on test")
    for k in ("log_rank:baseline", "log_rank:matchup", "elo:matchup"):
        print(f"\n  {k}:")
        print("  " + calibration_table(y, preds[k]).round(3).to_string().replace("\n", "\n  "))

    print("\nTest log loss by season:")
    per = pd.DataFrame({k: pd.Series(per_match_logloss(y, p), index=test_ix.index).groupby(test_ix["season"]).mean()
                        for k, p in preds.items()})
    print(per.round(4).to_string())

    print("\nWhere does the gain come from? Test log-loss gain (log_rank: matchup vs baseline) by surface and rank gap")
    gain = per_match_logloss(y, preds["log_rank:baseline"]) - per_match_logloss(y, preds["log_rank:matchup"])
    gdf = test_ix.assign(gain=gain)
    print(gdf.groupby("surface")["gain"].agg(["size", "mean"]).round(4).to_string())
    print(gdf.groupby(pd.cut(gdf["log_rank_diff"], [0, .5, 1, 1.5, 2, 10]), observed=True)["gain"]
          .agg(["size", "mean"]).round(4).to_string())

    # ---------------- diagnostic (post hoc): is the additive gain style or strength? ----------------
    # The additive coefficients look like overall quality (dog spw/rpw both raise, fav spw/rpw both
    # lower the upset odds). Test: replace the 14 profile columns with just two strength-style gaps.
    print("\nPOST-HOC DIAGNOSTIC: two quality gaps (d_spw, d_rpw = dog minus fav) vs all 14 profile columns")
    for s in STRENGTHS:
        gaps = BASELINES[s] + ["d_spw", "d_rpw"]
        p_gaps = fit_logit(train, gaps).predict_proba(test[gaps])[:, 1]
        print(f"  [{s}] log loss: baseline {res.loc[f'{s}:baseline', 'log_loss']:.4f}, "
              f"+2 gaps {evaluate(y, p_gaps)['log_loss']:.4f}, +14 cols {res.loc[f'{s}:additive', 'log_loss']:.4f}")
        bt = paired_bootstrap(y, p_gaps, preds[f"{s}:additive"]).loc["log_loss"]
        print(f"         14 cols vs 2 gaps improvement {bt['improvement']:.5f} (95% CI {bt['ci_low']:.5f} to {bt['ci_high']:.5f})")

    # ---------------- robustness: rolling origin ----------------
    print("\n" + "=" * 70)
    print("ROLLING ORIGIN: train on all seasons before Y, test on Y")
    print("=" * 70)
    roll = []
    for yr in range(2018, max(config.TEST_SEASONS) + 1):
        tr, te = g3[g3["season"] < yr], g3[g3["season"] == yr]
        _, pr, te = fit_predict_all(tr, te)
        yy = te["upset"].to_numpy()
        row = {"season": yr, "n": len(yy)}
        for s in STRENGTHS:
            ll = {kind: log_loss_val for kind, log_loss_val in
                  ((kind, per_match_logloss(yy, pr[f"{s}:{kind}"]).mean()) for kind in ("baseline", "additive", "matchup"))}
            row[f"{s}_base"] = ll["baseline"]
            row[f"{s}_add_gain"] = ll["baseline"] - ll["additive"]
            row[f"{s}_ix_gain"] = ll["additive"] - ll["matchup"]
        roll.append(row)
    roll = pd.DataFrame(roll).set_index("season")
    print(roll.round(4).to_string())
    for s in STRENGTHS:
        print(f"  [{s}] additive beats baseline in {(roll[f'{s}_add_gain'] > 0).sum()}/{len(roll)} seasons; "
              f"interactions beat additive in {(roll[f'{s}_ix_gain'] > 0).sum()}/{len(roll)} seasons")

    # ---------------- save ----------------
    out = test_ix[["match_id", "tourney_date", "season", "tourney_name", "surface", "round", "fav_id", "fav_name",
                   "fav_rank", "dog_id", "dog_name", "dog_rank", "elo_diff", "upset", "score"]].copy()
    for k, p in preds.items():
        out[f"p_{k.replace(':', '_')}"] = p
    out.to_csv(config.PROCESSED_DIR / "matchup_test_predictions.csv", index=False)
    res.to_csv(config.PROCESSED_DIR / "graph3_results.csv")
    pd.DataFrame(boot_rows).to_csv(config.PROCESSED_DIR / "graph3_bootstrap.csv", index=False)
    print("\nSaved -> matchup_test_predictions.csv, graph3_results.csv, graph3_bootstrap.csv (data/processed/)")
