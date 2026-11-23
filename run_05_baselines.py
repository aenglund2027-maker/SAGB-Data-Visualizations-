"""
Step 5 (Graph 3, part 1): baseline upset models — ranking vs. Elo — with a
chronological train/test split. The matchup model is intentionally not run yet.

Input:  data/processed/atp_model_dataset.csv
Output: data/processed/baseline_test_predictions.csv
Run from the project root:  python3 run_05_baselines.py
"""
import numpy as np
import pandas as pd

from src import config
from src.models import BASELINES, calibration_table, chronological_split, coefficients, evaluate, fit_logit, paired_bootstrap

pd.set_option("display.width", 160)

if __name__ == "__main__":
    d = pd.read_csv(config.PROCESSED_DIR / "atp_model_dataset.csv", parse_dates=["tourney_date"], low_memory=False)
    g3 = d[d["graph3_sample"]]
    train, test = chronological_split(g3, config.TRAIN_SEASONS, config.TEST_SEASONS)
    y = test["upset"].to_numpy()

    print("=" * 70)
    print(f"GRAPH 3 SAMPLE (upset definition = {config.UPSET_DEFINITION!r}, both profiles ok)")
    print("=" * 70)
    print(f"Train {config.TRAIN_SEASONS[0]}-{config.TRAIN_SEASONS[-1]}: {len(train):,} matches, upset rate {train['upset'].mean():.3f}")
    print(f"Test  {config.TEST_SEASONS[0]}-{config.TEST_SEASONS[-1]}: {len(test):,} matches, upset rate {test['upset'].mean():.3f}")

    models, preds, rows = {}, {}, {}
    for name, feats in BASELINES.items():
        models[name] = fit_logit(train, feats)
        preds[name] = models[name].predict_proba(test[feats])[:, 1]
        rows[name] = evaluate(y, preds[name])
    rows["constant_train_rate"] = evaluate(y, np.full(len(y), train["upset"].mean()))
    rows["raw_elo_formula (no fit)"] = evaluate(y, test["elo_prob_dog"].clip(1e-6, 1 - 1e-6))

    print("\nOUT-OF-SAMPLE (test) — lower log loss / Brier and higher AUC are better")
    print(pd.DataFrame(rows).T[["n", "log_loss", "brier", "auc", "accuracy", "mean_pred", "upset_rate"]].round(4).to_string())

    print("\nPaired bootstrap vs the raw-rank baseline (improvement > 0 = better than rank_raw; 95% CI):")
    for name in ("log_rank", "elo", "elo_rank"):
        print(f"\n  {name} vs rank_raw:")
        print("  " + paired_bootstrap(y, preds["rank_raw"], preds[name]).round(5).to_string().replace("\n", "\n  "))
    print("\n  elo vs log_rank:")
    print("  " + paired_bootstrap(y, preds["log_rank"], preds["elo"]).round(5).to_string().replace("\n", "\n  "))

    print("\nCOEFFICIENTS (fit on training seasons)")
    for name, feats in BASELINES.items():
        print(f"\n  {name}:")
        print("  " + coefficients(models[name], feats).round(5).to_string().replace("\n", "\n  "))

    print("\nCALIBRATION on test (predicted vs observed upset rate)")
    for name in ("rank_raw", "log_rank", "elo"):
        print(f"\n  {name}:")
        print("  " + calibration_table(y, preds[name]).round(3).to_string().replace("\n", "\n  "))

    print("\nBy test season (log loss):")
    per = pd.DataFrame({name: pd.Series(-(y * np.log(p) + (1 - y) * np.log(1 - p)), index=test.index)
                        .groupby(test["season"]).mean() for name, p in preds.items()})
    print(per.round(4).to_string())

    # Context: same baselines on ALL labelled matches (incl. thin profiles)
    tr_all, te_all = chronological_split(d, config.TRAIN_SEASONS, config.TEST_SEASONS)
    print(f"\nContext — baselines on ALL labelled matches (incl. thin profiles; test n={len(te_all):,}):")
    for name, feats in BASELINES.items():
        e = evaluate(te_all["upset"], fit_logit(tr_all, feats).predict_proba(te_all[feats])[:, 1])
        print(f"  {name:<9s} log loss {e['log_loss']:.4f}  Brier {e['brier']:.4f}  AUC {e['auc']:.4f}")

    out = test[["match_id", "tourney_date", "season", "tourney_name", "surface", "round", "fav_name", "dog_name",
                "fav_rank", "dog_rank", "elo_diff", "upset", "score"]].copy()
    for name, p in preds.items():
        out[f"p_{name}"] = p
    path = config.PROCESSED_DIR / "baseline_test_predictions.csv"
    out.to_csv(path, index=False)
    print(f"\nSaved -> {path}")
